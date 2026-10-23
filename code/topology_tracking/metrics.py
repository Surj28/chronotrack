"""Segmentation and tracking scores against annotated label maps.

Ground truth: one label map per frame in which a cell keeps the same id in
every frame (CTC "TRA" convention). Prediction: one label map per frame plus,
per frame, a {label: track_id} map from the tracker's output.

Matching per frame, one-to-one (Hungarian):
    mode="iou"     IoU >= 0.5 (our annotated crops: full masks)
    mode="marker"  prediction covers > 50% of the ground-truth object (CTC
                   tracking ground truth is markers, not full masks)

The Cell Tracking Challenge measures (SEG, DET, TRA, LNK, BIO, HOTA, IDF1,
MOTA, ...) come from ctc_eval.py, which runs the py-ctcmetrics package on
CTC-format exports. This module adds what the CTC does not report:
segmentation F1 / mean matched IoU, identity switches, fragmentations,
per-track completeness, link accuracy with a per-link ambiguity ratio, and
the measured error rates that calibrate the feature reliability curves.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def _ids_areas(lab: np.ndarray):
    ids, areas = np.unique(lab[lab > 0], return_counts=True)
    return ids.astype(np.int64), areas


def similarity(gt: np.ndarray, pred: np.ndarray, mode: str = "iou"):
    """(gt_ids, pred_ids, S) with S[i, j] the IoU (or marker coverage)."""
    g_ids, g_area = _ids_areas(gt)
    p_ids, p_area = _ids_areas(pred)
    S = np.zeros((len(g_ids), len(p_ids)))
    m = (gt > 0) & (pred > 0)
    if m.any() and len(g_ids) and len(p_ids):
        gi = np.searchsorted(g_ids, gt[m])
        pj = np.searchsorted(p_ids, pred[m])
        inter = np.zeros_like(S)
        np.add.at(inter, (gi, pj), 1)
        if mode == "iou":
            S = inter / (g_area[:, None] + p_area[None, :] - inter)
        elif mode == "marker":
            S = inter / g_area[:, None]
        else:
            raise ValueError(mode)
    return g_ids, p_ids, S


def _match(S: np.ndarray, mode: str):
    """Row/col indices of one-to-one matches."""
    if S.size == 0:
        return np.array([], int), np.array([], int)
    ok = S >= 0.5 if mode == "iou" else S > 0.5
    r, c = linear_sum_assignment(-np.where(ok, S, 0))
    keep = ok[r, c]
    return r[keep], c[keep]


def _centroids(lab: np.ndarray, ids: np.ndarray) -> np.ndarray:
    from scipy import ndimage as ndi
    if not len(ids):
        return np.zeros((0, 2))
    return np.array(ndi.center_of_mass(lab > 0, lab, ids)).reshape(-1, 2)


def evaluate(gt_frames: list, pred_frames: list, label_to_track: list, mode: str = "iou") -> dict:
    """Score one sequence. `label_to_track[t]` maps prediction labels in
    frame t to track ids; labels missing from it count as one-frame tracks.

    Returns a dict with 'scores' (dict), 'links' (DataFrame, one row per
    ground-truth continuation), 'per_gt' (DataFrame), 'rates' (measured
    error rates) and 'assign' ({gt id: track id} from the IDF1 matching)."""
    T = len(gt_frames)
    frames = []
    for t in range(T):
        g_ids, p_ids, S = similarity(gt_frames[t], pred_frames[t], mode)
        l2t = label_to_track[t]
        trk = np.array([l2t.get(int(p), -(t * 10_000_000 + int(p)) - 1) for p in p_ids], dtype=np.int64)
        r, c = _match(S, mode)
        frames.append(dict(g_ids=g_ids, p_ids=p_ids, trk=trk, S=S, r=r, c=c,
                           g_cent=_centroids(gt_frames[t], g_ids)))

    # -- detection ---------------------------------------------------------
    TP = sum(len(f["r"]) for f in frames)
    n_gt = sum(len(f["g_ids"]) for f in frames)
    n_pr = sum(len(f["p_ids"]) for f in frames)
    FN, FP = n_gt - TP, n_pr - TP
    seg_f1 = 2 * TP / (2 * TP + FN + FP) if (TP + FN + FP) else np.nan
    matched_sim = np.concatenate([f["S"][f["r"], f["c"]] for f in frames]) if TP else np.array([])

    # -- identity: IDF1 ----------------------------------------------------
    match_rows = []  # (t, gt_id, track_id)
    for t, f in enumerate(frames):
        for i, j in zip(f["r"], f["c"]):
            match_rows.append((t, int(f["g_ids"][i]), int(f["trk"][j])))
    M = pd.DataFrame(match_rows, columns=["t", "gt", "trk"])
    gt_count = pd.Series(np.concatenate([f["g_ids"] for f in frames])).value_counts()
    tr_count = pd.Series(np.concatenate([f["trk"] for f in frames])).value_counts()
    assign = {}
    idtp = 0
    if len(M):
        C = M.groupby(["gt", "trk"]).size()
        gts = C.index.get_level_values(0).unique()
        trs = C.index.get_level_values(1).unique()
        Cm = np.zeros((len(gts), len(trs)))
        Cm[gts.get_indexer(C.index.get_level_values(0)), trs.get_indexer(C.index.get_level_values(1))] = C.to_numpy()
        r, c = linear_sum_assignment(-Cm)
        for i, j in zip(r, c):
            if Cm[i, j] > 0:
                assign[int(gts[i])] = int(trs[j])
                idtp += Cm[i, j]
    idf1 = 2 * idtp / (gt_count.sum() + tr_count.sum()) if gt_count.sum() else np.nan

    # -- per ground-truth track: switches, fragmentations, completeness ---
    per_gt = []
    present = {}
    for t, f in enumerate(frames):
        for g in f["g_ids"]:
            present.setdefault(int(g), []).append(t)
    matched_at = {(t, g): k for t, g, k in match_rows}
    for g, ts in present.items():
        seq = [matched_at.get((t, g)) for t in ts]
        sw, last = 0, None
        for k in seq:
            if k is not None:
                if last is not None and k != last:
                    sw += 1
                last = k
        on = [k is not None for k in seq]
        # a fragmentation each time a matched run resumes after a miss
        frag = 0
        started = False
        for a, b in zip(on, on[1:]):
            started = started or a
            if started and not a and b:
                frag += 1
        comp = sum(1 for k in seq if k is not None and k == assign.get(g)) / len(ts)
        per_gt.append(dict(gt=g, n_frames=len(ts), switches=sw, fragmentations=frag, completeness=comp))
    per_gt = pd.DataFrame(per_gt)

    # -- links with ambiguity ----------------------------------------------
    links = []
    for t in range(T - 1):
        a, b = frames[t], frames[t + 1]
        pos_b = {int(g): p for g, p in zip(b["g_ids"], b["g_cent"])}
        for g, p in zip(a["g_ids"], a["g_cent"]):
            g = int(g)
            if g not in pos_b:
                continue
            d_true = float(np.linalg.norm(pos_b[g] - p))
            others = [q for h, q in pos_b.items() if h != g]
            d_other = float(np.min(np.linalg.norm(np.array(others) - p, axis=1))) if others else np.inf
            ka, kb = matched_at.get((t, g)), matched_at.get((t + 1, g))
            links.append(dict(t=t, gt=g, d_true=d_true, d_other=d_other,
                              ambiguity=d_other / d_true if d_true > 0 else np.inf,
                              checked=ka is not None and kb is not None,
                              correct=ka is not None and ka == kb))
    links = pd.DataFrame(links)

    # -- measured error rates (calibrate the reliability curves) ----------
    n_links = int(links["checked"].sum()) if len(links) else 0
    rates = dict(
        miss_rate=FN / n_gt if n_gt else np.nan,
        switch_rate=float(per_gt["switches"].sum()) / n_links if n_links else np.nan,
    )
    if mode == "iou" and TP:
        d_r = []
        for f, t in zip(frames, range(T)):
            if not len(f["r"]):
                continue
            ga = _ids_areas(gt_frames[t])[1][f["r"]]
            pa = _ids_areas(pred_frames[t])[1][f["c"]]
            d_r.append(np.sqrt(pa / np.pi) - np.sqrt(ga / np.pi))
        rates["radius_error_px"] = float(np.median(np.concatenate(d_r)))

    scores = dict(
        seg_f1=seg_f1, mean_matched_sim=float(matched_sim.mean()) if TP else np.nan,
        TP=TP, FP=FP, FN=FN, idf1=idf1,
        id_switches=int(per_gt["switches"].sum()) if len(per_gt) else 0,
        fragmentations=int(per_gt["fragmentations"].sum()) if len(per_gt) else 0,
        median_completeness=float(per_gt["completeness"].median()) if len(per_gt) else np.nan,
        link_accuracy=float(links.loc[links["checked"], "correct"].mean()) if n_links else np.nan,
        n_links_checked=n_links,
    )
    return dict(scores=scores, links=links, per_gt=per_gt, rates=rates, assign=assign)


def label_to_track_maps(tracks: pd.DataFrame, n_frames: int) -> list[dict]:
    """Per-frame {label: track_id} from a tracker output table."""
    maps = [dict() for _ in range(n_frames)]
    for t, lb, k in tracks[["frame_idx", "label", "track_id"]].itertuples(index=False):
        if 0 <= t < n_frames:
            maps[int(t)][int(lb)] = int(k)
    return maps
