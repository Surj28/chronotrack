"""Joint cell-vessel tracking, step 2a: a temporally consistent vascular scaffold.

Per-frame skeletons are unstable: about half of the skeleton moves more than
3 px per hour although consecutive vessel masks overlap at Dice 0.97
(results/vessel_motion). So the vessel network is estimated over the whole
sequence, not frame by frame:

  1. registration switch  cumulative deformation T1 -> T_last (TV-L1 on red,
                          drift removed, p95 on the T1 skeleton). Frames are
                          registered to T1 only if it exceeds half the median
                          vessel radius (a smaller shift keeps every centreline
                          inside its own vessel). Measured, never assumed.
  2. temporal consensus   c_t = majority of the masks t-2..t+2 (drops 1-2 frame noise)
  3. union scaffold       U = OR_t c_t: every vessel present for >= 3 frames;
                          boundary smoothed (Gaussian sigma 2, > 0.5)
  4. centreline graph     skeleton of U; junctions by crossing number; terminal
                          branches shorter than the local vessel diameter
                          (2 x radius at their junction) are pruned as boundary
                          spurs (3 passes); remaining branches = segments with
                          persistent ids; nodes = junction clusters
  5. presence             p_s(t) = fraction of segment s's centreline covered by
                          the raw mask of frame t
  6. remodelling          p_s median-filtered over 3 frames; one change point
                          (>= 3 frames each side). stable: present, |step| < 0.5;
                          regressing: step <= -0.5; emerging: step >= +0.5;
                          intermittent: mostly absent, or > 2 threshold crossings
  6b. red corroboration   r_s(t) = median red along the centreline, scaled per
                          frame (gel background 0, median vessel 1). A regressing
                          or emerging segment is confirmed only if r_s changes the
                          same way at its change point by more than 3x the
                          frame-to-frame noise of stable segments in that well;
                          otherwise it is 'mask-only' (likely a segmentation
                          artefact, e.g. a lumen hole flipping between filled
                          and unfilled)
  7. reproducibility      scaffolds from even and from odd frames, compared
                          (centreline within 3 px, both directions)

Outputs per well (<out>/<well>/): segment_ids.tif, territory_ids.tif,
union_mask.tif, segments.csv, nodes.csv, presence.csv, summary.json, and
<out>/fig_<well>_scaffold.png, <out>/fig_<well>_events.png.

Usage (from the repo root):
    python paper/code/vessel_scaffold.py --crops paper/results/roi_crops \
        --seg paper/results/segmentation --out paper/results/vessel_scaffold
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from skimage.registration import optical_flow_tvl1

from vessel_motion import DS, down, norm_red, warp

UM_PER_PX = 0.65
CLASSES = ["stable", "regressing", "emerging", "mask-only", "intermittent"]
# reference palette: categorical slots 1-4 in fixed order, muted grey for the residual class
COLOURS = {"stable": "#2a78d6", "regressing": "#eb6834", "emerging": "#1baf7a",
           "mask-only": "#eda100", "intermittent": "#9a9a94"}


# ---------------------------------------------------------------- inputs

def load_well(crops: Path, seg: Path, well: str):
    meta = json.loads((crops / well / "roi.json").read_text())
    roi = (tifffile.imread(crops / well / "roi_mask.tif") > 0) & \
          ~(tifffile.imread(crops / well / "pillar_mask.tif") > 0)
    ts = meta["timepoints"]
    V = np.stack([(tifffile.imread(seg / well / f"vessels_T{t:04d}.tif") > 0) & roi for t in ts])
    return ts, roi, V


def red_ds(crops: Path, well: str, t: int, roi):
    red = norm_red(tifffile.imread(crops / well / f"T{t:04d}_C1.tif").astype(np.float32), roi)
    return ndi.gaussian_filter(down(red * roi), 1)


def full_flow(ref_ds, mov_ds, shape):
    v, u = optical_flow_tvl1(ref_ds, mov_ds)
    flow = np.stack([ndi.zoom(v, DS, order=1), ndi.zoom(u, DS, order=1)]) * DS
    return np.pad(flow, ((0, 0), (0, shape[0] - flow.shape[1]), (0, shape[1] - flow.shape[2])), mode="edge")


# Skeleton-graph and scaffold geometry live in topology_tracking/scaffold.py
from topology_tracking.scaffold import (NB8, RING, branch_lengths, build_scaffold,  # noqa: E402,F401
                                        centreline_agreement, crossing_number, prune,
                                        prune_short, skeleton_parts)


# ---------------------------------------------------------------- remodelling

def change_points(P, min_side=3):
    """Best single step for each row of P (segments x frames): returns k, mean before, after."""
    n, T = P.shape
    S1 = np.cumsum(P, 1); S2 = np.cumsum(P ** 2, 1)
    best = np.full(n, np.inf); kb = np.zeros(n, int); mb = np.zeros(n); ma = np.zeros(n)
    for k in range(min_side, T - min_side + 1):
        s1, q1 = S1[:, k - 1], S2[:, k - 1]
        s2, q2 = S1[:, -1] - s1, S2[:, -1] - q1
        sse = q1 - s1 ** 2 / k + q2 - s2 ** 2 / (T - k)
        better = sse < best
        best[better] = sse[better]; kb[better] = k
        mb[better] = (s1 / k)[better]; ma[better] = (s2 / (T - k))[better]
    return kb, mb, ma


def classify(P_smooth, R, max_flips=2, noise_mult=3.0):
    """Presence classes, then red corroboration of each remodelling event.

    R: segments x frames, red along the centreline (background 0, vessel 1).
    """
    k, before, after = change_points(P_smooth)
    step = after - before
    mean_p = P_smooth.mean(1)
    flips = (np.diff((P_smooth >= 0.5).astype(int), axis=1) != 0).sum(1)
    cls = np.full(len(P_smooth), "intermittent", dtype=object)
    cls[(np.abs(step) < 0.5) & (mean_p >= 0.5)] = "stable"
    cls[(step <= -0.5) & (before >= 0.5)] = "regressing"
    cls[(step >= 0.5) & (after >= 0.5)] = "emerging"
    cls[(flips > max_flips) & (cls != "stable")] = "intermittent"
    # red change at the same change point
    T = R.shape[1]
    idx = np.arange(T)[None, :]
    r_before = np.nanmean(np.where(idx < k[:, None], R, np.nan), 1)
    r_after = np.nanmean(np.where(idx >= k[:, None], R, np.nan), 1)
    r_step = r_after - r_before
    # per-frame red noise of stable segments (robust SD of frame-to-frame differences / sqrt 2);
    # a step between the means of k and T-k frames has SD sigma * sqrt(1/k + 1/(T-k))
    stable = cls == "stable"
    sigma = float(np.nanmedian(np.abs(np.diff(R[stable], axis=1)))) * 1.4826 / np.sqrt(2) if stable.any() else np.nan
    thr = noise_mult * sigma * np.sqrt(1.0 / np.maximum(k, 1) + 1.0 / np.maximum(T - k, 1))
    unconfirmed = ((cls == "regressing") & ~(r_step <= -thr)) | ((cls == "emerging") & ~(r_step >= thr))
    cls[unconfirmed] = "mask-only"
    return cls, k, step, flips, r_step, sigma


# ---------------------------------------------------------------- figures

def gray(img, roi, lo_p=1, hi_p=99.5):
    lo, hi = np.percentile(img[roi], [lo_p, hi_p])
    return np.clip((img - lo) / (hi - lo + 1e-6), 0, 1)


def draw_classes(base, seg_ids, seg_class, scale, thick=7):
    rgb = np.repeat(base[::scale, ::scale, None], 3, axis=2) * 0.85
    for c in CLASSES:
        ids = [s for s, k in seg_class.items() if k == c]
        if not ids:
            continue
        m = ndi.binary_dilation(np.isin(seg_ids, ids), iterations=thick // 2)[::scale, ::scale]
        rgb[m] = matplotlib.colors.to_rgb(COLOURS[c])
    return rgb


def figure_scaffold(path, well, red_first, red_last, roi, seg_ids, seg_class, ts, summary):
    scale = 3
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.9))
    for ax, img, lab in ((axes[0], red_first, f"T{ts[0]:04d}"), (axes[1], red_last, f"T{ts[-1]:04d}")):
        ax.imshow(draw_classes(gray(img, roi), seg_ids, seg_class, scale), interpolation="nearest")
        ax.set_title(f"{well}  {lab}  (red channel, scaffold coloured by class)", fontsize=9, loc="left")
        ax.set_axis_off()
    fr = summary["length_fraction"]
    handles = [plt.Line2D([], [], color=COLOURS[c], lw=3, label=f"{c}  {fr[c]:.0%} of length") for c in CLASSES]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def figure_events(path, well, crops, ts, roi, seg_ids, seg_df, P_raw, R_raw):
    ev = seg_df[seg_df["class"].isin(["regressing", "emerging", "mask-only"])].sort_values("length_um", ascending=False)
    ev = pd.concat([ev[ev["class"] == "regressing"].head(4), ev[ev["class"] == "emerging"].head(2),
                    ev[ev["class"] == "mask-only"].head(3)])
    if ev.empty:
        return
    frames = [ts[0], ts[len(ts) // 2], ts[-1]]
    reds = {t: tifffile.imread(crops / well / f"T{t:04d}_C1.tif").astype(np.float32) for t in frames}
    fig, axes = plt.subplots(len(ev), 4, figsize=(10, 2.5 * len(ev)),
                             gridspec_kw=dict(width_ratios=[1, 1, 1, 1.4]))
    axes = np.atleast_2d(axes)
    for row, (_, s) in zip(axes, ev.iterrows()):
        r, c, half = int(s.centroid_r), int(s.centroid_c), 180
        sl = (slice(max(r - half, 0), r + half), slice(max(c - half, 0), c + half))
        seg = ndi.binary_dilation(seg_ids[sl] == s.segment_id, iterations=2)
        for ax, t in zip(row[:3], frames):
            g = gray(reds[t], roi)[sl]
            rgb = np.repeat(g[..., None], 3, 2)
            edge = seg & ~ndi.binary_erosion(seg)
            rgb[edge] = matplotlib.colors.to_rgb(COLOURS[s["class"]])
            ax.imshow(rgb, interpolation="nearest"); ax.set_axis_off()
            ax.set_title(f"T{t:04d}", fontsize=8)
        ax = row[3]
        ax.plot(ts, P_raw[int(s.segment_id)], color=COLOURS[s["class"]], lw=2, label="mask presence")
        ax.plot(ts, R_raw[int(s.segment_id)], color="#1f1f1e", lw=1.5, ls="--", label="red signal (gel 0, vessel 1)")
        ax.axhline(0.5, color="#c9c8c1", lw=0.8)
        ax.set_ylim(-0.05, max(1.05, float(np.nanmax(R_raw[int(s.segment_id)])) + 0.05)); ax.set_xlim(ts[0], ts[-1])
        if row is axes[0]:
            ax.legend(fontsize=7, frameon=False, loc="lower left")
        ax.set_title(f"segment {int(s.segment_id)}: {s['class']} at ~T{int(s.change_t):02d}, "
                     f"{s.length_um:.0f} µm", fontsize=8, loc="left")
        ax.set_ylabel("fraction", fontsize=8); ax.tick_params(labelsize=7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle(f"{well}: largest remodelling events (red channel; outlined segment)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------- main

def process_well(crops: Path, seg: Path, out: Path, well: str):
    t0 = time.time()
    ts, roi, V = load_well(crops, seg, well)
    T = len(ts)
    od = out / well
    od.mkdir(parents=True, exist_ok=True)

    # 1. registration switch
    sk0 = skeletonize(V[0])
    r_med = float(np.median(ndi.distance_transform_edt(V[0])[sk0]))
    flow = full_flow(red_ds(crops, well, ts[0], roi), red_ds(crops, well, ts[-1], roi), roi.shape)
    drift = np.array([np.median(flow[0][V[0]]), np.median(flow[1][V[0]])])
    cum = np.hypot(flow[0][sk0] - drift[0], flow[1][sk0] - drift[1])
    cum_p95 = float(np.percentile(cum, 95))
    registered = cum_p95 > 0.5 * r_med
    flows = [None] * T
    if registered:
        ref = red_ds(crops, well, ts[0], roi)
        ds_flows = np.zeros((T, 2) + ref.shape, np.float16)
        for i in range(1, T):
            v, u = optical_flow_tvl1(ref, red_ds(crops, well, ts[i], roi))
            ds_flows[i] = np.stack([v, u])
            f = np.stack([ndi.zoom(v, DS, order=1), ndi.zoom(u, DS, order=1)]) * DS
            flows[i] = np.pad(f, ((0, 0), (0, roi.shape[0] - f.shape[1]), (0, roi.shape[1] - f.shape[2])), mode="edge")
            V[i] = (warp(V[i], flows[i]) > 0.5) & roi
        # flow at downsampled resolution (x DS px): point x in T1 sits at x + DS*flow in frame t
        np.savez_compressed(od / "flows_to_T1.npz", flows=ds_flows, ds=DS, timepoints=np.array(ts))

    # 2-4. scaffold
    U, skel, radius = build_scaffold(V, roi)
    blab, n, jlab, nj, endpoints, pairs = skeleton_parts(skel)
    L = branch_lengths(blab, n)
    ids = np.arange(1, n + 1)
    cents = np.array(ndi.center_of_mass(blab > 0, blab, ids)) if n else np.zeros((0, 2))
    r_mean = np.asarray(ndi.mean(radius, blab, ids)) if n else np.zeros(0)
    ends = {b: [] for b in ids}
    for b, j in pairs:
        ends[b].append(int(j))
    _, territory_idx = ndi.distance_transform_edt(blab == 0, return_indices=True)
    territory = np.where(U, blab[territory_idx[0], territory_idx[1]], 0).astype(np.int32)

    # 5. presence, and red along each centreline (frame-scaled: gel 0, vessel 1)
    P = np.stack([np.asarray(ndi.mean(V[t].astype(np.float32), blab, ids)) for t in range(T)], 1) \
        if n else np.zeros((0, T))
    R = np.zeros((n, T))
    for i, t in enumerate(ts):
        red = tifffile.imread(crops / well / f"T{t:04d}_C1.tif").astype(np.float32)
        if flows[i] is not None:
            red = warp(red, flows[i], order=1)
        bg = np.median(red[roi & ~V[i]]); ves = np.median(red[V[i]])
        R[:, i] = (np.asarray(ndi.median(red, blab, ids)) - bg) / max(ves - bg, 1e-6)
    P_s = ndi.median_filter(P, size=(1, 3), mode="nearest")
    cls, kcp, step, flips, r_step, red_thr = classify(P_s, R)

    seg_df = pd.DataFrame(dict(
        segment_id=ids, length_um=L[1:] * UM_PER_PX, mean_radius_um=r_mean * UM_PER_PX,
        centroid_r=cents[:, 0] if n else [], centroid_c=cents[:, 1] if n else [],
        node_a=[ends[b][0] if len(ends[b]) > 0 else -1 for b in ids],
        node_b=[ends[b][1] if len(ends[b]) > 1 else -1 for b in ids],
        n_junctions=[len(ends[b]) for b in ids],
        mean_presence=P.mean(1), step=step, change_t=[ts[min(k, T - 1)] for k in kcp],
        flips=flips, red_step=r_step, **{"class": cls}))
    jids = np.arange(1, nj + 1)
    jc = np.array(ndi.center_of_mass(jlab > 0, jlab, jids)) if nj else np.zeros((0, 2))
    nodes_df = pd.DataFrame(dict(node_id=jids, r=jc[:, 0] if nj else [], c=jc[:, 1] if nj else [],
                                 degree=np.bincount(pairs[:, 1], minlength=nj + 1)[1:] if nj else []))

    # stability: per-hour segment persistence (raw presence, length-weighted)
    present = P >= 0.5
    w = L[1:][:, None]
    seg_persist = float((w * (present[:, :-1] & present[:, 1:])).sum() / max((w * present[:, :-1]).sum(), 1e-9))
    cl = skel
    cov = [V[t][cl] for t in range(T)]
    pix_persist = float(np.mean([(cov[t] & cov[t + 1]).sum() / max(cov[t].sum(), 1) for t in range(T - 1)]))

    # 7. reproducibility: even vs odd frames
    _, sk_even, _ = build_scaffold(V[0::2], roi)
    _, sk_odd, _ = build_scaffold(V[1::2], roi)
    agree_eo, agree_oe = centreline_agreement(sk_even, sk_odd, 3.0)
    # per-frame skeleton baseline at the same tolerance (consecutive frames, every 4th pair)
    base = [centreline_agreement(skeletonize(V[i]), skeletonize(V[i + 1]), 3.0) for i in range(0, T - 1, 4)]

    tot = seg_df.length_um.sum()
    summary = dict(
        well=well, n_frames=T, median_vessel_radius_um=r_med * UM_PER_PX,
        cumulative_deformation_p95_um=cum_p95 * UM_PER_PX, registered=bool(registered),
        n_segments=int(n), n_junctions=int(nj), total_length_mm=float(tot / 1000),
        length_fraction={c: float(seg_df.loc[seg_df["class"] == c, "length_um"].sum() / max(tot, 1e-9)) for c in CLASSES},
        n_by_class={c: int((seg_df["class"] == c).sum()) for c in CLASSES},
        segment_persistence_per_hour=seg_persist, centreline_coverage_persistence_per_hour=pix_persist,
        split_half_agreement_3px=[agree_eo, agree_oe],
        per_frame_skeleton_agreement_3px=float(np.mean([np.mean(b) for b in base])),
        red_noise_per_frame=float(red_thr),
        union_cover_of_roi=float(U[roi].mean()), seconds=round(time.time() - t0, 1))

    tifffile.imwrite(od / "segment_ids.tif", blab.astype(np.int32), compression="zlib")
    tifffile.imwrite(od / "territory_ids.tif", territory, compression="zlib")
    tifffile.imwrite(od / "union_mask.tif", U.astype(np.uint8), compression="zlib")
    seg_df.to_csv(od / "segments.csv", index=False)
    nodes_df.to_csv(od / "nodes.csv", index=False)
    pd.DataFrame(P, index=pd.Index(ids, name="segment_id"), columns=[f"T{t:04d}" for t in ts]).to_csv(od / "presence.csv")
    pd.DataFrame(R, index=pd.Index(ids, name="segment_id"), columns=[f"T{t:04d}" for t in ts]).to_csv(od / "red_profile.csv")
    (od / "summary.json").write_text(json.dumps(summary, indent=2))

    seg_class = dict(zip(seg_df.segment_id, seg_df["class"]))
    red_first = tifffile.imread(crops / well / f"T{ts[0]:04d}_C1.tif").astype(np.float32)
    red_last = tifffile.imread(crops / well / f"T{ts[-1]:04d}_C1.tif").astype(np.float32)
    figure_scaffold(out / f"fig_{well}_scaffold.png", well, red_first, red_last, roi, blab, seg_class, ts, summary)
    P_full = np.zeros((n + 1, T)); P_full[1:] = P
    R_full = np.zeros((n + 1, T)); R_full[1:] = R
    figure_events(out / f"fig_{well}_events.png", well, crops, ts, roi, blab, seg_df, P_full, R_full)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", type=Path, required=True)
    ap.add_argument("--seg", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--wells", nargs="+", default=None)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    wells = args.wells or sorted([p.name for p in args.crops.glob("ARTORG_w*")], key=lambda s: int(s.split("_")[-1]))
    for well in wells:
        s = process_well(args.crops, args.seg, args.out, well)
        print(json.dumps(s), flush=True)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------- generic (any sequence)

def segments_table(V: np.ndarray, vessel_img: np.ndarray, scaffold, roi: np.ndarray | None = None,
                   um_per_px: float = UM_PER_PX):
    """Segments, presence / red profiles and remodelling classes for any
    sequence, from its per-frame vessel masks V (T, H, W), vessel channel and
    Scaffold (topology_tracking.scaffold). Same rules as process_well.
    Returns (segments DataFrame, territory id map, presence (n, T), red (n, T))."""
    from topology_tracking.scaffold import _full_flow, _warp
    T = len(V)
    roi = np.ones(V.shape[1:], bool) if roi is None else roi
    blab, n, jlab, nj, endpoints, pairs = skeleton_parts(scaffold.skeleton)
    L = branch_lengths(blab, n)
    ids = np.arange(1, n + 1)
    _, ti = ndi.distance_transform_edt(blab == 0, return_indices=True)
    territory = np.where(scaffold.union, blab[ti[0], ti[1]], 0).astype(np.int32)
    P = np.zeros((n, T)); R = np.zeros((n, T))
    for t in range(T):
        v, img = V[t], vessel_img[t].astype(np.float32)
        if scaffold.flows is not None and t > 0:
            f = _full_flow(scaffold.flows[t].astype(np.float32), scaffold.flow_ds, v.shape)
            v, img = _warp(v, f) > 0.5, _warp(img, f, order=1)
        P[:, t] = ndi.mean(v.astype(np.float32), blab, ids)
        bg, ves = np.median(img[roi & ~v]), np.median(img[v]) if v.any() else 1.0
        R[:, t] = (np.asarray(ndi.median(img, blab, ids)) - bg) / max(ves - bg, 1e-6)
    cls, kcp, step, flips, r_step, _ = classify(ndi.median_filter(P, size=(1, 3), mode="nearest"), R)
    seg = pd.DataFrame(dict(segment_id=ids, length_um=L[1:] * um_per_px, mean_presence=P.mean(1), step=step,
                            change_t=kcp, flips=flips, red_step=r_step, **{"class": cls}))
    return seg, territory, P, R


def adaptive_windows(V: np.ndarray, vessel_img: np.ndarray, roi: np.ndarray | None = None, ds: int = 4,
                     um_per_px: float = UM_PER_PX, stable_min: float = 0.5, min_frames: int = 12) -> list[dict]:
    """Temporal support of the scaffold, chosen from the data. The whole
    sequence keeps one scaffold if at least `stable_min` of its centreline
    length is classed stable; otherwise each half is tried, recursively
    (halves shorter than `min_frames` are not split further). A persistent
    network (ARTORG, Song) keeps one window; a forming or remodelling network
    (Hachey) gets several. Each window is built exactly like a whole sequence
    (topology_tracking.scaffold.scaffold_from_stack, including its registration
    switch; segments_table). Returns one dict per window: start, stop,
    scaffold, info, segments, territory, P, R, stable_frac."""
    from topology_tracking.scaffold import scaffold_from_stack

    def build(a, b):
        sc, info = scaffold_from_stack(V[a:b], vessel_img[a:b], roi, ds)
        seg, terr, P, R = segments_table(V[a:b], vessel_img[a:b], sc, roi, um_per_px)
        stable = float(seg.loc[seg["class"] == "stable", "length_um"].sum() / max(seg.length_um.sum(), 1e-9))
        print(f"  window {a}-{b}: stable {stable:.2f}, registered {info['registered']}", flush=True)
        return dict(start=a, stop=b, scaffold=sc, info=info, segments=seg, territory=terr, P=P, R=R,
                    stable_frac=stable)

    def split(a, b):
        w = build(a, b)
        if w["stable_frac"] >= stable_min or (b - a) // 2 < min_frames:
            return [w]
        m = a + (b - a) // 2
        return split(a, m) + split(m, b)

    return split(0, len(V))
