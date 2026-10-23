"""Cell Tracking Challenge (CTC) format export and official measures.

The CTC measures (https://celltrackingchallenge.net/evaluation-methodology/)
are computed with py-ctcmetrics, which reproduces the CTC evaluation
software: SEG, DET, TRA, LNK, BIO (CT, TF, BC, CCA), OP_CSB, OP_CTB, OP_CLB,
plus HOTA, IDF1 and MOTA.

CTC layout:
    <res>/mask000.tif ...       one uint16 label map per frame, label = track
    <res>/res_track.txt         "L B E P" per track (label, begin, end, parent)
    <gt>/TRA/man_track000.tif   ground-truth tracking labels (+ man_track.txt)
    <gt>/SEG/man_seg000.tif     ground-truth masks on the annotated frames

A CTC track must be continuous in time: a track with a gap is split, and
the part after the gap becomes a new label whose parent is the part before
it (the CTC convention for gap closing).
"""
from __future__ import annotations

import contextlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff

from .tracker import _load_array


def _split_at_gaps(presence: dict[int, list[int]], parents: dict[int, int] | None = None):
    """presence: {label: sorted frames present}. Returns (segments, lookup):
    segments = list of (new_label, begin, end, parent); lookup[(label, t)]
    = new label at frame t."""
    parents = parents or {}
    segs, lookup, last_seg_of = [], {}, {}
    next_label = 1
    order = sorted(presence, key=lambda lb: presence[lb][0])
    for lb in order:
        ts = presence[lb]
        runs, start = [], ts[0]
        for a, b in zip(ts, ts[1:]):
            if b != a + 1:
                runs.append((start, a))
                start = b
        runs.append((start, ts[-1]))
        prev = last_seg_of.get(parents.get(lb, 0), 0)
        for b, e in runs:
            segs.append((next_label, b, e, prev))
            for t in range(b, e + 1):
                lookup[(lb, t)] = next_label
            prev = next_label
            next_label += 1
        last_seg_of[lb] = prev
    return segs, lookup


def write_ctc(out_dir: Path, label_maps: list, label_to_track: list[dict], *, gt: bool = False,
              parents: dict[int, int] | None = None, seg_frames: list[int] | None = None,
              seg_maps: list | None = None) -> Path:
    """Write a result folder (gt=False) or a ground-truth folder (gt=True).

    label_maps: per-frame label maps (arrays or paths). label_to_track[t]
    maps a label in frame t to its track id (for ground truth: the
    annotated cell id); labels missing from it are left out. parents:
    {track: parent track} (divisions, CTC data only). For ground truth,
    seg_maps/seg_frames give the frames with careful masks (SEG)."""
    out_dir = Path(out_dir)
    tra_dir = out_dir / "TRA" if gt else out_dir
    tra_dir.mkdir(parents=True, exist_ok=True)
    T = len(label_maps)
    nd = max(3, len(str(T - 1)))

    presence: dict[int, list[int]] = {}
    for t in range(T):
        for k in set(label_to_track[t].values()):
            presence.setdefault(int(k), []).append(t)
    segs, lookup = _split_at_gaps(presence, parents)
    if len(segs) >= 65535:
        raise ValueError(f"{len(segs)} tracks do not fit the CTC uint16 format")

    for t in range(T):
        lab = _load_array(label_maps[t])
        lut = np.zeros(int(lab.max()) + 1, dtype=np.uint16)
        for lb, k in label_to_track[t].items():
            if 0 < lb < len(lut):
                lut[lb] = lookup[(int(k), t)]
        name = f"man_track{t:0{nd}d}.tif" if gt else f"mask{t:0{nd}d}.tif"
        tiff.imwrite(str(tra_dir / name), lut[lab], compression="zlib")
    txt = "man_track.txt" if gt else "res_track.txt"
    with open(tra_dir / txt, "w") as f:
        for L, B, E, P in segs:
            f.write(f"{L} {B} {E} {P}\n")

    if gt and seg_frames:
        seg_dir = out_dir / "SEG"
        seg_dir.mkdir(exist_ok=True)
        for t, m in zip(seg_frames, seg_maps):
            tiff.imwrite(str(seg_dir / f"man_seg{t:0{nd}d}.tif"),
                         _load_array(m).astype(np.uint16), compression="zlib")
    return out_dir


def tracks_to_ctc(out_dir: Path, tracks: pd.DataFrame, label_maps: list) -> Path:
    """Export a tracker output table (with `label`) as a CTC result folder."""
    l2t = [dict() for _ in label_maps]
    for t, lb, k in tracks[["frame_idx", "label", "track_id"]].itertuples(index=False):
        if 0 <= t < len(label_maps) and lb > 0:
            l2t[int(t)][int(lb)] = int(k)
    return write_ctc(out_dir, label_maps, l2t)


def stack_to_ctc_gt(out_dir: Path, gt_stack: np.ndarray, seg_frames: list[int] | None = None) -> Path:
    """Export an annotated (T, Y, X) stack, in which a cell keeps its id in
    every frame, as a CTC ground-truth folder. Frames in seg_frames also go
    to SEG (all frames if None)."""
    T = gt_stack.shape[0]
    l2t = [{int(i): int(i) for i in np.unique(gt_stack[t]) if i > 0} for t in range(T)]
    seg_frames = list(range(T)) if seg_frames is None else list(seg_frames)
    return write_ctc(out_dir, list(gt_stack), l2t, gt=True,
                     seg_frames=seg_frames, seg_maps=[gt_stack[t] for t in seg_frames])


# CHOTA (also in py-ctcmetrics, not a CTC measure) is left out: it does not
# finish in reasonable time on dense sequences.
CTC_METRICS = ["Valid", "SEG", "DET", "TRA", "LNK", "CT", "TF", "BC", "CCA", "BIO",
               "OP_CSB", "OP_CTB", "OP_CLB", "HOTA", "IDF1", "MOTA"]


def ctc_measures(res_dir: Path, gt_dir: Path, metrics: list[str] | None = None,
                 threads: int = 1) -> dict:
    """Official CTC measures for one sequence (py-ctcmetrics). SEG is only
    requested when the ground truth has a SEG folder."""
    from ctc_metrics.scripts.evaluate import evaluate_sequence
    metrics = list(metrics or CTC_METRICS)
    if {"BIO", "OP_CLB"} & set(metrics):  # BIO needs all four parts, OP_CLB needs BIO and LNK
        metrics += [m for m in ("CT", "TF", "BC", "CCA", "BIO", "LNK") if m not in metrics]
    if "LNK" in metrics and not {"DET", "TRA"} & set(metrics):  # LNK reuses TRA's graph operations
        metrics.append("TRA")
    if not (Path(gt_dir) / "SEG").is_dir():
        metrics = [m for m in metrics if m not in ("SEG", "OP_CSB", "OP_CTB")]
    with contextlib.redirect_stdout(io.StringIO()):
        res = evaluate_sequence(str(res_dir), str(gt_dir), metrics=metrics, threads=threads)
    return {k: (float(v) if v is not None else np.nan) for k, v in res.items()
            if np.isscalar(v) or v is None}
