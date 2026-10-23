"""Label-free test of under-linking: continuity of near-stationary cells.

A detection at frame t with exactly one detection within one cell radius at
t + lag, and no other detection within three cell radii at t or t + lag, is an
unambiguous case: the same cell, practically not moved. Any tracker should
keep the two detections in one track. Per tracker and lag, the fraction kept
measures under-linking (track breaks that the detections do not support);
it says nothing about over-linking.

Sequences: the ARTORG wells (run_event_tests.Well), the Song videos
(run_external_events.SongVideo) and Hachey days 0-5 (run_hachey_events.HacheyVideo).
Outputs <results>/stationary_continuity/continuity.csv.

Usage (from the repo root):
    python paper/code/stationary_continuity.py [--wells ARTORG_w7] [--videos S1 S5] [--hachey] [--lags 1 2]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import run_event_tests as RET
from common import RESULTS
from run_external_events import SongVideo

OUT = RESULTS / "stationary_continuity"


def unambiguous_pairs(W, lag: int):
    """(t, label_a, label_b) for every unambiguous near-stationary pair t -> t + lag."""
    r1 = W.delta / W.um_per_px
    out = []
    for t in range(W.T - lag):
        a, b = W.frames[t], W.frames[t + lag]
        if not len(a.label) or not len(b.label):
            continue
        ta, tb = cKDTree(a.pos), cKDTree(b.pos)
        near_b = tb.query_ball_point(a.pos, 3 * r1)
        near_a = ta.query_ball_point(a.pos, 3 * r1)
        for i in range(len(a.label)):
            nb = near_b[i]
            if len(nb) != 1 or len(near_a[i]) != 1:
                continue
            j = nb[0]
            if np.hypot(*(b.pos[j] - a.pos[i])) <= r1:
                out.append((t, int(a.label[i]), int(b.label[j])))
    return out


def continuity(W, tracks: pd.DataFrame, pairs, lag: int) -> float:
    tid = dict(zip(zip(tracks.frame_idx.astype(int), tracks.label.astype(int)), tracks.track_id))
    kept = [tid.get((t, la)) is not None and tid.get((t, la)) == tid.get((t + lag, lb)) for t, la, lb in pairs]
    return float(np.mean(kept)) if kept else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wells", nargs="*", default=["ARTORG_w7"])
    ap.add_argument("--videos", nargs="*", default=["S1", "S3", "S5"])
    ap.add_argument("--hachey", action="store_true")
    ap.add_argument("--lags", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--trackers", nargs="+", default=None, help="subset of run_event_tests.TRACKERS")
    args = ap.parse_args()
    if args.trackers:
        RET.TRACKERS = tuple(args.trackers)
    OUT.mkdir(parents=True, exist_ok=True)
    seqs = [RET.Well(w) for w in args.wells] + [SongVideo(v) for v in args.videos]
    if args.hachey:
        from run_hachey_events import HacheyVideo
        seqs.append(HacheyVideo())
    rows = []
    for W in seqs:
        pairs = {lag: unambiguous_pairs(W, lag) for lag in args.lags}
        n_det = sum(len(fd.label) for fd in W.frames)
        for name in RET.TRACKERS:
            tr = RET.run_named(name, W.frames, W.labels)
            for lag in args.lags:
                rows.append(dict(sequence=W.well, tracker=name, lag=lag, n_pairs=len(pairs[lag]),
                                 pairs_per_detection=len(pairs[lag]) / max(n_det, 1),
                                 kept=continuity(W, tr, pairs[lag], lag)))
            print(W.well, name, {r["lag"]: round(r["kept"], 3) for r in rows[-len(args.lags):]}, flush=True)
        pd.DataFrame(rows).to_csv(OUT / "continuity.csv", index=False)
    df = pd.DataFrame(rows)
    print(df.pivot_table(index="tracker", columns=["sequence", "lag"], values="kept").round(3).to_string())
    print(df.groupby(["sequence", "lag"])[["n_pairs", "pairs_per_detection"]].first().round(3).to_string())


if __name__ == "__main__":
    main()
