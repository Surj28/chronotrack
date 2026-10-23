"""Extra frame-interval levels for ARTORG (ChronoTrack only), so that its curve in
the frame-interval panel has more than one point: every 3rd and 4th frame
(3 h and 4 h; 8 and 6 frames). Same protocol as run_event_tests.process.

Output <results>/event_tests/features_sampling_extra.csv (same columns as features.csv).

Usage (from the repo root):
    python paper/code/run_sampling_extra.py [--steps 3 4]
"""
from __future__ import annotations

import argparse

import pandas as pd

import run_event_tests as RET
from common import WELLS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", nargs="+", type=int, default=[3, 4])
    args = ap.parse_args()
    rows = []
    for w in WELLS:
        W = RET.Well(w)
        for k in args.steps:
            fr, lab = W.frames[::k], W.labels[::k]
            tk = RET.run_named("sym", fr, lab)
            Fk, _, _ = W.table(tk, {i: k * i for i in range(len(fr))}, len(fr), W.segments, frame_h=k * W.frame_h)
            rows.append(dict(well=w, tracker="sym", condition=f"every_{k}_frames", **Fk))
            print(w, k, flush=True)
        pd.DataFrame(rows).to_csv(RET.OUT / "features_sampling_extra.csv", index=False)


if __name__ == "__main__":
    main()
