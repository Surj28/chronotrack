"""Lost-detection test at further levels, so that robustness is summarized by
a dose-response slope instead of one arbitrary level. run_event_tests.py
(and its external / Hachey runners) already remove 10 % and 20 % of the
detections; this adds the levels given here (default 5 % and 30 %), with the
same protocol: detections removed at random before tracking, the whole
pipeline re-run (gate and missed-detection model re-estimated), relative
change of every feature against the forward run in features.csv.

Outputs <results dir>/removal_curve.csv (same columns as removal.csv).

Usage (from the repo root):
    python paper/code/run_dropout_curve.py --model artorg|song|hachey [--levels 0.05 0.30] [--repeats 3]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

import run_event_tests as RET
from common import WELLS


def curve(W, F_forward: pd.DataFrame, levels, repeats, rng):
    rows = []
    for name in RET.TRACKERS:
        t0 = time.time()
        F = F_forward[(F_forward.well == W.well) & (F_forward.tracker == name)].iloc[0].to_dict()
        F = {k: v for k, v in F.items() if k not in ("well", "tracker", "condition")}
        for X in levels:
            for rep in range(repeats):
                keep = [rng.random(len(fd.label)) >= X for fd in W.frames]
                fr = [RET.subset(fd, k) for fd, k in zip(W.frames, keep)]
                lab = [np.where(np.isin(L, fd.label[~k]), 0, L) for L, fd, k in zip(W.labels, W.frames, keep)]
                Fb, _, _ = W.table(RET.run_named(name, fr, lab), {i: i for i in range(W.T)}, W.T, W.segments)
                rows.append(dict(well=W.well, tracker=name, kind="detections", X=X, rep=rep, **RET.rel_change(F, Fb)))
        print(f"{W.well} {name}: {time.time() - t0:.0f}s", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["artorg", "song", "hachey"])
    ap.add_argument("--levels", nargs="+", type=float, default=[0.05, 0.30])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--trackers", nargs="+", default=None, help="subset of run_event_tests.TRACKERS")
    args = ap.parse_args()
    if args.trackers:
        RET.TRACKERS = tuple(args.trackers)
    rng = np.random.default_rng(1)
    if args.model == "artorg":
        out, seqs = RET.OUT, (RET.Well(w) for w in WELLS)
    elif args.model == "song":
        from run_external_events import OUT as out, SongVideo
        seqs = (SongVideo(v) for v in ("S1", "S3", "S5"))
    else:
        from run_hachey_events import OUT as out, HacheyVideo
        seqs = (HacheyVideo(),)
    Ffw = pd.read_csv(out / "features.csv")
    Ffw = Ffw[Ffw.condition == "forward"]
    rows = []
    for W in seqs:
        rows += curve(W, Ffw, args.levels, args.repeats, rng)
        pd.DataFrame(rows).to_csv(out / "removal_curve.csv", index=False)


if __name__ == "__main__":
    main()
