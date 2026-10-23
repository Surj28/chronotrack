"""Run the vessel-referenced, self-calibrating tracker on the ARTORG wells.

Per well (<out>/<well>/): tracks.csv, calib.json (final parameters, per-cue
separation s_false / s_true), history.csv (parameters per calibration round).

--cues lets an ablation drop cues (e.g. --cues persistence size bright for
"no vessel cues"); the output folder then gets a suffix.

Usage (from the repo root):
    python paper/code/run_vessel_tracker.py [--wells ARTORG_w7] [--cues ...] [--frame-step 2]
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

import numpy as np
import pandas as pd
import tifffile

from common import CROPS, RESULTS, SEG, WELLS, estimate_max_dist
from topology_tracking.scaffold import Scaffold
from topology_tracking.vessel_tracker import CUES, frame_detections, run, separation, wall_map

OUT = RESULTS / "vessel_tracker"


def load_frames(well: str, frame_step: int = 1):
    ts = json.loads((CROPS / well / "roi.json").read_text())["timepoints"][::frame_step]
    sc = Scaffold.load(RESULTS / "vessel_scaffold" / well)
    wm = wall_map(sc)
    all_ts = json.loads((CROPS / well / "roi.json").read_text())["timepoints"]
    labels, frames = [], []
    for t in ts:
        lab = tifffile.imread(SEG / well / f"cells_T{t:04d}.tif")
        img = tifffile.imread(CROPS / well / f"T{t:04d}_C2.tif")
        labels.append(lab)
        frames.append(frame_detections(lab, img, sc, all_ts.index(t), wm))
    return ts, labels, frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wells", nargs="+", default=WELLS)
    ap.add_argument("--cues", nargs="+", default=list(CUES))
    ap.add_argument("--frame-step", type=int, default=1)
    args = ap.parse_args()
    suffix = "" if set(args.cues) == set(CUES) else "_" + "-".join(sorted(set(CUES) - set(args.cues)) and
                                                                  ["no-" + c for c in sorted(set(CUES) - set(args.cues))])
    if args.frame_step > 1:
        suffix += f"_step{args.frame_step}"
    for well in args.wells:
        t0 = time.time()
        ts, labels, frames = load_frames(well, args.frame_step)
        gate = estimate_max_dist(labels)
        tracks, cal, hist = run(frames, gate, cues=tuple(args.cues), verbose=True)
        out = OUT / f"{well}{suffix}"
        out.mkdir(parents=True, exist_ok=True)
        tracks.to_csv(out / "tracks.csv", index=False)
        pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
        (out / "calib.json").write_text(json.dumps(dict(**{k: v for k, v in asdict(cal).items()}, gate=gate,
                                                       separation=separation(cal), frame_step=args.frame_step,
                                                       n_frames=len(ts)), indent=2, default=list))
        n_tr = tracks.track_id.nunique()
        L = tracks.groupby("track_id").size()
        print(f"{well}{suffix}: {n_tr} tracks, mean length {L.mean():.1f} frames, {len(hist)} rounds, "
              f"{time.time() - t0:.0f}s | separation " +
              json.dumps({k: round(v, 2) for k, v in separation(cal).items()}), flush=True)


if __name__ == "__main__":
    main()
