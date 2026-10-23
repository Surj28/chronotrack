"""Fair comparison of the trackers on synthetic twins of the ARTORG wells
(synth_twin.py), with ground truth. Every tracker gets the same detections,
gate and missed-detection model as on real data (run_event_tests.run_named),
and the vessel layer is the real well's.

Per sequence and tracker:
  precision        fraction of the tracker's links (consecutive detections of
                   a track, across gaps too) that join two detections of the
                   same cell
  recall           fraction of the true links (consecutive detections of each
                   cell, across missed frames) that the tracker makes
  f1               harmonic mean of the two
  complete         fraction of cells (>= 2 detections) whose detections form
                   exactly one track of the tracker, with nothing else in it
  feature_error    median over the linking-dependent features of the relative
                   error against the event table of the true tracks (the same
                   detections linked correctly, missed frames bridged)

Conditions: density rho (1: the real count; 2: twice), motion lambda (1: the
consensus steps; cal: steps scaled to match the real well's pilot-step
statistic, synth_twin.calibrate), seeds 0-2; twins generated on first use.

Outputs <results>/synthetic/benchmark.csv (one row per sequence and tracker;
with --published, the published trackers of published_trackers.py on the
existing twins: benchmark_published.csv).

Usage (from the repo root):
    python paper/code/synth_benchmark.py [--wells ...] [--rhos 1 2] [--lams 1 cal] [--seeds 0 1 2] [--jobs 6]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import get_context

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi

import run_event_tests as RET
from common import CROPS, RESULTS, SEG, WELLS
from summarize_event_tests import TRACK_FEATS
from synth_twin import OUT, calibrate, seq_dir, write
from topology_tracking.scaffold import Scaffold
from topology_tracking.vessel_tracker import frame_detections, wall_map

BENCH = OUT / "benchmark.csv"


class TwinWell(RET.Well):
    """A twin sequence with the real well's vessel layer (run_event_tests.Well interface)."""

    def __init__(self, well, d):
        self.well = d.name
        self.ts = json.loads((CROPS / well / "roi.json").read_text())["timepoints"]
        lab = tifffile.imread(d / "cells.tif")
        green = tifffile.imread(d / "green.tif").astype(np.float32)
        self.T = len(lab)
        self.labels = [lab[t] for t in range(self.T)]
        sd = RESULTS / "vessel_scaffold" / well
        self.sc = Scaffold.load(sd)
        wm = wall_map(self.sc)
        self.frames = [frame_detections(lab[t], green[t], self.sc, t, wm) for t in range(self.T)]
        self.territory = tifffile.imread(sd / "territory_ids.tif")
        self.segments = pd.read_csv(sd / "segments.csv")
        self.presence = pd.read_csv(sd / "presence.csv", index_col=0)
        self.red = pd.read_csv(sd / "red_profile.csv", index_col=0)
        self.dmap = []
        for t, fd in zip(self.ts, self.frames):
            mask = tifffile.imread(SEG / well / f"vessels_T{t:04d}.tif") > 0
            dist = ndi.distance_transform_edt(~mask)
            r = np.clip(np.round(fd.pos_img[:, 0]).astype(int), 0, mask.shape[0] - 1)
            c = np.clip(np.round(fd.pos_img[:, 1]).astype(int), 0, mask.shape[1] - 1)
            self.dmap.append(dict(zip(fd.label.astype(int), dist[r, c] * self.um_per_px)))
        areas = np.concatenate([fd.area for fd in self.frames])
        self.delta = float(np.sqrt(np.median(areas) / np.pi) * self.um_per_px)
        roi = (tifffile.imread(CROPS / well / "roi_mask.tif") > 0) & ~(tifffile.imread(CROPS / well / "pillar_mask.tif") > 0)
        self.roi = roi
        self.edge_um = ndi.distance_transform_edt(roi) * self.um_per_px


def links_of(tracks):
    t = tracks.sort_values(["track_id", "frame_idx"])
    same = t.track_id.values[1:] == t.track_id.values[:-1]
    a = list(zip(t.frame_idx.values[:-1][same], t.label.values[:-1][same].astype(int)))
    b = list(zip(t.frame_idx.values[1:][same], t.label.values[1:][same].astype(int)))
    return list(zip(a, b))


def truth_tracks(W, gt):
    g = gt[gt.label > 0]
    pos = {(t, int(l)): p for t, fd in enumerate(W.frames) for l, p in zip(fd.label, fd.pos_img)}
    rc = np.array([pos[(f, int(l))] for f, l in zip(g.frame, g.label)])
    return pd.DataFrame(dict(track_id=g.gt_id.values, frame_idx=g.frame.values, label=g.label.values.astype(int),
                             centroid_r=rc[:, 0], centroid_c=rc[:, 1]))


def score(tracks, true_tracks, F, F_true):
    owner = dict(zip(zip(true_tracks.frame_idx, true_tracks.label), true_tracks.track_id))
    L = links_of(tracks)
    good = sum(owner.get(a) is not None and owner.get(a) == owner.get(b) for a, b in L)
    true_links = set(links_of(true_tracks))
    found = sum((a, b) in true_links for a, b in L)
    prec = good / max(len(L), 1)
    rec = found / max(len(true_links), 1)
    tr = tracks.assign(owner=[owner.get((f, int(l))) for f, l in zip(tracks.frame_idx, tracks.label)])
    per_track = tr.groupby("track_id").owner.agg(lambda o: o.iloc[0] if o.nunique() == 1 else -1)
    size = tr.groupby("track_id").size()
    n_true = true_tracks.groupby("track_id").size()
    multi = n_true[n_true >= 2]
    ok = per_track[(per_track >= 0)]
    complete = sum(1 for tid, o in ok.items() if o in multi.index and size[tid] == n_true[o]) / max(len(multi), 1)
    errs = [abs(F[f] - F_true[f]) / abs(F_true[f]) for f in TRACK_FEATS
            if f in F and np.isfinite(F_true.get(f, np.nan)) and abs(F_true[f]) > 1e-12 and np.isfinite(F[f])]
    return dict(precision=prec, recall=rec, f1=2 * prec * rec / max(prec + rec, 1e-12), complete=complete,
                feature_error=float(np.median(errs)) if errs else np.nan, n_links=len(L), n_true_links=len(true_links))


def job(cfg, trackers=RET.TRACKERS):
    well, rho, lam_label, seed = cfg
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    d = seq_dir(well, rho, lam_label, seed)
    t0 = time.time()
    if not (d / "gt.csv").exists():
        lam = 1.0 if lam_label == "1" else calibrate(well, 0)
        write(well, rho, seed, lam, lam_label)
    W = TwinWell(well, d)
    gt = pd.read_csv(d / "gt.csv")
    true_tr = truth_tracks(W, gt)
    ident = {i: i for i in range(W.T)}
    F_true, _, _ = W.table(true_tr, ident, W.T, W.segments)
    rows = []
    for name in trackers:
        tr = RET.run_named(name, W.frames, W.labels)
        tr.to_csv(d / f"tracks_{name}.csv.gz", index=False)   # features can be recomputed without re-tracking
        F, _, _ = W.table(tr, ident, W.T, W.segments)
        rows.append(dict(well=well, rho=rho, lam=lam_label, seed=seed, tracker=name, **score(tr, true_tr, F, F_true)))
    print(f"{d.name}: {time.time() - t0:.0f}s " + " ".join(f"{r['tracker']} f1={r['f1']:.3f}" for r in rows), flush=True)
    return rows


def _call(a):
    fn, cfg, trackers = a
    return fn(cfg, trackers)


PUBLISHED_NAMES = ("btrack", "laptrack", "trackastra")


def job_published(cfg, trackers=PUBLISHED_NAMES):
    """The published trackers (published_trackers.py) on an existing twin."""
    from common import symmetric_gate
    from published_trackers import PUBLISHED, run_published
    from topology_tracking.detection_model import detection_model
    well, rho, lam_label, seed = cfg
    d = seq_dir(well, rho, lam_label, seed)
    t0 = time.time()
    W = TwinWell(well, d)
    gt = pd.read_csv(d / "gt.csv")
    true_tr = truth_tracks(W, gt)
    ident = {i: i for i in range(W.T)}
    F_true, _, _ = W.table(true_tr, ident, W.T, W.segments)
    gate = symmetric_gate(W.labels)["max_dist_px"]
    M = detection_model(W.frames)["max_gap"]
    green = tifffile.imread(d / "green.tif").astype(np.float32)
    rows = []
    for name in trackers:
        tr = run_published(name, W.frames, W.labels, green, gate, M)
        tr.to_csv(d / f"tracks_{name}.csv.gz", index=False)
        F, _, _ = W.table(tr, ident, W.T, W.segments)
        rows.append(dict(well=well, rho=rho, lam=lam_label, seed=seed, tracker=name, **score(tr, true_tr, F, F_true)))
    print(f"{d.name}: {time.time() - t0:.0f}s " + " ".join(f"{r['tracker']} f1={r['f1']:.3f}" for r in rows), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wells", nargs="+", default=WELLS)
    ap.add_argument("--rhos", nargs="+", type=float, default=[1.0, 2.0])
    ap.add_argument("--lams", nargs="+", default=["1", "cal"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--jobs", type=int, default=6, help="parallel workers; 1: in this process")
    ap.add_argument("--published", action="store_true", help="run the published trackers on existing twins")
    ap.add_argument("--trackers", nargs="+", default=list(RET.TRACKERS), help="our trackers to run")
    ap.add_argument("--replace", action="store_true", help="recompute these trackers where they already have results")
    ap.add_argument("--out", default=None, help="output CSV name in <results>/synthetic (default per mode)")
    ap.add_argument("--twins-after", type=float, default=None, help="only twins created after this time (epoch s)")
    args = ap.parse_args()
    out = OUT / (args.out or ("benchmark_published.csv" if args.published else BENCH.name))
    for w in args.wells:                               # calibrate lambda once per well, before the parallel jobs
        if "cal" in args.lams and not args.published:
            print(w, "lambda", round(calibrate(w, 0), 3), flush=True)
    cfgs = [(w, r, l, s) for w in args.wells for r in args.rhos for l in args.lams for s in args.seeds]
    if args.published:
        cfgs = [c for c in cfgs if (seq_dir(*c) / "gt.csv").exists()]
    if args.twins_after is not None:
        cfgs = [c for c in cfgs if (seq_dir(*c) / "gt.csv").exists() and (seq_dir(*c) / "gt.csv").stat().st_mtime > args.twins_after]
    done = pd.read_csv(out) if out.exists() else pd.DataFrame()
    want = PUBLISHED_NAMES if args.published else tuple(args.trackers)
    if len(done) and args.replace:
        done = done[~done.tracker.isin(want)]
    have = set(zip(done.well, done.rho, done.lam.astype(str), done.seed, done.tracker)) if len(done) else set()
    todo = [(c, tuple(t for t in want if (c[0], c[1], c[2], c[3], t) not in have)) for c in cfgs]
    todo = [(c, t) for c, t in todo if t]
    rows = done.to_dict("records") if len(done) else []
    fn = job_published if args.published else job
    tasks = [(fn, c, t) for c, t in todo]
    if args.jobs <= 1:                     # in this process, no pool: a stalled worker cannot hang the run
        for res in map(_call, tasks):
            rows += res
            pd.DataFrame(rows).to_csv(out, index=False)
        return
    with get_context("spawn").Pool(args.jobs) as pool:
        for res in pool.imap_unordered(_call, tasks):
            rows += res
            pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == "__main__":
    main()
