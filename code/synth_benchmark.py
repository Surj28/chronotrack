"""Fair comparison of the trackers on the synthetic sequences with ground
truth. Every tracker gets the same detections, gate and missed-detection
model, and the vessel layer is the sequence's own (<data>/vessels/seq<k>/).

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

Trackers (TRACKERS): sym (ChronoTrack: time-symmetric objective on motion,
persistence, size and intensity, self-calibration;
topology_tracking.symmetric_tracker), sym_vessel (the same plus the vessel
cues state, anisotropy and wall distance; an ablation), first_order (sym's
cues with a forward-only, frame-by-frame assignment;
topology_tracking.vessel_tracker), the distance-only Kalman tracker (w0) and
greedy nearest-neighbour linking (nn). Every tracker that bridges missed
detections uses the same per-sequence detection model
(topology_tracking.detection_model). With --published: btrack, LapTrack and
Trackastra (published_trackers.py).

Conditions: density (1, or 2: twice as many cells), motion (gentle, or
ambiguous: steps scaled up to a harder linking ambiguity), seeds 0-2.

Data (--data, default data/; see the data README):
  <data>/seq<k>_density<d>_<gentle|ambiguous>_seed<s>/  cells.tif, green.tif, gt.csv, info.json
  <data>/vessels/seq<k>/                                vessels.tif, union_mask.tif, segment_ids.tif,
                                                        territory_ids.tif, segments.csv, flows.npz

Outputs in --out (default results/): benchmark.csv (one row per sequence and
tracker; with --published: benchmark_published.csv) and the tracks,
tracks/<sequence>/tracks_<tracker>.csv.gz.

Usage (from the repo root):
    python code/synth_benchmark.py [--data data] [--seqs seq1 ...] [--densities 1 2]
        [--motions gentle ambiguous] [--seeds 0 1 2] [--trackers sym ...] [--published] [--jobs 6]
"""
from __future__ import annotations

import argparse
import os
import re
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi

from common import FRAME_H, K_EXIT, UM_PER_PX, run_tracker, symmetric_gate
from topology_tracking import symmetric_tracker as ST
from topology_tracking.detection_model import detection_model
from topology_tracking.events import cell_events, chip_features
from topology_tracking.scaffold import Scaffold
from topology_tracking.vessel_tracker import CUES, VESSEL_CUES, frame_detections, run, wall_map

REPO = Path(__file__).resolve().parent.parent
TRACKERS = ("sym", "sym_vessel", "first_order", "w0", "nn")
METHOD_CUES = tuple(c for c in CUES if c not in VESSEL_CUES)        # persistence, size, bright
PUBLISHED_NAMES = ("btrack", "laptrack", "trackastra")
SEQ_NAME = re.compile(r"^(seq\d+)_density(\d+)_(gentle|ambiguous)_seed(\d+)$")
# features that depend on the linking (cells per frame, state occupancy and the vessel fractions do not)
TRACK_FEATS = ["arrivals_per_100_cell_h", "departures_per_100_cell_h", "arrival_on_vessel_frac",
               "departure_on_vessel_frac", "arrival_vessel_enrichment", "departure_vessel_enrichment",
               "exits_per_100_cell_h", "entries_per_100_cell_h", "speed_on_vessel_um_h", "speed_off_vessel_um_h",
               "mean_track_duration_h", "exit_regression_colocation", "entry_emergence_colocation"]


def run_named(name, frames, labels):
    gate = symmetric_gate(labels)
    det = detection_model(frames)
    if name == "sym":
        return ST.run(frames, gate, labels[0].shape, cues=METHOD_CUES, gap_cost=det["gap_cost"])[0]
    if name == "sym_vessel":
        return ST.run(frames, gate, labels[0].shape, cues=CUES, gap_cost=det["gap_cost"])[0]
    if name == "first_order":
        return run(frames, gate, cues=METHOD_CUES, max_gap=det["max_gap"])[0]
    if name == "w0":
        return run_tracker(name, labels, None, gate["max_dist_px"], max_gap=det["max_gap"])
    return run_tracker(name, labels, None, gate["max_dist_px"])


def seq_name(seq, density, motion, seed):
    return f"{seq}_density{density}_{motion}_seed{seed}"


class Sequence:
    """A synthetic sequence with its vessel layer: detections, scaffold and event table."""
    um_per_px = UM_PER_PX
    frame_h = FRAME_H

    def __init__(self, d: Path, vessel_dir: Path):
        self.name = d.name
        lab = tifffile.imread(d / "cells.tif")
        green = tifffile.imread(d / "green.tif").astype(np.float32)
        self.T = len(lab)
        self.labels = [lab[t] for t in range(self.T)]
        flows = np.load(vessel_dir / "flows.npz")
        self.sc = Scaffold.from_masks(tifffile.imread(vessel_dir / "union_mask.tif") > 0,
                                      tifffile.imread(vessel_dir / "segment_ids.tif") > 0,
                                      flows["flows"], int(flows["ds"]))
        wm = wall_map(self.sc)
        self.frames = [frame_detections(lab[t], green[t], self.sc, t, wm) for t in range(self.T)]
        self.territory = tifffile.imread(vessel_dir / "territory_ids.tif")
        self.segments = pd.read_csv(vessel_dir / "segments.csv")
        V = tifffile.imread(vessel_dir / "vessels.tif") > 0
        self.dmap = []
        for mask, fd in zip(V, self.frames):
            dist = ndi.distance_transform_edt(~mask)
            r = np.clip(np.round(fd.pos_img[:, 0]).astype(int), 0, mask.shape[0] - 1)
            c = np.clip(np.round(fd.pos_img[:, 1]).astype(int), 0, mask.shape[1] - 1)
            self.dmap.append(dict(zip(fd.label.astype(int), dist[r, c] * self.um_per_px)))
        areas = np.concatenate([fd.area for fd in self.frames])
        self.delta = float(np.sqrt(np.median(areas) / np.pi) * self.um_per_px)

    def states(self, tracks, fmap):
        fo = np.array([fmap[f] for f in tracks.frame_idx])
        d = np.array([self.dmap[f][int(l)] for f, l in zip(fo, tracks.label)])
        out = tracks.assign(d_vessel_um=d)
        out["state"] = np.where(d == 0, "overlapping", np.where(d <= self.delta, "adjacent", "distant"))
        return out

    def table(self, tracks, fmap, n_frames, segments, frame_h=None):
        st = self.states(tracks, fmap)
        ev = cell_events(st, n_frames, K_EXIT, self.territory,
                         to_reference=lambda p, f: self.sc.to_reference(p, fmap[f]))
        return chip_features(st, ev, segments, n_frames, frame_h or self.frame_h, self.um_per_px), st, ev


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


def job(cfg, data, out, trackers=TRACKERS, published=False):
    """Our trackers (or, published=True, the published ones of published_trackers.py) on one sequence."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    t0 = time.time()
    d = data / seq_name(*cfg)
    W = Sequence(d, data / "vessels" / cfg[0])
    true_tr = truth_tracks(W, pd.read_csv(d / "gt.csv"))
    ident = {i: i for i in range(W.T)}
    F_true, _, _ = W.table(true_tr, ident, W.T, W.segments)
    td = out / "tracks" / W.name
    td.mkdir(parents=True, exist_ok=True)
    if published:
        from published_trackers import run_published
        gate = symmetric_gate(W.labels)["max_dist_px"]
        M = detection_model(W.frames)["max_gap"]
        green = tifffile.imread(d / "green.tif").astype(np.float32)
    rows = []
    for name in trackers:
        tr = run_published(name, W.frames, W.labels, green, gate, M) if published else run_named(name, W.frames, W.labels)
        tr.to_csv(td / f"tracks_{name}.csv.gz", index=False)   # features can be recomputed without re-tracking
        F, _, _ = W.table(tr, ident, W.T, W.segments)
        rows.append(dict(seq=cfg[0], density=cfg[1], motion=cfg[2], seed=cfg[3], tracker=name,
                         **score(tr, true_tr, F, F_true)))
    print(f"{W.name}: {time.time() - t0:.0f}s " + " ".join(f"{r['tracker']} f1={r['f1']:.3f}" for r in rows), flush=True)
    return rows


def _call(a):
    return job(*a)


def sequences(data: Path, seqs, densities, motions, seeds):
    """(seq, density, motion, seed) of every sequence folder in `data` that matches the selection."""
    out = []
    for d in sorted(data.iterdir()) if data.is_dir() else []:
        m = SEQ_NAME.match(d.name)
        if m and (d / "gt.csv").exists():
            c = (m[1], int(m[2]), m[3], int(m[4]))
            if (not seqs or c[0] in seqs) and c[1] in densities and c[2] in motions and c[3] in seeds:
                out.append(c)
    return out


def main():
    ap = argparse.ArgumentParser(description="Benchmark of the trackers on the synthetic sequences with ground truth.")
    ap.add_argument("--data", type=Path, default=REPO / "data", help="folder with the sequences (default: data/)")
    ap.add_argument("--out", type=Path, default=REPO / "results", help="output folder (default: results/)")
    ap.add_argument("--seqs", nargs="+", default=None, help="e.g. seq1 seq3 (default: all)")
    ap.add_argument("--densities", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--motions", nargs="+", default=["gentle", "ambiguous"], choices=["gentle", "ambiguous"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--jobs", type=int, default=6, help="parallel workers; 1: in this process")
    ap.add_argument("--published", action="store_true", help="run the published trackers (btrack, LapTrack, Trackastra)")
    ap.add_argument("--trackers", nargs="+", default=None,
                    help=f"subset of the trackers to run (default: {' '.join(TRACKERS)}; "
                         f"with --published: {' '.join(PUBLISHED_NAMES)})")
    ap.add_argument("--replace", action="store_true", help="recompute these trackers where they already have results")
    args = ap.parse_args()
    cfgs = sequences(args.data, args.seqs, args.densities, args.motions, args.seeds)
    if not cfgs:
        raise SystemExit(f"no matching sequence in {args.data} (folders seq<k>_density<d>_<motion>_seed<s>)")
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / ("benchmark_published.csv" if args.published else "benchmark.csv")
    want = tuple(args.trackers or (PUBLISHED_NAMES if args.published else TRACKERS))
    done = pd.read_csv(out) if out.exists() else pd.DataFrame()
    if len(done) and args.replace:
        done = done[~done.tracker.isin(want)]
    have = set(zip(done.seq, done.density, done.motion, done.seed, done.tracker)) if len(done) else set()
    todo = [(c, tuple(t for t in want if (*c, t) not in have)) for c in cfgs]
    tasks = [(c, args.data, args.out, t, args.published) for c, t in todo if t]
    rows = done.to_dict("records") if len(done) else []
    if args.jobs <= 1:                     # in this process, no pool: a stalled worker cannot hang the run
        for res in map(_call, tasks):
            rows += res
            pd.DataFrame(rows).to_csv(out, index=False)
    else:
        with get_context("spawn").Pool(args.jobs) as pool:
            for res in pool.imap_unordered(_call, tasks):
                rows += res
                pd.DataFrame(rows).to_csv(out, index=False)
    print(out)


if __name__ == "__main__":
    main()
