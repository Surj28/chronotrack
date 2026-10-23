"""Chip event table and its label-free validation on ARTORG, for several trackers.

Trackers: sym (the paper's method: time-symmetric objective on motion,
persistence, size and intensity, self-calibration;
topology_tracking.symmetric_tracker), sym_vessel (the same plus the vessel
cues state, anisotropy and wall distance; an ablation: on ARTORG they made the
tracker less robust to missed detections), first_order (sym's cues with a
forward-only, frame-by-frame assignment; topology_tracking.vessel_tracker),
the distance-only Kalman tracker (w0) and greedy nearest-neighbour linking
(nn). Every tracker that bridges missed detections uses the same per-sequence
detection model (topology_tracking.detection_model: longest gap and its cost,
from isolated cells that stay in place), re-estimated on each perturbed,
reversed or subsampled sequence like the motion gate. For each well and tracker: tracks ->
vessel states -> event table (topology_tracking.events) -> chip features, and
the tests below. None needs annotation.

  1 removal      (a) sym only: X % of tracks removed at random, features
                 recomputed (sampling robustness of each feature);
                 (b) every tracker: X % of detections dropped at random before
                 tracking (each pipeline re-run in full). Relative change of each
                 feature vs X.
  2 reversal     every tracker on the reversed movie (vessel classes re-derived
                 from the reversed presence profiles). A consistent pipeline
                 returns the mirror image (arrival <-> departure, exit <-> entry,
                 regression <-> emergence). Feature level: forward vs mirrored
                 reversed value. Event level: fraction of forward events whose
                 mirror event sits on the same detection.
  3 accounting   arrivals / departures explained by the edge of the analysed
                 region, or as a fragment (a departure 1-3 frames earlier within
                 reach of an arrival: a link refused, or a cell missed for 1-2
                 frames and not bridged); each with its chance level. Vessels: change of covered
                 centreline length first -> last frame explained by regressing /
                 emerging segments.
  4 sampling     every tracker at 2 h (every other frame) vs 1 h.

Outputs in <out>/: features.csv, removal.csv, reversal_events.csv,
accounting.csv, vessel_accounting.csv, detection_model.csv.

Usage (from the repo root):
    python paper/code/run_event_tests.py [--wells ARTORG_w7] [--repeats 3]
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import fields

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from common import (CROPS, FRAME_H, K_EXIT, RESULTS, SEG, UM_PER_PX, WELLS, estimate_max_dist, run_tracker,
                    symmetric_gate)
from run_vessel_tracker import load_frames
from topology_tracking.detection_model import detection_model
from topology_tracking.events import MIRROR, cell_events, chip_features
from topology_tracking.scaffold import Scaffold
from topology_tracking import symmetric_tracker as ST
from topology_tracking.vessel_tracker import CUES, VESSEL_CUES, FrameDets, run
from vessel_scaffold import classify

OUT = RESULTS / "event_tests"
TRACKERS = ("sym", "sym_vessel", "first_order", "w0", "nn")
METHOD = "sym"
METHOD_CUES = tuple(c for c in CUES if c not in VESSEL_CUES)        # persistence, size, bright
X_TRACKS = (0.05, 0.10, 0.20, 0.30)
X_DETS = (0.10, 0.20)


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


def subset(fd: FrameDets, keep: np.ndarray) -> FrameDets:
    return FrameDets(**{f.name: getattr(fd, f.name)[keep] for f in fields(FrameDets)})


def mirrored(feats: dict) -> dict:
    return {MIRROR.get(k, k): v for k, v in feats.items()}


def rel_change(a: dict, b: dict) -> dict:
    out = {}
    for k in a:
        x, y = a[k], b.get(k, np.nan)
        out[k] = abs(y - x) / abs(x) if (np.isfinite(x) and np.isfinite(y) and abs(x) > 1e-12) else np.nan
    return out


class Well:
    """One ARTORG well. Other datasets provide the same attributes (see
    run_external_events.py): well, frames, labels, T, sc, territory, segments,
    presence, red, dmap, delta, roi, edge_um, um_per_px, frame_h, steps."""
    um_per_px = UM_PER_PX
    frame_h = FRAME_H
    steps = (2,)                     # sampling test: every 2nd frame

    def __init__(self, well):
        self.well = well
        self.ts, self.labels, self.frames = load_frames(well)
        self.T = len(self.ts)
        d = RESULTS / "vessel_scaffold" / well
        self.sc = Scaffold.load(d)
        self.territory = tifffile.imread(d / "territory_ids.tif")
        self.segments = pd.read_csv(d / "segments.csv")
        self.presence = pd.read_csv(d / "presence.csv", index_col=0)
        self.red = pd.read_csv(d / "red_profile.csv", index_col=0)
        self.dmap = []
        for t, fd in zip(self.ts, self.frames):
            mask = tifffile.imread(SEG / well / f"vessels_T{t:04d}.tif") > 0
            dist = ndi.distance_transform_edt(~mask)
            r = np.clip(np.round(fd.pos_img[:, 0]).astype(int), 0, mask.shape[0] - 1)
            c = np.clip(np.round(fd.pos_img[:, 1]).astype(int), 0, mask.shape[1] - 1)
            self.dmap.append(dict(zip(fd.label.astype(int), dist[r, c] * UM_PER_PX)))
        areas = np.concatenate([fd.area for fd in self.frames])
        self.delta = float(np.sqrt(np.median(areas) / np.pi) * UM_PER_PX)
        roi = (tifffile.imread(CROPS / well / "roi_mask.tif") > 0) & ~(tifffile.imread(CROPS / well / "pillar_mask.tif") > 0)
        self.roi = roi
        self.edge_um = ndi.distance_transform_edt(roi) * UM_PER_PX

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

    def reversed_segments(self):
        P = self.presence.to_numpy()[:, ::-1]
        R = self.red.to_numpy()[:, ::-1]
        cls, *_ = classify(ndi.median_filter(P, size=(1, 3), mode="nearest"), R)
        return self.segments.assign(**{"class": cls})


def event_keys(ev, st, fmap, kind):
    """(original frame, label) of the detection carrying each event: the last
    overlapping detection for exits, the event's own detection otherwise."""
    lab = st.set_index(["track_id", "frame_idx"]).label
    keys = set()
    for e in ev[ev.type == kind].itertuples():
        f = e.frame_idx - 1 if kind == "exit" else e.frame_idx
        if (e.track_id, f) in lab.index:
            keys.add((fmap[f], int(lab.loc[(e.track_id, f)])))
    return keys


def accounting(W: Well, ev, st, gate_px):
    """Explain arrivals / departures; chance levels from all detections."""
    cell_d = 2 * W.delta
    ev = ev[ev.type.isin(["arrival", "departure"])]
    H, Wd = W.roi.shape
    at = lambda r, c: W.edge_um[int(np.clip(round(r), 0, H - 1)), int(np.clip(round(c), 0, Wd - 1))]
    near_edge = np.array([at(e.r, e.c) <= cell_d for e in ev.itertuples()])
    det_edge = np.mean([at(r, c) <= cell_d for r, c in st[["centroid_r", "centroid_c"]].to_numpy()[::7]])
    # fragments: arrival at f with a departure at f-1, f-2 or f-3 within reach (a refused link, or 1-2 missed frames)
    dep = ev[ev.type == "departure"]
    arr = ev[ev.type == "arrival"]
    frag = np.zeros(len(arr), bool)
    pos = lambda e: W.sc.to_reference(np.array([[e.r, e.c]]), int(e.frame_idx))[0]
    dep_by_f = {f: np.array([pos(e) for e in g.itertuples()]) for f, g in dep.groupby("frame_idx")}
    for n, e in enumerate(arr.itertuples()):
        p = pos(e)
        for lag in (1, 2, 3):
            q = dep_by_f.get(e.frame_idx - lag)
            if q is not None and (np.hypot(*(q - p).T) <= gate_px * np.sqrt(lag)).any():
                frag[n] = True
                break
    # chance level of a fragment match: expected departures within reach at random positions
    area_px = float(W.roi.sum())
    exp_frag = np.mean([min(1.0, sum(len(dep_by_f.get(f - lag, [])) * np.pi * gate_px ** 2 * lag / area_px
                                     for lag in (1, 2, 3)))
                        for f in arr.frame_idx]) if len(arr) else np.nan
    return dict(n_arrivals=len(arr), n_departures=len(dep),
                arrival_edge_frac=float(near_edge[(ev.type == "arrival").to_numpy()].mean()) if len(arr) else np.nan,
                departure_edge_frac=float(near_edge[(ev.type == "departure").to_numpy()].mean()) if len(dep) else np.nan,
                edge_chance=float(det_edge),
                arrival_fragment_frac=float(frag.mean()) if len(arr) else np.nan,
                fragment_chance=float(exp_frag))


def vessel_accounting(W: Well):
    P = W.presence.to_numpy()
    L = W.segments.set_index("segment_id").loc[W.presence.index, "length_um"].to_numpy()
    cls = W.segments.set_index("segment_id").loc[W.presence.index, "class"].to_numpy()
    d = (P[:, -1] - P[:, 0]) * L
    out = dict(well=W.well, covered_length_change_um=float(d.sum()))
    for c in ("regressing", "emerging", "mask-only", "intermittent", "stable"):
        out[f"change_from_{c}_um"] = float(d[cls == c].sum())
    out["frac_change_explained_by_events"] = (out["change_from_regressing_um"] + out["change_from_emerging_um"]) / \
        out["covered_length_change_um"] if out["covered_length_change_um"] != 0 else np.nan
    return out


def process(W, repeats, rng):
    t0 = time.time()
    well = W.well
    T = W.T
    ident, rmap = {i: i for i in range(T)}, {i: T - 1 - i for i in range(T)}
    seg_rev = W.reversed_segments()
    gate = symmetric_gate(W.labels)["max_dist_px"]
    feat_rows, rem_rows, rev_rows, acc_rows = [], [], [], []
    for name in TRACKERS:
        fwd = run_named(name, W.frames, W.labels)
        F, st, ev = W.table(fwd, ident, T, W.segments)
        feat_rows.append(dict(well=well, tracker=name, condition="forward", **F))
        acc_rows.append(dict(well=well, tracker=name, **accounting(W, ev, st, gate)))
        # reversal
        rv = run_named(name, W.frames[::-1], W.labels[::-1])
        Fr, str_, evr = W.table(rv, rmap, T, seg_rev)
        feat_rows.append(dict(well=well, tracker=name, condition="reversed_mirrored", **mirrored(Fr)))
        for a, b in (("arrival", "departure"), ("departure", "arrival"), ("exit", "entry"), ("entry", "exit")):
            ka, kb = event_keys(ev, st, ident, a), event_keys(evr, str_, rmap, b)
            rev_rows.append(dict(well=well, tracker=name, forward_event=a, n_forward=len(ka), n_mirror=len(kb),
                                 frac_matched=len(ka & kb) / len(ka) if ka else np.nan))
        # removal (b): detections
        for X in X_DETS:
            for rep in range(repeats):
                keep = [rng.random(len(fd.label)) >= X for fd in W.frames]
                fr = [subset(fd, k) for fd, k in zip(W.frames, keep)]
                lab = [np.where(np.isin(L, fd.label[~k]), 0, L) for L, fd, k in zip(W.labels, W.frames, keep)]
                Fb, _, _ = W.table(run_named(name, fr, lab), ident, T, W.segments)
                rem_rows.append(dict(well=well, tracker=name, kind="detections", X=X, rep=rep, **rel_change(F, Fb)))
        if name == METHOD:
            for X in X_TRACKS:                                  # removal (a): tracks
                for rep in range(repeats):
                    ids = fwd.track_id.unique()
                    kept = rng.choice(ids, size=int(round(len(ids) * (1 - X))), replace=False)
                    Fa, _, _ = W.table(fwd[fwd.track_id.isin(kept)], ident, T, W.segments)
                    rem_rows.append(dict(well=well, tracker=name, kind="tracks", X=X, rep=rep, **rel_change(F, Fa)))
        for k in W.steps:                                          # sampling: every k-th frame, every tracker
            tk = run_named(name, W.frames[::k], W.labels[::k])
            Fk, _, _ = W.table(tk, {i: k * i for i in range(len(W.frames[::k]))}, len(W.frames[::k]),
                               W.segments, frame_h=k * W.frame_h)
            feat_rows.append(dict(well=well, tracker=name, condition=f"every_{k}_frames", **Fk))
        print(f"{well} {name}: {time.time() - t0:.0f}s", flush=True)
    det = detection_model(W.frames)
    det_row = dict(well=well, **{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in det.items()})
    return feat_rows, rem_rows, rev_rows, acc_rows, vessel_accounting(W), det_row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wells", nargs="+", default=WELLS)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--trackers", nargs="+", default=None, help="subset of run_event_tests.TRACKERS")
    ap.add_argument("--out", default=None, help="output folder in <results> (default: event_tests)")
    args = ap.parse_args()
    out = RESULTS / args.out if args.out else OUT
    if args.trackers:
        global TRACKERS
        TRACKERS = tuple(args.trackers)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    R = [process(Well(w), args.repeats, rng) for w in args.wells]
    for i, name in enumerate(("features", "removal", "reversal_events", "accounting")):
        pd.DataFrame([r for x in R for r in x[i]]).to_csv(out / f"{name}.csv", index=False)
    pd.DataFrame([x[4] for x in R]).to_csv(out / "vessel_accounting.csv", index=False)
    pd.DataFrame([x[5] for x in R]).to_csv(out / "detection_model.csv", index=False)


if __name__ == "__main__":
    main()
