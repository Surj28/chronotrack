"""Chip event table and its label-free tests on Hachey et al. 2021 Video 1
(prepared by prepare_hachey.py), with exactly the trackers and tests of
run_event_tests.py.

Only days 0-5 (frames 0-267) are analysed: the tumor cells are still single,
slowly moving objects there, which is what a one-to-one cell tracker models.
Later they grow into clusters and merge into one mass (29 % of the field by
day 10), which needs merge / split logic instead (user, 2026-10-15).

Tracked objects: tumor objects of prepare_hachey.py. The ARTORG size prior
(fitted at T0) is not applied: the cells grow, and it would drop 26 % of the
objects over days 0-5 (more than half by day 4.5) as artificial departures.

Vessels: the network forms from single endothelial cells and keeps remodelling
(results/vessel_persistence), so one whole-sequence scaffold does not fit.
The scaffold's temporal support is chosen from the data
(vessel_scaffold.adaptive_windows: halve until most of the centreline is
stable). Each frame uses its window's scaffold for vessel context (on vessel,
wall distance, axis) and segment ids; cell positions stay in image
coordinates, so that tracks do not jump at window boundaries (tissue
deformation is ~0.1-0.3 px per frame here, against a p95 step of 3.3 px).
Vessel features and the reversal test are computed per window.

Frame interval (~26.9 min) and pixel size (~1.13 um) are ASSUMED (see
prepare_hachey.py); they only scale rates.

Outputs in <results>/event_tests_hachey/: features.csv, removal.csv,
reversal_events.csv, accounting.csv, vessel_accounting.csv, windows.csv,
segments.csv.

Usage (from the repo root):
    python paper/code/run_hachey_events.py [--repeats 3]
"""
from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi

import run_event_tests as RET
from common import K_EXIT, RESULTS
from prepare_hachey import FRAME_MIN, OUT as SRC, UM_PER_PX
from topology_tracking.events import cell_events, chip_features
from topology_tracking.vessel_tracker import frame_detections, wall_map
from vessel_scaffold import adaptive_windows, classify

OUT = RESULTS / "event_tests_hachey"
N_FRAMES = 268                   # days 0-5


class _ImageCoords:
    """Positions stay in image coordinates (see module docstring)."""
    @staticmethod
    def to_reference(p, t):
        return p


class HacheyVideo(RET.Well):
    um_per_px = UM_PER_PX
    frame_h = FRAME_MIN / 60
    steps = (2, 4, 8)                # ~0.9, 1.8, 3.6 h

    def __init__(self, n_frames: int = N_FRAMES):
        self.well = "hachey2021_V1"
        lab = tifffile.imread(SRC / "tumor_labels.tif")[:n_frames]
        img = tifffile.imread(SRC / "tumor.tif")[:n_frames].astype(np.float32)
        V = tifffile.imread(SRC / "vessel_masks.tif")[:n_frames] > 0
        ves = tifffile.imread(SRC / "vessel.tif")[:n_frames]
        self.T = len(lab)
        self.windows = adaptive_windows(V, ves, None, ds=2, um_per_px=self.um_per_px)
        segs, off = [], 0
        for k, w in enumerate(self.windows):           # segment ids unique over windows
            s = w["segments"].copy()
            s["segment_id"] += off
            s["window"] = k
            w["territory"] = np.where(w["territory"] > 0, w["territory"] + off, 0)
            w["wmap"] = wall_map(w["scaffold"])
            off += len(s)
            segs.append(s)
        self.segments = pd.concat(segs, ignore_index=True)
        self.win_of = np.concatenate([np.full(w["stop"] - w["start"], k) for k, w in enumerate(self.windows)])
        # vessel_accounting uses the first and last presence column: per window here
        self.presence = pd.DataFrame({0: np.concatenate([w["P"][:, 0] for w in self.windows]),
                                      1: np.concatenate([w["P"][:, -1] for w in self.windows])},
                                     index=pd.Index(self.segments.segment_id, name="segment_id"))
        self.sc = _ImageCoords()
        self.labels = [lab[t] for t in range(self.T)]
        self.frames = []
        for t in range(self.T):
            w = self.windows[self.win_of[t]]
            fd = frame_detections(lab[t], img[t], w["scaffold"], t - w["start"], w["wmap"])
            self.frames.append(replace(fd, pos=fd.pos_img.copy()))
        self.dmap = []
        for t, fd in enumerate(self.frames):
            dist = ndi.distance_transform_edt(~V[t])
            r = np.clip(np.round(fd.pos_img[:, 0]).astype(int), 0, V.shape[1] - 1)
            c = np.clip(np.round(fd.pos_img[:, 1]).astype(int), 0, V.shape[2] - 1)
            self.dmap.append(dict(zip(fd.label.astype(int), dist[r, c] * self.um_per_px)))
        areas = np.concatenate([fd.area for fd in self.frames if len(fd.area)])
        self.delta = float(np.sqrt(np.median(areas) / np.pi) * self.um_per_px)
        self.roi = np.ones(V.shape[1:], bool)
        border = np.zeros(V.shape[1:], bool)
        border[1:-1, 1:-1] = True
        self.edge_um = ndi.distance_transform_edt(border) * self.um_per_px   # distance to the field edge

    def table(self, tracks, fmap, n_frames, segments, frame_h=None):
        st = self.states(tracks, fmap)
        ev = cell_events(st, n_frames, K_EXIT)
        if len(ev):                                    # segment of each event: its frame's window
            seg = np.zeros(len(ev), int)
            pts = ev[["r", "c"]].to_numpy(float)
            for f, idx in ev.groupby("frame_idx").indices.items():
                t = fmap[int(f)]
                w = self.windows[self.win_of[t]]
                p = w["scaffold"].to_reference(pts[idx], t - w["start"])
                H, Wd = w["territory"].shape
                r = np.clip(np.round(p[:, 0]).astype(int), 0, H - 1)
                c = np.clip(np.round(p[:, 1]).astype(int), 0, Wd - 1)
                seg[idx] = w["territory"][r, c]
            ev["segment"] = seg
        return chip_features(st, ev, segments, n_frames, frame_h or self.frame_h, self.um_per_px), st, ev

    def reversed_segments(self):
        cls = [classify(ndi.median_filter(w["P"][:, ::-1], size=(1, 3), mode="nearest"), w["R"][:, ::-1])[0]
               for w in self.windows]
        return self.segments.assign(**{"class": np.concatenate(cls)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--trackers", nargs="+", default=None, help="subset of run_event_tests.TRACKERS")
    ap.add_argument("--out", default=None, help="output folder in <results> (default: event_tests_hachey)")
    args = ap.parse_args()
    out = RESULTS / args.out if args.out else OUT
    if args.trackers:
        RET.TRACKERS = tuple(args.trackers)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    W = HacheyVideo()
    pd.DataFrame([dict(window=k, start=w["start"], stop=w["stop"], hours=(w["stop"] - w["start"]) * W.frame_h,
                       stable_frac=w["stable_frac"], n_segments=len(w["segments"]), **w["info"])
                  for k, w in enumerate(W.windows)]).to_csv(out / "windows.csv", index=False)
    W.segments.to_csv(out / "segments.csv", index=False)
    R = RET.process(W, args.repeats, rng)
    for i, name in enumerate(("features", "removal", "reversal_events", "accounting")):
        pd.DataFrame(R[i]).to_csv(out / f"{name}.csv", index=False)
    pd.DataFrame([R[4]]).to_csv(out / "vessel_accounting.csv", index=False)
    pd.DataFrame([R[5]]).to_csv(out / "detection_model.csv", index=False)


if __name__ == "__main__":
    main()
