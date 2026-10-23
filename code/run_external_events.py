"""Chip event table and its label-free tests on the external microvasculature-on-chip
videos, with exactly the trackers and tests of run_event_tests.py.

Song et al. 2021 (Videos S1, S3, S5; prepared by prepare_song.py): the tracked
cells are the perfused NK cells; vessels from the red channel. Pixel size
2.44 um, 5 min per frame; the sampling test re-runs every pipeline at 10, 20,
40 and 60 min.

Outputs in <results>/event_tests_external/: features.csv, removal.csv,
reversal_events.csv, accounting.csv, vessel_accounting.csv.

Usage (from the repo root):
    python paper/code/run_external_events.py [--videos S1 S3 S5] [--repeats 3]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi

import run_event_tests as RET
from common import RESULTS
from prepare_song import FRAME_MIN, UM_PER_PX as SONG_UM, load_channels
from topology_tracking.scaffold import Scaffold
from topology_tracking.vessel_tracker import frame_detections, wall_map
from vessel_scaffold import segments_table

OUT = RESULTS / "event_tests_external"


class SongVideo(RET.Well):
    um_per_px = SONG_UM
    frame_h = FRAME_MIN / 60
    steps = (2, 4, 8, 12)            # 10, 20, 40, 60 min

    def __init__(self, video: str):
        d = RESULTS / "external" / "song2021" / video
        self.well = f"song2021_{video}"
        lab = tifffile.imread(d / "nk_labels.tif")
        white = tifffile.imread(d / "white.tif").astype(np.float32)
        V = tifffile.imread(d / "vessel_masks.tif") > 0
        U = tifffile.imread(d / "union_mask.tif") > 0
        sk = tifffile.imread(d / "segment_ids.tif") > 0
        flows, ds = None, 1
        if (d / "flows_to_T1.npz").exists():
            z = np.load(d / "flows_to_T1.npz")
            flows, ds = z["flows"], int(z["ds"])
        self.sc = Scaffold.from_masks(U, sk, flows, ds)
        _, vessel_img, _ = load_channels(video)
        seg, self.territory, P, R = segments_table(V, vessel_img, self.sc, um_per_px=self.um_per_px)
        self.segments = seg
        self.presence = pd.DataFrame(P, index=pd.Index(seg.segment_id, name="segment_id"))
        self.red = pd.DataFrame(R, index=pd.Index(seg.segment_id, name="segment_id"))
        wm = wall_map(self.sc)
        self.T = len(lab)
        self.labels = [lab[t] for t in range(self.T)]
        self.frames = [frame_detections(lab[t], white[t], self.sc, t, wm) for t in range(self.T)]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", default=["S1", "S3", "S5"])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--trackers", nargs="+", default=None, help="subset of run_event_tests.TRACKERS")
    ap.add_argument("--out", default=None, help="output folder in <results> (default: event_tests_external)")
    args = ap.parse_args()
    out = RESULTS / args.out if args.out else OUT
    if args.trackers:
        RET.TRACKERS = tuple(args.trackers)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    R = [RET.process(SongVideo(v), args.repeats, rng) for v in args.videos]
    for i, name in enumerate(("features", "removal", "reversal_events", "accounting")):
        pd.DataFrame([r for x in R for r in x[i]]).to_csv(out / f"{name}.csv", index=False)
    pd.DataFrame([x[4] for x in R]).to_csv(out / "vessel_accounting.csv", index=False)
    pd.DataFrame([x[5] for x in R]).to_csv(out / "detection_model.csv", index=False)


if __name__ == "__main__":
    main()
