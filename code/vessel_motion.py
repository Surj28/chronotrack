"""Joint tracking, step 1 — how much does the vessel network move and change per hour?

Per well, for every pair of consecutive frames (t, t+1):
  motion       dense TV-L1 optical flow of the red channel (ROI crops, 4x
               downsampled), split into a global shift (median flow over vessel
               pixels: stage drift) and local deformation (flow - shift),
               reported on vessel-skeleton pixels;
  registration vessel-mask Dice between t and t+1, raw and after warping t+1
               back onto t with the flow (higher after = flow captures motion);
  persistence  fraction of the t skeleton within `tol` px of the t+1 skeleton,
               raw and after warping; 1 - warped persistence ~ remodelling;
  cells        where tracks exist (results/tracking/<well>/tracks.csv): cell
               step length, and the step left after subtracting the vessel flow
               at the cell's position (how much of a cell's step is the vessel
               moving under it).
No parameters are tuned per sample; every number is measured.

Usage (from the repo root):
    python paper/code/vessel_motion.py --crops paper/results/roi_crops \
        --seg paper/results/segmentation --tracks paper/results/tracking \
        --out paper/results/vessel_motion
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from skimage.registration import optical_flow_tvl1

UM_PER_PX = 0.65
DS = 4          # registration on 4x-downsampled frames (cells ~5 px, vessels ~3-10 px wide there)


def norm_red(red, roi):
    lo, hi = np.percentile(red[roi], [1, 99.8])
    return np.clip((red - lo) / (hi - lo + 1e-6), 0, 1)


def down(a):
    h, w = (a.shape[0] // DS) * DS, (a.shape[1] // DS) * DS
    return a[:h, :w].reshape(h // DS, DS, w // DS, DS).mean((1, 3))


def warp(img, flow_full, order=0):
    """Sample img at x + flow (bring frame t+1 content back onto frame t's grid)."""
    rr, cc = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]), indexing="ij")
    return ndi.map_coordinates(img.astype(np.float32), [rr + flow_full[0], cc + flow_full[1]],
                               order=order, mode="constant")


def dice(a, b):
    return float(2 * (a & b).sum() / max(a.sum() + b.sum(), 1))


def persistence(skel_a, skel_b, tol):
    if not skel_a.any():
        return float("nan")
    d = ndi.distance_transform_edt(~skel_b)
    return float((d[skel_a] <= tol).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", type=Path, required=True)
    ap.add_argument("--seg", type=Path, required=True)
    ap.add_argument("--tracks", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tol", type=float, default=3.0, help="skeleton persistence tolerance (px, full res)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows, cell_rows = [], []
    for w in sorted(args.crops.glob("ARTORG_w*"), key=lambda p: int(p.name.split("_")[-1])):
        meta = json.loads((w / "roi.json").read_text())
        roi = (tifffile.imread(w / "roi_mask.tif") > 0) & ~(tifffile.imread(w / "pillar_mask.tif") > 0)
        ts = meta["timepoints"]
        tr_path = args.tracks / w.name / "tracks.csv" if args.tracks else None
        tracks = pd.read_csv(tr_path) if tr_path and tr_path.exists() else None
        prev = None
        for t in ts:
            red = norm_red(tifffile.imread(w / f"T{t:04d}_C1.tif").astype(np.float32), roi)
            ves = (tifffile.imread(args.seg / w.name / f"vessels_T{t:04d}.tif") > 0) & roi
            cur = dict(t=t, red_ds=ndi.gaussian_filter(down(red * roi), 1), ves=ves, skel=skeletonize(ves))
            if prev is not None:
                t0 = time.time()
                v, u = optical_flow_tvl1(prev["red_ds"], cur["red_ds"])
                flow = np.stack([ndi.zoom(v, DS, order=1), ndi.zoom(u, DS, order=1)]) * DS
                flow = np.pad(flow, ((0, 0), (0, roi.shape[0] - flow.shape[1]), (0, roi.shape[1] - flow.shape[2])),
                              mode="edge")
                sk = prev["skel"]
                fv, fu = flow[0][prev["ves"]], flow[1][prev["ves"]]
                drift = np.array([np.median(fv), np.median(fu)])
                defo = np.hypot(flow[0][sk] - drift[0], flow[1][sk] - drift[1])
                ves_back = warp(cur["ves"], flow) > 0.5
                skel_back = warp(cur["skel"], flow) > 0.5
                rows.append(dict(
                    well=w.name, t=prev["t"], t_next=t,
                    drift_um=float(np.hypot(*drift) * UM_PER_PX),
                    deform_median_um=float(np.median(defo) * UM_PER_PX),
                    deform_p95_um=float(np.percentile(defo, 95) * UM_PER_PX),
                    dice_raw=dice(prev["ves"], cur["ves"]), dice_registered=dice(prev["ves"], ves_back),
                    skel_persist_raw=persistence(sk, cur["skel"], args.tol),
                    skel_persist_registered=persistence(sk, skel_back, args.tol),
                    seconds=round(time.time() - t0, 1)))
                if tracks is not None:
                    a = tracks[tracks.frame_idx == ts.index(prev["t"])][["track_id", "centroid_r", "centroid_c", "inside_vessel"]]
                    b = tracks[tracks.frame_idx == ts.index(t)][["track_id", "centroid_r", "centroid_c"]]
                    m = a.merge(b, on="track_id", suffixes=("_a", "_b"))
                    if len(m):
                        r = np.clip(m.centroid_r_a.round().astype(int), 0, roi.shape[0] - 1)
                        c = np.clip(m.centroid_c_a.round().astype(int), 0, roi.shape[1] - 1)
                        dr = m.centroid_r_b - m.centroid_r_a; dc = m.centroid_c_b - m.centroid_c_a
                        step = np.hypot(dr, dc)
                        resid = np.hypot(dr - flow[0][r, c], dc - flow[1][r, c])
                        for s, rs, iv in zip(step, resid, m.inside_vessel):
                            cell_rows.append(dict(well=w.name, t=prev["t"], step_um=s * UM_PER_PX,
                                                  residual_um=rs * UM_PER_PX, on_vessel=bool(iv)))
                print(f"{w.name} T{prev['t']:02d}->T{t:02d}: drift {rows[-1]['drift_um']:.1f} um, "
                      f"deform median {rows[-1]['deform_median_um']:.1f} / p95 {rows[-1]['deform_p95_um']:.1f} um, "
                      f"Dice {rows[-1]['dice_raw']:.3f}->{rows[-1]['dice_registered']:.3f}, "
                      f"skeleton kept {rows[-1]['skel_persist_raw']:.2f}->{rows[-1]['skel_persist_registered']:.2f} "
                      f"({rows[-1]['seconds']}s)", flush=True)
            prev = cur
    pd.DataFrame(rows).to_csv(args.out / "vessel_motion.csv", index=False)
    if cell_rows:
        pd.DataFrame(cell_rows).to_csv(args.out / "cell_steps_vs_vessel_flow.csv", index=False)


if __name__ == "__main__":
    main()
