"""Crop a synthetic sequence (and its vessel layer) to a small square window:
the sample in data/sample/ was made with the defaults, from the full data set.

Cells whose true centre (gt.csv r, c) lies outside the window are removed from
the label maps, and their rows from gt.csv; the other rows are shifted to the
window's coordinates (a merge into a removed cell becomes merged_into = 0).
The vessel layer is cropped the same way (the flows on their downsampled grid,
so the window's origin and size are multiples of the flows' downsampling);
segments.csv keeps the segments that reach into the window, with their full
length. All frames are kept.

Usage (from the repo root):
    python code/make_sample.py --data <full data folder> [--seq seq1_density1_gentle_seed0]
        [--origin 1536 1536] [--size 1280] [--out data/sample]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

REPO = Path(__file__).resolve().parent.parent


def crop_sequence(src: Path, dst: Path, r0: int, c0: int, size: int):
    lab = tifffile.imread(src / "cells.tif")[:, r0:r0 + size, c0:c0 + size].copy()
    green = tifffile.imread(src / "green.tif")[:, r0:r0 + size, c0:c0 + size]
    gt = pd.read_csv(src / "gt.csv")
    inside = (gt.r >= r0) & (gt.r < r0 + size) & (gt.c >= c0) & (gt.c < c0 + size)
    det = gt.label > 0
    for t in range(len(lab)):                              # detections of cells centred outside the window
        drop = gt.label[det & ~inside & (gt.frame == t)].to_numpy()
        lab[t][np.isin(lab[t], drop)] = 0
    present = [set(np.unique(L)) for L in lab]
    keep = inside & (~det | np.array([l in present[f] for f, l in zip(gt.frame, gt.label)]))
    gt = gt[keep].copy()
    gt["merged_into"] = [m if m == 0 or m in present[f] else 0 for f, m in zip(gt.frame, gt.merged_into)]
    gt["r"] -= r0
    gt["c"] -= c0
    dst.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(dst / "cells.tif", lab, compression="zlib")
    tifffile.imwrite(dst / "green.tif", green, compression="zlib")
    gt.to_csv(dst / "gt.csv", index=False)
    info = json.loads((src / "info.json").read_text())
    info.update(sequence=dst.name, n_detections_per_frame=float((gt.label > 0).groupby(gt.frame).sum().mean()),
                frac_missed_or_merged=float((gt.label < 0).mean()),
                crop=dict(of=src.name, origin_rc=[r0, c0], size=size))
    (dst / "info.json").write_text(json.dumps(info, indent=2) + "\n")


def crop_vessels(src: Path, dst: Path, r0: int, c0: int, size: int):
    dst.mkdir(parents=True, exist_ok=True)
    sl = (slice(r0, r0 + size), slice(c0, c0 + size))
    tifffile.imwrite(dst / "vessels.tif", tifffile.imread(src / "vessels.tif")[(slice(None),) + sl], compression="zlib")
    for f in ("union_mask", "segment_ids", "territory_ids"):
        tifffile.imwrite(dst / f"{f}.tif", tifffile.imread(src / f"{f}.tif")[sl], compression="zlib")
    ids = set(np.unique(tifffile.imread(dst / "territory_ids.tif"))) | set(np.unique(tifffile.imread(dst / "segment_ids.tif")))
    seg = pd.read_csv(src / "segments.csv")
    seg[seg.segment_id.isin(ids)].to_csv(dst / "segments.csv", index=False)
    z = np.load(src / "flows.npz")
    ds = int(z["ds"])
    if r0 % ds or c0 % ds or size % ds:
        raise SystemExit(f"origin and size must be multiples of the flows' downsampling ({ds})")
    np.savez_compressed(dst / "flows.npz", ds=z["ds"],
                        flows=z["flows"][:, :, r0 // ds:(r0 + size) // ds, c0 // ds:(c0 + size) // ds])


def main():
    ap = argparse.ArgumentParser(description="Crop a synthetic sequence and its vessel layer to a square window.")
    ap.add_argument("--data", type=Path, required=True, help="folder with the full sequences")
    ap.add_argument("--seq", default="seq1_density1_gentle_seed0")
    ap.add_argument("--origin", nargs=2, type=int, default=[1536, 1536], metavar=("ROW", "COL"))
    ap.add_argument("--size", type=int, default=1280)
    ap.add_argument("--out", type=Path, default=REPO / "data" / "sample")
    args = ap.parse_args()
    (r0, c0), s = args.origin, args.size
    crop_vessels(args.data / "vessels" / args.seq.split("_")[0], args.out / "vessels" / args.seq.split("_")[0], r0, c0, s)
    crop_sequence(args.data / args.seq, args.out / args.seq, r0, c0, s)
    print(args.out / args.seq)


if __name__ == "__main__":
    main()
