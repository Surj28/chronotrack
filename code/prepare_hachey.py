"""Prepare Hachey et al. 2021 (Lab Chip, doi:10.1039/d0lc01216e; CC BY-NC 3.0)
Supplementary Video 1 for the chip event table: vessel network formation and
tumor-cluster growth (paper/data/external/hachey2021_hughes/movies/...s010.mp4).

Channels from the RGB composite (vessels red, tumor cells cyan):
    vessels  R - max(G, B)
    tumor    min(G, B) - R
Vessels: common.vessels_from_composite (smoothing, one Otsu threshold, the
ARTORG size rules in um^2); the composite is already contrast-adjusted, so the
Sato ridge filter used on raw ARTORG is not applied. No scaffold here: the
network forms and remodels during the movie (mask IoU 0.17-0.29 between frames
1-3 days apart), so the single whole-sequence scaffold used for ARTORG and Song
does not apply; its temporal support is decided downstream. Tumor: Otsu
threshold (every 10th frame), objects smaller than 20 px removed, connected
components = clusters (early frames: single cells).

ASSUMED, not stated in the source: pixel size ~1.13 um (10x objective on a
Leica SP8 at 1024 px, consistent with the ~1 mm-wide chamber filling the
frame) and frame interval ~26.9 min (10 days over 536 frames, evenly spaced).
Both only scale absolute rates.

Outputs (<results>/external/hachey2021/V1/): vessel_masks.tif, vessel.tif,
tumor.tif, tumor_labels.tif, info.json.

Usage (from the repo root):
    python paper/code/prepare_hachey.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import imageio.v3 as iio
import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects

from common import vessels_from_composite

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "data" / "external" / "hachey2021_hughes" / "movies" / "LC-021-D0LC01216E-s010.mp4"
OUT = HERE.parent / "results" / "external" / "hachey2021" / "V1"
UM_PER_PX = 1.13          # assumed
FRAME_MIN = 10 * 24 * 60 / 535   # assumed: 10 days, even spacing
MIN_TUMOR_PX = 20


def load_channels():
    rgb = np.stack([f[..., :3] for f in iio.imiter(SRC)])
    R, G, B = (rgb[..., k].astype(np.int16) for k in range(3))
    vessel = np.clip(R - np.maximum(G, B), 0, 255).astype(np.uint8)
    tumor = np.clip(np.minimum(G, B) - R, 0, 255).astype(np.uint8)
    return vessel, tumor


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    vessel, tumor = load_channels()
    T = len(vessel)
    V, vthr = vessels_from_composite(vessel, UM_PER_PX, sub=10)
    thr = float(threshold_otsu(tumor[::10]))
    tl = np.zeros(tumor.shape, np.int32)
    for t in range(T):
        tl[t] = ndi.label(remove_small_objects(tumor[t] > thr, MIN_TUMOR_PX))[0]
    tifffile.imwrite(OUT / "vessel_masks.tif", V.astype(np.uint8), compression="zlib")
    tifffile.imwrite(OUT / "vessel.tif", vessel, compression="zlib")
    tifffile.imwrite(OUT / "tumor.tif", tumor, compression="zlib")
    tifffile.imwrite(OUT / "tumor_labels.tif", tl, compression="zlib")
    info = dict(source=SRC.name, n_frames=T, frame_min_assumed=FRAME_MIN, um_per_px_assumed=UM_PER_PX,
                vessel_threshold=vthr, tumor_threshold=thr, clusters_first=int(tl[0].max()),
                clusters_last=int(tl[-1].max()), vessel_cover_first=float(V[0].mean()),
                vessel_cover_median=float(np.median(V.mean((1, 2)))), vessel_cover_last=float(V[-1].mean()))
    (OUT / "info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
