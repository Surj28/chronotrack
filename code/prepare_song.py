"""Prepare Song et al. 2021 (Front Immunol, doi:10.3389/fimmu.2021.733317; CC BY 4.0)
for the event pipeline: NK cells, vessels and tumor clusters from the published
RGB videos (Supplementary Videos S1, S3, S5; paper/data/external/song2021_jeon).

Each frame is cropped to remove the burned-in clock and scale bar (rows 68-467,
cols 56-455; 2.44 um per pixel from the 200 um bar = 82 px; 5 min per frame).
Channels from the RGB composite:
    NK cells   white   = min(R, G, B)
    vessels    red     = R - max(G, B)
    tumor      green   = G - max(R, B)
NK cells: Gaussian smoothing (sigma 1 px, as for the vessels) and one Otsu
threshold on the white channel (every 5th frame), connected components, then
the same size prior as ARTORG (two-component Gaussian mixture on log area of the
first frame, mu +- 3 sd). Without the smoothing, single-pixel noise specks
outnumbered the cells at T0 in S3 and S5 and the size prior selected them.
Objects are ~3-6 px across, below the size range of Cellpose-SAM, so
thresholding replaces it here.
Vessels: common.vessels_from_composite (smoothing, one Otsu threshold, the
ARTORG size rules in um^2); the composite is already contrast-adjusted, so the
Sato ridge filter used on raw ARTORG is not applied. Then the temporally
consistent scaffold (topology_tracking.scaffold.scaffold_from_stack, no
downsampling).
Tumor clusters: Otsu threshold on the tumor channel, components larger than
one median NK cell.

Outputs per video (<out>/<video>/): nk_labels.tif (T, H, W), white.tif,
vessel_masks.tif, tumor_labels.tif, union_mask.tif, segment_ids.tif (centreline),
info.json.

Usage (from the repo root):
    python paper/code/prepare_song.py
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

from cell_size_prior import fit_mixture
from common import vessels_from_composite
from topology_tracking.scaffold import scaffold_from_stack

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "data" / "external" / "song2021_jeon"
OUT = HERE.parent / "results" / "external" / "song2021"
CROP = (slice(68, 468), slice(56, 456))
UM_PER_PX = 200 / 82
FRAME_MIN = 5
VIDEOS = {"S1": "figshare_30873682.avi", "S3": "figshare_30873688.avi", "S5": "figshare_30873694.avi"}


def load_channels(name: str):
    """(white, vessel, tumor) stacks (T, H, W) of one video, cropped."""
    rgb = np.stack([f[CROP][..., :3] for f in iio.imiter(SRC / VIDEOS[name])]).astype(np.float32)
    white = rgb.min(axis=3)
    vessel = np.clip(rgb[..., 0] - rgb[..., 1:].max(axis=3), 0, None)
    tumor = np.clip(rgb[..., 1] - np.maximum(rgb[..., 0], rgb[..., 2]), 0, None)
    return white, vessel, tumor


def main():
    for name, fname in VIDEOS.items():
        out = OUT / name
        out.mkdir(parents=True, exist_ok=True)
        white, vessel, tumor = load_channels(name)
        T = len(white)

        # NK cells
        white_sm = np.stack([ndi.gaussian_filter(w, 1.0) for w in white])
        thr = float(threshold_otsu(white_sm[::5]))
        lab = np.zeros(white.shape, np.int32)
        for t in range(T):
            lab[t] = ndi.label(white_sm[t] > thr)[0]
        ids0 = np.unique(lab[0]); ids0 = ids0[ids0 > 0]
        a0 = ndi.sum(np.ones_like(lab[0]), lab[0], ids0)
        fit = fit_mixture(np.log10(a0))
        lo, hi = 10 ** fit["lo"], 10 ** fit["hi"]
        n_kept = 0
        for t in range(T):
            ids = np.unique(lab[t]); ids = ids[ids > 0]
            a = ndi.sum(np.ones_like(lab[t]), lab[t], ids)
            bad = ids[(a < lo) | (a > hi)]
            lab[t][np.isin(lab[t], bad)] = 0
            lab[t] = ndi.label(lab[t] > 0)[0]
            n_kept += int(lab[t].max())
        med_area = float(np.median(ndi.sum(np.ones_like(lab[0]), lab[0], np.arange(1, lab[0].max() + 1))))

        # vessels
        V, vthr = vessels_from_composite(vessel, UM_PER_PX)
        sc, sc_info = scaffold_from_stack(V, vessel, roi=None, ds=1)

        # tumor clusters
        tthr = float(threshold_otsu(tumor[::5]))
        tl = np.zeros(white.shape, np.int32)
        for t in range(T):
            tl[t] = ndi.label(remove_small_objects(tumor[t] > tthr, max(int(med_area), 2)))[0]

        tifffile.imwrite(out / "nk_labels.tif", lab, compression="zlib")
        tifffile.imwrite(out / "white.tif", np.clip(white, 0, 255).astype(np.uint8), compression="zlib")
        tifffile.imwrite(out / "vessel_masks.tif", V.astype(np.uint8), compression="zlib")
        tifffile.imwrite(out / "tumor_labels.tif", tl, compression="zlib")
        tifffile.imwrite(out / "union_mask.tif", sc.union.astype(np.uint8), compression="zlib")
        tifffile.imwrite(out / "segment_ids.tif", sc.skeleton.astype(np.uint8), compression="zlib")
        if sc.flows is not None:
            np.savez_compressed(out / "flows_to_T1.npz", flows=sc.flows, ds=sc.flow_ds)
        info = dict(video=name, source=fname, n_frames=T, frame_min=FRAME_MIN, um_per_px=UM_PER_PX,
                    white_threshold=thr, nk_area_px=[lo, hi], nk_per_frame=n_kept / T, nk_median_area_px=med_area,
                    vessel_threshold=vthr, tumor_threshold=tthr, tumor_clusters_per_frame=float(np.mean([tl[t].max() for t in range(T)])),
                    vessel_cover=float(V.mean()), **{f"scaffold_{k}": v for k, v in sc_info.items()})
        (out / "info.json").write_text(json.dumps(info, indent=2))
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in info.items()}), flush=True)


if __name__ == "__main__":
    main()
