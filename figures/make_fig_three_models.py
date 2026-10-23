"""Figure: the three microvasculature-on-chip models used in the paper, side by side.

Columns: the Guenat lab's model (ARTORG Center, University of Bern; sample
B34/ARTORG, one well, a crop of the analysed region so that single tumor cells
are visible at print size), Song et al. 2021 (Video S1) and
Hachey et al. 2021 (Video 1). Rows: start, middle and end of each time-lapse.

Our crop is chosen automatically from the step-4 masks: windows fully inside
the region of interest, clear of the central pillar, with 30-60 % vessel cover
at the last frame; among those, last-frame cell count closest to the 75th
percentile of all windows, then
  --crop-rule steady (default, used in the paper): last-frame count within 20 %
      of that percentile and the most cells already at the first frame, so
      every row shows cells;
  --crop-rule arrival: the largest increase in cells since the first frame, so
      the column shows tumor cells entering the network (empty at 0 h). Display contrast for our
panels is fixed over time (background = median of the region of interest, white
point = 99.7th/99.8th percentile at the last frame), so brightness changes are
real. The external movies are shown as published, cropped only to remove the
burned-in clock and scale bar (Song) and resized.

Usage (from the repo root):
    python paper/figures/make_fig_three_models.py
    python paper/figures/make_fig_three_models.py --well ARTORG_w8 --crop-um 400 --rows 2
"""
import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "STIXGeneral"], "mathtext.fontset": "stix"})   # match the paper (Times)
import tifffile
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
CROPS = PAPER / "results" / "roi_crops"
SEG = PAPER / "results" / "segmentation"
EXT = PAPER / "data" / "external"
UM_PER_PX = 0.65

# Song S1: burned-in clock (top-left) and 200 um scale bar (bottom-right, 82 px)
SONG_FILE = EXT / "song2021_jeon" / "figshare_30873682.avi"
SONG_CROP = (slice(68, 468), slice(56, 456))
SONG_UM_PER_PX = 200 / 82
SONG_MIN_PER_FRAME = 5
HACHEY_FILE = EXT / "hachey2021_hughes" / "movies" / "LC-021-D0LC01216E-s010.mp4"

ap = argparse.ArgumentParser()
ap.add_argument("--well", default="ARTORG_w7")
ap.add_argument("--crop-um", type=float, default=450.0)
ap.add_argument("--rows", type=int, choices=(2, 3), default=3, help="time points: 3 = start/middle/end, 2 = start/end")
ap.add_argument("--width-in", type=float, default=3.5, help="figure width (IEEE single column = 3.5 in)")
ap.add_argument("--crop-rule", choices=("arrival", "steady"), default="steady")
ap.add_argument("--our-title", default="ARTORG dataset", help="column title for the ARTORG data")
ap.add_argument("--out", type=Path, default=HERE / "fig_three_models")
ap.add_argument("--hachey-last-frame", type=int, default=None,
                help="last Hachey frame shown (default: the last; 267 = day 5, the analysed period)")
args = ap.parse_args()


# ------------------------------------------------------------------ our crop
def choose_window(well, size, rule):
    meta = json.loads((CROPS / well / "roi.json").read_text())
    ts = meta["timepoints"]
    roi = tifffile.imread(CROPS / well / "roi_mask.tif") > 0
    pillar = tifffile.imread(CROPS / well / "pillar_mask.tif") > 0
    ok = roi & ~ndi.binary_dilation(pillar, iterations=20)
    last = tifffile.imread(SEG / well / f"cells_T{ts[-1]:04d}.tif")
    first = tifffile.imread(SEG / well / f"cells_T{ts[0]:04d}.tif")
    ves = tifffile.imread(SEG / well / f"vessels_T{ts[-1]:04d}.tif") > 0

    def centroid_map(lab):
        ids = np.arange(1, lab.max() + 1)
        c = np.array(ndi.center_of_mass(lab > 0, lab, ids)).round().astype(int)
        m = np.zeros(lab.shape, np.int32)
        np.add.at(m, (c[:, 0], c[:, 1]), 1)
        return m

    S = lambda a: np.pad(a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)       # integral image
    box = lambda I, y, x: I[y + size, x + size] - I[y, x + size] - I[y + size, x] + I[y, x]
    I_ok, I_last, I_first, I_ves = S(ok.astype(np.int64)), S(centroid_map(last)), S(centroid_map(first)), S(ves.astype(np.int64))
    cands = []
    for y in range(0, roi.shape[0] - size, 32):
        for x in range(0, roi.shape[1] - size, 32):
            if box(I_ok, y, x) < size * size:
                continue
            vf = box(I_ves, y, x) / size ** 2
            if 0.30 <= vf <= 0.60:
                cands.append((y, x, box(I_last, y, x), box(I_first, y, x), vf))
    c = np.array(cands, float)
    target = np.percentile(c[:, 2], 75)
    if rule == "arrival":
        score = np.abs(c[:, 2] - target) - 1e-3 * (c[:, 2] - c[:, 3])   # ties -> largest increase
        y, x, n_last, n_first, vf = c[np.argmin(score)]
    else:
        near = np.abs(c[:, 2] - target) <= 0.2 * target
        pool = c[near] if near.any() else c
        y, x, n_last, n_first, vf = pool[np.argmax(pool[:, 3])]        # most cells already at the start
    return dict(well=well, crop_rule=rule, y0=int(y), x0=int(x), size_px=size, size_um=size * UM_PER_PX,
                n_cells_last=int(n_last), n_cells_first=int(n_first), vessel_frac_last=float(vf),
                n_candidates=len(c), timepoints=ts)


size = int(round(args.crop_um / UM_PER_PX))
win = choose_window(args.well, size, args.crop_rule)
ts = win["timepoints"]
pick = [ts[0], ts[len(ts) // 2], ts[-1]] if args.rows == 3 else [ts[0], ts[-1]]
sl = (slice(win["y0"], win["y0"] + size), slice(win["x0"], win["x0"] + size))
roi = tifffile.imread(CROPS / args.well / "roi_mask.tif") > 0
red_last = tifffile.imread(CROPS / args.well / f"T{ts[-1]:04d}_C1.tif").astype(np.float32)
grn_last = tifffile.imread(CROPS / args.well / f"T{ts[-1]:04d}_C2.tif").astype(np.float32)
r_lo, r_hi = np.percentile(red_last[roi], [50, 99.7])
g_lo, g_hi = np.percentile(grn_last[roi], [50, 99.8])
ours = []
for t in pick:
    r = tifffile.imread(CROPS / args.well / f"T{t:04d}_C1.tif").astype(np.float32)[sl]
    g = tifffile.imread(CROPS / args.well / f"T{t:04d}_C2.tif").astype(np.float32)[sl]
    r = np.clip((r - r_lo) / (r_hi - r_lo), 0, 1)
    g = np.clip((g - g_lo) / (g_hi - g_lo), 0, 1)
    ours.append((np.stack([r, g, np.zeros_like(r)], -1), f"{t - ts[0]} h"))

# ------------------------------------------------------------------ external movies
def frames_at(path, idx):
    out, want = {}, set(idx)
    for i, f in enumerate(iio.imiter(path)):
        if i in want:
            out[i] = f[..., :3].astype(np.float32) / 255
        if len(out) == len(want):
            break
    return [out[i] for i in idx]

n_song = sum(1 for _ in iio.imiter(SONG_FILE))
n_hachey = sum(1 for _ in iio.imiter(HACHEY_FILE))
s_idx = [0, n_song // 2, n_song - 1] if args.rows == 3 else [0, n_song - 1]
h_last = n_hachey - 1 if args.hachey_last_frame is None else args.hachey_last_frame
h_idx = [0, h_last // 2, h_last] if args.rows == 3 else [0, h_last]
hours = lambda i: f"{i * SONG_MIN_PER_FRAME / 60:.1f}".removesuffix(".0") + " h"
song = [(f[SONG_CROP], hours(i)) for f, i in zip(frames_at(SONG_FILE, s_idx), s_idx)]
h_lab = [f"day {10 * i / (n_hachey - 1):.0f}" for i in h_idx]      # 10-day movie, evenly spaced frames assumed
hachey = [(f, lab) for f, lab in zip(frames_at(HACHEY_FILE, h_idx), h_lab)]

# ------------------------------------------------------------------ figure
columns = [(args.our_title, ours, UM_PER_PX, 100),
           ("Song et al. 2021", song, SONG_UM_PER_PX, 200),
           ("Hachey et al. 2021", hachey, None, None)]
n_rows = args.rows
gap = 0.03
panel = (args.width_in - 2 * gap) / 3
head = 0.17
fig = plt.figure(figsize=(args.width_in, head + n_rows * panel + (n_rows - 1) * gap))
fig.patch.set_facecolor("white")
H = fig.get_figheight()
stroke = [pe.withStroke(linewidth=1.6, foreground="black")]
for c, (title, frames, um_px, bar_um) in enumerate(columns):
    for r, (img, tlabel) in enumerate(frames):
        left = c * (panel + gap) / args.width_in
        bottom = (H - head - (r + 1) * panel - r * gap) / H
        ax = fig.add_axes([left, bottom, panel / args.width_in, panel / H])
        ax.imshow(img, interpolation="lanczos")
        ax.set_axis_off()
        ax.text(0.04, 0.04, tlabel, transform=ax.transAxes, color="white", fontsize=6.5,
                ha="left", va="bottom", path_effects=stroke)
        if r == 0:
            ax.set_title(title, fontsize=7, pad=2.5, color="#1f1f1e")
        if r == n_rows - 1 and bar_um:
            h, w = img.shape[:2]
            L = bar_um / um_px
            x1, y1 = w * 0.95, h * 0.94
            ax.add_patch(plt.Rectangle((x1 - L, y1 - h * 0.018), L, h * 0.018, color="white", lw=0))
            ax.text(x1 - L / 2, y1 - h * 0.04, f"{bar_um:.0f} µm", color="white", fontsize=6,
                    ha="center", va="bottom", path_effects=stroke)
args.out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.out.with_suffix(".pdf"), dpi=600)
fig.savefig(args.out.with_suffix(".png"), dpi=600)
win.update(rows=n_rows, our_times=[f"T{t:04d}" for t in pick], song_frames=s_idx, hachey_frames=h_idx,
           song_crop_rows_cols=[[68, 468], [56, 456]], song_um_per_px=SONG_UM_PER_PX,
           display_red=[float(r_lo), float(r_hi)], display_green=[float(g_lo), float(g_hi)])
win.pop("timepoints")
args.out.with_suffix(".json").write_text(json.dumps(win, indent=2))
print(json.dumps(win))
