"""How long does each model's vessel network persist? One measure for every
model (ARTORG wells, Song videos, Hachey Video 1), from the per-frame vessel
masks only:

    IoU(lag)        mask IoU between frames t and t + lag, averaged over t
    agreement(lag)  centreline agreement within one median vessel radius
                    (symmetric mean of the two directions; vessel_scaffold's
                    centreline_agreement), so small displacements are not
                    counted as change

The whole-sequence scaffold (temporal consensus, union, presence profiles)
assumes a network that persists over the movie; these curves show for which
models that holds.

Outputs in <results>/vessel_persistence/: persistence.csv, persistence.png.

Usage (from the repo root):
    python paper/code/vessel_persistence.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

from common import CROPS, RESULTS, SEG, WELLS
from topology_tracking.scaffold import centreline_agreement
from vessel_scaffold import load_well

OUT = RESULTS / "vessel_persistence"
N_LAGS = 12


def sequences():
    for w in WELLS:
        _, _, V = load_well(CROPS, SEG, w)
        yield "ARTORG (ours)", w, V, 1.0
    for v in ("S1", "S3", "S5"):
        yield "Song 2021", f"song_{v}", tifffile.imread(RESULTS / "external" / "song2021" / v / "vessel_masks.tif") > 0, 5 / 60
    from prepare_hachey import FRAME_MIN
    yield "Hachey 2021", "hachey_V1", tifffile.imread(RESULTS / "external" / "hachey2021" / "V1" / "vessel_masks.tif") > 0, \
        FRAME_MIN / 60


def curves(V, frame_h, model, name, rng):
    T = len(V)
    sk = [skeletonize(v) for v in V]
    r = float(np.median(np.concatenate([ndi.distance_transform_edt(v)[s] for v, s in zip(V[::max(T // 10, 1)], sk[::max(T // 10, 1)])])))
    lags = np.unique(np.round(np.geomspace(1, T - 1, N_LAGS)).astype(int))
    rows = []
    for L in lags:
        ts = np.arange(T - L)
        ts = ts if len(ts) <= 40 else np.sort(rng.choice(ts, 40, replace=False))
        iou = [(V[t] & V[t + L]).sum() / max((V[t] | V[t + L]).sum(), 1) for t in ts]
        agr = [np.mean(centreline_agreement(sk[t], sk[t + L], tol=r)) for t in ts if sk[t].any() and sk[t + L].any()]
        rows.append(dict(model=model, sequence=name, lag_frames=int(L), lag_h=L * frame_h, iou=float(np.mean(iou)),
                         agreement=float(np.mean(agr)) if agr else np.nan, radius_px=r, n_frames=T,
                         duration_h=(T - 1) * frame_h))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    for model, name, V, fh in sequences():
        rows += curves(V, fh, model, name, rng)
        print(name, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "persistence.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"ARTORG (ours)": "#2a78d6", "Song 2021": "#eb6834", "Hachey 2021": "#1baf7a"}
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), sharex=True)
    for name, g in df.groupby("sequence"):
        for a, k in zip(ax, ("iou", "agreement")):
            a.plot(g.lag_h, g[k], "-o", ms=3, color=col[g.model.iloc[0]], label=g.model.iloc[0])
    for a, k in zip(ax, ("mask IoU", "centreline agreement (tol = 1 radius)")):
        a.set_xscale("log"); a.set_xlabel("lag (h)"); a.set_ylabel(k); a.set_ylim(0, 1); a.grid(alpha=.3)
    h, l = ax[0].get_legend_handles_labels()
    u = dict(zip(l, h)); ax[0].legend(u.values(), u.keys(), frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "persistence.png", dpi=150)
    print(df.groupby(["model", "sequence"]).apply(lambda g: g.iloc[[0, len(g) // 2, -1]][["lag_h", "iou", "agreement"]].round(2)
                                                 .to_dict("records")).to_string())


if __name__ == "__main__":
    main()
