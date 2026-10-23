"""Step 3 — cell-size prior from T0001.

Segments the first frame of each well (Cellpose-SAM, green only, whole ROI,
Cellpose's own normalisation), then fits a Gaussian mixture to log10(cell
area). The number of components is chosen by BIC; the dominant component is
taken as the real-cell population, and the accepted size range is its
mean +- 3 SD in log area (55-396 um^2 pooled on B34/ARTORG). Objects cut by the ROI edge are left out of
the fit (truncated, so artificially small) but still reported.

Fits are made per well and pooled over wells (same cell line, so the
distributions should agree).

Usage (from the repo root):
    uv run python paper/code/cell_size_prior.py --crops paper/results/roi_crops \
        --out paper/results/cell_size_prior
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy import ndimage as ndi
from sklearn.mixture import GaussianMixture

UM2_PER_PX = 0.65 ** 2
# reference palette, categorical slots 1-3, plus line style as secondary encoding
STYLE = {"cells": ("#2a78d6", "-"), "small objects": ("#eb6834", "--"), "large objects": ("#1baf7a", ":")}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def segment_t1(model, well: Path) -> np.ndarray:
    green = tifffile.imread(well / "T0001_C2.tif").astype(np.float32)
    m, _, _ = model.eval(green, normalize=True)
    return m.astype(np.int32)


def object_areas(masks, roi):
    ids = np.unique(masks); ids = ids[ids > 0]
    area = ndi.sum(np.ones_like(masks), masks, ids)
    inside = ndi.sum(roi.astype(np.int32), masks, ids)
    return ids, area, inside / area


def fit_mixture(log_area, max_k=4, seed=0):
    x = log_area.reshape(-1, 1)
    fits = [GaussianMixture(k, random_state=seed, n_init=5).fit(x) for k in range(1, max_k + 1)]
    best = min(fits, key=lambda g: g.bic(x))
    main = int(np.argmax(best.weights_))
    grid = np.linspace(log_area.min() - 0.3, log_area.max() + 0.3, 4000).reshape(-1, 1)
    post = best.predict_proba(grid)[:, main]
    ok = grid[post >= 0.5, 0]
    mu = float(best.means_[main, 0]); sd = float(np.sqrt(best.covariances_[main].ravel()[0]))
    # contiguous accepted interval around the main component's mean
    centre = np.argmin(np.abs(grid[:, 0] - mu))
    lo_i = centre
    while lo_i > 0 and post[lo_i - 1] >= 0.5: lo_i -= 1
    hi_i = centre
    while hi_i < len(post) - 1 and post[hi_i + 1] >= 0.5: hi_i += 1
    # lower bound: where the cell component takes over from smaller ones (never
    # below mean - 3 SD); upper bound: mean + 3 SD of the cell component, since
    # a broad small-object component can win again far out on the right
    # Accepted range: mean +- 3 SD of the cell component (log area). The
    # posterior crossing with the small-object component (lo_cross) sits higher
    # and cut real small cells on visual review, so it is reported, not used.
    lo_cross = float(grid[lo_i, 0])
    lo, hi = mu - 3 * sd, mu + 3 * sd
    return dict(model=best, k=best.n_components, main=main, mu=mu, sd=sd, lo=lo, hi=hi, lo_cross=lo_cross,
                bic={k + 1: float(f.bic(x)) for k, f in enumerate(fits)})


def component_roles(fit):
    """Name each component relative to the main (cell) one."""
    roles = {}
    for j in range(fit["k"]):
        m = fit["model"].means_[j, 0]
        roles[j] = "cells" if j == fit["main"] else ("small objects" if m < fit["mu"] else "large objects")
    return roles


def plot_panel(ax, log_area, fit, title, n_edge):
    bins = np.linspace(1.0, 3.6, 80)
    ax.hist(log_area, bins=bins, density=True, color="#c9c8c1", edgecolor="white", linewidth=0.6)
    xs = np.linspace(bins[0], bins[-1], 600)
    g = fit["model"]
    for j, role in component_roles(fit).items():
        w = g.weights_[j]; m = g.means_[j, 0]; s = np.sqrt(g.covariances_[j].ravel()[0])
        y = w * np.exp(-0.5 * ((xs - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
        c, ls = STYLE[role]
        ax.plot(xs, y, color=c, ls=ls, lw=2, label=role)
    ax.axvspan(fit["lo"], fit["hi"], color=STYLE["cells"][0], alpha=0.08, lw=0)
    keep = (log_area >= fit["lo"]) & (log_area <= fit["hi"])
    ax.set_title(title, loc="left", fontsize=10, color=INK, pad=4)
    lo_um, hi_um = 10 ** fit["lo"] * UM2_PER_PX, 10 ** fit["hi"] * UM2_PER_PX
    ax.text(0.02, 0.97, f"n = {log_area.size}\nkept {keep.mean():.0%}\n{lo_um:.0f}–{hi_um:.0f} µm²",
            transform=ax.transAxes, ha="left", va="top", fontsize=8, color=MUTED)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ticks = [30, 100, 300, 1000]
    ax.set_xticks([np.log10(t / UM2_PER_PX) for t in ticks]); ax.set_xticklabels([str(t) for t in ticks])
    ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    from cellpose import models

    args.out.mkdir(parents=True, exist_ok=True)
    wells = sorted(args.crops.glob("ARTORG_w*"), key=lambda p: int(p.name.split("_")[-1]))
    model = models.CellposeModel(gpu=True)
    per_well, pooled, report = {}, [], {}
    for w in wells:
        roi = (tifffile.imread(w / "roi_mask.tif") > 0) & ~(tifffile.imread(w / "pillar_mask.tif") > 0)
        masks = segment_t1(model, w)
        (args.out / w.name).mkdir(exist_ok=True)
        tifffile.imwrite(args.out / w.name / "cells_T0001.tif", masks, compression="zlib")
        ids, area, frac_in = object_areas(masks, roi)
        full = frac_in >= 0.999
        la = np.log10(area[full])
        per_well[w.name] = (la, int((~full & (frac_in > 0)).sum()), ids, area, full)
        pooled.append(la)

    fits = {name: fit_mixture(v[0]) for name, v in per_well.items()}
    pooled = np.concatenate(pooled)
    fit_all = fit_mixture(pooled)

    fig, axes = plt.subplots(1, len(wells) + 1, figsize=(3.3 * (len(wells) + 1), 3.0), sharex=True)
    for ax, (name, (la, n_edge, *_)) in zip(axes, per_well.items()):
        plot_panel(ax, la, fits[name], f"Well {name.split('_')[-1]}", n_edge)
    plot_panel(axes[-1], pooled, fit_all, "Pooled, 4 wells", sum(v[1] for v in per_well.values()))
    axes[-1].title.set_fontweight("bold")
    for ax in axes: ax.set_xlabel("cell area (µm², log scale)", fontsize=8, color=MUTED)
    used = {r for f in list(fits.values()) + [fit_all] for r in component_roles(f).values()}
    handles = [plt.Line2D([], [], color=c, ls=ls, lw=2, label=r) for r, (c, ls) in STYLE.items() if r in used]
    handles.append(plt.Rectangle((0, 0), 1, 1, color=STYLE["cells"][0], alpha=0.15, label="accepted range"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("B34/ARTORG, T0001: cell size distribution with Gaussian-mixture fit (log area)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(args.out / "fig_cell_size_T0001.png", dpi=200)
    fig.savefig(args.out / "fig_cell_size_T0001.pdf")

    for name, f in list(fits.items()) + [("pooled", fit_all)]:
        report[name] = dict(n_components=f["k"], bic=f["bic"],
                            cell_mean_um2=10 ** f["mu"] * UM2_PER_PX,
                            cell_log10_sd=f["sd"],
                            accept_px=[10 ** f["lo"], 10 ** f["hi"]],
                            accept_um2=[10 ** f["lo"] * UM2_PER_PX, 10 ** f["hi"] * UM2_PER_PX],
                            posterior_crossing_um2=10 ** f["lo_cross"] * UM2_PER_PX,
                            weights=[float(x) for x in f["model"].weights_],
                            means_um2=[float(10 ** m * UM2_PER_PX) for m in f["model"].means_[:, 0]])
        if name in per_well:
            la = per_well[name][0]
            keep = (la >= f["lo"]) & (la <= f["hi"])
            report[name].update(n_fitted=int(la.size), n_kept=int(keep.sum()),
                                n_small_removed=int((la < f["lo"]).sum()), n_large_removed=int((la > f["hi"]).sum()),
                                n_edge_cut=per_well[name][1])
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: {kk: v[kk] for kk in ("n_components", "cell_mean_um2", "accept_um2") + (("n_fitted", "n_kept", "n_small_removed", "n_large_removed") if k != "pooled" else ())} for k, v in report.items()}, indent=1))


if __name__ == "__main__":
    main()
