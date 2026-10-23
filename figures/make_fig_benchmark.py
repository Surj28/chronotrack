"""Figure and table: synthetic benchmark on the ARTORG twins (paper/code/synth_benchmark.py).

Figure: our tracker against the published trackers (btrack, LapTrack,
Trackastra), one panel per metric (link F1, complete tracks, feature
error), labelled horizontal bars grouped by condition (density: real or 2x;
motion: gentle = consensus steps, or steps scaled to the real well's linking
ambiguity). Bars: median over the 12 sequences of a condition (4 wells x 3
seeds). A bracket joins our bar to each published tracker's bar and carries
the significance of their difference (paired Wilcoxon signed-rank test over
those sequences, Holm-corrected over the 12 comparisons of a panel): ns, *,
**, ***. Bars over fewer sequences than ours (runs still missing) show their n
and no bracket.

Printed: every value and p-value, and the ablation (ours, +vessel,
first-order, Kalman, nearest neighbor; Holm over its own 16 comparisons of a
metric).

Usage (from the repo root):
    python paper/figures/make_fig_benchmark.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "STIXGeneral"], "mathtext.fontset": "stix"})   # match the paper (Times)
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
SYN = PAPER / "results" / "synthetic"
OUT = PAPER / "images" / "benchmark"
DATA = PAPER / "figure_data"                      # the plotted numbers, as CSV
SHOWN = ["sym", "btrack", "laptrack", "trackastra"]
ABLATION = ["sym", "sym_vessel", "first_order", "w0", "nn"]
NAMES = {"sym": "ChronoTrack", "btrack": "btrack", "laptrack": "LapTrack", "trackastra": "Trackastra",
         "sym_vessel": "+vessel", "first_order": "Frame-by-frame", "w0": "Kalman", "nn": "Nearest neighbor"}
COLOURS = {"sym": "#0072B2", "btrack": "#E69F00", "laptrack": "#009E73", "trackastra": "#CC79A7"}
CONDS = [(1.0, "1"), (1.0, "cal"), (2.0, "1"), (2.0, "cal")]
COND_NAME = {(1.0, "1"): "Real density,\ngentle motion", (1.0, "cal"): "Real density,\nambiguous motion",
             (2.0, "1"): "2× density,\ngentle motion", (2.0, "cal"): "2× density,\nambiguous motion"}
METRICS = [("f1", "(a) Link accuracy", "Link F1 (%)", 100, True),
           ("complete", "(b) Complete tracks", "Cells tracked in one piece (%)", 100, True),
           ("feature_error", "(c) Feature error", "Error of the MVoC features (%)", 100, False)]
ALPHA = 0.05
GAP, STEP, TICK = 3.0, 9.0, 1.5      # brackets, in points: gap after the value labels, spacing, tick length


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def compare(B):
    """Medians, n, and per family (published / ablation) Holm-corrected paired Wilcoxon tests.
    tests[(metric, cond, tracker)] = (p_holm, n, ours_better)."""
    key = ["well", "rho", "lam", "seed"]
    val, n_of, tests = {}, {}, {}
    for metric, _, _, _, higher in METRICS:
        fam = {"published": [], "ablation": []}
        for cond in CONDS:
            b = B[(B.rho == cond[0]) & (B.lam == cond[1])].pivot_table(index=key, columns="tracker", values=metric)
            for tr in b.columns:
                val[(metric, cond, tr)] = float(b[tr].median())
                n_of[(metric, cond, tr)] = int(b[tr].notna().sum())
            for tr in [t for t in b.columns if t != "sym"]:
                x = b[["sym", tr]].dropna()
                if len(x) < 5:
                    continue
                d = x["sym"] - x[tr]
                ours_better = bool((np.median(d) if np.median(d) != 0 else d.mean()) > 0) == higher
                fam["published" if tr in SHOWN else "ablation"].append(
                    ((metric, cond, tr), stats.wilcoxon(x["sym"], x[tr]).pvalue, len(x), ours_better))
        for rows in fam.values():
            if rows:
                adj = multipletests([r[1] for r in rows], method="holm")[1]
                for (k, _, n, better), p in zip(rows, adj):
                    tests[k] = (float(p), n, better)
    return val, n_of, tests


def label(ax, x, y, parts, fontsize=5.3):
    """Text pieces [(text, colour)] placed one after the other, starting just right of x; returns the last."""
    r = ax.figure.canvas.get_renderer()
    off = 1.2
    for s, c in parts:
        t = ax.annotate(s, (x, y), xytext=(off, 0), textcoords="offset points", va="center", ha="left",
                        fontsize=fontsize, color=c, annotation_clip=False)
        off += t.get_window_extent(r).width * 72 / ax.figure.dpi
    return t


def bracket(ax, x, y0, y1, text, fontsize=5.3):
    """Bracket at data x joining the bars at heights y0 and y1 (ticks pointing left), text right of its middle."""
    x0 = ax.transData.inverted().transform(ax.transData.transform((x, 0)) - (TICK * ax.figure.dpi / 72, 0))[0]
    ax.plot([x0, x, x, x0], [y0, y0, y1, y1], color="k", lw=0.5, clip_on=False, zorder=3)
    ax.annotate(text, (x, (y0 + y1) / 2), xytext=(1.0, 0), textcoords="offset points", va="center", ha="left",
                fontsize=fontsize, annotation_clip=False)


def main():
    B = pd.concat([pd.read_csv(f) for f in sorted(SYN.rglob("benchmark*.csv"))], ignore_index=True)
    B = B.drop_duplicates(subset=["well", "rho", "lam", "seed", "tracker"], keep="last")
    B["lam"] = B.lam.astype(str)
    val, n_of, tests = compare(B)
    pd.set_option("display.width", 200)
    for metric, *_ in METRICS:
        print(f"\n== {metric} (median over sequences; p vs ours, Holm within published / ablation)")
        for cond in CONDS:
            line = []
            for tr in ABLATION + [t for t in SHOWN if t != "sym"]:
                k = (metric, cond, tr)
                if k in val:
                    t = tests.get(k)
                    line.append(f"{NAMES[tr]} {100 * val[k]:.1f}" +
                                (f" (p={t[0]:.2g}{stars(t[0])}, n={t[1]}, {'ours better' if t[2] else 'ours worse'})"
                                 if t else f" (n={n_of[k]}, no test)"))
            print(f"  {COND_NAME[cond].replace(chr(10), ' '):38s} " + " | ".join(line))

    shown = [t for t in SHOWN if any((m[0], c, t) in val for m in METRICS for c in CONDS)]
    rivals = [t for t in shown if t != "sym"]
    plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6})
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.5), sharey=True)
    h = 0.19
    ys = {(c, tr): i * 1.0 - k * h for i, c in enumerate(reversed(CONDS)) for k, tr in enumerate(shown)}
    axes[0].set_ylim(min(ys.values()) - 0.55 * h - 0.1, max(ys.values()) + 0.55 * h + 0.1)   # shared; before labels
    ends = {}                                        # (panel, condition) -> value labels; the brackets go right of them
    axis_ticks = {}                                  # panel -> (x ticks, end of the drawn axis)
    for a, (ax, (metric, title, xlabel, scale, _)) in enumerate(zip(axes, METRICS)):
        vals = [scale * val.get((metric, c, t), np.nan) for c in CONDS for t in shown]
        lo = 0 if metric != "f1" else max(0, np.floor(np.nanmin(vals) / 5) * 5 - 5)
        hi = np.nanmax(vals)
        ax.set_xlim(lo, lo + (hi - lo) * 1.32)
        ticks = ax.get_xticks()                      # axis drawn up to the first tick past the data (F1: 100 at most)
        top = min(t for t in ticks if t >= hi)
        top = min(top, 100) if metric == "f1" else top
        axis_ticks[a] = ([t for t in ticks if lo <= t <= top], top)
        for tr in shown:
            for c in CONDS:
                v = scale * val.get((metric, c, tr), np.nan)
                if not np.isfinite(v):
                    continue
                y = ys[(c, tr)]
                ax.barh(y, v - lo, h * 0.88, left=lo, color=COLOURS[tr], label=NAMES[tr] if c == CONDS[0] else None, zorder=2)
                parts = [(f"{v:.1f}" if (metric == "f1" or v < 10) else f"{v:.0f}", "k")]
                if tr != "sym" and (metric, c, tr) not in tests:     # runs still missing: no test yet
                    parts.append((f" (n={n_of[(metric, c, tr)]})", "0.5"))
                ends.setdefault((a, c), []).append(label(ax, v, y, parts))
        ax.set_yticks([i * 1.0 - 1.5 * h for i in range(len(CONDS))])
        ax.set_yticklabels([COND_NAME[c] for c in reversed(CONDS)])
        ax.set_title(title, fontsize=7.5, pad=3, loc="left")
        ax.set_xlabel(xlabel, labelpad=2)
        ax.tick_params(width=0.6, length=2, pad=1)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    hd, lb = axes[0].get_legend_handles_labels()
    fig.legend(hd, lb, loc="upper center", ncol=len(shown), frameon=False, fontsize=7, handlelength=1.2,
               columnspacing=1.5, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93), pad=0.3, w_pad=1.0)
    # brackets (ChronoTrack vs each published tracker) right of each group's value labels; widen x until they fit
    px = fig.dpi / 72                                # pixels per point
    need = (GAP + (len(rivals) - 1) * STEP + 1.0 + 7.0) * px
    for _ in range(8):
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        fitted = True
        for a, ax in enumerate(axes):
            right = max(t.get_window_extent(r).x1 for c in CONDS for t in ends.get((a, c), []))
            excess = right + need - ax.get_window_extent(r).x1
            if excess > 0:
                lo, hi = ax.get_xlim()
                per_unit = ax.transData.transform((1, 0))[0] - ax.transData.transform((0, 0))[0]
                ax.set_xlim(lo, hi + 1.2 * excess / per_unit)
                fitted = False
        if fitted:
            break
    for a, ax in enumerate(axes):                    # the x range widened for the brackets: keep the axis to the data
        ticks, top = axis_ticks[a]
        ax.set_xticks(ticks)
        ax.spines["bottom"].set_bounds(ax.get_xlim()[0], top)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for a, (ax, (metric, *_)) in enumerate(zip(axes, METRICS)):
        inv = ax.transData.inverted()
        for c in CONDS:
            right = max(t.get_window_extent(r).x1 for t in ends.get((a, c), []))
            for j, tr in enumerate(rivals):
                t = tests.get((metric, c, tr))
                if t:
                    x = inv.transform((right + (GAP + j * STEP) * px, 0))[0]
                    bracket(ax, x, ys[(c, "sym")], ys[(c, tr)], stars(t[0]) or "ns")
    for ext in ("pdf", "png"):
        fig.savefig(OUT.with_suffix("." + ext), dpi=300, bbox_inches="tight", pad_inches=0.02)
    # the plotted numbers: bars (median, n, test vs ours) and the per-sequence values behind them
    DATA.mkdir(exist_ok=True)
    rows = []
    for metric, *_ in METRICS:
        for c in CONDS:
            for tr in ABLATION + [t for t in SHOWN if t != "sym"]:
                if (metric, c, tr) in val:
                    t = tests.get((metric, c, tr))
                    rows.append(dict(metric=metric, density=f"{c[0]:g}x", motion="gentle" if c[1] == "1" else "ambiguous",
                                     tracker=NAMES[tr], in_figure=tr in shown, median=val[(metric, c, tr)],
                                     n_sequences=n_of[(metric, c, tr)], p_holm_vs_chronotrack=t[0] if t else np.nan,
                                     holm_family=("published" if tr in SHOWN else "ablation") if t else "",
                                     chronotrack_better=t[2] if t else np.nan))
    pd.DataFrame(rows).to_csv(DATA / "benchmark_bars.csv", index=False)
    seq = B[B.tracker.isin(ABLATION + SHOWN)][["well", "rho", "lam", "seed", "tracker"] + [m[0] for m in METRICS]]
    seq.assign(tracker=seq.tracker.map(NAMES)).sort_values(["rho", "lam", "well", "seed", "tracker"]).to_csv(
        DATA / "benchmark_sequences.csv", index=False)
    print("\n" + str(OUT.with_suffix(".pdf")), "|", DATA / "benchmark_bars.csv", "|", DATA / "benchmark_sequences.csv")


if __name__ == "__main__":
    main()
