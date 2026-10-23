"""LaTeX rows for the ablation table (Supplementary Table S1, printed): link
F1, complete tracks and feature error on the synthetic sequences, ours vs its
variants (median over the 12 sequences of a condition; stars: paired Wilcoxon
vs ours, Holm over the 16 comparisons of a metric; ns: not significant).

Inputs: <results>/benchmark*.csv (code/synth_benchmark.py; --results, default
results/). The ablation's link F1 is also written as CSV by
figures/make_fig_benchmark.py (figure_data/table_ablation.csv).

Usage (from the repo root):
    python figures/make_tables.py [--results results]
"""
import argparse
import sys
from pathlib import Path

from scipy import stats
from statsmodels.stats.multitest import multipletests

PAPER = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PAPER / "figures"))
CONDS = [(1, "gentle", "Real, gentle"), (1, "ambiguous", "Real, ambiguous"), (2, "gentle", "2$\\times$, gentle"),
         (2, "ambiguous", "2$\\times$, ambiguous")]
VARIANTS = ["sym", "first_order", "sym_vessel", "w0", "nn"]   # the table's column order


def stars(p):
    return "$^{**}$" if p < 0.01 else "$^{*}$" if p < 0.05 else "~ns"


def ablation(results):
    from make_fig_benchmark import load_benchmark
    B = load_benchmark(results)
    for metric, scale, fmt in (("f1", 100, "{:.1f}"), ("complete", 100, "{:.1f}"), ("feature_error", 100, "{:.1f}")):
        tests, vals = [], {}
        for dens, motion, _ in CONDS:
            p = B[(B.density == dens) & (B.motion == motion)].pivot_table(index=["seq", "seed"], columns="tracker", values=metric)
            for tr in VARIANTS:
                vals[(dens, motion, tr)] = scale * p[tr].median()
            for tr in VARIANTS[1:]:
                tests.append(((dens, motion, tr), stats.wilcoxon(p["sym"], p[tr]).pvalue))
        adj = dict(zip([t[0] for t in tests], multipletests([t[1] for t in tests], method="holm")[1]))
        print(f"% {metric}")
        for dens, motion, name in CONDS:
            cells = [fmt.format(vals[(dens, motion, tr)]) + (stars(adj[(dens, motion, tr)]) if tr != "sym" else "") for tr in VARIANTS]
            print(f"{name} & " + " & ".join(cells) + " \\\\")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", type=Path, default=PAPER / "results", help="folder with the benchmark CSVs (default: results/)")
    ablation(ap.parse_args().results)
