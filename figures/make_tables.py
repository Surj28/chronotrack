"""LaTeX rows for the paper's tables (printed):
  ablation  synthetic twin: link F1 and event-table error, ours vs its variants
            (median over the 12 sequences of a condition; stars: paired
            Wilcoxon vs ours, Holm over the 16 comparisons of a metric)
  features  our method's chip event table on the three MVoC models: ARTORG
            mean +- SD of the 4 replicate wells, the published videos each on
            their own; stars from replicate_stats.contrasts

Usage (from the repo root):
    python paper/figures/make_tables.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

PAPER = Path(__file__).resolve().parent.parent
RES = PAPER / "results"
CONDS = [(1.0, "1", "Real, gentle"), (1.0, "cal", "Real, ambiguous"), (2.0, "1", "2$\\times$, gentle"),
         (2.0, "cal", "2$\\times$, ambiguous")]
VARIANTS = ["sym", "sym_vessel", "first_order", "w0", "nn"]


def stars(p):
    return "$^{***}$" if p < 0.001 else "$^{**}$" if p < 0.01 else "$^{*}$" if p < 0.05 else ""


def ablation():
    B = pd.read_csv(RES / "synthetic" / "benchmark.csv")
    B["lam"] = B.lam.astype(str)
    for metric, scale, fmt in (("f1", 100, "{:.1f}"), ("feature_error", 100, "{:.1f}")):
        tests, vals = [], {}
        for rho, lam, _ in CONDS:
            p = B[(B.rho == rho) & (B.lam == lam)].pivot_table(index=["well", "seed"], columns="tracker", values=metric)
            for tr in VARIANTS:
                vals[(rho, lam, tr)] = scale * p[tr].median()
            for tr in VARIANTS[1:]:
                tests.append(((rho, lam, tr), stats.wilcoxon(p["sym"], p[tr]).pvalue))
        adj = dict(zip([t[0] for t in tests], multipletests([t[1] for t in tests], method="holm")[1]))
        print(f"% {metric}")
        for rho, lam, name in CONDS:
            cells = [fmt.format(vals[(rho, lam, tr)]) + (stars(adj[(rho, lam, tr)]) if tr != "sym" else "") for tr in VARIANTS]
            print(f"{name} & " + " & ".join(cells) + " \\\\")


def features():
    """ARTORG: mean +- SD of the 4 replicate wells; published videos: each on its own
    (not known to be replicates). Rows with two values a / b: observed / chance, or on /
    off the vessels; a star says that a and b differ (replicate_stats.contrasts, Holm p < 0.05)."""
    import sys
    sys.path.insert(0, str(PAPER / "code"))
    from common import FRAME_H
    from prepare_hachey import FRAME_MIN as HACHEY_MIN
    from prepare_song import FRAME_MIN as SONG_MIN
    from replicate_stats import _load, contrasts
    M = {"ARTORG": "event_tests", "Song": "event_tests_external", "Hachey": "event_tests_hachey"}
    F = pd.concat([pd.read_csv(RES / d / "features.csv").query("tracker == 'sym' and condition == 'forward'").assign(model=m)
                   for m, d in M.items()])
    F = F.merge(_load("point")[["well", "arrival_edge_frac", "edge_chance"]], on="well")
    # Per frame, not per hour: the frame interval of the Hachey video is not reported (only estimated),
    # and per-frame values need no interval. The rates per 100 cell-hours become rates per 100
    # detections (one detection = one cell in one frame).
    frame_h = F.model.map({"ARTORG": FRAME_H, "Song": SONG_MIN / 60, "Hachey": HACHEY_MIN / 60})
    for k in ("speed_on_vessel", "speed_off_vessel"):
        F[f"{k}_um_frame"] = F[f"{k}_um_h"] * frame_h
    for k in ("arrivals", "exits"):
        F[f"{k}_per_100_detections"] = F[f"{k}_per_100_cell_h"] * frame_h
    F["track_length_frames"] = F.mean_track_duration_h / frame_h   # frames from first to last detection
    interval = {"ARTORG": "1 h", "Song": f"{SONG_MIN:g} min", "Hachey": "unknown"}
    C = contrasts()
    sig = {(r.unit if r.model != "ARTORG" else "ARTORG", r.contrast): r.p_holm < 0.05 for r in C.itertuples()}
    cols = [("ARTORG", None)] + [("Song", w) for w in F[F.model == "Song"].well] + [("Hachey", w) for w in F[F.model == "Hachey"].well]

    def fmt_of(x):
        """Two significant digits: one decimal below 10, integers above; exact zeros as 0."""
        return "{:.0f}" if x == 0 or abs(x) >= 10 else "{:.1f}"

    def value(keys, pct, star, model, w):
        """One cell: a value, or a pair a / b (observed / chance, on / off); ARTORG with the SD over its wells."""
        sub = F[F.model == model] if w is None else F[F.well == w]
        parts = []
        for k in keys:
            x = (100 if pct else 1) * sub[k]
            f = fmt_of(x.mean())
            sd = f.format(x.std()) if w is None else ""
            if sd and sd.strip("0.") == "":             # an SD that rounds to zero: one more decimal
                f = "{:.1f}" if f == "{:.0f}" else "{:.2f}"
                sd = f.format(x.std())
            parts.append(f.format(x.mean()) + (f"$\\pm${sd}" if sd else ""))
        v = " / ".join(parts)
        return v + ("$^*$" if star and sig.get((w or "ARTORG", star)) else "")

    # (feature, what it measures, keys: one value or a pair, percent, contrast behind the star)
    spec = [("Cells on vessels (\\%)", "detections overlapping the vessel mask", ["frac_overlapping"], True, None),
            ("Track length (frames)", "frames from first to last detection", ["track_length_frames"], False, None),
            ("Speed on / off vessels (\\textmu m/frame)", "median distance moved per frame",
             ["speed_on_vessel_um_frame", "speed_off_vessel_um_frame"], False, "speed"),
            ("Arrivals (per 100 detections)", "cells appearing after the first frame",
             ["arrivals_per_100_detections"], False, None),
            ("Arrivals on vessels (\\%)", "arrivals / all detections (chance) on a vessel",
             ["arrival_on_vessel_frac", "frac_overlapping"], True, "arrival_enrichment"),
            ("Arrivals at the edge (\\%)", "arrivals / all detections (chance) at the edge",
             ["arrival_edge_frac", "edge_chance"], True, "arrival_edge"),
            ("Extravasation (per 100 detections)", "a cell leaves a vessel for $\\geq$\\,2 frames",
             ["exits_per_100_detections"], False, None),
            ("Regressing vessels (\\%)", "share of vessel length that regresses", ["vessel_regression_frac"], True, None)]
    print("% features: columns", [c[1] or c[0] for c in cols])
    print("Frame interval & & " + " & ".join(interval[m] for m, _ in cols) + " \\\\")
    for name, what, keys, pct, star in spec:
        print(f"{name} & {what} & " + " & ".join(value(keys, pct, star, m, w) for m, w in cols) + " \\\\")
    # the table's numbers, and its tests, as CSV
    DATA = PAPER / "figure_data"
    DATA.mkdir(exist_ok=True)
    num = []
    for name, what, keys, pct, star in spec:
        for model, w in cols:
            sub = F[F.model == model] if w is None else F[F.well == w]
            for k in keys:
                scale = 100 if pct else 1
                num.append(dict(row=name, feature=k, model=model, column=w or "mean of 4 wells", n_sequences=len(sub),
                                mean=scale * sub[k].mean(), sd=scale * sub[k].std() if w is None else np.nan,
                                significant=bool(sig.get((w or "ARTORG", star))) if star else np.nan))
    pd.DataFrame(num).to_csv(DATA / "table_features.csv", index=False)
    C.to_csv(DATA / "table_features_tests.csv", index=False)
    print(C.to_string())


if __name__ == "__main__":
    ablation()
    features()
