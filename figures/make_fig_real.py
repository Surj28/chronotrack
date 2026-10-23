"""Figure: our tracker on the real MVoC data (robustness and reproducibility).

Sequences: ARTORG, 4 replicate wells; Song, 3 published videos; Hachey, 1
published video. Each sequence is summarized by the median over the
linking-dependent features of their relative change. ARTORG: mean +- SD over
the replicate wells. The published videos are shown one line each: whether
they are replicates, the same cells or different conditions is not known, so
they are not averaged.

  (a) lost detections   change against the fraction of detections removed at
                        random per frame (per feature: median over three
                        repeats); dotted: change = loss; triangle: the model's
                        measured miss rate (detection_model p_miss)
  (b) frame interval    change when only every k-th frame is used, against the
                        resulting frame interval; each curve starts at the
                        model's native interval (open circle)
  (c) reproducibility   of each feature across the four ARTORG replicate wells:
                        100% - coefficient of variation (100%: the same value in
                        every well; 0%: a spread as large as the mean). The
                        heterogeneity test (replicate_stats.heterogeneity) is
                        saved in the CSV, not drawn

Usage (from the repo root):
    python paper/figures/make_fig_real.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "STIXGeneral"], "mathtext.fontset": "stix"})   # match the paper (Times)
import pandas as pd

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
sys.path.insert(0, str(PAPER / "code"))
from summarize_event_tests import TRACK_FEATS  # noqa: E402
from common import FRAME_H  # noqa: E402
from prepare_hachey import FRAME_MIN as HACHEY_MIN  # noqa: E402
from prepare_song import FRAME_MIN as SONG_MIN  # noqa: E402

RES = PAPER / "results"
OUT = PAPER / "images" / "real_data"
DATA = PAPER / "figure_data"                      # the plotted numbers, as CSV
MODELS = {"ARTORG": "event_tests", "Song": "event_tests_external", "Hachey": "event_tests_hachey"}
COLOURS = {"ARTORG": "#2a78d6", "Song": "#eb6834", "Hachey": "#1baf7a"}
FRAME_MIN = {"ARTORG": 60 * FRAME_H, "Song": SONG_MIN, "Hachey": HACHEY_MIN}   # native frame interval
LEGEND = {"ARTORG": "ARTORG (4 wells)", "Song": "Song et al. (3 videos)", "Hachey": "Hachey et al. (1 video)"}
LABEL = {"arrivals_per_100_cell_h": "arrival rate", "departures_per_100_cell_h": "departure rate",
         "arrival_on_vessel_frac": "arrivals on vessels", "departure_on_vessel_frac": "departures from vessels",
         "arrival_vessel_enrichment": "arrivals on vessels vs. chance",
         "departure_vessel_enrichment": "departures from vessels vs. chance",
         "exits_per_100_cell_h": "extravasation rate", "entries_per_100_cell_h": "intravasation rate",
         "speed_on_vessel_um_h": "speed on vessels", "speed_off_vessel_um_h": "speed off vessels",
         "mean_track_duration_h": "track length",      # ARTORG: 1-h frames, so hours = frames
         "exit_regression_colocation": "extravasation at regressing vessels",
         "entry_emergence_colocation": "intravasation at emerging vessels"}
ERR = dict(capsize=1.6, elinewidth=0.7, capthick=0.7)
NOT_IN_C = {"entry_emergence_colocation"}             # not shown in the reproducibility panel


def rel(a, b):
    return np.abs(b - a) / np.abs(a).where(np.abs(a) > 1e-12)


def mean_sd(per_seq):
    """per_seq: (sequences x levels) -> mean, SD over sequences (NaN for one sequence)."""
    a = np.asarray(per_seq, float)
    return a.mean(0), (a.std(0, ddof=1) if len(a) > 1 else np.full(a.shape[1], np.nan))


def curves(ax, m, x, per, miss=None):
    """ARTORG: mean +- SD over the replicate wells; published videos: one line each."""
    per = np.asarray(per, float)
    if m == "ARTORG":
        mu, sd = mean_sd(per)
        ax.errorbar(x, mu, yerr=sd, fmt="-o", ms=2.5, lw=1.1, color=COLOURS[m], label=LEGEND[m], **ERR)
        lines, misses = [mu], [np.mean(miss)] if miss is not None else []
    else:
        for i, y in enumerate(per):
            ax.plot(x, y, "-o", ms=2.0, lw=0.8, color=COLOURS[m], label=LEGEND[m] if i == 0 else None)
        lines, misses = list(per), list(miss) if miss is not None else []
    for y, q in zip(lines, misses):                       # measured miss rate on each plotted line
        ax.plot([q], [float(np.interp(q, x, y))], marker="v", ms=4.5, color=COLOURS[m], mec="k", mew=0.4,
                zorder=4, ls="none")


def main():
    plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6})
    # constrained layout: each gap is sized to its own labels (one shared spacing left a wide gap
    # between (a) and (b) to make room for the feature names of (c))
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 1.85), gridspec_kw=dict(width_ratios=[1, 1, 1.35]),
                             layout="constrained")
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.02, wspace=0.07)
    fw_be, feats_be = None, None
    rows_a, rows_m, rows_b = [], [], []
    for m, d in MODELS.items():
        p = RES / d
        F = pd.concat([pd.read_csv(f) for f in (p / "features.csv", p / "features_sampling_extra.csv") if f.exists()])
        F = F[F.tracker == "sym"]
        R = pd.concat([pd.read_csv(p / "removal.csv"), pd.read_csv(p / "removal_curve.csv")])
        R = R[(R.tracker == "sym") & (R.kind == "detections")]
        feats = [f for f in TRACK_FEATS if f in F and F[f].notna().any()]
        fw = F[F.condition == "forward"].set_index("well")
        wells = list(fw.index)
        # (a) lost detections: per sequence, median over features of the change (each: median over repeats)
        xs = sorted(R.X.unique())
        per = [[0.0] + [float(np.nanmedian(R[(R.X == x) & (R.well == w)][feats].median())) for x in xs] for w in wells]
        miss = pd.read_csv(p / "detection_model.csv").set_index("well").p_miss.reindex(wells).to_numpy()
        curves(axes[0], m, [0] + [100 * x for x in xs], 100 * np.asarray(per), 100 * miss)
        mu, sd = mean_sd(per)
        for w, row, q in zip(wells, per, miss):
            rows_a += [dict(model=m, sequence=w, lost_pct=100 * x, change_pct=100 * y) for x, y in zip([0] + xs, row)]
            rows_m.append(dict(model=m, sequence=w, miss_rate_pct=100 * q, change_at_miss_rate_pct=100 * np.interp(q, [0] + xs, row)))
        if m == "ARTORG":
            rows_a += [dict(model=m, sequence="mean of 4 wells", lost_pct=100 * x, change_pct=100 * a, sd_pct=100 * b)
                       for x, a, b in zip([0] + xs, mu, sd)]
            rows_m.append(dict(model=m, sequence="mean of 4 wells", miss_rate_pct=100 * miss.mean(),
                               change_at_miss_rate_pct=100 * np.interp(miss.mean(), [0] + xs, mu)))
        for w, row, q in zip(wells, per, miss):
            print(f"{w}: lost " + ", ".join(f"{100 * x:.0f}%: {100 * y:.1f}" for x, y in zip(xs, row[1:]))
                  + f"; miss rate {100 * q:.1f}% -> {100 * np.interp(q, [0] + xs, row):.1f}")
        if m == "ARTORG":
            print(f"ARTORG mean+-SD: " + ", ".join(f"{100 * x:.0f}%: {100 * a:.1f}+-{100 * b:.1f}" for x, a, b in
                                                 zip(xs, mu[1:], sd[1:]))
                  + f"; miss rate {100 * miss.mean():.1f}% -> {100 * np.interp(miss.mean(), [0] + xs, mu):.1f}")
        # (b) frame interval: per sequence, median over features of the change
        ks = sorted(int(c.split("_")[1]) for c in F.condition.unique() if c.startswith("every_"))
        per = []
        for w in wells:
            row = [0.0]
            for k in ks:
                Fk = F[(F.condition == f"every_{k}_frames") & (F.well == w)].iloc[0]
                row.append(float(np.nanmedian(rel(fw.loc[w, feats].astype(float), Fk[feats].astype(float)))))
            per.append(row)
        dt = FRAME_MIN[m]
        curves(axes[1], m, [dt] + [k * dt for k in ks], 100 * np.asarray(per))
        axes[1].plot([dt], [0], "o", ms=3.2, mfc="white", mec=COLOURS[m], mew=0.9, zorder=4, clip_on=False)
        mu, sd = mean_sd(per)
        for w, row in zip(wells, per):
            rows_b += [dict(model=m, sequence=w, every_kth_frame=k, frame_interval_min=k * dt, change_pct=100 * y)
                       for k, y in zip([1] + ks, row)]
        if m == "ARTORG":
            rows_b += [dict(model=m, sequence="mean of 4 wells", every_kth_frame=k, frame_interval_min=k * dt,
                            change_pct=100 * a, sd_pct=100 * b) for k, a, b in zip([1] + ks, mu, sd)]
        for w, row in zip(wells, per):
            print(f"{w}: interval native {dt:.1f} min; " + ", ".join(f"x{k} ({k * dt:.0f} min): {100 * y:.1f}"
                                                                    for k, y in zip(ks, row[1:])))
        if m == "ARTORG":
            print("ARTORG mean+-SD: " + ", ".join(f"x{k}: {100 * a:.1f}+-{100 * b:.1f}" for k, a, b in zip(ks, mu[1:], sd[1:])))
        if m == "ARTORG":
            fw_be, feats_be = fw, feats
    ax = axes[0]
    ax.plot([0, 30], [0, 30], ":", color="0.4", lw=0.8, label="change = removed")
    ax.plot([], [], "v", ms=4.5, mfc="white", mec="k", mew=0.5, label="measured miss rate")
    ax.set_xlabel("Detections removed at random (%)", labelpad=2)
    ax.set_ylabel("Change in features (%)", labelpad=2)
    ax.set_title("(a) Robustness to missed detections", fontsize=6.8, pad=3, loc="left")
    ax.set_xticks([0, 5, 10, 20, 30])
    hd, lb = ax.get_legend_handles_labels()
    order = sorted(range(len(lb)), key=lambda i: [LEGEND["ARTORG"], LEGEND["Song"], LEGEND["Hachey"]].index(lb[i])
                   if lb[i] in LEGEND.values() else 3 + i)
    ax.legend([hd[i] for i in order], [lb[i] for i in order], frameon=False, fontsize=6, loc="upper left",
              handlelength=1.5, borderaxespad=0.2, labelspacing=0.3)
    ax = axes[1]
    ax.set_xscale("log")
    ax.set_xticks([5, 15, 60, 240])
    ax.set_xticklabels(["5 min", "15 min", "1 h", "4 h"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Frame interval (log scale)", labelpad=2)
    ax.set_ylabel("Change in features (%)", labelpad=2)
    ax.set_title("(b) Robustness to a longer frame interval", fontsize=6.8, pad=3, loc="left")
    # (c) ARTORG reproducibility of each feature across the 4 replicate wells: 100% - CV
    # (100%: the same value in every well; 0%: a spread between wells as large as the mean)
    ax = axes[2]
    X = fw_be[[f for f in feats_be if f not in NOT_IN_C]].astype(float)
    rel_mean = 100 * X / X.mean()
    cv = X.std() / X.mean().abs()
    repro = (100 * (1 - cv)).clip(lower=0).sort_values()
    H = None
    try:
        from replicate_stats import heterogeneity
        H = heterogeneity(list(cv.index)).set_index("feature")
        print(H[["Q", "p", "p_holm", "I2", "cv", "median_within_cv"]].round(4).to_string())
    except (FileNotFoundError, ValueError):
        print("replicate_stats not run yet: no heterogeneity test in the CSV")
    ax.barh(range(len(repro)), repro.to_numpy(), height=0.7, color=COLOURS["ARTORG"], zorder=2)
    for i, v in enumerate(repro):
        ax.text(v + 1.5, i, f"{v:.0f}%", va="center", ha="left", fontsize=5.5)
    ax.set_yticks(range(len(repro)))
    ax.set_yticklabels([LABEL.get(k, k) for k in repro.index], fontsize=6)
    ax.set_ylim(-0.6, len(repro) - 0.4)
    ax.set_xlim(0, 114)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Reproducibility across 4 ARTORG wells (%)", labelpad=2)
    ax.set_title("(c) Reproducibility across replicate wells", fontsize=6.8, pad=3, loc="left")
    for a in axes:
        a.grid(alpha=0.3, lw=0.4, zorder=0)
        a.tick_params(width=0.6, length=2, pad=1)
        for s_ in ("top", "right"):
            a.spines[s_].set_visible(False)
    axes[0].set_ylim(0, 50)                               # headroom for the legend above the curves
    axes[1].set_ylim(bottom=0)
    for ext in ("pdf", "png"):
        fig.savefig(OUT.with_suffix("." + ext), dpi=300, bbox_inches="tight", pad_inches=0.02)
    # the plotted numbers
    DATA.mkdir(exist_ok=True)
    pd.DataFrame(rows_a).to_csv(DATA / "real_data_a_lost_detections.csv", index=False)
    pd.DataFrame(rows_m).to_csv(DATA / "real_data_a_miss_rate.csv", index=False)
    pd.DataFrame(rows_b).to_csv(DATA / "real_data_b_frame_interval.csv", index=False)
    wells_c = X.stack().rename("value").reset_index().rename(columns={"level_1": "feature"})
    wells_c = wells_c.rename(columns={wells_c.columns[0]: "well"})
    wells_c["value_rel_to_mean_pct"] = [rel_mean.loc[w, f] for w, f in zip(wells_c.well, wells_c.feature)]
    wells_c.insert(2, "label", wells_c.feature.map(LABEL))
    wells_c.to_csv(DATA / "real_data_c_wells.csv", index=False)
    summ = pd.DataFrame(dict(feature=cv.index, label=[LABEL.get(k, k) for k in cv.index], mean=X.mean()[cv.index].values,
                             sd=X.std()[cv.index].values, cv_pct=100 * cv.values,
                             reproducibility_pct=(100 * (1 - cv)).clip(lower=0).values))
    if H is not None:
        summ = summ.merge(H.reset_index()[["feature", "Q", "df", "p", "p_holm", "I2", "median_within_cv"]], on="feature",
                          how="left")
        summ["wells_differ_beyond_sampling_noise"] = summ.p_holm < 0.05
    summ.to_csv(DATA / "real_data_c_summary.csv", index=False)
    print(OUT.with_suffix(".pdf"), "| CSVs in", DATA)


if __name__ == "__main__":
    main()
