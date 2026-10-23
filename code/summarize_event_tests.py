"""Compact tables from the label-free event tests (run_event_tests.py,
run_external_events.py): one block per test, one row per tracker.

Only features that depend on the linking are scored (TRACK_FEATS): cells per
frame, state occupancy and the vessel fractions are the same for every tracker
(and cells per frame changes by exactly X under detection removal).

Usage (from the repo root):
    python paper/code/summarize_event_tests.py [results/event_tests] [results/event_tests_external]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


TRACK_FEATS = ["arrivals_per_100_cell_h", "departures_per_100_cell_h", "arrival_on_vessel_frac",
               "departure_on_vessel_frac", "arrival_vessel_enrichment", "departure_vessel_enrichment",
               "exits_per_100_cell_h", "entries_per_100_cell_h", "speed_on_vessel_um_h", "speed_off_vessel_um_h",
               "mean_track_duration_h", "exit_regression_colocation", "entry_emergence_colocation"]
ORDER = ["sym", "sym_vessel", "first_order", "w0", "nn"]


def rel(a, b):
    return np.abs(b - a) / np.abs(a).where(np.abs(a) > 1e-12)


def summarize(d: Path) -> dict:
    F = pd.read_csv(d / "features.csv")
    R = pd.read_csv(d / "removal.csv")
    E = pd.read_csv(d / "reversal_events.csv")
    A = pd.read_csv(d / "accounting.csv")
    feats = [f for f in TRACK_FEATS if f in F.columns]
    fw = F[F.condition == "forward"].set_index(["well", "tracker"])
    out = {}
    # forward values (median over sequences)
    out["forward"] = fw[["cells_per_frame", "mean_track_duration_h", "arrivals_per_100_cell_h", "exits_per_100_cell_h",
                         "speed_on_vessel_um_h", "speed_off_vessel_um_h"]].groupby("tracker").median().reindex(ORDER)
    # 1b detection removal: median relative change over sequences, repeats and features
    det = R[R.kind == "detections"].melt(["well", "tracker", "kind", "X", "rep"], feats, "feature", "rel")
    out["removal_detections"] = det.groupby(["tracker", "X"]).rel.median().unstack().reindex(ORDER)
    # 1a track removal (sym): per feature
    trk = R[R.kind == "tracks"].melt(["well", "tracker", "kind", "X", "rep"], feats, "feature", "rel")
    out["removal_tracks_sym"] = trk.groupby(["feature", "X"]).rel.median().unstack()
    # 2 reversal, feature level: forward vs mirrored reversed
    rv = F[F.condition == "reversed_mirrored"].set_index(["well", "tracker"])
    rr = pd.DataFrame({f: rel(fw[f], rv.loc[fw.index, f]) for f in feats})
    out["reversal_features"] = rr.groupby("tracker").median().median(axis=1).rename("median_rel_diff").to_frame() \
        .join(rr.groupby("tracker").median()[["mean_track_duration_h", "arrivals_per_100_cell_h", "exits_per_100_cell_h"]]) \
        .reindex(ORDER)
    # 2 reversal, event level
    out["reversal_events"] = E.groupby(["tracker", "forward_event"]).frac_matched.mean().unstack().reindex(ORDER)
    # 3 accounting
    a = A.groupby("tracker")[["n_arrivals", "arrival_edge_frac", "edge_chance", "arrival_fragment_frac",
                              "fragment_chance"]].median()
    a["fragment_excess"] = a.arrival_fragment_frac - a.fragment_chance
    out["accounting"] = a.reindex(ORDER)
    # 4 sampling: every k-th frame vs every frame
    sm = F[F.condition.str.startswith("every_")]
    rows = []
    for (w, tr, c), g in sm.groupby(["well", "tracker", "condition"]):
        r = rel(fw.loc[(w, tr), feats].astype(float), g.iloc[0][feats].astype(float))
        rows.append(dict(well=w, tracker=tr, condition=c, median_rel=float(np.nanmedian(r)),
                         **{f: float(r[f]) for f in ("mean_track_duration_h", "arrivals_per_100_cell_h",
                                                      "speed_off_vessel_um_h")}))
    out["sampling"] = pd.DataFrame(rows).groupby(["condition", "tracker"]).median(numeric_only=True) \
        .reindex(ORDER, level="tracker")
    # 5 replicate consistency (meaningful for replicate wells only, i.e. ARTORG): CV across sequences
    if fw.index.get_level_values("well").nunique() >= 3:
        g = fw[feats].groupby("tracker")
        cv = (g.std() / g.mean().abs()).replace([np.inf, -np.inf], np.nan)
        out["replicate_cv"] = cv.median(axis=1).rename("median_cv").to_frame() \
            .join(cv[["mean_track_duration_h", "arrivals_per_100_cell_h", "exits_per_100_cell_h"]]).reindex(ORDER)
    return out


def main():
    dirs = [Path(p) for p in sys.argv[1:]] or [Path(__file__).resolve().parent.parent / "results" / "event_tests"]
    pd.set_option("display.width", 200)
    for d in dirs:
        print(f"\n######## {d}")
        for k, v in summarize(d).items():
            print(f"\n== {k}\n{v.round(3).to_string()}")


if __name__ == "__main__":
    main()
