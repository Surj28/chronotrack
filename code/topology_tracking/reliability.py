"""Per-feature reliability estimate (paper Sec. II-D).

For each operator and level, perturb the pipeline's own tracks, recompute
per-track features, and compare them with the unperturbed features of the
same track id:

    E_f(level) = mean_tracks |f_perturbed - f_clean| / IQR_f

normalised by the feature's interquartile range across tracks (a relative
deviation is undefined for features that are often 0, such as n_exits; the
standard deviation is used where the IQR is 0). The mean, not the median,
is used because most perturbations change only a minority of tracks, which
a median would not register.
The x-axis is the *realised* error rate (fraction of detections dropped,
fraction of links whose identity changed, or mask radius change in px), so
it is directly comparable with the rates measured against annotation
(metrics.evaluate()['rates']).

The tolerated rate p*_f is the realised rate at which E_f first exceeds tau
(linear interpolation between levels); a feature is reliable at the
pipeline's measured rate p_hat if p_hat < p*_f.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import perturb
from .states import FEATURES, add_vessel_distance, assign_states, track_features
from .tracker import _load_array


def feature_scales(clean: pd.DataFrame) -> pd.Series:
    q = clean[FEATURES].quantile([0.25, 0.75])
    s = q.loc[0.75] - q.loc[0.25]
    s = s.where(s > 0, clean[FEATURES].std())
    return s.where(s > 0, 1.0)


def feature_error(clean: pd.DataFrame, pert: pd.DataFrame, scale: pd.Series) -> pd.Series:
    m = clean[["track_id"] + FEATURES].merge(pert[["track_id"] + FEATURES], on="track_id",
                                              suffixes=("_c", "_p"))
    return pd.Series({f: float(np.nanmean(np.abs(m[f + "_p"] - m[f + "_c"]))) / scale[f]
                      for f in FEATURES})


def switch_fraction(pert: pd.DataFrame) -> float:
    """Fraction of a cell's consecutive observations whose track id differs
    (needs the `orig_track_id` column set before perturbing)."""
    s = pert.sort_values(["orig_track_id", "frame_idx"])
    same_cell = s["orig_track_id"].to_numpy()[1:] == s["orig_track_id"].to_numpy()[:-1]
    changed = s["track_id"].to_numpy()[1:] != s["track_id"].to_numpy()[:-1]
    return float(changed[same_cell].mean()) if same_cell.any() else 0.0


def error_curves(tracks: pd.DataFrame, *, levels: dict, n_rep: int, seed: int,
                 um_per_px: float, frame_interval_h: float, k: int, delta_um: float,
                 max_gap: int, cell_label_maps: list | None = None,
                 vessel_masks: list | None = None) -> pd.DataFrame:
    """`tracks` must already carry d_vessel_um and state (states.py).

    levels: {"dropout": [p...], "swap": [p...], "cell_mask": [r...],
    "vessel_mask": [r...]}; the mask operators need the label maps / vessel
    masks (arrays or paths, indexed by frame)."""
    feat = dict(frame_interval_h=frame_interval_h, um_per_px=um_per_px, k=k)
    clean = track_features(tracks, **feat)
    scale = feature_scales(clean)
    rng = np.random.default_rng(seed)
    base = tracks.assign(orig_track_id=tracks["track_id"])
    rows = []

    def add(op, level, rep, rate, pert_feats):
        for f, e in feature_error(clean, pert_feats, scale).items():
            rows.append(dict(operator=op, level=level, rep=rep, realised_rate=rate, feature=f, error=e))

    for p in levels.get("dropout", []):
        for rep in range(n_rep):
            pt = perturb.drop_detections(base, p, max_gap, rng)
            add("dropout", p, rep, 1 - len(pt) / len(base), track_features(pt, **feat))
    for p in levels.get("swap", []):
        for rep in range(n_rep):
            pt = perturb.swap_identities(base, p, rng)
            add("swap", p, rep, switch_fraction(pt), track_features(pt, **feat))
    for r in levels.get("cell_mask", []):
        maps = [perturb.erode_dilate_labels(_load_array(m), r) for m in cell_label_maps]
        pt = perturb.remeasure_tracks(base, maps)
        pt = assign_states(add_vessel_distance(pt, vessel_masks, um_per_px), delta_um)
        add("cell_mask", r, 0, float(r), track_features(pt, **feat))
    for r in levels.get("vessel_mask", []):
        pt = add_vessel_distance(base, vessel_masks, um_per_px,
                                 transform=lambda m, r=r: perturb.erode_dilate_mask(m, r))
        pt = assign_states(pt, delta_um)
        add("vessel_mask", r, 0, float(r), track_features(pt, **feat))
    return pd.DataFrame(rows)


def tolerated_rates(curves: pd.DataFrame, tau: float) -> pd.DataFrame:
    """p*_f per operator and feature. Mask operators are two-sided (erode /
    dilate), so their tolerance is reported separately per sign as the
    largest |radius| still within tau."""
    out = []
    mean = curves.groupby(["operator", "feature", "level"], as_index=False)[["realised_rate", "error"]].mean()
    for (op, f), g in mean.groupby(["operator", "feature"]):
        sides = [("", g)] if op in ("dropout", "swap") else [
            ("-", g[g["level"] <= 0].assign(realised_rate=lambda d: -d["realised_rate"])),
            ("+", g[g["level"] >= 0])]
        for sign, s in sides:
            s = s.sort_values("realised_rate")
            x, e = s["realised_rate"].to_numpy(), s["error"].to_numpy()
            over = np.flatnonzero(e > tau)
            if not len(over):
                p_star = np.inf  # within tolerance over the whole tested range
            elif over[0] == 0:
                p_star = 0.0
            else:
                i = over[0]
                p_star = float(np.interp(tau, [e[i - 1], e[i]], [x[i - 1], x[i]]))
            out.append(dict(operator=op + sign, feature=f, p_star=p_star,
                             max_tested=float(x.max()) if len(x) else np.nan))
    return pd.DataFrame(out)
