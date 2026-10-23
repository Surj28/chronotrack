"""Vessel-referenced states and per-track features (paper Sec. II-C).

Per track position: distance d_t to the vessel mask (0 on the mask), then
    overlapping  d_t = 0
    adjacent     0 < d_t <= delta
    distant      d_t > delta
with delta = one median cell radius. A vessel exit is a run of at least k
non-overlapping observations after an overlapping one. Because a cell that
has left the lumen but sits on the outer wall still overlaps the vessel in
2D, exits are a lower bound.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from .tracker import _load_array

STATES = ("overlapping", "adjacent", "distant")

FEATURES = [
    "speed_um_h", "net_disp_um", "straightness", "duration_h",
    "frac_overlapping", "frac_adjacent", "frac_distant",
    "mean_d_vessel_um", "n_exits",
]


def default_delta_um(tracks: pd.DataFrame, um_per_px: float) -> float:
    """One median cell radius (um), from the tracked cells' own areas."""
    return float(np.sqrt(np.median(tracks["area"]) / np.pi) * um_per_px)


def add_vessel_distance(tracks: pd.DataFrame, vessel_masks: list, um_per_px: float,
                        transform=None) -> pd.DataFrame:
    """Add `d_vessel_um`, the distance from each track position to the
    nearest vessel pixel (0 on the vessel). `vessel_masks` holds one mask
    (array or path) per frame, indexed by frame_idx. `transform`, if given,
    is applied to each mask first (used by the vessel null test and the
    vessel-mask perturbation)."""
    out = tracks.copy()
    d = np.full(len(out), np.nan)
    for t, idx in out.groupby("frame_idx").indices.items():
        mask = _load_array(vessel_masks[int(t)]) > 0
        if transform is not None:
            mask = transform(mask)
        dist = ndi.distance_transform_edt(~mask)
        r = np.clip(out["centroid_r"].to_numpy()[idx].round().astype(int), 0, mask.shape[0] - 1)
        c = np.clip(out["centroid_c"].to_numpy()[idx].round().astype(int), 0, mask.shape[1] - 1)
        d[idx] = dist[r, c] * um_per_px
    out["d_vessel_um"] = d
    return out


def assign_states(tracks: pd.DataFrame, delta_um: float) -> pd.DataFrame:
    out = tracks.copy()
    d = out["d_vessel_um"].to_numpy()
    out["state"] = np.where(d == 0, "overlapping", np.where(d <= delta_um, "adjacent", "distant"))
    return out


def count_exits(overlapping: np.ndarray, k: int) -> int:
    """Number of runs of >= k non-overlapping observations that follow an
    overlapping one."""
    n, seen_on, run = 0, False, 0
    for on in overlapping:
        if on:
            seen_on, run = True, 0
        elif seen_on:
            run += 1
            if run == k:
                n += 1
                seen_on = False  # a new exit needs a return to the vessel first
    return n


def track_features(tracks: pd.DataFrame, frame_interval_h: float, um_per_px: float,
                   k: int, min_obs: int = 3) -> pd.DataFrame:
    """One row per track with at least `min_obs` observations.

    speed_um_h is the mean step length divided by the time between the two
    observations (so a gap counts as one longer step). straightness is net
    displacement over path length. `started_on_vessel` marks tracks taken to
    be intravascular under the loading protocol."""
    rows = []
    for tid, g in tracks.sort_values("frame_idx").groupby("track_id", sort=False):
        if len(g) < min_obs:
            continue
        t = g["frame_idx"].to_numpy()
        p = g[["centroid_r", "centroid_c"]].to_numpy() * um_per_px
        steps = np.linalg.norm(np.diff(p, axis=0), axis=1)
        dt_h = np.diff(t) * frame_interval_h
        path = float(steps.sum())
        net = float(np.linalg.norm(p[-1] - p[0]))
        state = g["state"].to_numpy()
        on = state == "overlapping"
        rows.append(dict(
            track_id=tid, first_frame=int(t[0]), n_obs=len(g),
            start_r=float(g["centroid_r"].iloc[0]), start_c=float(g["centroid_c"].iloc[0]),
            speed_um_h=float(np.mean(steps / dt_h)),
            net_disp_um=net,
            straightness=net / path if path > 0 else np.nan,
            duration_h=float((t[-1] - t[0]) * frame_interval_h),
            frac_overlapping=float(on.mean()),
            frac_adjacent=float((state == "adjacent").mean()),
            frac_distant=float((state == "distant").mean()),
            mean_d_vessel_um=float(g["d_vessel_um"].mean()),
            n_exits=count_exits(on, k),
            started_on_vessel=bool(on[0]),
        ))
    return pd.DataFrame(rows, columns=["track_id", "first_frame", "n_obs", "start_r", "start_c",
                                       *FEATURES, "started_on_vessel"])


def features_from_tracks(tracks: pd.DataFrame, vessel_masks: list, *, um_per_px: float,
                         frame_interval_h: float, k: int, delta_um: float,
                         vessel_transform=None, min_obs: int = 3) -> pd.DataFrame:
    """Vessel distance -> states -> per-track features, in one call."""
    t = add_vessel_distance(tracks, vessel_masks, um_per_px, vessel_transform)
    t = assign_states(t, delta_um)
    return track_features(t, frame_interval_h, um_per_px, k, min_obs)
