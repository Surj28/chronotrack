"""Shared settings of ChronoTrack's experiments.

All parameters are fixed in advance (paper Sec. III-A): w from the earlier
CTC work, d_max estimated from each sequence's own detections, and g, k, tau
set once here. Nothing is tuned on the benchmark.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from topology_tracking.tracker import _cell_props, _load_array, build_tracks_neighbor_kf  # noqa: E402

UM_PER_PX = 0.65    # pixel size (um) and frame interval (h) of the sequences
FRAME_H = 1.0

W = 0.35          # topology weight (from the earlier CTC work, not tuned here)
MAX_GAP = 1       # g: frames a track may be missed before it ends
K_EXIT = 2        # k: non-overlapping frames needed to call a vessel exit
TAU = 0.10        # reliability tolerance, in IQR units
SEED = 0


def cell_spacing(label_maps: list) -> float:
    """Median distance to the nearest other cell, over frames."""
    sp = []
    for m in label_maps:
        c = np.array([v[:2] for v in _cell_props(_load_array(m)).values()])
        if len(c) > 2:
            sp.append(np.median(cKDTree(c).query(c, k=2)[0][:, 1]))
    return float(np.median(sp))


def estimate_max_dist(label_maps: list, pilot_frames: int = 8) -> dict:
    """d_max from the data alone: a distance-only pilot pass with a gate of
    twice the cell spacing; d_max = max(1.5 x its 95th-percentile step,
    half the spacing). On the CTC sequences the pilot's 95th percentile is
    within a few percent of the ground-truth one at every frame interval."""
    maps = [_load_array(m) for m in label_maps[:pilot_frames]]
    spacing = cell_spacing(maps)
    df = build_tracks_neighbor_kf(maps, None, max_dist=2 * spacing, neighbor_weight=0.0,
                                  border_px=None, max_gap=0)
    d = df.sort_values(["track_id", "frame_idx"])
    same = d["track_id"].to_numpy()[1:] == d["track_id"].to_numpy()[:-1]
    steps = np.hypot(np.diff(d["centroid_r"].to_numpy()), np.diff(d["centroid_c"].to_numpy()))[same]
    p95 = float(np.percentile(steps, 95)) if steps.size else spacing
    return dict(spacing_px=spacing, pilot_step_p95_px=p95,
                max_dist_px=max(1.5 * p95, 0.5 * spacing),
                ambiguity_index=p95 / spacing)


def symmetric_gate(label_maps: list, n_windows: int = 4, window: int = 3) -> dict:
    """estimate_max_dist on short windows spread evenly over the sequence
    (centred at (2k+1)/(2 n_windows) of its length), so that a movie and its
    reverse get the same gate. estimate_max_dist alone uses the first frames,
    which differ between the two directions when the movie is not stationary
    (e.g. cells accumulating)."""
    T = len(label_maps)
    if T < n_windows * window:
        return estimate_max_dist(label_maps, pilot_frames=T)
    spac, steps = [], []
    for k in range(n_windows):
        c = int(round((2 * k + 1) / (2 * n_windows) * (T - 1)))
        lo = int(np.clip(c - window // 2, 0, T - window))
        g = estimate_max_dist(label_maps[lo:lo + window], pilot_frames=window)
        spac.append(g["spacing_px"]); steps.append(g["pilot_step_p95_px"])
    spacing, p95 = float(np.median(spac)), float(np.median(steps))
    return dict(spacing_px=spacing, pilot_step_p95_px=p95, max_dist_px=max(1.5 * p95, 0.5 * spacing),
                ambiguity_index=p95 / spacing)


# From trivial to established: no linking, greedy nearest neighbour, our
# tracker without the topology term, btrack, and ours.
TRACKERS = ("none", "nn", "w0", "btrack", "ours")


def run_tracker(name: str, label_maps: list, vessel_masks: list | None, max_dist: float,
                w: float = W, max_gap: int = MAX_GAP) -> pd.DataFrame:
    """'ours' (topology term, weight w), 'w0' (same code, distance only),
    'btrack', 'nn' (greedy nearest neighbour) or 'none' (no linking). New
    tracks may start anywhere (border_px=None)."""
    from topology_tracking import baselines
    if name == "btrack":
        return baselines.run_btrack(label_maps, max_dist)
    if name == "nn":
        return baselines.run_nearest_neighbour(label_maps, max_dist)
    if name == "none":
        return baselines.run_no_linking(label_maps)
    return build_tracks_neighbor_kf(label_maps, vessel_masks, max_dist=max_dist, max_gap=max_gap,
                                    border_px=None, neighbor_weight=(w if name == "ours" else 0.0))
