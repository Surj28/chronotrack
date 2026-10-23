"""Baselines, returned in the same table format as build_tracks_neighbor_kf
so they go through the same evaluation: btrack (Bayesian cell tracker,
Ulicna et al. 2021) with its distance thresholds set to the shared gate and
link-only hypotheses, greedy nearest-neighbour linking, and no linking.

btrack's output positions are mapped back to our detection labels by
centroid (btrack reports the object centroids it was given; positions it
predicts for missed frames, flagged `dummy`, are dropped).
"""
from __future__ import annotations

import contextlib
import os
import sys

import numpy as np
import pandas as pd

from .tracker import _cell_props, _load_array


@contextlib.contextmanager
def _quiet():
    """Silence stdout/stderr at the file-descriptor level (btrack's solver
    prints from C)."""
    sys.stdout.flush(); sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    with open(os.devnull, "w") as null:
        os.dup2(null.fileno(), 1); os.dup2(null.fileno(), 2)
        try:
            yield
        finally:
            sys.stdout.flush(); sys.stderr.flush()
            os.dup2(saved[0], 1); os.dup2(saved[1], 2)
            os.close(saved[0]); os.close(saved[1])


def run_btrack(label_maps: list, max_dist: float, config_path: str | None = None) -> pd.DataFrame:
    with _quiet():
        return _run_btrack(label_maps, max_dist, config_path)


def _run_btrack(label_maps: list, max_dist: float, config_path: str | None) -> pd.DataFrame:
    import btrack
    from btrack import datasets as btrack_datasets

    stack = np.stack([_load_array(m) for m in label_maps]).astype(np.int32)
    lookup = {}
    for t in range(stack.shape[0]):
        for lb, (r, c, _a) in _cell_props(stack[t]).items():
            lookup[(t, round(r, 1), round(c, 1))] = (lb, r, c, _a)
    objects = btrack.utils.segmentation_to_objects(stack, properties=("area",))
    rows = []
    with btrack.BayesianTracker(verbose=False) as tracker:
        tracker.configure(config_path or btrack_datasets.cell_config())
        tracker.max_search_radius = max_dist
        # Same gate as every other tracker: the default config caps link
        # hypotheses at a fixed 40 px (theta_dist 20 px), which cuts true
        # links at long frame intervals. Division hypotheses (P_branch,
        # P_dead) are left out, as no compared tracker models division; with
        # them the optimiser does not finish on dense sequences.
        hm = tracker.hypothesis_model
        hm.dist_thresh = max_dist
        hm.theta_dist = max_dist / 2
        hm.hypotheses = ["P_FP", "P_init", "P_term", "P_link"]
        tracker.append(objects)
        tracker.volume = ((0, stack.shape[2]), (0, stack.shape[1]))
        tracker.track()
        tracker.optimize()
        for trk in tracker.tracks:
            dummy = getattr(trk, "dummy", [False] * len(trk))
            for t, x, y, d in zip(trk.t, trk.x, trk.y, dummy):
                if d:
                    continue
                hit = lookup.get((int(t), round(float(y), 1), round(float(x), 1)))
                if hit is None:
                    continue
                lb, r, c, a = hit
                rows.append(dict(track_id=int(trk.ID), frame_idx=int(t), label=int(lb),
                                 centroid_r=r, centroid_c=c, area=a,
                                 inside_vessel=False, recovered_from_green=False))
    return pd.DataFrame(rows, columns=["track_id", "frame_idx", "label", "centroid_r", "centroid_c",
                                       "area", "inside_vessel", "recovered_from_green"])


# ---------------------------------------------------------------------------
# Trivial floors: any linking method has to beat these.
# ---------------------------------------------------------------------------

def _rows_for(t: int, props: dict, ids) -> list[dict]:
    return [dict(track_id=int(k), frame_idx=t, label=int(lb), centroid_r=props[lb][0],
                 centroid_c=props[lb][1], area=props[lb][2], inside_vessel=False,
                 recovered_from_green=False) for lb, k in ids]


def run_no_linking(label_maps: list) -> pd.DataFrame:
    """Every detection is its own one-frame track: the score of detection
    alone, with no linking at all."""
    rows, k = [], 1
    for t, m in enumerate(label_maps):
        props = _cell_props(_load_array(m))
        rows += _rows_for(t, props, [(lb, k + i) for i, lb in enumerate(props)])
        k += len(props)
    return pd.DataFrame(rows)


def run_nearest_neighbour(label_maps: list, max_dist: float) -> pd.DataFrame:
    """Greedy nearest-neighbour linking between consecutive frames: no motion
    model, no gap closing; pairs are taken in order of distance, each track
    and detection at most once, pairs beyond max_dist never."""
    from scipy.spatial import cKDTree
    rows, next_id, prev = [], 1, None  # prev: (positions, track ids)
    for t, m in enumerate(label_maps):
        props = _cell_props(_load_array(m))
        labels = list(props)
        pos = np.array([props[lb][:2] for lb in labels]).reshape(-1, 2)
        ids = np.zeros(len(labels), dtype=np.int64)
        if prev is not None and len(prev[0]) and len(pos):
            pairs = cKDTree(prev[0]).sparse_distance_matrix(cKDTree(pos), max_dist, output_type="ndarray")
            used_p, used_d = set(), set()
            for i, j, _d in sorted(pairs.tolist(), key=lambda x: x[2]):
                if i not in used_p and j not in used_d:
                    used_p.add(i); used_d.add(j)
                    ids[j] = prev[1][i]
        for j in range(len(labels)):
            if ids[j] == 0:
                ids[j] = next_id
                next_id += 1
        rows += _rows_for(t, props, zip(labels, ids))
        prev = (pos, ids)
    return pd.DataFrame(rows)
