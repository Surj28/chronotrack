"""tracker.py — standalone implementation of the topology-preserving cell tracker
described in "Topology-Preserving Cell Tracking via Delaunay Neighbourhood
Consistency for Dense, Long-Interval Time-Lapse Microscopy".

Self-contained: only numpy, pandas, tifffile, scipy, scikit-image are required.
No dependency on the rest of the ChronoTrack pipeline.

Public entry point: build_tracks_neighbor_kf(). Setting neighbor_weight=0.0
recovers a pure distance/Kalman baseline (no topology term) using the exact
same code path -- this is the comparison condition used throughout the paper's
ablation, chosen specifically to avoid confounding the topology term's effect
with any other tracker design choice (see paper Section IV-A).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff
from scipy.optimize import linear_sum_assignment
from scipy.spatial import Delaunay as _Delaunay
from skimage.measure import regionprops

# ---------------------------------------------------------------------------
# Kalman filter matrices (module-level, shared across all tracks)
# State vector: [row, col, d_row, d_col]
#
# _KF_F's off-diagonal 1s add the FULL estimated velocity to position each
# step (a standard constant-velocity model). velocity_damping below controls
# how much of the estimated velocity is added to the position prediction
# (1.0 = original undamped behaviour, 0.0 = position-only / zero-velocity
# model).
#
# CAUTION -- this is a genuinely mixed result, not a validated improvement:
# in isolation, per-track prediction error at alpha=0 (no velocity trust) is
# lower than at alpha=1 on the two CTC sequences it was first checked
# against (n=8123, n=68686 real trajectory triples; ~1.5x worse at full
# trust on both). But once embedded in the full joint Hungarian assignment
# (where multiple tracks' predictions compete simultaneously, not just one
# track's isolated error), damping's effect on actual tracking accuracy was
# mixed across all four CTC sequences tested: better on 3, meaningfully
# *worse* on one (Fluo-N2DL-HeLa-02: hard-case accuracy dropped from 0.643
# to 0.573 at w=0, -11 of 157 cases). It also did not change the
# neighbourhood-consistency term's own null result on hard cases (still no
# consistent benefit, before or after damping -- see paper Discussion).
# Treat this parameter as an unresolved, dataset-dependent tuning knob, not
# a default improvement; it has NOT been validated at all on longer-interval
# data (e.g. hourly imaging), where a real velocity signal is more likely to
# dominate measurement noise.
# ---------------------------------------------------------------------------
_KF_H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
_KF_Q = np.diag([25.0, 25.0, 500.0, 500.0])  # process noise
_KF_R = np.diag([25.0, 25.0])                # measurement noise
_KF_P0 = np.eye(4) * 500.0                   # initial covariance (high uncertainty at birth)


def _kf_F(velocity_damping: float = 1.0) -> np.ndarray:
    """State transition matrix with the estimated velocity's contribution to
    the position update scaled by velocity_damping (see module docstring)."""
    a = float(velocity_damping)
    return np.array([[1, 0, a, 0], [0, 1, 0, a], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)


@dataclass
class _TrackState:
    """Per-track Kalman filter state + last confirmed mask for IoU matching."""
    track_id: int
    state: np.ndarray    # [r, c, dr, dc], float64
    cov: np.ndarray       # 4x4 error covariance
    mask_flat: np.ndarray | None  # unused by any cost term; kept for API compatibility
    last_area: int = 0
    frames_lost: int = 0
    velocity_damping: float = 1.0  # see module docstring above _kf_F

    @property
    def centroid(self) -> tuple[float, float]:
        return (float(self.state[0]), float(self.state[1]))

    def kf_predict(self) -> None:
        F = _kf_F(self.velocity_damping)
        self.state = F @ self.state
        self.cov = F @ self.cov @ F.T + _KF_Q

    def kf_update(self, r: float, c: float, mask_flat: np.ndarray | None, area: int) -> None:
        z = np.array([r, c], dtype=np.float64)
        S = _KF_H @ self.cov @ _KF_H.T + _KF_R
        K = self.cov @ _KF_H.T @ np.linalg.inv(S)
        self.state = self.state + K @ (z - _KF_H @ self.state)
        self.cov = (np.eye(4) - K @ _KF_H) @ self.cov
        self.mask_flat = mask_flat
        self.last_area = int(area)
        self.frames_lost = 0


def _cell_props(label_map: np.ndarray) -> dict[int, tuple[float, float, int]]:
    """Return {cell_label: (centroid_r, centroid_c, area)} for all cells."""
    props = {}
    for rp in regionprops(label_map):
        props[rp.label] = (rp.centroid[0], rp.centroid[1], rp.area)
    return props


def _extract_flat_masks(label_map: np.ndarray) -> dict[int, np.ndarray]:
    """Return {label: flat_pixel_indices} for each non-background cell."""
    flat = label_map.ravel()
    masks = {}
    for lb in np.unique(flat):
        if lb == 0:
            continue
        masks[int(lb)] = np.where(flat == lb)[0]
    return masks


def _centroid_in_mask(cr: float, cc: float, mask: np.ndarray) -> bool:
    r, c = int(cr), int(cc)
    return bool(mask[r, c]) if 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1] else False


def _load_array(src) -> np.ndarray:
    """A label map / mask given as an array, a .npy path, or a .tif path."""
    if isinstance(src, np.ndarray):
        return src
    src = Path(src)
    return np.load(str(src)) if src.suffix == ".npy" else tiff.imread(str(src))


# ---------------------------------------------------------------------------
# Neighbourhood-graph helpers -- the paper's core contribution
# ---------------------------------------------------------------------------

def _build_delaunay_neighbors(
    centroids: np.ndarray, max_edge_factor: float | None = None,
) -> dict[int, set[int]]:
    """Delaunay-triangulation adjacency dict for a set of 2-D points.

    Returns {i: set_of_neighbour_indices}. Falls back to an empty adjacency
    dict when fewer than 3 points are supplied (Delaunay undefined) or the
    triangulation raises (e.g. all points collinear).

    max_edge_factor: if given, discard any edge longer than
    max_edge_factor * median(all edge lengths in this triangulation).
    Plain Delaunay triangulation always includes the convex hull boundary,
    which frequently connects points that are not meaningfully "neighbours"
    in any physical sense (e.g. two cells on opposite sides of a sparse
    region, joined only because nothing lies between them) -- exactly the
    kind of spurious long edge that would corrupt the neighbourhood-
    consistency cost. None (default) preserves the original, unfiltered
    behaviour.
    """
    n = len(centroids)
    adj: dict[int, set[int]] = {i: set() for i in range(n)}
    if n < 3:
        return adj
    try:
        tri = _Delaunay(centroids)
        edges: set[tuple[int, int]] = set()
        for simplex in tri.simplices:
            a, b, c = int(simplex[0]), int(simplex[1]), int(simplex[2])
            for u, v in ((a, b), (a, c), (b, c)):
                edges.add((min(u, v), max(u, v)))

        if max_edge_factor is not None and edges:
            lengths = {
                (u, v): float(np.linalg.norm(centroids[u] - centroids[v]))
                for u, v in edges
            }
            cutoff = max_edge_factor * float(np.median(list(lengths.values())))
            edges = {e for e, L in lengths.items() if L <= cutoff}

        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
    except Exception:
        pass
    return adj


def _orientation_cost_matrix(
    track_positions: np.ndarray,      # (n_tracks, 2) tracks' own last-confirmed positions
    det_cents: np.ndarray,            # (n_dets, 2) current detection centroids
    track_adj: dict[int, set[int]],   # track-index -> neighbour track-indices
    nearest_det: np.ndarray,          # (n_tracks,) each track's nearest current detection
) -> np.ndarray:
    """(n_dets, n_tracks) orientation-inconsistency cost matrix -- a genuinely
    topological alternative to _neighbor_cost_matrix's Delaunay-membership
    check (binary or soft), motivated by the observation that "preserved
    topology" should NOT require preserved *distances*: a local cluster can
    expand, contract, or drift as a whole while still being the same
    configuration, and neither the binary nor the soft distance-based cost
    above can tell that case apart from a genuine identity swap (both are,
    underneath, still distance metrics -- which is likely why neither showed
    a measurable benefit; see paper Discussion).

    What genuinely distinguishes a real identity swap from ordinary motion is
    whether neighbours change their *relative angular order* around the
    point in question -- e.g. two neighbours crossing to opposite sides is a
    real topological change; both drifting further away without crossing is
    not. For each pair of track i's neighbours (n1, n2), we compare the sign
    of the cross product of (n1 - i, n2 - i) in the previous frame against
    the same sign using (nearest_det[n1] - det_j, nearest_det[n2] - det_j)
    for a candidate match j -- i.e. did n1 and n2 swap sides around i as seen
    from candidate j? cost[j, i] is the fraction of i's neighbour pairs whose
    orientation is reversed; invariant to uniform scaling, rotation, and
    translation of the local neighbourhood, unlike a distance-based cost.
    Tracks with fewer than 2 neighbours (orientation undefined) get a
    neutral cost of 0, as does any pair whose points are collinear (cross
    product exactly 0, no defined orientation).
    """
    n_t, n_d = len(track_positions), len(det_cents)
    cost = np.zeros((n_d, n_t), dtype=np.float32)
    if n_t == 0 or n_d == 0:
        return cost

    for i in range(n_t):
        nbrs_i = sorted(track_adj.get(i, set()))
        if len(nbrs_i) < 2:
            continue
        pi = track_positions[i]
        prev_vecs = {n: track_positions[n] - pi for n in nbrs_i}
        pairs = [(nbrs_i[a], nbrs_i[b]) for a in range(len(nbrs_i)) for b in range(a + 1, len(nbrs_i))]
        prev_signs = {
            (a, b): np.sign(prev_vecs[a][0] * prev_vecs[b][1] - prev_vecs[a][1] * prev_vecs[b][0])
            for a, b in pairs
        }
        for j in range(n_d):
            pj = det_cents[j]
            curr_vecs = {n: det_cents[nearest_det[n]] - pj for n in nbrs_i}
            n_valid = n_flipped = 0
            for a, b in pairs:
                ps = prev_signs[(a, b)]
                cs = np.sign(curr_vecs[a][0] * curr_vecs[b][1] - curr_vecs[a][1] * curr_vecs[b][0])
                if ps != 0 and cs != 0:
                    n_valid += 1
                    if ps != cs:
                        n_flipped += 1
            cost[j, i] = (n_flipped / n_valid) if n_valid > 0 else 0.0
    return cost


def _neighbor_cost_matrix(
    track_preds: np.ndarray,          # (n_tracks, 2) KF-predicted centroids
    det_cents: np.ndarray,            # (n_dets, 2) current detection centroids
    track_adj: dict[int, set[int]],   # track-index -> neighbour track-indices
    det_adj: dict[int, set[int]],     # det-index -> neighbour det-indices
    soft: bool = False,
    gate: np.ndarray | None = None,   # (n_dets, n_tracks) bool, pairs that can be matched
) -> np.ndarray:
    """(n_dets, n_tracks) neighbourhood-inconsistency cost matrix.

    cost[j, i] in [0, 1]: how inconsistent is the candidate match track_i ->
    det_j with the surrounding neighbourhood structure? For each neighbour n
    of track i we find the detection nearest n's KF prediction
    (nearest_det[n]).

    Binary variant (soft=False, the paper's Eq. 2): the consistency score is
    the fraction of track i's neighbours n for which nearest_det[n] is also
    a Delaunay neighbour of det j; cost = 1 - consistency. This discards
    graded distance information -- nearest_det[n] just outside j's Delaunay
    adjacency scores identically to one far outside it.

    Soft variant (soft=True): for each neighbour n, cost contribution is
    dist(nearest_det[n], det_j) / local_scale(j), clipped to [0, 1], where
    local_scale(j) is det_j's own median Delaunay-edge length (a
    self-calibrating local spacing scale, falling back to the frame's global
    median edge length for detections with no Delaunay neighbours of their
    own). Averaged over track i's neighbours. Tested as an alternative to
    the binary variant; see paper Discussion for whether it changed the
    hard-case result.

    Tracks with no Delaunay neighbours receive a neutral cost of 0 either way.
    """
    n_t, n_d = len(track_preds), len(det_cents)
    cost = np.zeros((n_d, n_t), dtype=np.float32)
    if n_t == 0 or n_d == 0:
        return cost

    all_d = np.linalg.norm(det_cents[None, :, :] - track_preds[:, None, :], axis=2)
    nearest_det = np.argmin(all_d, axis=1)

    if not soft:
        # Only pairs inside the distance gate can ever be matched, so with a
        # gate given the cost is computed for those pairs only (identical
        # values there; other entries stay 0 and are overwritten as
        # infeasible by the caller). Without it, every pair is computed.
        if gate is None:
            pairs = ((j, i) for i in range(n_t) for j in range(n_d))
        else:
            pairs = zip(*np.nonzero(gate))
        for j, i in pairs:
            nbrs_i = track_adj.get(i, set())
            if not nbrs_i:
                continue
            nbrs_j = det_adj.get(j, set())
            consistent = sum(1 for n in nbrs_i if nearest_det[n] in nbrs_j)
            cost[j, i] = 1.0 - consistent / len(nbrs_i)
        return cost

    # -- soft variant --
    all_edge_lengths = [
        float(np.linalg.norm(det_cents[j] - det_cents[k]))
        for j, nbrs in det_adj.items() for k in nbrs if k > j
    ]
    global_scale = float(np.median(all_edge_lengths)) if all_edge_lengths else 1.0
    local_scale = np.full(n_d, max(global_scale, 1e-6))
    for j, nbrs in det_adj.items():
        if nbrs:
            lengths = [np.linalg.norm(det_cents[j] - det_cents[k]) for k in nbrs]
            local_scale[j] = max(float(np.median(lengths)), 1e-6)

    for i in range(n_t):
        nbrs_i = track_adj.get(i, set())
        if not nbrs_i:
            continue
        for j in range(n_d):
            penalties = [
                min(1.0, float(np.linalg.norm(det_cents[nearest_det[n]] - det_cents[j])) / local_scale[j])
                for n in nbrs_i
            ]
            cost[j, i] = float(np.mean(penalties))
    return cost


# ---------------------------------------------------------------------------
# Main tracker
# ---------------------------------------------------------------------------

def build_tracks_neighbor_kf(
    cell_npy_paths: list,
    vessel_tif_paths: list | None,
    max_dist: float = 75.0,
    max_gap: int = 1,
    chip_mask: np.ndarray | None = None,
    border_px: int | None = 20,
    neighbor_weight: float = 0.35,
    green_npy_paths: list[Path] | None = None,
    velocity_damping: float = 1.0,
    max_edge_factor: float | None = None,
    topology_mode: str = "binary",
    area_weight: float = 0.0,
) -> pd.DataFrame:
    """Track cells with topology-preserving Hungarian + Kalman matching.

    Cost (Eq. 3 in the paper): cost[j,i] = (1-w)*dist_cost[j,i] + w*nbr_cost[j,i],
    w = neighbor_weight, gated to infeasible (never matched) outside max_dist.
    w=0 recovers pure distance-based matching (the paper's baseline condition).

    velocity_damping: how much of each track's estimated velocity is trusted
        in the position prediction (see module docstring above _kf_F).
        Default 1.0 preserves original undamped constant-velocity behaviour;
        CTC ground-truth trajectories favour values near 0.0 at short
        imaging intervals, but this is unvalidated on longer-interval data
        -- do not assume it transfers without checking on your own dataset.

    Args:
        cell_npy_paths: one integer label map per timepoint (0 = background,
            unique positive int per cell), each given as an array or as a
            .npy / .tif path.
        vessel_tif_paths: one binary vessel mask per timepoint (array or
            path), same length as cell_npy_paths, or None. Only used to
            populate the `inside_vessel` output column (False when None).
        max_dist: centroid distance gate (px). Pairs beyond this are never
            matched, regardless of cost.
        max_gap: max consecutive missed frames before a track is dropped.
        chip_mask: optional boolean array; cells whose centroid falls outside
            are excluded from tracking. Pass None to disable (default here;
            used for a dataset-specific region-of-interest filter in the
            original deployment, not part of this paper's core method).
        border_px: if an int, a fresh track ID is only assigned to an
            unmatched detection at t>0 if it lies within border_px of any
            image edge (the earlier border-only rule). None allows new tracks
            anywhere, which is what the paper uses: tumour cells keep
            arriving through the vessels during the first hours, so they
            appear inside the chamber, not at its edge. The default stays 20
            so that the earlier scripts reproduce their committed results.
        neighbor_weight: weight w in [0, 1] for the topology term (Eq. 3).
        green_npy_paths: optional secondary detection source used to recover
            tracks that were expected (Kalman-predicted) but unmatched this
            frame (a dataset-specific low-confidence-detection rescue, not part
            of the paper's core method); pass None to disable.

    Returns:
        DataFrame[track_id, frame_idx, label, centroid_r, centroid_c, area,
                  inside_vessel, recovered_from_green]; `label` is the
                  detection's id in that frame's label map (needed to match
                  tracks to annotated masks).
    """
    rows: list[dict] = []
    next_track_id = 1
    alive: list[_TrackState] = []
    last_cent: dict[int, np.ndarray] = {}

    if vessel_tif_paths is None:
        vessel_tif_paths = [None] * len(cell_npy_paths)

    for frame_idx, (npy_path, vtif_path) in enumerate(zip(cell_npy_paths, vessel_tif_paths)):
        label_map = _load_array(npy_path)
        vessel_mask = None if vtif_path is None else _load_array(vtif_path) > 0
        H, W = label_map.shape[:2]

        props_all = _cell_props(label_map)

        if chip_mask is not None:
            props = {lb: v for lb, v in props_all.items()
                     if _centroid_in_mask(v[0], v[1], chip_mask)}
        else:
            props = props_all

        def _inside(cr: float, cc: float) -> bool:
            return vessel_mask is not None and _centroid_in_mask(cr, cc, vessel_mask)

        for trk in alive:
            trk.kf_predict()

        curr_labels = list(props.keys())
        curr_centroids = (
            np.array([(props[lb][0], props[lb][1]) for lb in curr_labels], dtype=np.float64)
            if curr_labels else np.zeros((0, 2), dtype=np.float64)
        )
        n_snapshot = len(alive)

        unmatched_curr: set[int] = set(range(len(curr_labels)))
        matched_track_ids: set[int] = set()

        # -- Joint matching pass (all alive tracks) --------------------------
        if alive and curr_labels:
            pred_cents = np.array([t.centroid for t in alive], dtype=np.float64)

            dist_mat = np.linalg.norm(curr_centroids[:, None, :] - pred_cents[None, :, :], axis=2)
            dist_cost = np.clip(dist_mat / max_dist, 0.0, 1.0)

            alive_last = np.array(
                [last_cent.get(t.track_id, np.array(t.centroid)) for t in alive],
                dtype=np.float64,
            )
            if len(alive) >= 3:
                trk_adj = _build_delaunay_neighbors(alive_last, max_edge_factor)
            else:
                trk_adj = {i: set() for i in range(len(alive))}

            det_adj = _build_delaunay_neighbors(curr_centroids, max_edge_factor)

            w_n_gate = float(neighbor_weight) > 0.0
            if not w_n_gate:
                # w = 0: the neighbourhood term has no effect on the cost.
                nbr_cost = np.zeros(dist_mat.shape)
            elif topology_mode == "orientation":
                _all_d = np.linalg.norm(curr_centroids[None, :, :] - pred_cents[:, None, :], axis=2)
                _nearest_det = np.argmin(_all_d, axis=1)
                nbr_cost = _orientation_cost_matrix(alive_last, curr_centroids, trk_adj, _nearest_det)
            else:
                nbr_cost = _neighbor_cost_matrix(
                    pred_cents, curr_centroids, trk_adj, det_adj,
                    soft=(topology_mode == "soft"),
                    gate=(dist_mat <= max_dist) if w_n_gate else None,
                )

            w_n = float(neighbor_weight)
            w_d = 1.0 - w_n
            base_cost = w_d * dist_cost + w_n * nbr_cost

            w_a = float(area_weight)
            if w_a > 0.0:
                curr_areas = np.array([props[lb][2] for lb in curr_labels], dtype=np.float64)
                track_areas = np.array([t.last_area for t in alive], dtype=np.float64)
                a_min = np.minimum(curr_areas[:, None], track_areas[None, :])
                a_max = np.maximum(curr_areas[:, None], track_areas[None, :])
                area_cost = 1.0 - a_min / np.maximum(a_max, 1e-6)
                base_cost = (1.0 - w_a) * base_cost + w_a * area_cost

            cost = np.full(dist_mat.shape, fill_value=2.0)
            in_gate = dist_mat <= max_dist
            cost[in_gate] = base_cost[in_gate]

            row_ind, col_ind = linear_sum_assignment(cost)
            for ri, ci in zip(row_ind, col_ind):
                if cost[ri, ci] < 2.0:
                    lb = curr_labels[ri]
                    cr, cc, area = props[lb]
                    alive[ci].kf_update(cr, cc, None, area)
                    last_cent[alive[ci].track_id] = np.array([cr, cc])
                    unmatched_curr.discard(ri)
                    matched_track_ids.add(alive[ci].track_id)
                    rows.append(dict(
                        track_id=alive[ci].track_id, frame_idx=frame_idx, label=lb,
                        centroid_r=cr, centroid_c=cc, area=area,
                        inside_vessel=_inside(cr, cc), recovered_from_green=False,
                    ))

        # -- Optional secondary-source recovery (see docstring) --------------
        if (green_npy_paths is not None and frame_idx < len(green_npy_paths)
                and green_npy_paths[frame_idx] is not None):
            candidates = [t for t in alive[:n_snapshot] if t.track_id not in matched_track_ids]
            if candidates:
                green_label_map = _load_array(green_npy_paths[frame_idx])
                green_props = _cell_props(green_label_map)
                if green_props:
                    green_ids = list(green_props.keys())

                    matched_labels = [curr_labels[ri] for ri in range(len(curr_labels)) if ri not in unmatched_curr]
                    if matched_labels:
                        claimed = np.isin(label_map, matched_labels)
                        green_ids = [g for g in green_ids
                                     if not _centroid_in_mask(green_props[g][0], green_props[g][1], claimed)]

                    if green_ids:
                        green_cents = np.array([(green_props[g][0], green_props[g][1]) for g in green_ids], dtype=np.float64)
                        pred_cents2 = np.array([t.centroid for t in candidates], dtype=np.float64)
                        dist = np.linalg.norm(green_cents[:, None, :] - pred_cents2[None, :, :], axis=2)
                        cost = np.where(dist <= max_dist, dist, 1e6)

                        row_ind, col_ind = linear_sum_assignment(cost)
                        for ri, ci in zip(row_ind, col_ind):
                            if cost[ri, ci] >= 1e6:
                                continue
                            g, trk = green_ids[ri], candidates[ci]
                            cr, cc, area = green_props[g]
                            trk.kf_update(cr, cc, None, area)
                            last_cent[trk.track_id] = np.array([cr, cc])
                            matched_track_ids.add(trk.track_id)
                            rows.append(dict(
                                track_id=trk.track_id, frame_idx=frame_idx, label=-g,
                                centroid_r=cr, centroid_c=cc, area=area,
                                inside_vessel=_inside(cr, cc), recovered_from_green=True,
                            ))

        # -- Spawn new tracks (anywhere if border_px is None) ------------------
        for ri in sorted(unmatched_curr):
            lb = curr_labels[ri]
            cr, cc, area = props[lb]
            at_border = border_px is None or (
                cr < border_px or cr > H - border_px or cc < border_px or cc > W - border_px)
            if frame_idx == 0 or at_border:
                trk = _TrackState(
                    track_id=next_track_id,
                    state=np.array([cr, cc, 0.0, 0.0], dtype=np.float64),
                    cov=_KF_P0.copy(), mask_flat=None, last_area=int(area),
                    velocity_damping=velocity_damping,
                )
                alive.append(trk)
                last_cent[next_track_id] = np.array([cr, cc])
                next_track_id += 1
                rows.append(dict(
                    track_id=trk.track_id, frame_idx=frame_idx, label=lb,
                    centroid_r=cr, centroid_c=cc, area=area,
                    inside_vessel=_inside(cr, cc), recovered_from_green=False,
                ))
            # else: mid-frame appearance at t>0 discarded (see border_px docstring)

        # -- Age pre-existing unmatched tracks; drop at max_gap ---------------
        new_alive = []
        for i in range(n_snapshot):
            trk = alive[i]
            if trk.track_id in matched_track_ids:
                new_alive.append(trk)
            else:
                trk.frames_lost += 1
                if trk.frames_lost <= max_gap:
                    new_alive.append(trk)
                else:
                    last_cent.pop(trk.track_id, None)
        new_alive.extend(alive[n_snapshot:])
        alive = new_alive

    return pd.DataFrame(rows, columns=["track_id", "frame_idx", "label", "centroid_r",
                                        "centroid_c", "area", "inside_vessel",
                                        "recovered_from_green"])
