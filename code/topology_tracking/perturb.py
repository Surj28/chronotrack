"""Controlled error injection.

Track-level operators (paper Sec. II-D, feature reliability) act on a
finished track table and keep every other column (d_vessel_um, state, ...)
untouched, so features can be recomputed without re-reading images:
    drop_detections   remove detections at rate p; gaps longer than the
                      tracker's max_gap split the track (fragmentation)
    swap_identities   with probability p per track and frame, exchange the
                      continuations of two Delaunay-neighbouring tracks

Label-map operators (paper Sec. III-E, tracker robustness) act on the
detections before tracking:
    drop_labels, add_false_positives, merge_neighbours

Mask operators (both): erode_dilate_labels, erode_dilate_mask (signed
radius in px; negative erodes).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.segmentation import expand_labels, find_boundaries

from .tracker import _build_delaunay_neighbors, _cell_props


# ---------------------------------------------------------------------------
# Track-level
# ---------------------------------------------------------------------------

def drop_detections(tracks: pd.DataFrame, p: float, max_gap: int,
                    rng: np.random.Generator) -> pd.DataFrame:
    """Remove each detection with probability p. Where a track then has a gap
    of more than max_gap frames, the part after the gap gets a new track id,
    as the tracker would have terminated it there. The first part keeps the
    original id."""
    out = tracks[rng.random(len(tracks)) >= p].sort_values(["track_id", "frame_idx"]).copy()
    if out.empty:
        return out
    too_long = out.groupby("track_id")["frame_idx"].diff() > max_gap + 1
    seg = too_long.groupby(out["track_id"]).cumsum().to_numpy()
    frag = seg > 0
    if frag.any():
        keys = pd.MultiIndex.from_arrays([out["track_id"].to_numpy()[frag], seg[frag]])
        codes, _ = pd.factorize(keys)
        tid = out["track_id"].to_numpy().copy()
        tid[frag] = int(tracks["track_id"].max()) + 1 + codes
        out["track_id"] = tid
    return out


def swap_identities(tracks: pd.DataFrame, p: float, rng: np.random.Generator) -> pd.DataFrame:
    """For each frame t, every track present at t and t+1 is, with
    probability p, paired with a random Delaunay neighbour (among such
    tracks) not yet paired in this frame; the two exchange their ids from
    t+1 onwards. Positions and per-detection columns are unchanged; only the
    linking is corrupted, as in a real identity switch."""
    out = tracks.sort_values("frame_idx").copy()
    frames = np.sort(out["frame_idx"].unique())
    by_t = {t: g for t, g in out.groupby("frame_idx")}
    owner = {}  # original track id -> id written from now on
    new_ids = np.empty(len(out), dtype=np.int64)
    pos = {t: idx for t, idx in out.reset_index(drop=True).groupby("frame_idx").indices.items()}
    orig = out["track_id"].to_numpy()
    for fi, t in enumerate(frames):
        idx = pos[t]
        new_ids[idx] = [owner.get(o, o) for o in orig[idx]]
        if fi + 1 == len(frames) or p <= 0:
            continue
        nxt = set(by_t[frames[fi + 1]]["track_id"])
        g = by_t[t][by_t[t]["track_id"].isin(nxt)]
        if len(g) < 3:
            continue
        ids = g["track_id"].to_numpy()
        adj = _build_delaunay_neighbors(g[["centroid_r", "centroid_c"]].to_numpy())
        used = np.zeros(len(ids), bool)
        for i in rng.permutation(len(ids)):
            if used[i] or rng.random() >= p:
                continue
            cands = [j for j in adj[i] if not used[j]]
            if not cands:
                continue
            j = cands[rng.integers(len(cands))]
            used[i] = used[j] = True
            a, b = ids[i], ids[j]
            owner[a], owner[b] = owner.get(b, b), owner.get(a, a)
    out = out.reset_index(drop=True)
    out["track_id"] = new_ids
    return out


# ---------------------------------------------------------------------------
# Mask operators
# ---------------------------------------------------------------------------

def erode_dilate_labels(labels: np.ndarray, r: int) -> np.ndarray:
    """Grow (r > 0) or shrink (r < 0) every label by |r| px without letting
    labels overlap. Labels that vanish are removed."""
    if r == 0:
        return labels
    if r > 0:
        return expand_labels(labels, distance=r)
    inner = (labels > 0) & ~find_boundaries(labels, mode="inner")
    keep = ndi.distance_transform_edt(inner) >= -r
    return np.where(keep, labels, 0)


def erode_dilate_mask(mask: np.ndarray, r: int) -> np.ndarray:
    if r == 0:
        return mask
    d = ndi.distance_transform_edt(~mask if r > 0 else mask)
    return (mask | (d <= r)) if r > 0 else (d > -r)


def remeasure_tracks(tracks: pd.DataFrame, label_maps: list) -> pd.DataFrame:
    """Recompute centroid and area of each track row from (perturbed) label
    maps, matching rows by their `label`. Rows whose label vanished are
    dropped."""
    parts = []
    for t, g in tracks.groupby("frame_idx"):
        props = _cell_props(label_maps[int(t)])
        g = g[g["label"].isin(props.keys())].copy()
        if g.empty:
            continue
        vals = np.array([props[lb] for lb in g["label"]])
        g["centroid_r"], g["centroid_c"], g["area"] = vals[:, 0], vals[:, 1], vals[:, 2]
        parts.append(g)
    return pd.concat(parts, ignore_index=True) if parts else tracks.iloc[:0]


# ---------------------------------------------------------------------------
# Label-map operators (before tracking)
# ---------------------------------------------------------------------------

def drop_labels(labels: np.ndarray, p: float, rng: np.random.Generator) -> np.ndarray:
    ids = np.unique(labels)
    ids = ids[ids > 0]
    gone = ids[rng.random(len(ids)) < p]
    return np.where(np.isin(labels, gone), 0, labels)


def add_false_positives(labels: np.ndarray, p: float, rng: np.random.Generator,
                        allowed: np.ndarray | None = None) -> np.ndarray:
    """Add round(p * n_cells) discs of the median cell radius at random free
    positions (inside `allowed` if given)."""
    ids = np.unique(labels)
    ids = ids[ids > 0]
    n_add = int(round(p * len(ids)))
    if n_add == 0:
        return labels
    radius = max(2, int(round(np.sqrt(np.median(ndi.sum(labels > 0, labels, ids)) / np.pi))))
    free = labels == 0
    if allowed is not None:
        free &= allowed
    free = ndi.binary_erosion(free, iterations=radius)
    cand = np.flatnonzero(free)
    if cand.size == 0:
        return labels
    out = labels.copy()
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    disc = yy ** 2 + xx ** 2 <= radius ** 2
    next_id = int(ids.max()) + 1 if ids.size else 1
    for c in rng.choice(cand, size=min(n_add, cand.size), replace=False):
        r0, c0 = divmod(int(c), labels.shape[1])
        sl = (slice(r0 - radius, r0 + radius + 1), slice(c0 - radius, c0 + radius + 1))
        win = out[sl]
        if win.shape != disc.shape:
            continue
        win[disc & (win == 0)] = next_id
        next_id += 1
    return out


def merge_neighbours(labels: np.ndarray, p: float, rng: np.random.Generator) -> np.ndarray:
    """Under-segmentation: each cell, with probability p, absorbs one of its
    Delaunay neighbours (the merged object's centroid falls between them)."""
    props = _cell_props(labels)
    if len(props) < 3 or p <= 0:
        return labels
    ids = np.array(list(props))
    adj = _build_delaunay_neighbors(np.array([props[i][:2] for i in ids]))
    used = np.zeros(len(ids), bool)
    remap = {}
    for i in rng.permutation(len(ids)):
        if used[i] or rng.random() >= p:
            continue
        cands = [j for j in adj[i] if not used[j]]
        if not cands:
            continue
        j = cands[rng.integers(len(cands))]
        used[i] = used[j] = True
        remap[ids[j]] = ids[i]
    if not remap:
        return labels
    lut = np.arange(int(labels.max()) + 1)
    for src, dst in remap.items():
        lut[src] = dst
    return lut[labels]
