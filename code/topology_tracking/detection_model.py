"""Per-sequence detection model: how often, and for how long, a cell that is
there goes undetected. Estimated from isolated cells that stay in place, so it
needs neither tracks nor annotation; every tracker of a sequence uses the same
model (like the motion gate).

Base population: isolated detections (no other detection within 3 cell radii
r) found at the same spot (within r) in the previous frame. Those not found at
their spot in the next frame are followed: a return after exactly m missed
frames is a detection within r of the spot at t + m + 1. Returns are compared
with chance (local density of other detections x pi r^2, summed over the cells
still followed):

  max_gap      longest run of missed frames with more returns than chance at
               z > 3 (excess over the Poisson SD of the chance count),
               contiguous from m = 1. Cells that moved away also leave their
               spot and raise the chance count, so in a motile population
               only clear excesses count.
  gap_cost[m]  -log(pi_m / pi_0): pi_m = excess returns after m missed frames
               per cell of the base population, pi_0 = fraction found at their
               spot in the next frame. The cost of bridging m missed frames,
               relative to a link between consecutive frames.
  p_miss       per-frame miss probability (reported): among isolated cells at
               the same spot at t-1 and t+1, the fraction not found at t.

With too few followed cells to tell (< 30), max_gap = 1 at cost -log p_miss.
Both directions of time are pooled, so a movie and its reverse get the same model.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def detection_model(frames, r_px: float | None = None, max_m: int = 8, iso: float = 3.0, outer: float = 10.0,
                    z_min: float = 3.0, min_at_risk: int = 30) -> dict:
    """frames: vessel_tracker.FrameDets per frame (positions in .pos)."""
    T = len(frames)
    if r_px is None:
        areas = np.concatenate([f.area for f in frames if len(f.area)])
        r_px = float(np.sqrt(np.median(areas) / np.pi))
    r = r_px
    trees = [cKDTree(f.pos) if len(f.label) else None for f in frames]

    def count(t, p, rad):
        if trees[t] is None or not len(p):
            return np.zeros(len(p), int)
        return np.asarray(trees[t].query_ball_point(p, rad, return_length=True))

    # p_miss
    n_pair = n_miss = 0
    for t in range(1, T - 1):
        if trees[t - 1] is None or trees[t + 1] is None:
            continue
        A, B = frames[t - 1], frames[t + 1]
        dd, jj = trees[t + 1].query(A.pos, k=1, distance_upper_bound=r)
        ok = np.isfinite(dd) & (count(t - 1, A.pos, iso * r) == 1)
        if not ok.any():
            continue
        i, j = np.flatnonzero(ok), jj[ok]
        keep = count(t + 1, B.pos[j], iso * r) == 1
        mid = (A.pos[i[keep]] + B.pos[j[keep]]) / 2
        n_pair += int(keep.sum())
        n_miss += int((count(t, mid, r) == 0).sum())
    p_miss = n_miss / n_pair if n_pair else np.nan

    # runs of missed frames
    n_base = n_next = 0
    at_risk, back, chance = np.zeros(max_m + 1), np.zeros(max_m + 1), np.zeros(max_m + 1)
    ring = np.pi * ((outer * r) ** 2 - (iso * r) ** 2)
    for d in (1, -1):
        for t in range(T):
            h, n = t - d, t + d
            if not (0 <= h < T and 0 <= n < T) or trees[t] is None:
                continue
            P = frames[t].pos
            base = (count(t, P, iso * r) == 1) & (count(h, P, r) >= 1)
            found = count(n, P[base], r) >= 1
            n_base += int(base.sum())
            n_next += int(found.sum())
            p = P[base][~found]
            for m in range(1, max_m + 1):
                u = t + d * (m + 1)
                if not (0 <= u < T) or not len(p):
                    break
                at_risk[m] += len(p)
                hit = count(u, p, r) >= 1
                dens = (count(u, p, outer * r) - count(u, p, iso * r)) / ring
                chance[m] += float((1 - np.exp(-dens * np.pi * r ** 2)).sum())
                back[m] += int(hit.sum())
                p = p[~hit]
    excess = back - chance
    z = excess / np.sqrt(np.maximum(chance, 1.0))
    if at_risk[1] < min_at_risk or not n_base:
        max_gap = 1
        gap_cost = [float(-np.log(p_miss)) if np.isfinite(p_miss) and p_miss > 0 else float(np.log(T))]
    else:
        max_gap = 0
        for m in range(1, max_m + 1):
            if at_risk[m] >= min_at_risk and z[m] > z_min:
                max_gap = m
            else:
                break
        pi0 = n_next / n_base
        gap_cost = [float(-np.log((excess[m] / n_base) / pi0)) for m in range(1, max_gap + 1)]
    frac = lambda a: np.divide(a, at_risk, out=np.full(max_m + 1, np.nan), where=at_risk > 0)[1:].round(4).tolist()
    return dict(p_miss=float(p_miss), max_gap=int(max_gap), gap_cost=gap_cost, r_px=r, n_pairs=n_pair,
                n_base=n_base, hazard=frac(back), chance=frac(chance), z=z[1:].round(1).tolist(),
                at_risk=at_risk[1:].astype(int).tolist())
