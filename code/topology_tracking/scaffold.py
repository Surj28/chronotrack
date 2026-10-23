"""Vascular scaffold geometry used by the vessel-anchored tracker.

A scaffold is the temporally consistent vessel network of one sequence (built
by scaffold_from_stack): a union mask U, a pruned centreline
(skeleton, labelled by segment), and optionally the deformation fields that map
every frame onto the scaffold's reference frame. This module turns it into
the quantities the tracker evaluates at any point:

  on_vessel(p)        p inside U
  tangent(p)          unit vector along the local vessel axis (from the
                      nearest centreline pixel; local PCA of the centreline)
  gap_length(a, b)    length of the straight path a -> b that lies outside U
  to_reference(p, t)  frame-t position -> reference-frame position
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

NB8 = np.ones((3, 3), int)
RING = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]


# ---------------------------------------------------------------- skeleton graph

def crossing_number(skel):
    s = np.pad(skel.astype(np.int8), 1)
    H, W = skel.shape
    ring = [s[1 + dy:1 + dy + H, 1 + dx:1 + dx + W] for dy, dx in RING]
    cn = sum(np.abs(ring[i] - ring[(i + 1) % 8]) for i in range(8)) // 2
    return cn * skel


def skeleton_parts(skel):
    """Branches (labelled), junction clusters (labelled, incl. 1 px rim) and endpoints."""
    nb = ndi.convolve(skel.astype(np.uint8), NB8, mode="constant") - 1
    endpoints = skel & (nb == 1)
    junction = skel & (crossing_number(skel) >= 3)
    jrim = ndi.binary_dilation(junction, NB8) & skel
    blab, nbr = ndi.label(skel & ~jrim, NB8)
    jlab, njc = ndi.label(jrim, NB8)
    # branch <-> junction-cluster adjacency (8-neighbourhood)
    H, W = skel.shape
    bp = np.pad(blab, 1)
    pairs = []
    for dy, dx in RING:
        nbl = bp[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
        m = (jlab > 0) & (nbl > 0)
        pairs.append(np.stack([nbl[m], jlab[m]], 1))
    pairs = np.unique(np.concatenate(pairs), axis=0) if pairs else np.zeros((0, 2), int)
    return blab, nbr, jlab, njc, endpoints, pairs


def branch_lengths(blab, n):
    """Geometric length per branch: orthogonal links + sqrt(2) x diagonal links (+1 px)."""
    L = np.zeros(n + 1)
    for (dy, dx), wgt in (((0, 1), 1.0), ((1, 0), 1.0), ((1, 1), np.sqrt(2)), ((1, -1), np.sqrt(2))):
        a = blab[max(0, -dy):blab.shape[0] - max(0, dy), max(0, -dx):blab.shape[1] - max(0, dx)]
        b = blab[max(0, dy):, max(0, dx):][:a.shape[0], :a.shape[1]] if dx >= 0 else \
            blab[max(0, dy):, :dx][:a.shape[0], :a.shape[1]]
        same = (a > 0) & (a == b)
        L += np.bincount(a[same], minlength=n + 1) * wgt
    return L + (np.bincount(blab.ravel(), minlength=n + 1) > 0)


def prune_short(skel, k=3):
    """Classic morphological pruning: drop spurs of <= k px, then regrow true branch tips."""
    x = skel.copy()
    for _ in range(k):
        nb = ndi.convolve(x.astype(np.uint8), NB8, mode="constant") - 1
        x &= ~(nb <= 1)
    nb = ndi.convolve(x.astype(np.uint8), NB8, mode="constant") - 1
    grow = x & (nb == 1)
    for _ in range(k):
        grow = ndi.binary_dilation(grow, NB8) & skel
    return x | grow


def prune(skel, radius, passes=3):
    """Remove terminal branches shorter than the local vessel diameter, and small islands."""
    skel = prune_short(skel)
    for _ in range(passes):
        blab, n, jlab, nj, endpoints, pairs = skeleton_parts(skel)
        if n == 0:
            break
        L = branch_lengths(blab, n)
        has_end = np.bincount(blab[endpoints], minlength=n + 1) > 0
        n_junc = np.bincount(pairs[:, 0], minlength=n + 1) if len(pairs) else np.zeros(n + 1, int)
        r_junc = ndi.maximum(radius, jlab, np.arange(1, nj + 1)) if nj else np.zeros(0)
        r_at = np.zeros(n + 1)
        for b, j in pairs:
            r_at[b] = max(r_at[b], r_junc[j - 1])
        r_branch = ndi.maximum(radius, blab, np.arange(1, n + 1))
        spur = has_end & (n_junc == 1) & (L < 2 * r_at)
        island = np.zeros(n + 1, bool)
        island[1:] = (n_junc[1:] == 0) & (L[1:] < 2 * np.asarray(r_branch) + 1)
        drop = np.flatnonzero(spur | island)
        if drop.size == 0:
            break
        skel = skel & ~np.isin(blab, drop)
        # junction clusters left without branches become loose pixels: clean by re-thinning
        skel = skeletonize(skel)
        lab, k = ndi.label(skel, NB8)
        sizes = np.bincount(lab.ravel())
        skel &= ~np.isin(lab, np.flatnonzero(sizes < 3))
    return skel


def build_scaffold(V, roi, smooth_sigma=2.0, window=2):
    """Union of temporal-consensus masks -> smoothed mask, pruned centreline, radius map."""
    T = len(V)
    union = np.zeros(V.shape[1:], bool)
    for t in range(T):
        lo, hi = max(0, t - window), min(T, t + window + 1)
        union |= V[lo:hi].sum(0) * 2 > (hi - lo)
    U = (ndi.gaussian_filter(union.astype(np.float32), smooth_sigma) > 0.5) & roi
    radius = ndi.distance_transform_edt(U)
    skel = prune(skeletonize(U), radius)
    return U, skel, radius


def centreline_agreement(a, b, tol=3.0):
    """Fraction of a's centreline within tol px of b's, and vice versa."""
    da = ndi.distance_transform_edt(~b)
    db = ndi.distance_transform_edt(~a)
    return float((da[a] <= tol).mean()), float((db[b] <= tol).mean())


# ---------------------------------------------------------------- tangent field

def tangent_field(skeleton: np.ndarray, radius: int = 6) -> np.ndarray:
    """(2, H, W) unit tangent at every centreline pixel (zeros elsewhere).

    Local PCA of the centreline pixels within `radius`: the principal axis of
    the offsets (dr, dc) of neighbouring centreline pixels. Computed with
    correlations of the centreline indicator against moment kernels.
    """
    S = skeleton.astype(np.float64)
    r = np.arange(-radius, radius + 1)
    DR, DC = np.meshgrid(r, r, indexing="ij")
    K = (DR ** 2 + DC ** 2 <= radius ** 2).astype(np.float64)
    corr = lambda k: ndi.correlate(S, k, mode="constant")
    n = corr(K)
    n_safe = np.maximum(n, 1.0)
    mr, mc = corr(K * DR) / n_safe, corr(K * DC) / n_safe
    srr = corr(K * DR * DR) / n_safe - mr * mr
    scc = corr(K * DC * DC) / n_safe - mc * mc
    src = corr(K * DR * DC) / n_safe - mr * mc
    # principal eigenvector of [[srr, src], [src, scc]]
    theta = 0.5 * np.arctan2(2 * src, srr - scc)          # angle from the row axis
    t = np.stack([np.cos(theta), np.sin(theta)])
    return t * skeleton[None]


@dataclass
class Scaffold:
    union: np.ndarray                  # bool (H, W)
    skeleton: np.ndarray               # bool (H, W)
    tangent: np.ndarray                # (2, H, W) unit tangent of the nearest centreline pixel
    flows: np.ndarray | None = None    # (T, 2, h, w) float16, frame t -> reference (downsampled)
    flow_ds: int = 1

    @classmethod
    def from_masks(cls, union: np.ndarray, skeleton: np.ndarray, flows=None, flow_ds: int = 1,
                   radius: int = 6) -> "Scaffold":
        t_sk = tangent_field(skeleton, radius)
        _, (ir, ic) = ndi.distance_transform_edt(~skeleton, return_indices=True)
        tan = t_sk[:, ir, ic]
        return cls(union=union.astype(bool), skeleton=skeleton.astype(bool), tangent=tan,
                   flows=flows, flow_ds=flow_ds)

    @classmethod
    def load(cls, scaffold_dir: Path, radius: int = 6) -> "Scaffold":
        """Load a saved scaffold: union_mask.tif, segment_ids.tif and, if registered, flows_to_T1.npz."""
        d = Path(scaffold_dir)
        U = tifffile.imread(d / "union_mask.tif") > 0
        sk = tifffile.imread(d / "segment_ids.tif") > 0
        flows, ds = None, 1
        if (d / "flows_to_T1.npz").exists():
            z = np.load(d / "flows_to_T1.npz")
            flows, ds = z["flows"], int(z["ds"])
        return cls.from_masks(U, sk, flows, ds, radius)

    # -------------------------------------------------------------- queries
    def _idx(self, p: np.ndarray):
        r = np.clip(np.round(p[:, 0]).astype(int), 0, self.union.shape[0] - 1)
        c = np.clip(np.round(p[:, 1]).astype(int), 0, self.union.shape[1] - 1)
        return r, c

    def on_vessel(self, p: np.ndarray) -> np.ndarray:
        r, c = self._idx(p)
        return self.union[r, c]

    def tangent_at(self, p: np.ndarray) -> np.ndarray:
        r, c = self._idx(p)
        return self.tangent[:, r, c].T                    # (n, 2)

    def to_reference(self, p: np.ndarray, t: int) -> np.ndarray:
        """Frame-t positions -> reference-frame positions (first-order inverse
        of the registration flow; identity when the sequence was not registered)."""
        if self.flows is None or t == 0:
            return p
        f = self.flows[t].astype(np.float32)
        ds = self.flow_ds
        rr = np.clip(p[:, 0] / ds, 0, f.shape[1] - 1)
        cc = np.clip(p[:, 1] / ds, 0, f.shape[2] - 1)
        dv = ndi.map_coordinates(f[0], [rr, cc], order=1) * ds
        du = ndi.map_coordinates(f[1], [rr, cc], order=1) * ds
        return np.stack([p[:, 0] - dv, p[:, 1] - du], 1)

    def gap_length(self, a: np.ndarray, b: np.ndarray, step: float = 1.0) -> np.ndarray:
        """Length (px) of each straight path a[k] -> b[k] lying outside the union mask."""
        L = np.hypot(*(b - a).T)
        n = np.maximum(np.ceil(L / step).astype(int), 1)
        out = np.zeros(len(a))
        H, W = self.union.shape
        for k in range(len(a)):
            s = np.linspace(0.0, 1.0, n[k] + 1)
            rr = np.clip(np.round(a[k, 0] + s * (b[k, 0] - a[k, 0])).astype(int), 0, H - 1)
            cc = np.clip(np.round(a[k, 1] + s * (b[k, 1] - a[k, 1])).astype(int), 0, W - 1)
            out[k] = (~self.union[rr, cc]).mean() * L[k]
        return out


# ---------------------------------------------------------------- generic builder

def _down(a: np.ndarray, ds: int) -> np.ndarray:
    if ds == 1:
        return a
    h, w = (a.shape[0] // ds) * ds, (a.shape[1] // ds) * ds
    return a[:h, :w].reshape(h // ds, ds, w // ds, ds).mean((1, 3))


def _full_flow(fl_ds: np.ndarray, ds: int, shape) -> np.ndarray:
    f = np.stack([ndi.zoom(fl_ds[0], ds, order=1), ndi.zoom(fl_ds[1], ds, order=1)]) * ds if ds > 1 else fl_ds
    pad = ((0, 0), (0, max(shape[0] - f.shape[1], 0)), (0, max(shape[1] - f.shape[2], 0)))
    return np.pad(f, pad, mode="edge")[:, :shape[0], :shape[1]]


def _warp(img: np.ndarray, flow: np.ndarray, order: int = 0) -> np.ndarray:
    rr, cc = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]), indexing="ij")
    return ndi.map_coordinates(img.astype(np.float32), [rr + flow[0], cc + flow[1]], order=order, mode="constant")


def scaffold_from_stack(V: np.ndarray, vessel_img: np.ndarray, roi: np.ndarray | None = None, ds: int = 4,
                        smooth_sigma: float = 2.0, window: int = 2, radius: int = 6):
    """Scaffold of one sequence from its per-frame vessel masks V (T, H, W) and vessel channel (T, H, W).

    Frames are registered to the first one (TV-L1 optical flow on the vessel
    channel, downsampled by `ds`) only if the cumulative deformation first ->
    last frame (95th percentile on the first frame's centreline, drift
    removed) exceeds half the median vessel radius; then temporal consensus,
    union, pruned centreline.
    Returns (Scaffold, info dict).
    """
    from skimage.registration import optical_flow_tvl1

    T, H, W = V.shape
    roi = np.ones((H, W), bool) if roi is None else roi.astype(bool)
    V = V.astype(bool) & roi

    def prep(x):
        lo, hi = np.percentile(x[roi], [1, 99.8])
        return ndi.gaussian_filter(_down(np.clip((x - lo) / (hi - lo + 1e-6), 0, 1) * roi, ds), 1)

    sk0 = skeletonize(V[0])
    r_med = float(np.median(ndi.distance_transform_edt(V[0])[sk0])) if sk0.any() else 0.0
    ref = prep(vessel_img[0].astype(np.float32))
    v, u = optical_flow_tvl1(ref, prep(vessel_img[-1].astype(np.float32)))
    f = _full_flow(np.stack([v, u]), ds, (H, W))
    drift = np.array([np.median(f[0][V[0]]), np.median(f[1][V[0]])]) if V[0].any() else np.zeros(2)
    cum = np.hypot(f[0][sk0] - drift[0], f[1][sk0] - drift[1]) if sk0.any() else np.zeros(1)
    cum_p95 = float(np.percentile(cum, 95))
    registered = bool(cum_p95 > 0.5 * r_med)
    flows = None
    if registered:
        flows = np.zeros((T, 2) + ref.shape, np.float16)
        V = V.copy()
        for i in range(1, T):
            v, u = optical_flow_tvl1(ref, prep(vessel_img[i].astype(np.float32)))
            flows[i] = np.stack([v, u])
            V[i] = (_warp(V[i], _full_flow(flows[i].astype(np.float32), ds, (H, W))) > 0.5) & roi
    U, skel, _ = build_scaffold(V, roi, smooth_sigma, window)
    sc = Scaffold.from_masks(U, skel, flows, ds, radius)
    info = dict(registered=registered, cumulative_deformation_p95_px=cum_p95,
                median_vessel_radius_px=r_med, union_cover=float(U[roi].mean()),
                centreline_px=int(skel.sum()))
    return sc, info
