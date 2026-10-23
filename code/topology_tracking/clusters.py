"""Cell clusters (e.g. growing tumor masses) as tracked entities of the chip event table.

Clusters are the connected components of a per-frame binary mask. Between
consecutive frames, two clusters are linked when they overlap by at least
`min_frac` of the smaller one. The overlap graph gives, per frame pair:
    continuation   one cluster -> one cluster
    merge          several clusters -> one cluster
    split          one cluster -> several clusters
    birth / death  a cluster with no link backward / forward
Under time reversal merge <-> split and birth <-> death, so the chip event
tests (run_event_tests.py) apply unchanged.

Features (one row per sequence): clusters per frame, median cluster area,
growth rate (median log-area change per hour along continuations), merge,
split, birth and death rates per cluster-hour, and the fraction of cluster
boundary pixels that touch a vessel.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix

CLUSTER_MIRROR = {
    "cluster_merges_per_cluster_h": "cluster_splits_per_cluster_h",
    "cluster_splits_per_cluster_h": "cluster_merges_per_cluster_h",
    "cluster_births_per_cluster_h": "cluster_deaths_per_cluster_h",
    "cluster_deaths_per_cluster_h": "cluster_births_per_cluster_h",
}


def overlap_links(lab0: np.ndarray, lab1: np.ndarray, min_frac: float = 0.2):
    """(i, j) pairs of clusters in consecutive frames overlapping by >= min_frac of the smaller one."""
    m = (lab0 > 0) & (lab1 > 0)
    if not m.any():
        return np.zeros((0, 2), int)
    a, b = lab0[m].astype(np.int64), lab1[m].astype(np.int64)
    ov = coo_matrix((np.ones(a.size), (a, b)), shape=(lab0.max() + 1, lab1.max() + 1)).tocsr()
    ov.sum_duplicates()
    i, j = ov.nonzero()
    n = np.asarray(ov[i, j]).ravel()
    s0 = np.bincount(lab0.ravel(), minlength=lab0.max() + 1)
    s1 = np.bincount(lab1.ravel(), minlength=lab1.max() + 1)
    keep = n >= min_frac * np.minimum(s0[i], s1[j])
    return np.stack([i[keep], j[keep]], 1)


def cluster_events(labels: list[np.ndarray], min_frac: float = 0.2) -> pd.DataFrame:
    """One row per cluster event: frame (of the later cluster for births, merges,
    splits; of the earlier one for deaths), type, cluster id(s), sizes."""
    rows = []
    T = len(labels)
    for t in range(T - 1):
        L = overlap_links(labels[t], labels[t + 1], min_frac)
        n0, n1 = labels[t].max(), labels[t + 1].max()
        out_deg = np.bincount(L[:, 0], minlength=n0 + 1) if len(L) else np.zeros(n0 + 1, int)
        in_deg = np.bincount(L[:, 1], minlength=n1 + 1) if len(L) else np.zeros(n1 + 1, int)
        for j in range(1, n1 + 1):
            if in_deg[j] == 0:
                rows.append((t + 1, "birth", j))
            elif in_deg[j] >= 2:
                rows.append((t + 1, "merge", j))
        for i in range(1, n0 + 1):
            if out_deg[i] == 0:
                rows.append((t, "death", i))
            elif out_deg[i] >= 2:
                rows.append((t, "split", i))
    return pd.DataFrame(rows, columns=["frame_idx", "type", "cluster"])


def cluster_features(labels: list[np.ndarray], vessel_masks: list[np.ndarray] | None, frame_h: float,
                     um_per_px: float, min_frac: float = 0.2) -> dict:
    T = len(labels)
    n_per = np.array([lab.max() for lab in labels], float)
    cluster_h = n_per.sum() * frame_h
    ev = cluster_events(labels, min_frac)
    cnt = lambda k: int((ev.type == k).sum())
    areas, growth, contact = [], [], []
    for t in range(T):
        lab = labels[t]
        ids = np.arange(1, lab.max() + 1)
        if ids.size:
            areas.append(np.bincount(lab.ravel())[1:] * um_per_px ** 2)
            if vessel_masks is not None:
                edge = (lab > 0) & ~ndi.binary_erosion(lab > 0)
                near = ndi.binary_dilation(vessel_masks[t] > 0, iterations=1)
                contact.append(float((edge & near).sum() / max(edge.sum(), 1)))
        if t + 1 < T:
            L = overlap_links(labels[t], labels[t + 1], min_frac)
            if len(L):
                out_deg = np.bincount(L[:, 0], minlength=lab.max() + 1)
                in_deg = np.bincount(L[:, 1], minlength=labels[t + 1].max() + 1)
                one = (out_deg[L[:, 0]] == 1) & (in_deg[L[:, 1]] == 1)
                s0 = np.bincount(lab.ravel())[L[one, 0]]
                s1 = np.bincount(labels[t + 1].ravel())[L[one, 1]]
                growth.append(np.log(s1 / s0) / frame_h)
    g = np.concatenate(growth) if growth else np.array([np.nan])
    a = np.concatenate(areas) if areas else np.array([np.nan])
    return dict(
        clusters_per_frame=float(n_per.mean()),
        cluster_median_area_um2=float(np.median(a)),
        cluster_growth_per_h=float(np.median(g)),
        cluster_merges_per_cluster_h=cnt("merge") / cluster_h,
        cluster_splits_per_cluster_h=cnt("split") / cluster_h,
        cluster_births_per_cluster_h=cnt("birth") / cluster_h,
        cluster_deaths_per_cluster_h=cnt("death") / cluster_h,
        cluster_vessel_contact=float(np.mean(contact)) if contact else np.nan,
    )
