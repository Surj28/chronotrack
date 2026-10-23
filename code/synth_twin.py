"""Synthetic twin of a ARTORG well, with ground-truth identities, for a fair
comparison of the trackers (paper Sec. III). Everything that is not the
ground truth itself comes from the real well:

  stage      the well's region of interest, its hourly vessel masks, and its
             number of detections per frame (scaled by the density factor rho)
  start      cells at the real detections of the first frame; with rho > 1,
             extra cells at real detection positions of later frames
  motion     every step is resampled from the consensus-link library
             (synth_library.py: links on which all five trackers agree, all
             four wells): among library links that start in the same vessel
             state, one of the K = 20 nearest in previous step (persistence)
             is drawn at random; a draw whose vessel transition (on/off ->
             on/off) disagrees with the real vessel mask at the new position
             is redrawn (up to 10 times), as is a draw that brings the cell
             closer to another cell than real cells come (the 5th percentile
             of real nearest-neighbor distances). No tracker's motion model is
             used. Consensus links miss the hardest, longest steps, so all
             steps are multiplied by one factor lambda, set by bisection so
             that the twin's 95th-percentile pilot step (common.symmetric_gate,
             the tracker-independent statistic behind the gate and the
             ambiguity index) matches the real well's.
  births     cells appear where real cells start consensus tracks in that
             frame (any real detection if there are too few), so that the
             number of cells follows rho x the real count
  deaths     cells that leave the region; and each cell ends with the real
             per-frame probability that a detection ends its track in every
             tracker (consensus track end) away from the region's edge
  rendering  each cell carries the masks and green intensities of a real
             consensus chain (its donor), frame by frame (back and forth when
             the cell outlives the chain); pixels already taken by another
             cell stay with it, and a cell left with < 30 % of its mask merges
             into its neighbor (one detection, as when touching cells merge)
  misses     each detection is dropped with the well's measured per-frame miss
             probability (topology_tracking.detection_model, real frames)

Outputs <results>/synthetic/<well>_x<rho>_l<lambda: 1 or cal>_s<seed>/: cells.tif (label maps),
green.tif, gt.csv (gt_id, frame, label: detection carrying the cell, -1 if
missed or merged; r, c, on_vessel; merged_into: the label it merged into, else 0),
info.json.

Usage (from the repo root):
    python paper/code/synth_twin.py --well ARTORG_w7 --rho 1 --seed 0
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from common import CROPS, RESULTS, SEG
from run_vessel_tracker import load_frames
from topology_tracking.detection_model import detection_model

OUT = RESULTS / "synthetic"
K_NN = 20
REDRAWS = 10
MERGE_FRAC = 0.30


def consensus(well, trackers=("sym", "sym_vessel", "first_order", "w0", "nn")):
    """Consensus links and consensus track starts / ends (frame, label) of a well."""
    L, starts, ends = {}, None, None
    for n in trackers:
        tr = pd.read_csv(OUT / f"tracks_{well}_{n}.csv").sort_values(["track_id", "frame_idx"])
        same = (tr.track_id.values[1:] == tr.track_id.values[:-1])
        nxt = dict(zip(zip(tr.frame_idx.values[:-1][same], tr.label.values[:-1][same].astype(int)),
                       zip(tr.frame_idx.values[1:][same], tr.label.values[1:][same].astype(int))))
        L[n] = {k: v for k, v in nxt.items() if v[0] == k[0] + 1}
        first = tr.groupby("track_id").head(1); last = tr.groupby("track_id").tail(1)
        s = set(zip(first.frame_idx, first.label.astype(int))); e = set(zip(last.frame_idx, last.label.astype(int)))
        starts = s if starts is None else starts & s
        ends = e if ends is None else ends & e
    cons = {k: v for k, v in L[trackers[0]].items() if all(L[n].get(k) == v for n in trackers)}
    return cons, starts, ends


def chains(cons):
    """Maximal chains of consensus links: lists of (frame, label)."""
    succ = dict(cons)
    has_pred = set(cons.values())
    out = []
    for k in cons:
        if k in has_pred:
            continue
        c = [k]
        while c[-1] in succ:
            c.append(succ[c[-1]])
        out.append(c)
    return out


def sprites(well, ts, chain_list, min_len=3, max_chains=4000, rng=None):
    """Mask and green patches of donor chains (patch offsets relative to the centroid)."""
    chain_list = [c for c in chain_list if len(c) >= min_len]
    if len(chain_list) > max_chains:
        chain_list = [chain_list[i] for i in rng.choice(len(chain_list), max_chains, replace=False)]
    need = {}
    for ci, c in enumerate(chain_list):
        for j, (t, lab) in enumerate(c):
            need.setdefault(t, []).append((ci, j, lab))
    out = [[None] * len(c) for c in chain_list]
    for t, items in need.items():
        L = tifffile.imread(SEG / well / f"cells_T{ts[t]:04d}.tif")
        G = tifffile.imread(CROPS / well / f"T{ts[t]:04d}_C2.tif").astype(np.float32)
        bg = float(np.median(G[L == 0]))
        objs = ndi.find_objects(L)
        for ci, j, lab in items:
            sl = objs[lab - 1]
            m = L[sl] == lab
            rr, cc = np.nonzero(m)
            out[ci][j] = (m, np.where(m, np.clip(G[sl] - bg, 0, None), 0).astype(np.float32),
                          float(rr.mean()), float(cc.mean()))
    return out


def simulate(well, rho, seed, lam=1.0):
    rng = np.random.default_rng(seed)
    ts, labels, frames = load_frames(well)
    T = len(ts)
    roi = (tifffile.imread(CROPS / well / "roi_mask.tif") > 0) & ~(tifffile.imread(CROPS / well / "pillar_mask.tif") > 0)
    H, W = roi.shape
    edge = ndi.distance_transform_edt(roi)
    V = [tifffile.imread(SEG / well / f"vessels_T{t:04d}.tif") > 0 for t in ts]
    lib = pd.read_csv(OUT / "library.csv")
    p_miss = detection_model(frames)["p_miss"]
    cons, cstarts, cends = consensus(well)
    n_real = np.array([len(f.label) for f in frames])
    r_px = float(np.sqrt(np.median(np.concatenate([f.area for f in frames])) / np.pi))
    # death probability: consensus track ends away from the edge and the last frame, per detection
    interior = lambda t, p: edge[int(np.clip(round(p[0]), 0, H - 1)), int(np.clip(round(p[1]), 0, W - 1))] > 2 * r_px
    idx = [dict(zip(f.label.astype(int), range(len(f.label)))) for f in frames]
    n_end = sum(1 for (t, lab) in cends if t < T - 1 and interior(t, frames[t].pos_img[idx[t][lab]]))
    death_p = n_end / max(sum(n_real[:-1]), 1)
    don = sprites(well, ts, chains(cons), rng=rng)
    d_excl = float(np.percentile(np.concatenate([cKDTree(f.pos_img).query(f.pos_img, k=2)[0][:, 1] for f in frames]), 5))
    grid = {}                                        # spatial hash of current cell positions (cell size d_excl)
    key = lambda p: (int(p[0] // d_excl), int(p[1] // d_excl))

    def crowded(q, me):
        kr, kc = key(q)
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                for cid, pp in grid.get((kr + i, kc + j), {}).items():
                    if cid != me and (pp[0] - q[0]) ** 2 + (pp[1] - q[1]) ** 2 < d_excl ** 2:
                        return True
        return False

    def place(cid, p, old=None):
        if old is not None:
            grid.get(key(old), {}).pop(cid, None)
        grid.setdefault(key(p), {})[cid] = p
    on_at = lambda t, p: bool(V[t][int(np.clip(round(p[0]), 0, H - 1)), int(np.clip(round(p[1]), 0, W - 1))])
    in_roi = lambda p: 0 <= p[0] < H and 0 <= p[1] < W and roi[int(p[0]), int(p[1])]
    # library per starting state, KD-tree on the previous step
    L = {}
    for s in (True, False):
        g = lib[lib.on_a == s]
        gp = g[np.isfinite(g.pdr)]
        L[s] = dict(all=g[["dr", "dc", "on_b"]].to_numpy(), prev=gp[["dr", "dc", "on_b"]].to_numpy(),
                    tree=cKDTree(gp[["pdr", "pdc"]].to_numpy()))

    def step(p, v, t, me):
        s = on_at(t, p)
        lb = L[s]
        if v is None:
            cand = lb["all"][rng.integers(0, len(lb["all"]), REDRAWS)]
        else:
            _, j = lb["tree"].query(v / lam, k=K_NN)
            cand = lb["prev"][rng.choice(np.atleast_1d(j), REDRAWS)]
        for dr, dc, on_b in cand:
            q = p + lam * np.array([dr, dc])
            if in_roi(q) and on_at(t + 1, q) == bool(on_b) and not crowded(q, me):
                return q
        return p + lam * cand[-1][:2]

    cells = []                                       # dict(id, p, v, donor, age)
    nid = 0

    def new_cell(t, pool):
        nonlocal nid
        f = frames[t]
        for _ in range(REDRAWS):
            if pool:
                p = f.pos_img[idx[t][pool[rng.integers(len(pool))]]].copy()
            else:
                p = f.pos_img[rng.integers(len(f.label))].copy()
            if not crowded(p, -1):
                break
        cells.append(dict(id=nid, p=p, v=None, donor=int(rng.integers(len(don))), age=0))
        place(nid, p)
        nid += 1

    target = lambda t: int(round(rho * n_real[t] / (1 - p_miss)))
    start_pool = lambda t: [lab for (tt, lab) in cstarts if tt == t]
    for p in frames[0].pos_img:                      # one cell per real detection of the first frame
        cells.append(dict(id=nid, p=p.copy(), v=None, donor=int(rng.integers(len(don))), age=0))
        place(nid, p)
        nid += 1
    while len(cells) < target(0):                    # extra cells: real positions of later frames
        new_cell(int(rng.integers(1, T)), None)
    lab_stack = np.zeros((T, H, W), np.int32)
    green = np.zeros((T, H, W), np.float32)
    gt = []
    bg = np.float32(np.median(tifffile.imread(CROPS / well / f"T{ts[0]:04d}_C2.tif")[~roi]))
    for t in range(T):
        if t > 0:                                    # move, die, be born
            alive = []
            for k in rng.permutation(len(cells)):
                c = cells[k]
                if rng.random() < death_p:
                    grid.get(key(c["p"]), {}).pop(c["id"], None)
                    continue
                q = step(c["p"], c["v"], t - 1, c["id"])
                if not in_roi(q):
                    grid.get(key(c["p"]), {}).pop(c["id"], None)
                    continue
                place(c["id"], q, c["p"])
                c["v"], c["p"], c["age"] = q - c["p"], q, c["age"] + 1
                alive.append(c)
            cells = alive
            pool = start_pool(t)
            while len(cells) < target(t):
                new_cell(t, pool)
        # render in random order; earlier cells keep their pixels
        lab, img = lab_stack[t], green[t]
        nxt_label = 1
        for k in rng.permutation(len(cells)):
            c = cells[k]
            ch = don[c["donor"]]
            n = len(ch)
            j = c["age"] % (2 * n - 2) if n > 1 else 0
            m, g, cr, cc_ = ch[j if j < n else 2 * n - 2 - j]
            r0 = int(round(c["p"][0] - cr)); c0 = int(round(c["p"][1] - cc_))
            r1, c1 = r0 + m.shape[0], c0 + m.shape[1]
            if r0 < 0 or c0 < 0 or r1 > H or c1 > W:
                gt.append((c["id"], t, -1, *c["p"], on_at(t, c["p"]), 0))
                continue
            sub = lab[r0:r1, c0:c1]
            free = m & (sub == 0)
            img[r0:r1, c0:c1] += g                   # the cell is there even if it is not detected
            if rng.random() < p_miss:                # missed by segmentation
                gt.append((c["id"], t, -1, *c["p"], on_at(t, c["p"]), 0))
                continue
            if free.sum() < MERGE_FRAC * m.sum():    # merged into the neighbor that covers it most
                nb = int(np.bincount(sub[m & (sub > 0)]).argmax())
                sub[free] = nb
                gt.append((c["id"], t, -1, *c["p"], on_at(t, c["p"]), nb))
                continue
            sub[free] = nxt_label
            gt.append((c["id"], t, nxt_label, *c["p"], on_at(t, c["p"]), 0))
            nxt_label += 1
        img += bg
    gt = pd.DataFrame(gt, columns=["gt_id", "frame", "label", "r", "c", "on_vessel", "merged_into"])
    info = dict(well=well, rho=rho, seed=seed, lam=lam, p_miss=p_miss, death_p=death_p, n_library=len(lib),
                n_donors=len(don), r_px=r_px, d_excl_px=d_excl)
    return lab_stack, green, gt, info


def calibrate(well, seed, lo=1.0, hi=2.5, iters=6):
    """Step scale lambda such that the twin (rho = 1) matches the real well's
    95th-percentile pilot step (common.symmetric_gate)."""
    from common import symmetric_gate
    _, labels, _ = load_frames(well)
    target = symmetric_gate(labels)["pilot_step_p95_px"]
    f = OUT / f"lambda_{well}.json"
    if f.exists():
        return json.loads(f.read_text())["lambda"]
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lab = simulate(well, 1.0, seed, mid)[0]
        p95 = symmetric_gate([lab[t] for t in range(len(lab))])["pilot_step_p95_px"]
        print(f"  lambda {mid:.3f}: pilot p95 {p95:.1f} px (real {target:.1f})", flush=True)
        lo, hi = (mid, hi) if p95 < target else (lo, mid)
    lam = 0.5 * (lo + hi)
    f.write_text(json.dumps(dict(well=well, **{"lambda": lam}, target_p95_px=target)))
    return lam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--well", default="ARTORG_w7")
    ap.add_argument("--rho", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lam", type=float, default=None, help="step scale; default: calibrated at rho = 1")
    args = ap.parse_args()
    lam = args.lam if args.lam is not None else calibrate(args.well, args.seed)
    print(json.dumps(write(args.well, args.rho, args.seed, lam, "cal" if args.lam is None else f"{lam:g}")), flush=True)


def seq_dir(well, rho, lam_label, seed):
    return OUT / f"{well}_x{rho:g}_l{lam_label}_s{seed}"


def write(well, rho, seed, lam, lam_label):
    lab, green, gt, info = simulate(well, rho, seed, lam)
    d = seq_dir(well, rho, lam_label, seed)
    d.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(d / "cells.tif", lab, compression="zlib")
    tifffile.imwrite(d / "green.tif", np.clip(green, 0, 65535).astype(np.uint16), compression="zlib")
    gt.to_csv(d / "gt.csv", index=False)
    info.update(lam_label=lam_label, n_detections_per_frame=float((gt.label > 0).groupby(gt.frame).sum().mean()),
                frac_missed_or_merged=float((gt.label < 0).mean()))
    (d / "info.json").write_text(json.dumps(info, indent=2))
    return info


if __name__ == "__main__":
    main()
