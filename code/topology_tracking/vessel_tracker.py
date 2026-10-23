"""Vessel-referenced, self-calibrating cell tracker (the paper's method).

Each detection is described in vessel-referenced coordinates taken from the
sequence's temporally consistent vascular scaffold (scaffold.py): position in
the scaffold's reference frame, signed distance to the vessel wall, local
vessel axis, and whether it lies on a vessel ("state"). Area and mean
intensity are added.

Link cost: the negative log-likelihood that detection j is the same cell as
track i (last seen k frames ago), summed over cues:
    motion  residual r after a persistent prediction p + alpha * v * k, with a
            per-track scale that grows with the track's recent speed,
            sigma_i^2 = sigma0^2 + beta^2 |v|^2 (sigma_new for tracks without
            a previous step); Student-t (nu = 4) to keep heavy tails; in the
            vessel frame (along / across the local axis, ratio gamma) for
            cells on a vessel, isotropic in the gel; separate parameters for
            cells on vessels and in the gel
    size    change of log area          (Gaussian, z^2 capped at 9)
    bright  change of log intensity     (Gaussian, z^2 capped at 9)
    wall    change of signed wall distance (Gaussian, z^2 capped at 9)
Caps keep one segmentation error from vetoing a link.

Linking: frame by frame, one sparse linear assignment with birth and death
(the optimum of the augmented matrix of Jaqaman et al. 2008, solved exactly
per connected component of the candidate graph), over the 12 nearest
detections to each track's last position. A link is refused only when the
step is longer than the sequence's shared distance gate (common.symmetric_gate,
times sqrt(k) after k frames; the same gate for every tracker); motion,
appearance and vessel cues then decide between competing candidates through
the assignment, never refuse on their own. A refusal rule that depends on the
current links would tighten round after round (each round cuts the longest
links, which then no longer inform the next calibration); a z-gate on the
fitted motion scale did exactly that on dense, mostly stationary data (Song
S5: scale 0.81 -> 0.57 px over the rounds, moving cells cut off). Tracks may be
missed for max_gap frames. New tracks may start anywhere.

Self-calibration (no labels): every parameter is re-estimated from the links
of the current tracks (robust statistics; sigma0, beta by a fit of the residual
spread against the previous step's length), and the sequence is re-tracked,
until no parameter changes by more than 3 %. For each cue, the ratio of its
spread to nearest competing candidates over its spread along links is
reported as the cue's separation (1 = no information).

Cues can be switched off for ablation: persistence (alpha = beta = 0),
state (one parameter set for all cells), anisotropy (gamma = 1), size,
bright, wall. "state", "anisotropy" and "wall" are the vessel-derived cues.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.spatial import cKDTree

CUES = ("persistence", "state", "anisotropy", "size", "bright", "wall")
VESSEL_CUES = ("state", "anisotropy", "wall")
NU = 4.0
Z2_CAP = 9.0
Z2_GATE = NU * (0.001 ** (-2.0 / NU) - 1.0)   # 2-D Student-t: P(z^2 > gate) = (1 + gate/nu)^(-nu/2) = 0.001
K_NEAREST = 12      # candidates per track: the 12 nearest detections to its prediction


def _rsd(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if x.size else np.nan


# ------------------------------------------------------------------ detections

@dataclass
class FrameDets:
    label: np.ndarray
    pos_img: np.ndarray      # (n, 2) frame coordinates
    pos: np.ndarray          # (n, 2) scaffold reference coordinates
    area: np.ndarray
    inten: np.ndarray        # background-subtracted mean intensity (nan if no image)
    wall: np.ndarray         # signed distance to the vessel wall, px (+ inside); 0 without scaffold
    on: np.ndarray           # bool
    tan: np.ndarray          # (n, 2) unit local vessel axis (0 off-vessel / without scaffold)


def wall_map(scaffold) -> np.ndarray | None:
    if scaffold is None:
        return None
    U = scaffold.union
    return ndi.distance_transform_edt(U) - ndi.distance_transform_edt(~U)


def frame_detections(label_map: np.ndarray, image: np.ndarray | None, scaffold, t: int,
                     wmap: np.ndarray | None) -> FrameDets:
    ids = np.unique(label_map)
    ids = ids[ids > 0]
    if ids.size == 0:
        z = np.zeros((0, 2))
        return FrameDets(ids, z, z, np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0, bool), z)
    cen = np.array(ndi.center_of_mass(label_map > 0, label_map, ids)).reshape(-1, 2)
    area = np.asarray(ndi.sum(np.ones_like(label_map, np.int32), label_map, ids), float)
    if image is not None:
        bg = float(np.median(image[label_map == 0]))
        inten = np.asarray(ndi.mean(image.astype(np.float32), label_map, ids), float) - bg
    else:
        inten = np.full(ids.size, np.nan)
    if scaffold is not None:
        pos = scaffold.to_reference(cen, t)
        on = scaffold.on_vessel(pos)
        tan = scaffold.tangent_at(pos) * on[:, None]
        r = np.clip(np.round(pos[:, 0]).astype(int), 0, wmap.shape[0] - 1)
        c = np.clip(np.round(pos[:, 1]).astype(int), 0, wmap.shape[1] - 1)
        wall = wmap[r, c]
    else:
        pos, on, tan, wall = cen, np.zeros(ids.size, bool), np.zeros((ids.size, 2)), np.zeros(ids.size)
    return FrameDets(ids, cen, pos, area, inten, wall, on, tan)


# ------------------------------------------------------------------ model

@dataclass
class Calib:
    alpha_on: float = 0.0
    alpha_gel: float = 0.0
    sig0_on: float = 10.0
    beta_on: float = 0.0
    signew_on: float = 10.0
    sig0_gel: float = 10.0
    beta_gel: float = 0.0
    signew_gel: float = 10.0
    gamma: float = 1.0          # along / across spread ratio on vessels
    s_area: float = np.inf
    s_int: float = np.inf
    s_wall: float = np.inf
    c_max: float = 0.0          # diagnostic: fraction of current links beyond the motion gate
    # diagnostics: spread to the nearest competing candidate (not used for linking)
    sf_motion: float = np.nan
    sf_area: float = np.nan
    sf_int: float = np.nan
    sf_wall: float = np.nan
    n_links: int = 0
    cues: tuple = CUES

    def vector(self) -> np.ndarray:
        v = np.array([self.alpha_on, self.alpha_gel, self.sig0_on, self.beta_on, self.signew_on,
                      self.sig0_gel, self.beta_gel, self.signew_gel, self.gamma,
                      self.s_area, self.s_int, self.s_wall], float)
        return np.where(np.isfinite(v), v, 0.0)


def _sigma(cal: Calib, on, speed, has_v):
    """Per-pair motion scale (geometric mean of along/across on vessels)."""
    s0 = np.where(on, cal.sig0_on, cal.sig0_gel)
    be = np.where(on, cal.beta_on, cal.beta_gel)
    sn = np.where(on, cal.signew_on, cal.signew_gel)
    return np.where(has_v, np.sqrt(s0 ** 2 + (be * speed) ** 2), sn)


def pair_cost(tp, tv, has_v, ton, ttan, tarea, tint, twall, k, dp, darea, dint, dwall, cal: Calib):
    """Negative log-likelihood of each candidate pair (arrays over pairs)."""
    use = set(cal.cues)
    state = "state" in use
    on = ton if state else np.zeros_like(ton)
    if "persistence" in use:
        alpha = np.where(on, cal.alpha_on, cal.alpha_gel)
        speed = np.hypot(*tv.T)
    else:
        alpha, speed = np.zeros(len(tp)), np.zeros(len(tp))
        has_v = np.zeros_like(has_v)
    pred = tp + alpha[:, None] * tv * k[:, None]
    r = dp - pred
    sig = _sigma(cal, on, speed, has_v) * np.sqrt(k)
    g = cal.gamma if ("anisotropy" in use and state) else 1.0
    sa, sb = sig * np.sqrt(g), sig / np.sqrt(g)
    rp = (r * ttan).sum(1)
    rq = r[:, 0] * ttan[:, 1] - r[:, 1] * ttan[:, 0]
    z2 = np.where(on, (rp / sa) ** 2 + (rq / sb) ** 2, (r ** 2).sum(1) / sig ** 2)
    cost = 0.5 * (NU + 2) * np.log1p(z2 / NU) + 2 * np.log(sig)
    pair_cost.last_z2 = z2
    if "size" in use and np.isfinite(cal.s_area):
        cost = cost + 0.5 * np.minimum((np.log(darea / tarea) / cal.s_area) ** 2, Z2_CAP)
    if "bright" in use and np.isfinite(cal.s_int):
        ok = (tint > 0) & (dint > 0)
        x = np.where(ok, np.log(np.where(ok, dint, 1.0) / np.where(ok, tint, 1.0)), 0.0)
        cost = cost + np.where(ok, 0.5 * np.minimum((x / cal.s_int) ** 2, Z2_CAP), 0.0)
    if "wall" in use and np.isfinite(cal.s_wall):
        cost = cost + 0.5 * np.minimum(((dwall - twall) / cal.s_wall) ** 2, Z2_CAP)
    return cost, pred, sig


# ------------------------------------------------------------------ linking

def solve_links(n_d: int, n_t: int, dj: np.ndarray, ti: np.ndarray, cost: np.ndarray,
                min_total: bool = False):
    """Optimal links for one frame pair. min_total=False: among matchings that
    use the most feasible (in-gate) pairs, the one of least total cost.
    min_total=True: the set of links of least total cost, a link being worth
    making only if its cost is negative (costs given relative to "no link"). With the birth and
    death costs used here (any feasible link is preferred to a birth plus a
    death) this is the optimum of the augmented assignment of Jaqaman et al.
    Solved exactly per connected component of the candidate graph with a dense
    linear assignment, infeasible pairs at a prohibitive cost. (The sparse
    solver scipy.sparse.csgraph.min_weight_full_bipartite_matching was found
    to stall on some frames.)"""
    from scipy.optimize import linear_sum_assignment
    from scipy.sparse.csgraph import connected_components
    A = coo_matrix((np.ones(dj.size), (dj, n_d + ti)), shape=(n_d + n_t, n_d + n_t))
    _, comp = connected_components(A, directed=False)
    ec = comp[dj]
    order = np.argsort(ec, kind="stable")
    bounds = np.flatnonzero(np.diff(ec[order])) + 1
    out = []
    for e in np.split(order, bounds):
        rows, r_inv = np.unique(dj[e], return_inverse=True)
        cols, c_inv = np.unique(ti[e], return_inverse=True)
        if min_total:
            # "no link" costs 0: the least-total-cost set of links (all link costs < 0),
            # exactly the augmented assignment with zero birth / death costs
            C = np.zeros((rows.size, cols.size))
            C[r_inv, c_inv] = np.minimum(cost[e], 0.0)
            r, c = linear_sum_assignment(C)
            ok = C[r, c] < 0
        else:
            big = (float(np.abs(cost[e]).max()) + 1.0) * (min(rows.size, cols.size) + 1)
            C = np.full((rows.size, cols.size), big)
            C[r_inv, c_inv] = cost[e]
            r, c = linear_sum_assignment(C)
            ok = C[r, c] < big
        out += [(int(rows[a]), int(cols[b]), float(C[a, b])) for a, b in zip(r[ok], c[ok])]
    return out


def track(frames: list[FrameDets], cal: Calib, gate_px: float, max_gap: int = 1) -> pd.DataFrame:
    """Track one sequence with fixed parameters. Every detection is assigned
    (unlinked detections start new tracks). Returns one row per detection."""
    rows = []
    alive = []
    next_id = 1
    for t, fd in enumerate(frames):
        n_d, n_t = len(fd.label), len(alive)
        link_of_det = np.full(n_d, -1)
        cost_of_det = np.full(n_d, np.nan)
        if n_t and n_d:
            tree = cKDTree(fd.pos)
            P = np.array([a["p"] for a in alive]); Vv = np.array([a["v"] for a in alive])
            HV = np.array([a["has_v"] for a in alive]); ON = np.array([a["on"] for a in alive])
            TAN = np.array([a["tan"] for a in alive]); K = np.array([t - a["t_last"] for a in alive], float)
            radius = gate_px * np.sqrt(K)                 # shared distance gate, not the fitted scale
            kq = min(K_NEAREST, n_d)
            dd, jj = tree.query(P, k=kq, distance_upper_bound=float(radius.max()))
            dd, jj = dd.reshape(n_t, kq), jj.reshape(n_t, kq)
            ok = np.isfinite(dd) & (dd <= radius[:, None])
            ti = np.repeat(np.arange(n_t), kq).reshape(n_t, kq)[ok]
            dj = jj[ok]
            if ti.size:
                A = lambda key: np.array([alive[i][key] for i in ti])
                cost, _, _ = pair_cost(P[ti], Vv[ti], HV[ti], ON[ti], TAN[ti], A("area"), A("int"), A("wall"),
                                       K[ti], fd.pos[dj], fd.area[dj], fd.inten[dj], fd.wall[dj], cal)
            if ti.size:
                # augmented assignment: a link replaces one death and one birth, each costing
                # half the largest feasible link cost (+1), so every feasible link is worth
                # making on its own while conflicting links still trade off by cost
                nolink = float(cost.max()) + 1.0
                for j, i, c in solve_links(n_d, n_t, dj, ti, cost - nolink, min_total=True):
                    link_of_det[j] = i
                    cost_of_det[j] = c + nolink
        matched = set()
        for j in range(n_d):
            i = link_of_det[j]
            if i >= 0:
                a = alive[i]
                a["v"] = (fd.pos[j] - a["p"]) / (t - a["t_last"])
                a["has_v"] = True
                matched.add(i)
            else:
                a = dict(id=next_id, v=np.zeros(2), has_v=False)
                next_id += 1
                alive.append(a)
                matched.add(len(alive) - 1)
            a.update(p=fd.pos[j].copy(), on=bool(fd.on[j]), tan=fd.tan[j].copy(), area=float(fd.area[j]),
                     int=float(fd.inten[j]), wall=float(fd.wall[j]), t_last=t, lost=0)
            rows.append((a["id"], t, int(fd.label[j]), fd.pos_img[j, 0], fd.pos_img[j, 1], fd.area[j],
                         bool(fd.on[j]), fd.wall[j], fd.inten[j], cost_of_det[j]))
        nxt = []
        for idx, a in enumerate(alive):
            if idx in matched:
                nxt.append(a)
            else:
                a["lost"] += 1
                if a["lost"] <= max_gap:
                    nxt.append(a)
        alive = nxt
    return pd.DataFrame(rows, columns=["track_id", "frame_idx", "label", "centroid_r", "centroid_c", "area",
                                       "inside_vessel", "wall_px", "intensity", "link_cost"])


# ------------------------------------------------------------------ calibration

def _links(frames, tracks):
    """Consecutive-frame links of `tracks` with features of both ends and of the previous step."""
    idx = [{int(l): i for i, l in enumerate(fd.label)} for fd in frames]
    tr = tracks.sort_values(["track_id", "frame_idx"])
    tid, fr, lab = tr.track_id.to_numpy(), tr.frame_idx.to_numpy(), tr.label.to_numpy()
    di = np.array([idx[f][int(l)] for f, l in zip(fr, lab)])
    s = np.flatnonzero((tid[1:] == tid[:-1]) & (fr[1:] == fr[:-1] + 1))
    sp = np.maximum(s - 1, 0)
    prev_ok = (s >= 1) & (tid[sp] == tid[s]) & (fr[sp] == fr[s] - 1)
    F0, J0, J1 = fr[s], di[s], di[s + 1]
    g = lambda attr, F, J: np.array([getattr(frames[f], attr)[j] for f, j in zip(F, J)])
    p0, p1 = g("pos", F0, J0).reshape(-1, 2), g("pos", F0 + 1, J1).reshape(-1, 2)
    v = np.zeros_like(p0)
    if prev_ok.any():
        v[prev_ok] = p0[prev_ok] - g("pos", fr[sp][prev_ok], di[sp][prev_ok]).reshape(-1, 2)
    return dict(F0=F0, J0=J0, J1=J1, p0=p0, p1=p1, v=v, has_v=prev_ok,
                on=g("on", F0, J0).astype(bool), tan=g("tan", F0, J0).reshape(-1, 2),
                a0=g("area", F0, J0), a1=g("area", F0 + 1, J1), i0=g("inten", F0, J0), i1=g("inten", F0 + 1, J1),
                w0=g("wall", F0, J0), w1=g("wall", F0 + 1, J1))


def calibrate(frames, tracks, prev: Calib, gate_px: float, min_samples: int = 50) -> Calib:
    L = _links(frames, tracks)
    cal = Calib(**asdict(prev))
    use = set(prev.cues)
    on = L["on"] if "state" in use else np.zeros(len(L["on"]), bool)
    step = L["p1"] - L["p0"]

    def alpha_fit(mask):
        m = mask & L["has_v"]
        if m.sum() < min_samples:
            return 0.0
        a, b = L["v"][m], step[m]
        lim = np.percentile(np.hypot(*b.T), 99)
        ok = (np.hypot(*a.T) < lim) & (np.hypot(*b.T) < lim)
        den = (a[ok] ** 2).sum()
        return float(np.clip((a[ok] * b[ok]).sum() / den, 0.0, 0.95)) if den > 0 else 0.0

    if "persistence" in use:
        cal.alpha_on, cal.alpha_gel = alpha_fit(on), alpha_fit(~on)
    else:
        cal.alpha_on = cal.alpha_gel = 0.0
    alpha = np.where(on, cal.alpha_on, cal.alpha_gel)
    r = step - alpha[:, None] * L["v"]
    tan = L["tan"]
    rp = (r * tan).sum(1)
    rq = r[:, 0] * tan[:, 1] - r[:, 1] * tan[:, 0]
    if "anisotropy" in use and "state" in use and on.sum() >= min_samples:
        cal.gamma = float(np.clip(_rsd(rp[on]) / max(_rsd(rq[on]), 1e-6), 0.5, 4.0))
    else:
        cal.gamma = 1.0
    # per-link isotropic spread: geometric mean of the along/across components on vessels
    comp = np.where(on[:, None], np.stack([rp / np.sqrt(cal.gamma), rq * np.sqrt(cal.gamma)], 1), r)
    speed = np.hypot(*L["v"].T)

    def scale_fit(mask):
        """sigma0, beta from residual spread vs previous speed; sigma_new from first links."""
        mv = mask & L["has_v"]
        sig0, beta = np.nan, 0.0
        if mv.sum() >= min_samples and "persistence" in use:
            q = np.quantile(speed[mv], np.linspace(0, 1, 9))
            xs, ys = [], []
            for lo, hi in zip(q[:-1], q[1:]):
                b = mv & (speed >= lo) & (speed <= hi)
                if b.sum() >= 20:
                    xs.append(np.median(speed[b]) ** 2)
                    ys.append(_rsd(np.concatenate([comp[b, 0], comp[b, 1]])) ** 2)
            if len(xs) >= 3:
                Amat = np.stack([np.ones(len(xs)), xs], 1)
                coef, *_ = np.linalg.lstsq(Amat, np.array(ys), rcond=None)
                sig0 = float(np.sqrt(max(coef[0], 1e-2)))
                beta = float(np.sqrt(max(coef[1], 0.0)))
        if not np.isfinite(sig0):
            m = mask
            sig0 = _rsd(np.concatenate([comp[m, 0], comp[m, 1]])) if m.sum() >= min_samples else np.nan
        mn = mask & ~L["has_v"]
        if "persistence" not in use:
            mn = mask
        signew = _rsd(np.concatenate([comp[mn, 0], comp[mn, 1]])) if mn.sum() >= min_samples else sig0
        return sig0, beta, signew

    s_on = scale_fit(on) if on.sum() >= min_samples else None
    s_gel = scale_fit(~on) if (~on).sum() >= min_samples else None
    if s_on is None:
        s_on = s_gel
    if s_gel is None:
        s_gel = s_on
    (cal.sig0_on, cal.beta_on, cal.signew_on), (cal.sig0_gel, cal.beta_gel, cal.signew_gel) = s_on, s_gel

    # appearance spreads along links
    cal.s_area = _rsd(np.log(L["a1"] / L["a0"])) if "size" in use else np.inf
    ok = (L["i0"] > 0) & (L["i1"] > 0)
    cal.s_int = _rsd(np.log(L["i1"][ok] / L["i0"][ok])) if ("bright" in use and ok.sum() >= min_samples) else np.inf
    cal.s_wall = _rsd(L["w1"] - L["w0"]) if "wall" in use else np.inf
    for nm in ("s_area", "s_int", "s_wall"):
        if not np.isfinite(getattr(cal, nm)) or getattr(cal, nm) <= 1e-6:
            setattr(cal, nm, np.inf)

    k1 = np.ones(len(r))
    cost, pred, _ = pair_cost(L["p0"], L["v"], L["has_v"], L["on"], tan, L["a0"], L["i0"], L["w0"], k1,
                              L["p1"], L["a1"], L["i1"], L["w1"], cal)
    cal.c_max = float(np.mean(pair_cost.last_z2 > Z2_GATE))   # diagnostic: fraction of current links beyond the gate

    # diagnostics: nearest competing candidate of each link (spread ratios reported, not used)
    trees, cj = {}, np.full(len(r), -1)
    for n, (f1, pr, j1) in enumerate(zip(L["F0"] + 1, pred, L["J1"])):
        if f1 not in trees:
            trees[f1] = cKDTree(frames[f1].pos)
        dd, jj = trees[f1].query(pr, k=2, distance_upper_bound=gate_px)
        for d_, j_ in zip(np.atleast_1d(dd), np.atleast_1d(jj)):
            if np.isfinite(d_) and j_ != j1:
                cj[n] = j_
                break
    h = cj >= 0
    if h.sum() >= min_samples:
        F1 = L["F0"][h] + 1
        g = lambda attr: np.array([getattr(frames[f], attr)[j] for f, j in zip(F1, cj[h])])
        pc, ac, ic, wc = g("pos").reshape(-1, 2), g("area"), g("inten"), g("wall")
        rf = pc - pred[h]
        cal.sf_motion = _rsd(np.concatenate([rf[:, 0], rf[:, 1]]))
        cal.sf_area = _rsd(np.log(ac / L["a0"][h]))
        okc = (L["i0"][h] > 0) & (ic > 0)
        cal.sf_int = _rsd(np.log(ic[okc] / L["i0"][h][okc])) if okc.sum() >= min_samples else np.nan
        cal.sf_wall = _rsd(wc - L["w0"][h])
    cal.n_links = int(len(r))
    return cal


def separation(cal: Calib) -> dict:
    """Spread to the nearest competing candidate / spread along links, per cue (1 = no information)."""
    sm = float(np.sqrt(cal.sig0_on * cal.sig0_gel))
    return dict(motion=cal.sf_motion / sm if sm > 0 else np.nan, anisotropy=cal.gamma,
                size=cal.sf_area / cal.s_area, bright=cal.sf_int / cal.s_int, wall=cal.sf_wall / cal.s_wall)


def run(frames: list[FrameDets], gate: dict, cues=CUES, max_gap: int = 1, max_rounds: int = 8,
        tol: float = 0.03, verbose: bool = False):
    """Self-calibrating tracking of one sequence. `gate` is common.estimate_max_dist
    on the same label maps. Returns (tracks, final Calib, per-round history)."""
    s0 = gate["pilot_step_p95_px"] / 2.45
    cal = Calib(sig0_on=s0, signew_on=s0, sig0_gel=s0, signew_gel=s0,
                cues=tuple(c for c in cues if c in ("state",)))
    tracks = track(frames, cal, gate["max_dist_px"], max_gap)
    history = []
    for rnd in range(1, max_rounds + 1):
        new = calibrate(frames, tracks, Calib(**{**asdict(cal), "cues": tuple(cues)}), gate["max_dist_px"])
        change = float(np.max(np.abs(new.vector() - cal.vector()) / np.maximum(np.abs(cal.vector()), 1e-3)))
        history.append(dict(round=rnd, max_rel_change=change, **{k: v for k, v in asdict(new).items() if k != "cues"}))
        cal = new
        tracks = track(frames, cal, gate["max_dist_px"], max_gap)
        if verbose:
            L = tracks.groupby("track_id").size()
            print(f"  round {rnd}: change {change:.3f} | tracks {len(L)}, mean length {L.mean():.2f} | "
                  + json.dumps({k: round(v, 3) for k, v in asdict(cal).items()
                                if isinstance(v, float) and np.isfinite(v)}), flush=True)
        if change < tol and rnd > 1:
            break
    return tracks, cal, history
