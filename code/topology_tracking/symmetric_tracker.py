"""Time-symmetric, self-calibrating association: the paper's tracking objective.

Detections are described in vessel-referenced coordinates (vessel_tracker.FrameDets).
A tracking solution is a set of links between consecutive frames. Its energy
is the negative log-likelihood of a generative model that is invariant under
time reversal:

  births / deaths   each detection starts a track with probability beta, at a
                    position uniform over the image (density 1/A); each
                    detection ends its track with probability delta. Both are
                    fixed at 1/T (T frames): a cell is expected to persist
                    through the movie unless the data say otherwise. Estimating
                    them from the current tracks feeds back (fragmented tracks ->
                    higher rates -> cheaper births -> more fragmentation) and
                    lets a movie and its reverse settle on different rates.
  motion            for consecutive steps a, b of a track (meeting at a junction
                    detection), e(a, b) = 1/2 [-log p(b | a) - log p(a | b)],
                    where p(b | a) is a 2-D Student-t (nu = 4) with mean alpha * a
                    and scale sigma(|a|) = sqrt(sigma0^2 + beta_v^2 |a|^2), in the
                    vessel frame (along / across the local axis, ratio gamma) at
                    a junction on a vessel, isotropic in the gel; separate
                    parameters for cells on vessels and in the gel. The first and
                    last step of a track carry half a marginal term m(s)
                    (Student-t, scale sigma_new) each.
  appearance        per link, change of log area, log intensity and signed
                    wall distance, scored against the change to a nearby
                    different cell (log-likelihood ratio). Each cue's penalty is
                    bounded by the evidence it can give, log(s_false / s_true),
                    so an uninformative or noisy cue can neither make nor veto
                    a link.
Averaging the forward and backward factorizations of each track makes the
energy identical for a movie and its reverse. The total energy is minimized by
block coordinate descent: one frame pair at a time, the links between frames t
and t+1 are re-solved exactly (linear assignment, solve_links) given all other
links, sweeping forward and backward alternately until no link changes. Every
block update can only lower the energy, so the descent terminates. A link is
feasible when its step is within the sequence's shared distance gate
(common.symmetric_gate, the same for every tracker; times sqrt(m+1) across m
missed frames); a gate on the self-calibrated motion scale tightened round
after round on dense, mostly stationary data. The start
is the first-order solution (motion = marginal term only), in which frame
pairs are independent and the solution is exactly time-symmetric. A final
gap-closing step joins a track ending at t to a track starting at t+m+1 (m
missed frames, m = 1..max_gap, shortest gaps first) with the same symmetric
terms: the displacement over m+1 frames (Student-t, scale sigma * sqrt(m+1)),
the appearance terms, and the cost of an m-frame miss, both max_gap and that
cost measured per sequence from isolated cells that stay in place
(detection_model). Parameters are re-estimated from the current tracks
(robust statistics) between rounds.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .vessel_tracker import CUES, NU, FrameDets, _rsd, solve_links



@dataclass
class Model:
    alpha_on: float = 0.0
    alpha_gel: float = 0.0
    sig0_on: float = 10.0
    betav_on: float = 0.0
    signew_on: float = 10.0
    sig0_gel: float = 10.0
    betav_gel: float = 0.0
    signew_gel: float = 10.0
    gamma: float = 1.0
    s_area: float = np.inf
    sf_area: float = np.inf
    s_int: float = np.inf
    sf_int: float = np.inf
    s_wall: float = np.inf
    sf_wall: float = np.inf
    birth: float = 0.05
    death: float = 0.05
    log_area: float = 14.0
    radius: float = 60.0         # candidate radius (px), symmetric
    gate_px: float = np.inf      # feasibility: shared distance gate (px)
    cues: tuple = CUES

    def vector(self):
        v = np.array([self.alpha_on, self.alpha_gel, self.sig0_on, self.betav_on, self.signew_on, self.sig0_gel,
                      self.betav_gel, self.signew_gel, self.gamma, self.s_area, self.s_int, self.s_wall,
                      self.birth, self.death], float)
        return np.where(np.isfinite(v), v, 0.0)


# ------------------------------------------------------------------ energy terms

def _t_nll(r, on, tan, sig, gamma):
    """2-D Student-t NLL (up to a constant) of residual r (n, 2); vessel frame where `on`."""
    sa, sb = sig * np.sqrt(gamma), sig / np.sqrt(gamma)
    rp = (r * tan).sum(1)
    rq = r[:, 0] * tan[:, 1] - r[:, 1] * tan[:, 0]
    z2 = np.where(on, (rp / sa) ** 2 + (rq / sb) ** 2, (r ** 2).sum(1) / sig ** 2)
    return 0.5 * (NU + 2) * np.log1p(z2 / NU) + 2 * np.log(sig), z2


class Terms:
    """Vectorized energy terms for a Model."""

    def __init__(self, m: Model):
        self.m = m
        u = set(m.cues)
        self.state = "state" in u
        self.pers = "persistence" in u
        self.gamma = m.gamma if ("anisotropy" in u and self.state) else 1.0
        self.size, self.bright, self.wall = ("size" in u), ("bright" in u), ("wall" in u)

    def _on(self, on):
        return on if self.state else np.zeros_like(on)

    def marginal(self, s, on, tan):
        on = self._on(on)
        sig = np.where(on, self.m.signew_on, self.m.signew_gel)
        return _t_nll(s, on, tan, sig, self.gamma)

    def conditional(self, a, b, on, tan):
        """-log p(b | a): b is the step after a, junction state on / tan."""
        on = self._on(on)
        if self.pers:
            alpha = np.where(on, self.m.alpha_on, self.m.alpha_gel)
            s0 = np.where(on, self.m.sig0_on, self.m.sig0_gel)
            bv = np.where(on, self.m.betav_on, self.m.betav_gel)
            sig = np.sqrt(s0 ** 2 + (bv * np.hypot(*a.T)) ** 2)
        else:
            alpha = np.zeros(len(a))
            sig = np.where(on, self.m.signew_on, self.m.signew_gel)
        return _t_nll(b - alpha[:, None] * a, on, tan, sig, self.gamma)[0]

    def pairwise(self, a, b, on, tan):
        """e(a, b) = 1/2 [-log p(b | a) - log p(a | b)] (b after a)."""
        return 0.5 * (self.conditional(a, b, on, tan) + self.conditional(b, a, on, tan))

    def appearance(self, Fi, I, Fj, J):
        c = np.zeros(len(I))

        def llr(x, st, sf):
            if not (np.isfinite(st) and np.isfinite(sf)) or st >= 0.95 * sf:
                return np.zeros_like(x)
            v = 0.5 * (x / st) ** 2 - 0.5 * (x / sf) ** 2 + np.log(st / sf)   # -LLR
            return np.minimum(v, np.log(sf / st))       # bounded influence
        if self.size:
            c += llr(np.log(Fj.area[J] / Fi.area[I]), self.m.s_area, self.m.sf_area)
        if self.bright:
            a, b = Fi.inten[I], Fj.inten[J]
            ok = (a > 0) & (b > 0)
            x = np.where(ok, np.log(np.where(ok, b, 1.0) / np.where(ok, a, 1.0)), 0.0)
            c += np.where(ok, llr(x, self.m.s_int, self.m.sf_int), 0.0)
        if self.wall:
            c += llr(Fj.wall[J] - Fi.wall[I], self.m.s_wall, self.m.sf_wall)
        return c

    def link_const(self):
        """Energy of a link minus that of the death + birth it replaces (motion excluded)."""
        b, d = self.m.birth, self.m.death
        return -np.log1p(-d) - np.log1p(-b) + np.log(d) + np.log(b) - self.m.log_area


# ------------------------------------------------------------------ solver

class Solver:
    def __init__(self, frames: list[FrameDets], model: Model, k_nearest: int = 12):
        self.F = frames
        self.T = len(frames)
        self.m = model
        self.trees = [cKDTree(f.pos) if len(f.label) else None for f in frames]
        self.cand = []                              # per frame pair t: (I, J) candidate pairs
        for t in range(self.T - 1):
            A, B = frames[t], frames[t + 1]
            if not len(A.label) or not len(B.label):
                self.cand.append((np.zeros(0, int), np.zeros(0, int)))
                continue
            k = min(k_nearest, len(B.label))
            d1, j1 = self.trees[t + 1].query(A.pos, k=k, distance_upper_bound=model.radius)
            d1, j1 = d1.reshape(len(A.label), k), j1.reshape(len(A.label), k)
            k2 = min(k_nearest, len(A.label))
            d2, i2 = self.trees[t].query(B.pos, k=k2, distance_upper_bound=model.radius)
            d2, i2 = d2.reshape(len(B.label), k2), i2.reshape(len(B.label), k2)
            ok1, ok2 = np.isfinite(d1), np.isfinite(d2)
            I = np.concatenate([np.repeat(np.arange(len(A.label)), k).reshape(-1, k)[ok1], i2[ok2]])
            J = np.concatenate([j1[ok1], np.repeat(np.arange(len(B.label)), k2).reshape(-1, k2)[ok2]])
            pairs = np.unique(np.stack([I, J], 1), axis=0)          # symmetric union of both directions
            self.cand.append((pairs[:, 0], pairs[:, 1]))
        self.succ = [np.full(len(f.label), -1) for f in frames]
        self.pred = [np.full(len(f.label), -1) for f in frames]
        self.gap_succ = {}                                          # (t, i) -> (t + m + 1, j)
        self.gap_out = [np.zeros(len(f.label), bool) for f in frames]   # detection starts a gap link
        self.gap_in = [np.zeros(len(f.label), bool) for f in frames]    # detection ends a gap link

    # -- costs of the candidate links of frame pair t, given the other links
    def costs(self, t, terms: Terms, second_order: bool):
        I, J = self.cand[t]
        A, B = self.F[t], self.F[t + 1]
        s = B.pos[J] - A.pos[I]
        half_mi = 0.5 * terms.marginal(s, A.on[I], A.tan[I])[0]
        half_mj = 0.5 * terms.marginal(s, B.on[J], B.tan[J])[0]
        motion = half_mi + half_mj
        if second_order and t > 0:
            h = self.pred[t][I]
            has = h >= 0
            if has.any():
                a = A.pos[I[has]] - self.F[t - 1].pos[h[has]]
                ii = I[has]
                e = terms.pairwise(a, s[has], A.on[ii], A.tan[ii])
                m_prev = 0.5 * terms.marginal(a, A.on[ii], A.tan[ii])[0]
                motion[has] += e - m_prev - half_mi[has]
        if second_order and t + 2 < self.T:
            k = self.succ[t + 1][J]
            has = k >= 0
            if has.any():
                jj = J[has]
                b = self.F[t + 2].pos[k[has]] - B.pos[jj]
                e = terms.pairwise(s[has], b, B.on[jj], B.tan[jj])
                m_next = 0.5 * terms.marginal(b, B.on[jj], B.tan[jj])[0]
                motion[has] += e - m_next - half_mj[has]
        cost = motion + terms.appearance(A, I, B, J) + terms.link_const()
        feasible = np.hypot(*s.T) <= self.m.gate_px
        return I, J, cost, feasible

    def assign(self, t, terms, second_order):
        I, J, cost, feas = self.costs(t, terms, second_order)
        nA, nB = len(self.F[t].label), len(self.F[t + 1].label)
        feas = feas & ~self.gap_out[t][I] & ~self.gap_in[t + 1][J]   # held by a gap link
        neg = feas & (cost < 0)                     # a link must beat its death + birth
        old = self.succ[t].copy()
        self.succ[t][:] = -1
        self.pred[t + 1][:] = -1
        if neg.any():
            for i, j, _ in solve_links(nB, nA, J[neg], I[neg], cost[neg], min_total=True):   # i at t+1, j at t
                self.succ[t][j] = i
                self.pred[t + 1][i] = j
        return int((old != self.succ[t]).sum())

    def solve(self, terms, max_sweeps=6, verbose=False, init=True):
        if init:
            for t in range(self.T - 1):             # first-order start: frame pairs independent
                self.assign(t, terms, second_order=False)
        hist = []
        for sweep in range(max_sweeps):
            order = range(self.T - 1) if sweep % 2 == 0 else range(self.T - 2, -1, -1)
            changed = sum(self.assign(t, terms, second_order=True) for t in order)
            hist.append(changed)
            if verbose:
                print(f"    sweep {sweep + 1}: {changed} links changed", flush=True)
            if changed == 0:
                break
        return hist

    def set_links(self, tracks: pd.DataFrame, max_gap: int = 0):
        """Start from the links of `tracks` (frame_idx, label): consecutive-frame
        links, and links across up to max_gap missed frames as gap links."""
        idx = [dict(zip(f.label.astype(int), range(len(f.label)))) for f in self.F]
        for a in self.succ:
            a[:] = -1
        for b in self.pred:
            b[:] = -1
        self._clear_gaps()
        t = tracks.sort_values(["track_id", "frame_idx"])
        same = t.track_id.values[1:] == t.track_id.values[:-1]
        dt = np.diff(t.frame_idx.values)
        for ok, d, f, la, lb in zip(same, dt, t.frame_idx.values[:-1], t.label.values[:-1], t.label.values[1:]):
            if not ok:
                continue
            i, j = idx[f][int(la)], idx[f + d][int(lb)]
            if d == 1:
                self.succ[f][i] = j
                self.pred[f + 1][j] = i
            elif d - 1 <= max_gap:
                self._add_gap(f, i, f + d, j)

    def _clear_gaps(self):
        self.gap_succ = {}
        for a, b in zip(self.gap_out, self.gap_in):
            a[:] = False
            b[:] = False

    def _add_gap(self, t, i, u, j):
        self.gap_succ[(t, i)] = (u, j)
        self.gap_out[t][i] = True
        self.gap_in[u][j] = True

    def optimize(self, terms, gap_cost=None, max_sweeps=6, init=True, max_alternations=4):
        """Minimize the energy over frame-to-frame and gap links together:
        descent over frame-to-frame links with the current gap links held,
        then all gap links re-solved given the frame-to-frame links, repeated
        until neither changes. Each step can only lower the energy."""
        hist = self.solve(terms, max_sweeps=max_sweeps, init=init)
        if gap_cost is not None and len(gap_cost) == 0:
            return hist
        for _ in range(max_alternations):
            before = dict(self.gap_succ)
            self.close_gaps(terms, gap_cost)
            h = self.solve(terms, max_sweeps=max_sweeps, init=False)
            hist += h
            if self.gap_succ == before and sum(h) == 0:
                break
        self.close_gaps(terms, gap_cost)
        return hist

    def close_gaps(self, terms, gap_cost=None):
        """Join a track ending at t to a track starting at t+m+1 (m missed frames,
        m = 1..len(gap_cost)), shortest gaps first. gap_cost[m-1] is the cost of an
        m-frame miss (detection_model); default: one missed frame at -log(death)."""
        self._clear_gaps()
        if gap_cost is None:
            gap_cost = [-np.log(self.m.death)]
        into = set()
        for m, g in enumerate(gap_cost, start=1):
            q = np.sqrt(m + 1.0)
            for t in range(self.T - m - 1):
                u = t + m + 1
                ends = np.array([i for i in np.flatnonzero(self.succ[t] < 0) if (t, i) not in self.gap_succ], int)
                starts = np.array([j for j in np.flatnonzero(self.pred[u] < 0) if (u, j) not in into], int)
                if not ends.size or not starts.size:
                    continue
                A, C = self.F[t], self.F[u]
                lists = cKDTree(C.pos[starts]).query_ball_point(A.pos[ends], self.m.radius * q)
                I = np.fromiter((ends[n] for n, js in enumerate(lists) for _ in js), int)
                J = np.fromiter((starts[j] for js in lists for j in js), int)
                if not I.size:
                    continue
                d = (C.pos[J] - A.pos[I]) / q             # displacement over m+1 frames, in one-step units
                motion = 0.5 * (terms.marginal(d, A.on[I], A.tan[I])[0] + terms.marginal(d, C.on[J], C.tan[J])[0]) \
                    + 2 * np.log(q)
                cost = motion + terms.appearance(A, I, C, J) + terms.link_const() + g
                ok = (np.hypot(*d.T) * q <= self.m.gate_px * q) & (cost < 0)
                for i, j, _ in solve_links(len(C.label), len(A.label), J[ok], I[ok], cost[ok], min_total=True):
                    self._add_gap(t, j, u, i)
                    into.add((u, i))

    def tracks(self) -> pd.DataFrame:
        rows, tid = [], 0
        started = [np.zeros(len(f.label), bool) for f in self.F]
        gap_into = set(self.gap_succ.values())
        for t in range(self.T):
            for i in range(len(self.F[t].label)):
                if started[t][i] or self.pred[t][i] >= 0 or (t, i) in gap_into:
                    continue
                tid += 1
                tt, ii = t, i
                while True:
                    started[tt][ii] = True
                    f = self.F[tt]
                    rows.append((tid, tt, int(f.label[ii]), f.pos_img[ii, 0], f.pos_img[ii, 1], f.area[ii],
                                 bool(f.on[ii]), f.wall[ii], f.inten[ii]))
                    if tt + 1 < self.T and self.succ[tt][ii] >= 0:
                        tt, ii = tt + 1, int(self.succ[tt][ii])
                    elif (tt, ii) in self.gap_succ:
                        tt, ii = self.gap_succ[(tt, ii)]
                        ii = int(ii)
                    else:
                        break
        return pd.DataFrame(rows, columns=["track_id", "frame_idx", "label", "centroid_r", "centroid_c", "area",
                                           "inside_vessel", "wall_px", "intensity"])


def energy(frames: list[FrameDets], tracks: pd.DataFrame, m: Model, gap_cost=None) -> float:
    """Total energy (eq. 1 of the paper) of a set of tracks under model m:
    per track the start / end term, half marginals at the ends of every run of
    consecutive frames, pairwise terms inside runs, appearance and survival
    terms per link, and for each link across n missed frames the motion over
    n + 1 frames (scale x sqrt(n + 1)) plus the gap cost g_n."""
    terms = Terms(m)
    gap_cost = [-np.log(m.death)] if gap_cost is None else list(gap_cost)
    idx = [dict(zip(f.label.astype(int), range(len(f.label)))) for f in frames]
    c_track = -np.log(m.birth * m.death) + m.log_area
    c_link = -np.log((1 - m.birth) * (1 - m.death))
    E = 0.0
    for _, g in tracks.sort_values("frame_idx").groupby("track_id"):
        f = g.frame_idx.to_numpy()
        ii = [idx[t][int(l)] for t, l in zip(f, g.label.to_numpy())]
        P = np.array([frames[t].pos[i] for t, i in zip(f, ii)])
        on = np.array([frames[t].on[i] for t, i in zip(f, ii)])
        tan = np.array([frames[t].tan[i] for t, i in zip(f, ii)]).reshape(-1, 2)
        E += c_track
        k0 = 0
        for k in range(1, len(f) + 1):
            if k < len(f) and f[k] - f[k - 1] == 1:
                continue
            seg = np.arange(k0, k)
            if len(seg) >= 2:
                u = np.diff(P[seg], axis=0)
                E += 0.5 * (terms.marginal(u[:1], on[seg[:1]], tan[seg[:1]])[0][0]
                            + terms.marginal(u[-1:], on[seg[-1:]], tan[seg[-1:]])[0][0])
                if len(u) >= 2:
                    E += float(terms.pairwise(u[:-1], u[1:], on[seg[1:-1]], tan[seg[1:-1]]).sum())
            if k < len(f):                              # link across f[k] - f[k-1] - 1 missed frames
                n = int(f[k] - f[k - 1] - 1)
                q = np.sqrt(n + 1.0)
                dv = (P[k] - P[k - 1])[None] / q
                E += 0.5 * (terms.marginal(dv, on[k - 1:k], tan[k - 1:k])[0][0]
                            + terms.marginal(dv, on[k:k + 1], tan[k:k + 1])[0][0]) + 2 * np.log(q)
                E += gap_cost[min(n, len(gap_cost)) - 1] if gap_cost else 0.0
            k0 = k
        for a, b in zip(range(len(f) - 1), range(1, len(f))):
            E += float(terms.appearance(frames[f[a]], np.array([ii[a]]), frames[f[b]], np.array([ii[b]]))[0]) + c_link
    return float(E)


# ------------------------------------------------------------------ calibration

def calibrate(frames: list[FrameDets], tracks: pd.DataFrame, prev: Model, min_samples: int = 50,
              freeze_rates: bool = False) -> Model:
    """Re-estimate the model from the consecutive-frame links of `tracks`."""
    from .vessel_tracker import _links
    L = _links(frames, tracks)
    m = Model(**asdict(prev))
    u = set(prev.cues)
    on = L["on"] if "state" in u else np.zeros(len(L["on"]), bool)
    step = L["p1"] - L["p0"]
    v, hv, tan = L["v"], L["has_v"], L["tan"]

    def alpha_fit(mask):
        mm = mask & hv
        if mm.sum() < min_samples:
            return 0.0
        a, b = v[mm], step[mm]
        lim = np.percentile(np.hypot(*b.T), 99)
        ok = (np.hypot(*a.T) < lim) & (np.hypot(*b.T) < lim)
        den = (a[ok] ** 2).sum()
        return float(np.clip((a[ok] * b[ok]).sum() / den, 0.0, 0.95)) if den > 0 else 0.0

    if "persistence" in u:
        m.alpha_on, m.alpha_gel = alpha_fit(on), alpha_fit(~on)
    alpha = np.where(on, m.alpha_on, m.alpha_gel) * ("persistence" in u)
    r = step - alpha[:, None] * v
    rp = (r * tan).sum(1)
    rq = r[:, 0] * tan[:, 1] - r[:, 1] * tan[:, 0]
    m.gamma = float(np.clip(_rsd(rp[on]) / max(_rsd(rq[on]), 1e-6), 0.5, 4.0)) \
        if ("anisotropy" in u and "state" in u and on.sum() >= min_samples) else 1.0
    comp = np.where(on[:, None], np.stack([rp / np.sqrt(m.gamma), rq * np.sqrt(m.gamma)], 1), r)
    scomp = np.where(on[:, None], np.stack([(step * tan).sum(1) / np.sqrt(m.gamma),
                                            (step[:, 0] * tan[:, 1] - step[:, 1] * tan[:, 0]) * np.sqrt(m.gamma)], 1), step)
    speed = np.hypot(*v.T)

    def fit(mask):
        mv = mask & hv
        sig0, bv = np.nan, 0.0
        if "persistence" in u and mv.sum() >= min_samples:
            q = np.quantile(speed[mv], np.linspace(0, 1, 9))
            xs, ys = [], []
            for lo, hi in zip(q[:-1], q[1:]):
                bb = mv & (speed >= lo) & (speed <= hi)
                if bb.sum() >= 20:
                    xs.append(np.median(speed[bb]) ** 2)
                    ys.append(_rsd(np.concatenate([comp[bb, 0], comp[bb, 1]])) ** 2)
            if len(xs) >= 3:
                coef, *_ = np.linalg.lstsq(np.stack([np.ones(len(xs)), xs], 1), np.array(ys), rcond=None)
                sig0, bv = float(np.sqrt(max(coef[0], 1e-2))), float(np.sqrt(max(coef[1], 0.0)))
        signew = _rsd(np.concatenate([scomp[mask, 0], scomp[mask, 1]])) if mask.sum() >= min_samples else np.nan
        if not np.isfinite(sig0):
            sig0 = signew
        return sig0, bv, signew

    f_on = fit(on) if on.sum() >= min_samples else None
    f_gel = fit(~on) if (~on).sum() >= min_samples else None
    f_on = f_on or f_gel
    f_gel = f_gel or f_on
    (m.sig0_on, m.betav_on, m.signew_on), (m.sig0_gel, m.betav_gel, m.signew_gel) = f_on, f_gel

    # appearance: spread along links and to the nearest competitor of each link
    m.s_area = _rsd(np.log(L["a1"] / L["a0"])) if "size" in u else np.inf
    ok = (L["i0"] > 0) & (L["i1"] > 0)
    m.s_int = _rsd(np.log(L["i1"][ok] / L["i0"][ok])) if ("bright" in u and ok.sum() >= min_samples) else np.inf
    m.s_wall = _rsd(L["w1"] - L["w0"]) if "wall" in u else np.inf
    trees, cj = {}, np.full(len(r), -1)
    pred = L["p0"] + alpha[:, None] * v
    for n, (f1, pr, j1) in enumerate(zip(L["F0"] + 1, pred, L["J1"])):
        if f1 not in trees:
            trees[f1] = cKDTree(frames[f1].pos)
        dd, jj = trees[f1].query(pr, k=2, distance_upper_bound=m.radius)
        for d_, j_ in zip(np.atleast_1d(dd), np.atleast_1d(jj)):
            if np.isfinite(d_) and j_ != j1:
                cj[n] = j_
                break
    h = cj >= 0
    if h.sum() >= min_samples:
        F1 = L["F0"][h] + 1
        g = lambda attr: np.array([getattr(frames[f], attr)[j] for f, j in zip(F1, cj[h])])
        ac, ic, wc = g("area"), g("inten"), g("wall")
        m.sf_area = _rsd(np.log(ac / L["a0"][h]))
        okc = (L["i0"][h] > 0) & (ic > 0)
        m.sf_int = _rsd(np.log(ic[okc] / L["i0"][h][okc])) if okc.sum() >= min_samples else np.inf
        m.sf_wall = _rsd(wc - L["w0"][h])
    for nm in ("s_area", "s_int", "s_wall", "sf_area", "sf_int", "sf_wall"):
        x = getattr(m, nm)
        if not np.isfinite(x) or x <= 1e-6:
            setattr(m, nm, np.inf)

    m.birth = m.death = 1.0 / len(frames)        # fixed prior, see module docstring
    return m


def run(frames: list[FrameDets], gate: dict, image_shape, cues=CUES, max_rounds: int = 6, tol: float = 0.03,
        max_sweeps: int = 6, verbose: bool = False, gap_cost=None, multistart: bool = True):
    """Self-calibrating, time-symmetric tracking of one sequence. `gate` is
    common.estimate_max_dist on the same label maps; `gap_cost` the per-length
    cost of missed frames (detection_model; None: one missed frame at
    -log(death)). With `multistart`, the final descent is also started from the
    forward and from the backward frame-by-frame solution
    (vessel_tracker.run with the same cues), and the lowest-energy result is
    kept; the set of starts is the same for a movie and its reverse.
    Returns (tracks, Model, history)."""
    s0 = gate["pilot_step_p95_px"] / 2.45
    m = Model(sig0_on=s0, signew_on=s0, sig0_gel=s0, signew_gel=s0, radius=1.5 * gate["max_dist_px"],
              gate_px=gate["max_dist_px"],
              log_area=float(np.log(image_shape[0] * image_shape[1])), cues=("state",),
              birth=1.0 / len(frames), death=1.0 / len(frames))
    history = []
    tracks = None
    for rnd in range(max_rounds + 1):
        S = Solver(frames, m)
        if rnd > 0:
            sweeps = S.optimize(Terms(m), gap_cost, max_sweeps=max_sweeps)
        else:
            sweeps = S.solve(Terms(m), max_sweeps=0)
            S.close_gaps(Terms(m), gap_cost)
        tracks = S.tracks()
        if rnd == max_rounds:
            break
        new = calibrate(frames, tracks, Model(**{**asdict(m), "cues": tuple(cues)}), freeze_rates=rnd >= 2)
        change = float(np.max(np.abs(new.vector() - m.vector()) / np.maximum(np.abs(m.vector()), 1e-3)))
        fr = tracks.groupby("track_id").frame_idx
        Ln = fr.max() - fr.min()                       # track length: frames from first to last detection
        history.append(dict(round=rnd, max_rel_change=change, n_tracks=len(Ln), mean_length=float(Ln.mean()),
                            sweeps=sweeps, **{k: v for k, v in asdict(new).items() if k != "cues"}))
        if verbose:
            print(f"  round {rnd}: change {change:.3f} | tracks {len(Ln)}, mean length {Ln.mean():.2f} | sweeps {sweeps}",
                  flush=True)
        m = new
        if change < tol and rnd >= 2:
            S = Solver(frames, m)
            S.optimize(Terms(m), gap_cost, max_sweeps=max_sweeps)
            tracks = S.tracks()
            break
    if multistart:
        from .vessel_tracker import run as run_first_order
        T = len(frames)
        M = len(gap_cost) if gap_cost is not None else 1
        fw = run_first_order(frames, gate, cues=tuple(cues), max_gap=M)[0]
        bw = run_first_order(frames[::-1], gate, cues=tuple(cues), max_gap=M)[0]
        bw = bw.assign(frame_idx=T - 1 - bw.frame_idx)
        best = (energy(frames, tracks, m, gap_cost), "symmetric", tracks)
        for name, start in (("forward", fw), ("backward", bw)):
            S = Solver(frames, m)
            S.set_links(start, max_gap=M)
            S.optimize(Terms(m), gap_cost, max_sweeps=max_sweeps, init=False)
            cand = S.tracks()
            e = energy(frames, cand, m, gap_cost)
            history.append(dict(start=name, energy=e))
            if e < best[0]:
                best = (e, name, cand)
        history.append(dict(start="symmetric", energy=energy(frames, tracks, m, gap_cost), chosen=best[1]))
        tracks = best[2]
    return tracks, m, history
