"""Replicate-level statistics for ChronoTrack on the real MVoC data (paper Fig. 3
and the chip event table).

Sequences: ARTORG, 4 replicate wells; Song et al. 2021, 3 videos (NK cells
against HT29, SW48 and LoVo tumors: independent chips, not replicates of one
condition); Hachey et al. 2021, 1 video.

run      ChronoTrack forward on every sequence (as run_event_tests.process), then
         a cluster bootstrap over tracks: tracks are resampled with replacement
         (B times) and the chip features recomputed, giving each feature's
         sampling distribution within the sequence. Also the edge fraction of
         arrivals and its chance level (fraction of all detections within one
         cell diameter of the region edge; as run_event_tests.accounting).
         Outputs <results>/replicate_stats/: point_<model>.csv (one row per
         sequence), bootstrap_<model>.csv (one row per sequence and resample),
         tables/ (forward tracks with states, and events).

Tests (functions, used by the figure and table scripts):
  heterogeneity   do the ARTORG wells differ by more than the sampling noise
                  within wells? Cochran's Q over the 4 wells, with the bootstrap
                  standard errors; Holm over the features
  contrasts       the readouts the paper interprets: speed on vs off vessel,
                  vessel enrichment of arrivals vs 1, edge fraction of
                  arrivals vs chance. Two-sided bootstrap p-value per
                  sequence. ARTORG: a claim needs every replicate well
                  (intersection-union: max p). Published videos: each on its
                  own (not known to be replicates). Holm over all tests
  fragments       arrivals with a departure within reach 1-3 frames before, vs
                  the chance level (accounting.csv): binomial test per sequence,
                  max over a model's sequences
  reversal        events without a mirror event on the reversed movie,
                  ChronoTrack vs frame-by-frame linking (first_order): paired
                  Wilcoxon signed-rank test over all sequences

Usage (from the repo root):
    python paper/code/replicate_stats.py run --model artorg|song|hachey [--boot 2000]
    python paper/code/replicate_stats.py tests
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

import run_event_tests as RET
from common import RESULTS, WELLS
from topology_tracking.events import chip_features

OUT = RESULTS / "replicate_stats"
MODEL_DIRS = {"ARTORG": "event_tests", "Song": "event_tests_external", "Hachey": "event_tests_hachey"}
CONTRASTS = [("speed", "speed_on_vessel_um_h", "speed_off_vessel_um_h"),       # (name, a, b): a - b, or a - 1 if b is None
             ("arrival_enrichment", "arrival_vessel_enrichment", None),
             ("departure_enrichment", "departure_vessel_enrichment", None),
             ("arrival_edge", "arrival_edge_frac", "edge_chance")]


def sequences(model):
    if model == "artorg":
        return "ARTORG", (RET.Well(w) for w in WELLS)
    if model == "song":
        from run_external_events import SongVideo
        return "Song", (SongVideo(v) for v in ("S1", "S3", "S5"))
    from run_hachey_events import HacheyVideo
    return "Hachey", (HacheyVideo(),)


def _groups(codes, n):
    order = np.argsort(codes, kind="stable")
    counts = np.bincount(codes, minlength=n)
    return order, np.r_[0, np.cumsum(counts)[:-1]], counts


def _take(order, starts, counts, draw):
    """Row indices of the drawn tracks (with repeats) and their new track ids."""
    n = counts[draw]
    within = np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n)
    return order[np.repeat(starts[draw], n) + within], np.repeat(np.arange(len(draw)), n)


def bootstrap(W, st, ev, B, rng):
    H, Wd = W.edge_um.shape
    at_edge = lambda r, c: W.edge_um[np.clip(np.round(r).astype(int), 0, H - 1),
                                     np.clip(np.round(c).astype(int), 0, Wd - 1)] <= 2 * W.delta
    det_edge = at_edge(st.centroid_r.to_numpy(), st.centroid_c.to_numpy())
    ev_edge = at_edge(ev.r.to_numpy(), ev.c.to_numpy())
    is_arr = (ev.type == "arrival").to_numpy()
    feats = lambda s, e, rs, re_: dict(chip_features(s, e, W.segments, W.T, W.frame_h, W.um_per_px),
                                       arrival_edge_frac=float(ev_edge[re_][is_arr[re_]].mean()) if is_arr[re_].any() else np.nan,
                                       edge_chance=float(det_edge[rs].mean()))
    point = feats(st, ev, np.arange(len(st)), np.arange(len(ev)))
    tids = pd.Index(st.track_id.unique())
    gs, ge = _groups(tids.get_indexer(st.track_id), len(tids)), _groups(tids.get_indexer(ev.track_id), len(tids))
    st_ = st[["track_id", "frame_idx", "centroid_r", "centroid_c", "state"]]
    rows = []
    for b in range(B):
        draw = rng.integers(0, len(tids), len(tids))
        rs, ids_s = _take(*gs, draw)
        re_, ids_e = _take(*ge, draw)
        rows.append(dict(b=b, **feats(st_.iloc[rs].assign(track_id=ids_s), ev.iloc[re_].assign(track_id=ids_e), rs, re_)))
    return point, rows


def run(model, B):
    name, seqs = sequences(model)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    F0 = pd.read_csv(RESULTS / MODEL_DIRS[name] / "features.csv").query("tracker == 'sym' and condition == 'forward'")
    rng = np.random.default_rng(7)
    points, boots = [], []
    for W in seqs:
        t0 = time.time()
        fs, fe = OUT / "tables" / f"{W.well}_states.csv.gz", OUT / "tables" / f"{W.well}_events.csv.gz"
        if fs.exists() and fe.exists():                    # cached forward run (delete to re-track)
            st, ev = pd.read_csv(fs), pd.read_csv(fe)
        else:
            tk = RET.run_named("sym", W.frames, W.labels)
            _, st, ev = W.table(tk, {i: i for i in range(W.T)}, W.T, W.segments)
            st.to_csv(fs, index=False)
            ev.to_csv(fe, index=False)
        point, rows = bootstrap(W, st, ev, B, rng)
        ref = F0[F0.well == W.well].iloc[0]
        dev = max(abs(point[k] - ref[k]) / max(abs(ref[k]), 1e-9) for k in ref.index
                  if k in point and pd.notna(ref[k]) and pd.notna(point[k]))
        print(f"{W.well}: {st.track_id.nunique()} tracks, {len(ev)} events, max deviation from features.csv "
              f"{dev:.2e}, {time.time() - t0:.0f}s", flush=True)
        points.append(dict(model=name, well=W.well, n_tracks=st.track_id.nunique(),
                           n_arrivals=int((ev.type == "arrival").sum()), **point))
        boots += [dict(model=name, well=W.well, **r) for r in rows]
        pd.DataFrame(points).to_csv(OUT / f"point_{model}.csv", index=False)
        pd.DataFrame(boots).to_csv(OUT / f"bootstrap_{model}.csv", index=False)


def _load(kind):
    return pd.concat([pd.read_csv(OUT / f"{kind}_{m}.csv") for m in ("artorg", "song", "hachey")
                      if (OUT / f"{kind}_{m}.csv").exists()], ignore_index=True)


def _boot_p(d, null=0.0):
    """Two-sided bootstrap p-value of H0: statistic = null."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    lo, hi = (d <= null).sum(), (d >= null).sum()
    return min(1.0, 2 * (1 + min(lo, hi)) / (len(d) + 1))


def heterogeneity(features, model="ARTORG"):
    """Cochran's Q over the sequences of a model, per feature (bootstrap SEs); Holm."""
    P = _load("point").query("model == @model").set_index("well")
    Bt = _load("bootstrap").query("model == @model")
    rows = []
    for f in features:
        x, se = P[f].astype(float), Bt.groupby("well")[f].std()
        ok = x.notna() & se.gt(0)
        w = 1 / se[ok] ** 2
        mu = float((w * x[ok]).sum() / w.sum())
        Q = float((w * (x[ok] - mu) ** 2).sum())
        df = int(ok.sum()) - 1
        rows.append(dict(feature=f, Q=Q, df=df, p=float(stats.chi2.sf(Q, df)), I2=max(0.0, (Q - df) / Q) if Q > 0 else 0.0,
                         cv=float(x.std() / abs(x.mean())), median_within_cv=float((se / x.abs()).median())))
    T = pd.DataFrame(rows)
    T["p_holm"] = multipletests(T.p, method="holm")[1]
    return T


def contrasts(names=("speed", "arrival_enrichment", "arrival_edge")):
    """The interpreted readouts: ARTORG per claim over its replicate wells (significant
    in every well: max p); published videos each on their own. Holm over all."""
    P = _load("point")
    Bt = _load("bootstrap")
    rows = []
    for name, a, b in [c for c in CONTRASTS if c[0] in names]:
        for model, Pm in P.groupby("model", sort=False):
            per = []
            for r in Pm.itertuples():
                bt = Bt[(Bt.model == model) & (Bt.well == r.well)]
                est = getattr(r, a) - (getattr(r, b) if b else 1.0)
                per.append((r.well, est, _boot_p(bt[a] - (bt[b] if b else 1.0))))
            groups = [("4 wells", per)] if model == "ARTORG" else [(q[0], [q]) for q in per]
            for unit, g in groups:
                rows.append(dict(model=model, unit=unit, contrast=name, estimate=float(np.mean([q[1] for q in g])),
                                 same_sign=bool(len({np.sign(q[1]) for q in g}) == 1),
                                 p_each=";".join(f"{q[2]:.3g}" for q in g), p=max(q[2] for q in g)))
    T = pd.DataFrame(rows)
    T["p_holm"] = multipletests(T.p, method="holm")[1]
    return T


def fragments():
    rows = []
    for model, d in MODEL_DIRS.items():
        A = pd.read_csv(RESULTS / d / "accounting.csv").query("tracker == 'sym'")
        ps = []
        for r in A.itertuples():
            k = int(round(r.arrival_fragment_frac * r.n_arrivals))
            ps.append(stats.binomtest(k, int(r.n_arrivals), min(max(r.fragment_chance, 1e-9), 1 - 1e-9)).pvalue)
        rows.append(dict(model=model, frac=";".join(f"{x:.3f}" for x in A.arrival_fragment_frac),
                         chance=";".join(f"{x:.3f}" for x in A.fragment_chance),
                         below=bool((A.arrival_fragment_frac < A.fragment_chance).all()),
                         p_each=";".join(f"{p:.2g}" for p in ps), p=max(ps)))
    return pd.DataFrame(rows)


def reversal():
    rows = []
    for model, d in MODEL_DIRS.items():
        for src, tr in ((d, "sym"), (d + "_framebyframe_2026-10-17", "first_order")):   # same pipeline version
            R = pd.read_csv(RESULTS / src / "reversal_events.csv").query("tracker == @tr")
            R = R.assign(n_miss=R.n_forward * (1 - R.frac_matched)).groupby("well")[["n_forward", "n_miss"]].sum()
            rows += [dict(model=model, well=w, tracker=tr, D=r.n_miss / r.n_forward) for w, r in R.iterrows()]
    T = pd.DataFrame(rows).pivot_table(index=["model", "well"], columns="tracker", values="D")
    return T, stats.wilcoxon(T["sym"], T["first_order"]).pvalue


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True, choices=["artorg", "song", "hachey"])
    r.add_argument("--boot", type=int, default=2000)
    sub.add_parser("tests")
    args = ap.parse_args()
    if args.cmd == "run":
        run(args.model, args.boot)
        return
    pd.set_option("display.width", 220)
    from summarize_event_tests import TRACK_FEATS
    print("== heterogeneity of the ARTORG wells\n", heterogeneity([f for f in TRACK_FEATS]).round(4).to_string())
    print("\n== contrasts\n", contrasts().to_string())
    print("\n== fragments vs chance\n", fragments().to_string())
    T, p = reversal()
    print("\n== reversal (fraction of events without mirror)\n", T.round(4).to_string(), f"\nWilcoxon p = {p:.4f}")


if __name__ == "__main__":
    main()
