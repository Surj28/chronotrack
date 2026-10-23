"""Motion library for the synthetic ARTORG twin (synth_twin.py): consecutive-frame
links on which all five trackers of run_event_tests.py agree, in the four ARTORG
wells. Agreement of trackers with different models is the closest we get to
correct links without annotation; resampling their steps gives the twin real
motion without using any one tracker's motion model.

Per consensus link (frame t -> t+1): image-coordinate step, the previous step
when the link before it is also a consensus link, and the vessel state
(on the vessel mask or not) at both ends.

Outputs <results>/synthetic/: tracks_<well>_<tracker>.csv (cached tracks),
library.csv.

Usage (from the repo root):
    python paper/code/synth_library.py [--wells ARTORG_w7 ...]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import tifffile

import run_event_tests as RET
from common import RESULTS, SEG, WELLS

OUT = RESULTS / "synthetic"


def links(tr: pd.DataFrame) -> dict:
    t = tr.sort_values(["track_id", "frame_idx"])
    same = (t.track_id.values[1:] == t.track_id.values[:-1]) & (np.diff(t.frame_idx.values) == 1)
    a = zip(t.frame_idx.values[:-1][same], t.label.values[:-1][same].astype(int))
    return dict(zip(a, t.label.values[1:][same].astype(int)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wells", nargs="+", default=WELLS)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for well in args.wells:
        W = RET.Well(well)
        L = {}
        for name in RET.TRACKERS:
            f = OUT / f"tracks_{well}_{name}.csv"
            tr = pd.read_csv(f) if f.exists() else RET.run_named(name, W.frames, W.labels)
            tr.to_csv(f, index=False)
            L[name] = links(tr)
        cons = {k: v for k, v in L["sym"].items() if all(L[n].get(k) == v for n in RET.TRACKERS)}
        idx = [dict(zip(fd.label.astype(int), range(len(fd.label)))) for fd in W.frames]
        V = [tifffile.imread(SEG / well / f"vessels_T{t:04d}.tif") > 0 for t in W.ts]
        on = lambda t, p: bool(V[t][int(np.clip(round(p[0]), 0, V[t].shape[0] - 1)), int(np.clip(round(p[1]), 0, V[t].shape[1] - 1))])
        prev = {(t + 1, b): (t, a) for (t, a), b in cons.items()}
        for (t, a), b in cons.items():
            pa = W.frames[t].pos_img[idx[t][a]]
            pb = W.frames[t + 1].pos_img[idx[t + 1][b]]
            row = dict(well=well, t=t, a=a, b=b, dr=pb[0] - pa[0], dc=pb[1] - pa[1], on_a=on(t, pa), on_b=on(t + 1, pb),
                       area=float(W.frames[t].area[idx[t][a]]), pdr=np.nan, pdc=np.nan)
            if (t, a) in prev:
                tp, z = prev[(t, a)]
                pz = W.frames[tp].pos_img[idx[tp][z]]
                row.update(pdr=pa[0] - pz[0], pdc=pa[1] - pz[1])
            rows.append(row)
        n_all = len(L["sym"])
        print(f"{well}: {len(cons)} consensus links of {n_all} ({100 * len(cons) / n_all:.1f}%)", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "library.csv", index=False)


if __name__ == "__main__":
    main()
