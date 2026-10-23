"""Published trackers for the synthetic benchmark (synth_benchmark.py), run on
the same detections as ours and returning tracks in our format (track_id,
frame_idx, label, centroid_r, centroid_c).

  btrack      Ulicna et al. 2021 (Bayesian tracker), topology_tracking.baselines
              .run_btrack: default cell config, search radius and link-hypothesis
              distance set to the shared gate, no division hypotheses
  laptrack    Fukai & Kawaguchi 2023 (the TrackMate / u-track LAP framework,
              Jaqaman et al. 2008): squared-distance costs, linking cutoff =
              gate^2, gap closing up to M missed frames with cutoff
              gate^2 (M + 1), no splitting or merging
  trackastra  Gallusser & Weigert 2024 (transformer association): pretrained
              general_2d model on CPU, greedy linking without divisions,
              candidate distance gate * sqrt(M + 1), frame gaps up to M + 1

The gate is common.symmetric_gate and M the longest bridged gap of
topology_tracking.detection_model, both estimated on the sequence, as for our
trackers.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from topology_tracking.baselines import run_btrack

PUBLISHED = ("btrack", "laptrack", "trackastra")
_ASTRA = None


def _detections(frames):
    rows = [(t, int(l), p[0], p[1]) for t, fd in enumerate(frames) for l, p in zip(fd.label, fd.pos_img)]
    return pd.DataFrame(rows, columns=["frame", "label", "r", "c"])


def run_laptrack(frames, gate_px: float, max_gap: int) -> pd.DataFrame:
    from laptrack import LapTrack
    det = _detections(frames)
    lt = LapTrack(metric="sqeuclidean", cutoff=gate_px ** 2,
                  gap_closing_metric="sqeuclidean", gap_closing_cutoff=gate_px ** 2 * (max_gap + 1) if max_gap else False,
                  gap_closing_max_frame_count=max_gap + 1 if max_gap else 1,
                  splitting_cutoff=False, merging_cutoff=False)
    track_df, _, _ = lt.predict_dataframe(det, coordinate_cols=["r", "c"], frame_col="frame", only_coordinate_cols=False)
    return pd.DataFrame(dict(track_id=track_df.track_id.values, frame_idx=track_df.frame.values,
                             label=track_df.label.values.astype(int), centroid_r=track_df.r.values,
                             centroid_c=track_df.c.values))


def run_trackastra(labels, images, gate_px: float, max_gap: int) -> pd.DataFrame:
    global _ASTRA
    from trackastra.model import Trackastra
    if _ASTRA is None:
        _ASTRA = Trackastra.from_pretrained("general_2d", device="cpu")
    masks = np.stack(labels).astype(np.int32)
    graph, _ = _ASTRA.track(np.asarray(images, np.float32), masks, mode="greedy_nodiv",
                            max_distance=int(round(gate_px * np.sqrt(max_gap + 1))), delta_t=max_gap + 1)
    rows = []
    for tid, comp in enumerate(nx.weakly_connected_components(graph), start=1):
        for n in comp:
            d = graph.nodes[n]
            rows.append((tid, int(d["time"]), int(d["label"]), float(d["coords"][0]), float(d["coords"][1])))
    return pd.DataFrame(rows, columns=["track_id", "frame_idx", "label", "centroid_r", "centroid_c"])


def run_published(name, frames, labels, images, gate_px: float, max_gap: int) -> pd.DataFrame:
    if name == "btrack":
        return run_btrack(labels, gate_px)
    if name == "laptrack":
        return run_laptrack(frames, gate_px, max_gap)
    if name == "trackastra":
        return run_trackastra(labels, images, gate_px, max_gap)
    raise ValueError(name)
