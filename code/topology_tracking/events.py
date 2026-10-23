"""Chip event table: what happens in one microvasculature-on-chip time-lapse.

Entities and events
    cells     arrival (track starts after the first frame), departure (track
              ends before the last frame), vessel exit (overlapping -> k
              non-overlapping frames), vessel entry (k non-overlapping ->
              overlapping), each with the vessel state and scaffold segment
              where it happens
    vessels   segment regression and emergence (the scaffold segments' classes,
              with vessel-signal support), with their change time
The event list is turned into one row of chip-level features (rates per 100
cell-hours, so that removing tracks does not change them trivially; on-vessel
enrichment of events over chance; state occupancy; speeds; vessel remodelling;
joint cell-vessel co-location).

Time reversal maps every event onto its mirror image: arrival <-> departure,
exit <-> entry, regression <-> emergence. MIRROR gives that map for the
features, used by the time-reversal test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .states import add_vessel_distance, assign_states

MIRROR = {
    "arrivals_per_100_cell_h": "departures_per_100_cell_h",
    "departures_per_100_cell_h": "arrivals_per_100_cell_h",
    "arrival_vessel_enrichment": "departure_vessel_enrichment",
    "departure_vessel_enrichment": "arrival_vessel_enrichment",
    "arrival_on_vessel_frac": "departure_on_vessel_frac",
    "departure_on_vessel_frac": "arrival_on_vessel_frac",
    "exits_per_100_cell_h": "entries_per_100_cell_h",
    "entries_per_100_cell_h": "exits_per_100_cell_h",
    "vessel_regression_frac": "vessel_emergence_frac",
    "vessel_emergence_frac": "vessel_regression_frac",
    "exit_regression_colocation": "entry_emergence_colocation",
    "entry_emergence_colocation": "exit_regression_colocation",
}


def _runs(on: np.ndarray, k: int, kind: str) -> list[int]:
    """Indices where an exit (on -> k x off) or entry (k x off -> on) is completed.
    Exit index: first off frame of the run; entry index: first on frame."""
    out = []
    if kind == "exit":
        seen_on, run = False, 0
        for i, v in enumerate(on):
            if v:
                seen_on, run = True, 0
            elif seen_on:
                run += 1
                if run == k:
                    out.append(i - k + 1)
                    seen_on = False
    else:  # entry: mirror of exit
        rev = _runs(on[::-1], k, "exit")
        out = sorted(len(on) - 1 - (i - 1) for i in rev)   # first on frame in forward time
    return out


def cell_events(tracks: pd.DataFrame, n_frames: int, k: int, territory: np.ndarray | None = None,
                to_reference=None) -> pd.DataFrame:
    """Event list for cells. `tracks` must carry `state` (states.assign_states).
    `territory` (scaffold segment id per pixel, reference frame) and
    `to_reference(points, frame)` locate each event on a vessel segment."""
    rows = []
    for tid, g in tracks.sort_values("frame_idx").groupby("track_id", sort=False):
        f = g.frame_idx.to_numpy()
        on = (g.state.to_numpy() == "overlapping")
        p = g[["centroid_r", "centroid_c"]].to_numpy(float)
        if f[0] > 0:
            rows.append((tid, int(f[0]), "arrival", bool(on[0]), p[0, 0], p[0, 1]))
        if f[-1] < n_frames - 1:
            rows.append((tid, int(f[-1]), "departure", bool(on[-1]), p[-1, 0], p[-1, 1]))
        for i in _runs(on, k, "exit"):
            # located where the cell last overlapped the vessel
            j = max(i - 1, 0)
            rows.append((tid, int(f[i]), "exit", True, p[j, 0], p[j, 1]))
        for i in _runs(on, k, "entry"):
            rows.append((tid, int(f[i]), "entry", True, p[i, 0], p[i, 1]))
    ev = pd.DataFrame(rows, columns=["track_id", "frame_idx", "type", "on_vessel", "r", "c"])
    ev["segment"] = -1
    if territory is not None and len(ev):
        pts = ev[["r", "c"]].to_numpy(float, copy=True)
        if to_reference is not None:
            for fr, idx in ev.groupby("frame_idx").indices.items():
                pts[idx] = to_reference(pts[idx], int(fr))
        r = np.clip(np.round(pts[:, 0]).astype(int), 0, territory.shape[0] - 1)
        c = np.clip(np.round(pts[:, 1]).astype(int), 0, territory.shape[1] - 1)
        ev["segment"] = territory[r, c]
    return ev


def with_states(tracks: pd.DataFrame, vessel_masks: list, um_per_px: float, delta_um: float) -> pd.DataFrame:
    return assign_states(add_vessel_distance(tracks, vessel_masks, um_per_px), delta_um)


def chip_features(tracks: pd.DataFrame, events: pd.DataFrame, segments: pd.DataFrame | None,
                  n_frames: int, frame_h: float, um_per_px: float) -> dict:
    """One row of chip-level features from tracks (with states) and events."""
    cell_h = len(tracks) * frame_h                         # detections x frame interval
    n = lambda t: int((events.type == t).sum())
    frac_on = lambda t: float(events.loc[events.type == t, "on_vessel"].mean()) if n(t) else np.nan
    st = tracks.state.to_numpy()
    g = tracks.sort_values(["track_id", "frame_idx"])
    same = (g.track_id.to_numpy()[1:] == g.track_id.to_numpy()[:-1]) & \
           (np.diff(g.frame_idx.to_numpy()) == 1)
    step = np.hypot(np.diff(g.centroid_r.to_numpy()), np.diff(g.centroid_c.to_numpy()))[same] * um_per_px / frame_h
    st0 = g.state.to_numpy()[:-1][same]
    fr = g.groupby("track_id").frame_idx
    span = fr.max() - fr.min()                             # frames from first to last detection; missed frames count
    feats = dict(
        cells_per_frame=len(tracks) / n_frames,
        arrivals_per_100_cell_h=100 * n("arrival") / cell_h,
        departures_per_100_cell_h=100 * n("departure") / cell_h,
        arrival_on_vessel_frac=frac_on("arrival"),
        departure_on_vessel_frac=frac_on("departure"),
        # on-vessel fraction of arrivals / departures over that of all detections (1 = chance)
        arrival_vessel_enrichment=frac_on("arrival") / max(float((tracks.state == "overlapping").mean()), 1e-9),
        departure_vessel_enrichment=frac_on("departure") / max(float((tracks.state == "overlapping").mean()), 1e-9),
        exits_per_100_cell_h=100 * n("exit") / cell_h,
        entries_per_100_cell_h=100 * n("entry") / cell_h,
        frac_overlapping=float((st == "overlapping").mean()),
        frac_adjacent=float((st == "adjacent").mean()),
        frac_distant=float((st == "distant").mean()),
        speed_on_vessel_um_h=float(np.median(step[st0 == "overlapping"])) if (st0 == "overlapping").any() else np.nan,
        speed_off_vessel_um_h=float(np.median(step[st0 != "overlapping"])) if (st0 != "overlapping").any() else np.nan,
        mean_track_duration_h=float(span.mean() * frame_h),   # track length (frames) x frame interval
    )
    if segments is not None and len(segments):
        tot = segments.length_um.sum()
        reg = segments["class"] == "regressing"
        emg = segments["class"] == "emerging"
        feats["vessel_regression_frac"] = float(segments.loc[reg, "length_um"].sum() / tot)
        feats["vessel_emergence_frac"] = float(segments.loc[emg, "length_um"].sum() / tot)

        def colocation(event_type, seg_mask):
            """Events per mm on the given segments / events per mm on all segments (1 = no association)."""
            e = events[(events.type == event_type) & (events.segment > 0)]
            if not len(e) or not seg_mask.any():
                return np.nan
            ids = set(segments.loc[seg_mask, "segment_id"])
            on_sel = e.segment.isin(ids).sum() / segments.loc[seg_mask, "length_um"].sum()
            return float(on_sel / (len(e) / tot))
        feats["exit_regression_colocation"] = colocation("exit", reg)
        feats["entry_emergence_colocation"] = colocation("entry", emg)
    return feats
