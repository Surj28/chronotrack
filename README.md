# ChronoTrack: Spatiotemporal Cell–Vessel Tracking for Interpretable Microvasculature-on-Chip Readouts

Code, result tables and supplementary material for the paper

> S. Jagtap, S. Zeinali, O. Guenat, T. Heinemann. *ChronoTrack: Spatiotemporal Cell–Vessel Tracking for Interpretable Microvasculature-on-Chip Readouts.* (under review)

ChronoTrack tracks cells and vessels jointly in microvasculature-on-chip (MVoC) time-lapse images:

- **Time-symmetric cell tracking:** cells are linked by minimizing a cost (energy) that is identical for a movie and its time-reversed counterpart, jointly over links between consecutive frames and across missed frames.
- **Data-driven missed-detection model:** how long a cell may go undetected, and what bridging a gap costs, are measured on each sequence.
- **Vessel graph:** vessel segments are identified consistently over time and labeled stable, regressing, emerging or intermittent; the time window adapts to stable and forming networks.
- **Spatiotemporal MVoC features:** interpretable readouts such as arrivals on vessels relative to chance, candidate extravasation and intravasation rates, and speeds on and off the vessels.
- **Evaluation without ground truth:** synthetic twins of real recordings (with known tracks) and label-free tests on real data (time reversal, missed detections, frame interval, broken tracks, reproducibility).

## Supplementary material

[`supplement/supplement.pdf`](supplement/supplement.pdf): robustness to missed detections and longer frame intervals, reproducibility across replicate wells, the full ablation on the synthetic twins, and the label-free tests on real data.

## Repository layout

| Folder | Content |
|---|---|
| `code/topology_tracking/` | The ChronoTrack library: time-symmetric tracker (`symmetric_tracker.py`), missed-detection model (`detection_model.py`), vessel states and events (`states.py`, `events.py`), baselines and metrics |
| `code/` | Scripts that produce the paper's results (see below) |
| `figures/` | Scripts that draw the paper's figures and tables from the result files |
| `figure_data/` | The numbers behind every figure and table, as CSV |
| `supplement/` | Supplementary material (PDF and LaTeX source) |

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Segmentation is an input to ChronoTrack: cells were segmented with [Cellpose-SAM](https://github.com/MouseLand/cellpose) and vessels with a multi-scale ridge (Sato) filter; the published RGB videos were segmented by Otsu thresholding (`code/prepare_song.py`, `code/prepare_hachey.py`). ChronoTrack expects one cell label map and one vessel mask per frame.

## Data

- **ARTORG dataset** (four replicate wells, hourly frames): the raw recordings are not included in this repository.
- **Song et al. 2021** (Front. Immunol. 12:733317, CC BY 4.0): Supplementary Videos S1, S3 and S5.
- **Hachey et al. 2021** (Lab Chip 21:1333–1351, CC BY-NC 3.0): Supplementary Video 1, days 0–5.

Download the published videos from the journals' supplementary material and prepare them with `code/prepare_song.py` and `code/prepare_hachey.py`.

## Reproducing the results

Run from the repository root; outputs are written to `results/`.

| Step | Script |
|---|---|
| Vessel graph (adaptive windows) | `code/vessel_scaffold.py` |
| Event tests on real data (time reversal, missed detections, frame interval, broken tracks) | `code/run_event_tests.py`, `code/run_external_events.py`, `code/run_hachey_events.py`, `code/run_dropout_curve.py`, `code/run_sampling_extra.py` |
| Stationary cells | `code/stationary_continuity.py` |
| Bootstrap tests and reproducibility statistics | `code/replicate_stats.py` |
| Synthetic twins and benchmark | `code/synth_library.py`, `code/synth_twin.py`, `code/synth_benchmark.py` (published trackers: `--published`, via `code/published_trackers.py`) |
| Figures and tables | `figures/make_fig_three_models.py`, `figures/make_fig_benchmark.py`, `figures/make_fig_real.py`, `figures/make_tables.py` |

The figure scripts write the plotted numbers to `figure_data/`; the CSVs in this repository are the ones used in the paper.

## Citation

Citation details will be added after publication.

## License

Apache License 2.0, see [`LICENSE`](LICENSE).
