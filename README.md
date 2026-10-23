# ChronoTrack: Spatiotemporal Cell–Vessel Tracking for Interpretable Microvasculature-on-Chip Readouts

Code, benchmark data and supplementary material for the paper

> S. Jagtap, S. Zeinali, O. Guenat, T. Heinemann. *ChronoTrack: Spatiotemporal Cell–Vessel Tracking for Interpretable Microvasculature-on-Chip Readouts.* (under review)

ChronoTrack tracks cells and vessels jointly in microvasculature-on-chip (MVoC) time-lapse images:

- **Time-symmetric cell tracking:** cells are linked by minimizing a cost (energy) that is identical for a movie and its time-reversed counterpart, jointly over links between consecutive frames and across missed frames.
- **Data-driven missed-detection model:** how long a cell may go undetected, and what bridging a gap costs, are measured on each sequence.
- **Vessel graph:** vessel segments are identified consistently over time and labeled stable, regressing, emerging or intermittent; the time window adapts to stable and forming networks.
- **Spatiotemporal MVoC features:** interpretable readouts such as arrivals on vessels relative to chance, candidate extravasation and intravasation rates, and speeds on and off the vessels.
- **Benchmark with ground truth:** synthetic MVoC sequences with known tracks, on which ChronoTrack is compared with its ablations and with btrack, LapTrack and Trackastra.

## Repository layout

| Folder | Content |
|---|---|
| `code/topology_tracking/` | The ChronoTrack library: time-symmetric tracker (`symmetric_tracker.py`), missed-detection model (`detection_model.py`), vessel scaffold (`scaffold.py`), vessel states and events (`states.py`, `events.py`), baselines and metrics |
| `code/` | `synth_benchmark.py` (runs and scores the trackers), `published_trackers.py` (btrack, LapTrack, Trackastra), `common.py` (shared settings), `make_sample.py` (crops the sample) |
| `figures/` | Scripts that draw Fig. 2 and the ablation table from the benchmark results (figures written to `images/`) |
| `figure_data/` | The numbers behind Fig. 2 and the ablation table, as CSV (the ones used in the paper) |
| `data/` | A small sample sequence (`data/sample/`) and the data format (`data/README.md`) |
| `supplement/` | Supplementary material (PDF and LaTeX source) |

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Everything runs on the CPU.

## Data

Each sequence consists of cell label maps (one detection per label and frame), the cell-channel intensity, the ground truth and a vessel layer (per-frame vessel masks and the vessel graph); the format is described in [`data/README.md`](data/README.md).

- **Sample (in this repository):** `data/sample/` holds one sequence (`seq1_density1_gentle_seed0`), cropped to 1280 × 1280 pixels with all 24 frames (15 MB), to test the code without a download.
- **Full synthetic set (48 sequences, 2.9 GB):** https://doi.org/10.5281/zenodo.XXXXXXX (TODO). Four archives, one per vessel network; extract them into `data/`:

```bash
mkdir -p data && for f in chronotrack_synthetic_seq*.tar.gz; do tar -xzf "$f" -C data; done
```

## Running

Run from the repository root; every script prints its options with `--help`.

**Track one sequence.** ChronoTrack (`sym`) on the sample, scored against the ground truth:

```bash
python code/synth_benchmark.py --data data/sample --out results/sample --trackers sym --jobs 1
```

The scores are written to `results/sample/benchmark.csv` (precision, recall and F1 of the links, complete tracks, feature error), and the tracks to `results/sample/tracks/<sequence>/tracks_sym.csv.gz` (`track_id, frame_idx, label, centroid_r, centroid_c`, plus per-detection cues). Sequences of the full set are selected with `--seqs`, `--densities`, `--motions` and `--seeds`, e.g. `--seqs seq1 --densities 1 --motions gentle --seeds 0`.

**Benchmark.** Without `--trackers`, all of ChronoTrack's variants run (`sym`, `sym_vessel`, `first_order`, `w0`, `nn`); `--published` runs btrack, LapTrack and Trackastra instead (written to `benchmark_published.csv`):

```bash
python code/synth_benchmark.py --data data/sample --out results/sample --jobs 1
python code/synth_benchmark.py --data data/sample --out results/sample --published --jobs 1
```

On the full set (`--data data` and `--out results` are the defaults) the five variants take a few minutes per sequence and the three published trackers about 40 minutes; `--jobs` runs sequences in parallel:

```bash
python code/synth_benchmark.py --jobs 6
python code/synth_benchmark.py --published --jobs 6
```

**Fig. 2 and the ablation table.** From the full benchmark in `results/`:

```bash
python figures/make_fig_benchmark.py   # images/benchmark.pdf/.png; figure_data/benchmark_bars.csv, benchmark_sequences.csv, table_ablation.csv
python figures/make_tables.py          # LaTeX rows of the ablation table (Supplementary Table S1)
```

Both take `--results <folder>` to read the benchmark CSVs from another folder.

## Supplementary material

[`supplement/supplement.pdf`](supplement/supplement.pdf) (LaTeX source: `supplement/supplement.tex`): robustness to missed detections and longer frame intervals, the label-free tests, and the full ablation on the synthetic sequences.

## Citation

Citation details will be added after publication (TODO).

## License

Apache License 2.0, see [`LICENSE`](LICENSE).
