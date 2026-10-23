# ChronoTrack synthetic benchmark data

Synthetic microvasculature-on-chip time-lapse sequences with ground-truth cell identities, used for the tracking benchmark of

> S. Jagtap, S. Zeinali, O. Guenat, T. Heinemann. *ChronoTrack: Spatiotemporal Cell–Vessel Tracking for Interpretable Microvasculature-on-Chip Readouts.* (under review)

Code: https://github.com/Surj28/chronotrack (`code/synth_benchmark.py` runs the trackers on these sequences and scores them).
DOI of this data set: https://doi.org/10.5281/zenodo.XXXXXXX (TODO)

## Content

48 sequences: 4 vessel networks (`seq1`–`seq4`) × 2 cell densities × 2 motion regimes × 3 random seeds. Each sequence has 24 frames (1 h apart, 0.65 µm per pixel) of about 3000 × 3000 pixels, with about 2,100–5,500 cell detections per frame.

Sequence folders are named `seq<k>_density<d>_<motion>_seed<s>`:

| Part | Values | Meaning |
|---|---|---|
| `seq<k>` | 1–4 | vessel network (the four sequences of a network share its vessel layer, `vessels/seq<k>/`) |
| `density<d>` | 1, 2 | cell density (2: twice as many cells) |
| `<motion>` | `gentle`, `ambiguous` | cell steps as generated, or scaled up (`step_scale` in `info.json`) to a harder linking ambiguity |
| `seed<s>` | 0, 1, 2 | random seed of the simulation |

## Files

Archives (one per vessel network, 12 sequences each, plus that network's vessel layer):

| File | Size |
|---|---|
| `chronotrack_synthetic_seq1.tar.gz` | 610 MB |
| `chronotrack_synthetic_seq2.tar.gz` | 658 MB |
| `chronotrack_synthetic_seq3.tar.gz` | 821 MB |
| `chronotrack_synthetic_seq4.tar.gz` | 777 MB |
| `SHA256SUMS` | SHA-256 of the archives |
| `SHA256SUMS_files` | SHA-256 of every extracted file (paths relative to the extraction folder) |

Total: 2.87 GB compressed, 3.07 GB extracted. Extract all four into one folder, e.g. the repository's `data/`:

```bash
sha256sum -c SHA256SUMS
mkdir -p data && for f in chronotrack_synthetic_seq*.tar.gz; do tar -xzf "$f" -C data; done
(cd data && sha256sum -c ../SHA256SUMS_files)
```

After extraction:

```
data/
  seq<k>_density<d>_<motion>_seed<s>/   (48 folders)
    cells.tif     label maps (24, H, W) int32: 0 background, > 0 one detection (labels are per frame)
    green.tif     cell-channel intensity (24, H, W) uint16
    gt.csv        ground truth, one row per cell and frame
    info.json     parameters of the sequence
  vessels/seq<k>/   (4 folders)
    vessels.tif        per-frame vessel masks (24, H, W) uint8 (1 = vessel)
    union_mask.tif     vessel network in the reference frame (frame 0), (H, W) uint8
    segment_ids.tif    centreline pixels labelled by vessel segment, (H, W) int32
    territory_ids.tif  segment of the nearest centreline pixel, inside the network (0 outside), (H, W) int32
    segments.csv       one row per segment: segment_id, length_um, mean_radius_um, centroid_r, centroid_c,
                       class (stable, regressing, emerging, intermittent, mask-only)
    flows.npz          flows (24, 2, H // ds, W // ds) float16 and ds (downsampling, 4): displacement (row, col),
                       in downsampled pixels, from frame t to the reference frame (zeros for frame 0)
```

All images are zlib-compressed TIFF; coordinates are in pixels, (row, column), frames numbered from 0.

`gt.csv` columns:

| Column | Meaning |
|---|---|
| `gt_id` | identity of the cell (constant over its life) |
| `frame` | frame index (0–23) |
| `label` | label of the detection carrying the cell in `cells.tif` of that frame; -1 if the cell is missed or merged |
| `r`, `c` | true position of the cell centre (px) |
| `on_vessel` | the centre lies on a vessel in that frame |
| `merged_into` | label of the detection the cell merged into (touching cells segmented as one), else 0 |

The true tracks are the rows with `label > 0`, linked by `gt_id` (across missed frames too).

`info.json`: `sequence`, `vessels` (folder of the vessel layer), `density`, `motion`, `seed`, `n_frames`, `um_per_px`, `frame_interval_h`, `step_scale` (factor on all steps; 1 for gentle motion), `p_miss` (per-detection probability of a missed detection), `death_p` (per-frame probability that a cell ends), `cell_radius_px`, `min_cell_distance_px` (closest approach of two cell centres), `n_detections_per_frame`, `frac_missed_or_merged`.

## Sample in the code repository

The repository ships a small cropped sequence in `data/sample/` (15 MB), made with `code/make_sample.py` from `seq1_density1_gentle_seed0`: a 1280 × 1280 pixel window at (row, column) = (1536, 1536), all 24 frames. Cells whose true centre lies outside the window are removed from the label maps and from `gt.csv`; the remaining positions are shifted to the window. The vessel layer is cropped to the same window (the flows on their 4× downsampled grid); `segments.csv` keeps the segments that reach into the window, with their full length. Scores on the sample are therefore close to, but not the same as, those on the full sequence. `info.json` records the crop.

## License

TODO
