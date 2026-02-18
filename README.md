# qf-compare

QUIC-Fire simulation output comparison tool (v3.0). Compares project PNG outputs between any two groups of runs using GPU-accelerated image metrics, then generates multi-format reports with fire-metric-based pass/warn/fail verdicts.

v3.0 replaces SSIM-based z-score verdicts with **absolute fire-metric thresholds** (IoU, centroid drift, area diff). SSIM is retained as informational. Images are classified by type (fire/wind/emissions/other); only fire-category images drive verdicts.

---

## Installation

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

---

## Quick start

```bash
# 1. Validate your data and GPU before running
python3 scripts/qf-compare.py \
    --group-a projects/Mac \
    --group-b projects/Linux \
    --check-setup

# 2. Run the comparison (masking and edge metrics enabled by default)
python3 scripts/qf-compare.py \
    --group-a projects/Mac \
    --group-b projects/Linux \
    --output all \
    --output-dir results/
```

---

## CLI reference

```
usage: qf-compare [-h] [--group-a DIR] [--group-b DIR] [--project-a DIR]
                  [--project-b DIR] [--check-setup] [--output OUTPUT]
                  [--output-dir OUTPUT_DIR] [--sigma SIGMA]
                  [--batch-size BATCH_SIZE] [--io-workers IO_WORKERS]
                  [--fire-threshold FIRE_THRESHOLD]
                  [--fire-channels FIRE_CHANNELS] [--no-heatmaps]
                  [--no-mask-rendering] [--no-mask-boundaries]
                  [--border-top BORDER_TOP] [--border-bottom BORDER_BOTTOM]
                  [--border-left BORDER_LEFT] [--border-right BORDER_RIGHT]
                  [--erode-boundary ERODE_BOUNDARY] [--no-edge-metrics]
                  [--multiscale-levels MULTISCALE_LEVELS]
                  [--iou-threshold IOU_THRESHOLD] [--iou-warn IOU_WARN]
                  [--drift-threshold DRIFT_THRESHOLD] [--drift-warn DRIFT_WARN]
                  [--area-diff-threshold AREA_DIFF_THRESHOLD]
                  [--area-diff-warn AREA_DIFF_WARN]

General-purpose QUICFire simulation output comparison tool (GPU-accelerated, v3.0)

options:
  -h, --help            show this help message and exit
  --check-setup         Print setup diagnostics and exit (no comparison)
  --output OUTPUT       Comma-separated output formats: html,md,csv,all
                        (default: html)
  --output-dir OUTPUT_DIR
                        Directory for output files (default: cwd)
  --sigma SIGMA         SSIM deviation threshold in sigma, informational only
                        (default: 2.0)
  --batch-size BATCH_SIZE
                        GPU batch size (default: 32)
  --io-workers IO_WORKERS
                        IO thread pool size (default: 8)

group mode:
  --group-a DIR         Directory containing Group A run folders
  --group-b DIR         Directory containing Group B run folders

project mode:
  --project-a DIR       Single run directory for project A
  --project-b DIR       Single run directory for project B

fire segmentation:
  --fire-threshold FIRE_THRESHOLD
                        Intensity threshold for fire pixel detection (default: 30)
  --fire-channels FIRE_CHANNELS
                        Channels to check for fire: r, g, b, rg, rgb (default: rg)
  --no-heatmaps         Disable generation of difference heatmaps

rendering mask options:
  --no-mask-rendering   Disable masking of plot borders (axes, colorbar, title)
  --no-mask-boundaries  Disable masking of anti-aliased fire boundaries
  --border-top BORDER_TOP
                        Top border mask height in pixels (default: 60)
  --border-bottom BORDER_BOTTOM
                        Bottom border mask height (default: 60)
  --border-left BORDER_LEFT
                        Left border mask width (default: 80)
  --border-right BORDER_RIGHT
                        Right border mask width (default: 100)
  --erode-boundary ERODE_BOUNDARY
                        Pixels to erode from fire boundary (default: 2)

edge comparison options:
  --no-edge-metrics     Disable edge/boundary comparison (faster)

multi-scale analysis options:
  --multiscale-levels MULTISCALE_LEVELS
                        Number of pyramid levels for multi-scale analysis (default: 4)

fire-metric verdict thresholds:
  --iou-threshold IOU_THRESHOLD
                        IoU PASS threshold (default: 0.95)
  --iou-warn IOU_WARN   IoU WARN threshold (default: 0.85)
  --drift-threshold DRIFT_THRESHOLD
                        Centroid drift PASS threshold in pixels (default: 5.0)
  --drift-warn DRIFT_WARN
                        Centroid drift WARN threshold in pixels (default: 10.0)
  --area-diff-threshold AREA_DIFF_THRESHOLD
                        Area diff PASS threshold as fraction (default: 0.05)
  --area-diff-warn AREA_DIFF_WARN
                        Area diff WARN threshold as fraction (default: 0.15)

output formats:
  html   - rich HTML report with methodology, summary, per-project details
  md     - Markdown report (same structure as HTML)
  csv    - raw comparison data (one row per pair)
  all    - shorthand for html,md,csv

examples:
  qf-compare --group-a projects/Mac --group-b projects/Linux
  qf-compare --project-a Canyon-1 --project-b linux-Canyon-1 --output csv,html
  qf-compare --group-a projects/A --group-b projects/B --check-setup
  qf-compare --group-a projects/A --group-b projects/B --output all --output-dir results/
  qf-compare --group-a projects/A --group-b projects/B --fire-threshold 50
  qf-compare --group-a projects/A --group-b projects/B --no-mask-rendering --no-edge-metrics
  qf-compare --group-a projects/A --group-b projects/B --iou-threshold 0.90 --drift-threshold 10
```

---

## Modes of operation

### Group mode (`--group-a` / `--group-b`)

Point each flag at a directory that contains run folders.  The tool scans
both directories, normalises folder names (strips platform prefixes like
`linux-` or `mac-` and numeric suffixes like `-3`), then matches project
types by name.  Every matched project type is compared across all run
permutations.

```
projects/Mac/
  Canyon-1/PlotsFire/
  Canyon-2/PlotsFire/
  Canyon-3/PlotsFire/

projects/Linux/
  linux-Canyon-1/PlotsFire/
  linux-Canyon-2/PlotsFire/
```

Both groups normalise to `Canyon` and are matched automatically.

### Project mode (`--project-a` / `--project-b`)

Compare two individual run directories (or directories containing multiple
runs).  The tool auto-detects whether each path is a single run with a
`PlotsFire/` subdirectory, a directory of PNG files itself, or a parent
containing multiple run subdirectories.

```bash
python3 scripts/qf-compare.py \
    --project-a projects/Mac/Canyon-1 \
    --project-b projects/Linux/linux-Canyon-1 \
    --output csv,html
```

---

## What it measures

### Perceptual metrics (GPU-accelerated)

Every pair of images is evaluated with five complementary metrics.  All
computation happens on GPU tensors (or CPU tensors as fallback) via
PyTorch.

#### MAE (Mean Absolute Error)

The simplest pixel-level metric.  For two images A and B with N total
pixel-channel values:

```
MAE = (1/N) * sum(|A_i - B_i|)
```

Operates on float32 tensors.  Range: [0, 255]; 0 = identical.

#### RMSE (Root Mean Square Error)

Same as MAE but squares differences before averaging, so large localised
errors are penalised more heavily:

```
RMSE = sqrt( (1/N) * sum((A_i - B_i)^2) )
```

Range: [0, 255].

#### PSNR (Peak Signal-to-Noise Ratio)

Derived directly from RMSE:

```
PSNR = 20 * log10(255 / RMSE)
```

Higher values indicate more similarity.  Identical images produce infinity.

#### SSIM (Structural Similarity Index)

This is the most involved metric and the primary reason the tool benefits
from GPU acceleration.

SSIM (Wang et al., 2004) goes beyond per-pixel differences by comparing
**luminance**, **contrast**, and **structure** within local windows.  The
implementation works as follows:

1. **Gaussian kernel construction.**  A 1-D Gaussian with `window_size=11`
   and `sigma=1.5` is created, then the outer product forms an 11x11 2-D
   kernel.  This kernel is replicated for each RGB channel to produce a
   depthwise convolution filter of shape `(3, 1, 11, 11)`.

2. **Local statistics via depthwise convolution.**  The kernel is convolved
   over both images using `F.conv2d` with `groups=C` (one filter per
   channel).  Six convolutions produce the local means (mu1, mu2), local
   variances (sigma1_sq, sigma2_sq), and local covariance (sigma12) at
   every spatial position.

3. **SSIM map.**  The per-pixel SSIM is computed from the local statistics:

   ```
   SSIM(x,y) = (2*mu1*mu2 + C1)(2*sigma12 + C2)
               / (mu1^2 + mu2^2 + C1)(sigma1_sq + sigma2_sq + C2)
   ```

   where `C1 = (0.01 * 255)^2` and `C2 = (0.03 * 255)^2` are stabilisation
   constants.

4. **Per-image score.**  The SSIM map is averaged over all spatial positions
   and channels to produce a single scalar per image pair.

SSIM is computed in **sub-batches of 8** (grouped by image size) because the
six intermediate convolution buffers require roughly 6x the memory of the
source images.  Between sub-batches the device cache is cleared to avoid
out-of-memory errors.

If an image dimension is smaller than 3 pixels, SSIM is set to 1.0 (the
Gaussian window cannot fit).

Range: [-1, 1]; 1 = identical.

#### Histogram correlation

Computes the Pearson correlation coefficient between the normalised colour
histograms of two images:

1. Each image channel (R, G, B) is binned into 256 bins via `torch.histc`,
   then the three channel histograms are concatenated into a single 768-bin
   vector.
2. Each histogram is normalised to sum to 1.
3. Mean-centred histograms are correlated: `r = sum(h1c * h2c) / sqrt(sum(h1c^2) * sum(h2c^2))`.

This metric captures global colour-distribution agreement regardless of
spatial arrangement.  Range: [-1, 1]; 1 = identical distributions.

On Apple Silicon (MPS), `torch.histc` may not be supported.  The tool
catches the `RuntimeError` and falls back to computing the histogram on
CPU before transferring the result back to the device.

### Fire region metrics

| Metric | Description |
|--------|-------------|
| **Fire IoU** | Intersection over Union of segmented fire regions. Range: [0, 1]; 1 = identical fire boundaries. |
| **Fire Area Diff** | Absolute difference in fire pixel count. Large values indicate spread-rate bugs. |
| **Centroid Drift** | Euclidean distance between fire region centroids (pixels). |

Fire pixels are detected by thresholding the specified channels
(`--fire-channels`, default `rg`) against `--fire-threshold` (default 30).

### Rendering-region masking (v2.1)

Cross-platform comparisons between macOS and Linux reveal that most
"failures" are matplotlib rendering differences (font rendering,
anti-aliasing), not simulation bugs.  This enhancement masks out known
rendering regions before computing metrics.

**Enabled by default.**  Disable with `--no-mask-rendering` and/or
`--no-mask-boundaries`.

#### What is masked

| Region | Default size | Rationale |
|--------|-------------|-----------|
| **Top border** | 60 px | Title text (CoreText vs FreeType rendering) |
| **Bottom border** | 60 px | X-axis labels and tick marks |
| **Left border** | 80 px | Y-axis labels and tick marks |
| **Right border** | 100 px | Colorbar, colorbar labels |
| **Fire boundaries** | 2 px erosion | Anti-aliased edge pixels differ at sub-pixel level |

#### How it works

1. **`create_rendering_mask()`** builds a boolean mask the size of the
   image, setting `False` for the border strips (top, bottom, left, right).
   All pixels inside the remaining rectangle are `True` (included in
   comparison).

2. **`create_interior_mask()`** segments fire regions in both images using
   the same threshold as fire metrics, then applies
   `scipy.ndimage.binary_erosion` to both the fire masks and background
   masks.  Only pixels that are confidently interior (fire-in-both or
   background-in-both) after erosion are marked `True`.

3. **`compute_masked_metrics()`** combines both masks via logical AND, then
   computes MAE and RMSE only on the `True` pixels.  It also reports
   **mask coverage** (fraction of total pixels included, typically 0.4-0.7).

4. A separate pass computes the **rendering-region-only MAE** (metrics on
   the `~rendering_mask` pixels) as a diagnostic to show how much error
   lives in the rendering regions.

Both masked and unmasked metrics are reported in all output formats for
transparency.  The deviation score and verdict use **masked metrics** by
default.

#### Tuning border sizes

If your plots have non-standard layouts (e.g., no colorbar, extra-large
titles), adjust the border sizes:

```bash
python3 scripts/qf-compare.py \
    --group-a projects/Mac --group-b projects/Linux \
    --border-top 80 --border-right 0 --erode-boundary 3
```

### Edge and boundary comparison (v2.1)

Compares structural features between images using Sobel edge detection and
fire-boundary analysis.  Disabled with `--no-edge-metrics`.

#### Edge metrics

| Metric | Description |
|--------|-------------|
| **Edge MAE** | Mean absolute difference of Sobel edge magnitude maps. |
| **Edge Correlation** | Pearson correlation of edge magnitude maps. 1 = identical structure. |
| **Edge IoU** | Overlap of thresholded edge pixels (threshold = mean + 1 std). |

**Implementation:** Each image is converted to grayscale, then
`scipy.ndimage.sobel` is applied along both axes.  Edge magnitude is
`sqrt(sx^2 + sy^2)`.  The two magnitude maps are compared via MAE,
Pearson correlation, and thresholded IoU.

#### Fire boundary metrics

| Metric | Description |
|--------|-------------|
| **Boundary IoU** | Overlap of fire-front perimeter pixels. |
| **Boundary Length Diff** | Absolute difference in fire perimeter pixel count. |
| **Hausdorff (approx)** | Maximum distance from any boundary pixel in A to the nearest boundary pixel in B. |

**Implementation:** Fire regions are segmented, then boundaries are
extracted as `mask & ~erode(mask)`.  The approximate Hausdorff distance
iteratively dilates one boundary and checks when it fully covers the
other, up to a maximum of 50 pixels.

### Multi-scale analysis (v2.1)

For flagged images, errors are analyzed at multiple resolutions to
distinguish structural bugs from fine-grained rendering differences.

**Implementation:** The image pair is repeatedly downsampled by 2x using
`PIL.Image.resize` (bilinear), and MAE + correlation are computed at each
level.  The `analyze_scale_pattern()` function then classifies the error:

| Pattern | `fine_to_coarse_ratio` | Interpretation |
|---------|------------------------|----------------|
| **Fine-grained** | > 2.0 | Errors concentrate at original resolution (rendering/anti-aliasing) |
| **Structural** | < 0.5 | Errors concentrate at coarse resolution (algorithm/computation bug) |
| **Uniform** | 0.5 - 2.0 | Errors span all scales (systematic difference) |

Configure pyramid depth with `--multiscale-levels` (default 4).  Analysis
stops early if either dimension falls below 32 pixels.

### Temporal analysis

#### Monotonicity checks

Fire area should not decrease significantly over time.  The tool groups
fire area by run and timestep, then flags any timestep where fire area
drops by more than 5%.  Violations may indicate numerical instability,
integration bugs, or state corruption.

#### MAE temporal gradient

A linear fit of cross-group MAE vs timestep detects error accumulation
over simulation time.  A positive gradient suggests compounding numerical
errors.

#### Detailed temporal gradient analysis (v2.1)

For projects with 6+ cross-group timesteps, the tool fits both linear and
exponential models:

- **Linear:** `MAE = a*t + b` via `numpy.polyfit`
- **Exponential:** `MAE = a * exp(b*t) + c` via `scipy.optimize.curve_fit`

It reports which model fits better (R-squared comparison), segments the
simulation into early/mid/late phases, and computes acceleration (second
derivative via quadratic fit).  Exponential error growth triggers a warning
in the HTML report.

### Difference heatmaps

For flagged images, pixel-wise difference heatmaps are generated showing
where the two images differ most.  The color scale uses a fire-like
colormap: black (no difference) to red to yellow to white (maximum
difference).  Heatmaps are saved as PNG files and embedded as base64 in
the HTML report.

Disable with `--no-heatmaps`.

---

## GPU, MPS, and CPU execution

### Device selection

The tool automatically selects the best available compute backend at
startup, in priority order:

| Priority | Backend | PyTorch device | When selected |
|----------|---------|----------------|---------------|
| 1 | **CUDA** | `cuda` | `torch.cuda.is_available()` returns True (NVIDIA GPU with CUDA drivers) |
| 2 | **MPS** | `mps` | `torch.backends.mps.is_available()` returns True (Apple Silicon Mac) |
| 3 | **CPU** | `cpu` | Neither GPU backend available |

No flags are needed -- the tool logs which device it selected.

### What runs on the GPU

All tensor-heavy work runs on the selected device:

- **MAE/RMSE/PSNR**: element-wise operations on `(B, 3, H, W)` batches
- **SSIM**: six depthwise `F.conv2d` calls per sub-batch
- **Histogram correlation**: `torch.histc` per channel per image

The following run on CPU (not worth GPU overhead for their data sizes):

- **Fire segmentation and IoU**: simple thresholding on numpy arrays
- **Rendering-region masking**: boolean mask construction and masked MAE/RMSE
- **Edge metrics**: `scipy.ndimage.sobel` on grayscale numpy arrays
- **Multi-scale analysis**: PIL resize + numpy MAE at each level

Image loading (PIL decode, numpy conversion) always runs on CPU using a
`ThreadPoolExecutor` with `--io-workers` threads (default 8).  This
overlaps disk I/O with GPU computation.

### Tuning parameters

| Flag | Default | Effect |
|------|---------|--------|
| `--batch-size` | 32 | Number of image pairs loaded, padded, and transferred to GPU per batch.  Larger batches improve throughput but increase VRAM usage.  With 16 GB VRAM, 32 is conservative; 64-128 may work for smaller images. |
| `--io-workers` | 8 | Thread pool size for parallel image loading.  Increase on systems with fast NVMe storage; decrease if you see CPU memory pressure. |

### Memory management

- Images within a batch are cropped to matching dimensions, then zero-padded
  to the batch-maximum height and width, so a single contiguous tensor can
  be transferred to the GPU.
- SSIM is sub-batched (8 pairs at a time, grouped by image size) to limit
  intermediate memory.
- After MAE/RMSE computation and after each SSIM sub-batch, the device
  cache is cleared (`torch.cuda.empty_cache()` on CUDA,
  `torch.mps.empty_cache()` on MPS if available).
- Numpy arrays are explicitly deleted after tensor construction to free
  CPU memory.

### MPS-specific considerations

Apple's Metal Performance Shaders backend uses the same PyTorch tensor API
as CUDA, so all convolution and arithmetic operations work without
modification.  The one known gap is `torch.histc`, which may raise a
`RuntimeError` on MPS.  The histogram correlation function catches this
and transparently falls back to computing the histogram on CPU, then
transfers the result tensor back to the MPS device.

### CPU fallback

When no GPU is available the tool logs a warning and runs everything on
`torch.device("cpu")`.  The same batched tensor code executes -- PyTorch
handles dispatch -- but throughput drops significantly because the SSIM
convolutions and bulk arithmetic cannot be parallelised across GPU cores.
Reducing `--batch-size` to 8-16 is recommended on CPU to avoid excessive
memory use from padding.

---

## Comparison design

For each matched project type the tool constructs three sets of pair-wise
comparisons:

| Pair type | Pairs drawn from | Purpose |
|-----------|------------------|---------|
| **Intra-A** | `combinations(group_A_runs, 2)` | Baseline variability within Group A |
| **Intra-B** | `combinations(group_B_runs, 2)` | Baseline variability within Group B |
| **Cross** | `product(group_A_runs, group_B_runs)` | Cross-group differences |

Only PNG files present in **all** runs of a project are compared (set
intersection).  Static frames (terrain elevation, fuel height, initial
ignitions) are excluded.

### Image categorization (v3.0)

Each output PNG is classified by type based on its filename category:

| Type | Categories |
|------|-----------|
| **fire** | `perc_mass_burnt`, `bw_perc_mass_burnt`, `fuel_dens_Plane`, `wplume_Plane` |
| **wind** | `u_qu_*`, `v_qu_*`, `w_qu_*` |
| **emissions** | `co_emissions`, `pm_emissions` |
| **other** | Everything else |

**Only fire-category images drive verdicts.** Wind field images use
colormap auto-scaling that amplifies tiny float differences into large
pixel changes, making SSIM unreliable for those categories.

### Fire-metric verdicts (v3.0)

Verdicts are based on absolute fire-metric thresholds applied to
fire-category cross-group comparisons:

| Criterion | PASS | WARN | FAIL |
|-----------|------|------|------|
| **Fire IoU** | >= 0.95 | >= 0.85 | < 0.85 |
| **Centroid Drift** (px) | <= 5.0 | <= 10.0 | > 10.0 |
| **Area Diff %** | <= 5% | <= 15% | > 15% |
| **Temporal** | No violations | Violations | -- |

The overall project verdict is the **worst** of the per-criterion verdicts.
If no fire-category images exist, the verdict is **SKIP**.

All thresholds are configurable via CLI flags (`--iou-threshold`,
`--drift-threshold`, `--area-diff-threshold`, etc.).

### SSIM deviation score (informational)

The SSIM-based deviation score is still computed and reported for
reference, but it no longer drives verdicts:

```
deviation = (intra_SSIM_mean - cross_SSIM_mean) / intra_SSIM_std
```

### Flagged images

Fire-category images whose average cross-group IoU falls below the IoU
threshold (`--iou-threshold`, default 0.95) are flagged for manual review.
Up to 20 flagged images per project are reported, sorted by ascending IoU
(worst first).  Each flagged image receives:

- A difference heatmap (unless `--no-heatmaps`)
- Multi-scale analysis classifying errors as fine-grained, structural, or uniform

---

## Output formats

| Format | File | Description |
|--------|------|-------------|
| Console | (stdout) | Always printed.  Summary table with fire IoU, drift, area diff %, per-criterion verdicts, SSIM (info), and flagged-image details. |
| HTML | `report.html` | Full report with methodology section, fire-metric verdict table, summary table, per-project metrics with per-criterion verdicts, time-series tables, flagged images with heatmaps and scale analysis, temporal gradient details, and per-category breakdowns. |
| Markdown | `report.md` | Same logical structure as HTML in GFM pipe-table format. |
| CSV | `comparison_results.csv` | One row per image pair with all metrics (see below). |

Select formats with `--output`:

```bash
--output html              # default
--output csv,html,md       # multiple
--output all               # all formats
```

### CSV columns

The CSV includes the following columns:

| Column group | Columns |
|-------------|---------|
| **Identity** | `project`, `png_name`, `category`, `image_type`, `timestep`, `run_a`, `run_b`, `pair_type` |
| **Perceptual** | `ssim`, `hist_corr` |
| **Fire** | `fire_iou`, `fire_area_a`, `fire_area_b`, `fire_area_diff`, `centroid_drift` |
| **Masked** | `masked_ssim`, `mask_coverage` |
| **Edge** | `edge_correlation`, `edge_iou`, `boundary_iou`, `boundary_length_diff`, `hausdorff_approx` |

---

## Setup check

```bash
python3 scripts/qf-compare.py \
    --group-a projects/Mac \
    --group-b projects/Linux \
    --check-setup
```

Prints a diagnostic report covering:

- **System**: PyTorch version, CPU cores, GPU name and VRAM
- **Enhanced features**: Lists v2.1 capabilities and scipy availability
- **Groups**: project types discovered in each directory, run counts
- **Matching**: matched and unmatched project types, PNG counts per project,
  pair counts, and total estimated comparisons

No comparison is performed.  Exit code is always 0.

---

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | All projects PASS (or `--check-setup`) |
| 1 | No data found or argument error |
| 2 | One or more projects received a FAIL verdict |

---

## Requirements

- Python 3.10+
- PyTorch (any recent version with `torch.nn.functional.conv2d`)
- NumPy
- Pillow
- SciPy >= 1.10.0

SciPy is a hard requirement for v2.1.  It provides:

| Module | Used for |
|--------|----------|
| `scipy.ndimage.binary_erosion` | Interior mask construction (fire boundary erosion) |
| `scipy.ndimage.binary_dilation` | Boundary extraction, approximate Hausdorff distance |
| `scipy.ndimage.sobel` | Edge detection for edge comparison metrics |
| `scipy.optimize.curve_fit` | Exponential model fitting for temporal gradient analysis |

Install all dependencies:

```bash
pip install torch numpy pillow scipy
```
