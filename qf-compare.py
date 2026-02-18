#!/usr/bin/env python3
"""
qf-compare — General-Purpose QUIC-Fire Output Comparison Tool (GPU-accelerated)

Enhanced version (v2.1) with:
  - Difference heatmaps for flagged images
  - Fire region segmentation and IoU metrics
  - Temporal monotonicity checks for physics violations
  - Rendering-region masking (border/colorbar/title exclusion)
  - Interior-only comparison (anti-aliased boundary erosion)
  - Edge & boundary comparison (Sobel, Hausdorff distance)
  - Multi-scale analysis (pyramid downsampling)

Compares PlotsFire PNG outputs between any two groups of QUIC-Fire simulation
runs, detecting statistically significant deviations.  Supports CUDA, Apple
Silicon (MPS), and CPU backends.

Modes
-----
  Group mode:   --group-a <dir> --group-b <dir>
  Project mode:  --project-a <dir> --project-b <dir>

Output formats: console (always), html, md, csv  (--output)
"""

# ───────────────────────────────────────────────────────────────────────────
#  Imports
# ───────────────────────────────────────────────────────────────────────────

import csv
import html as html_mod
import itertools
import logging
import math
import os
import re
import sys
import time
import argparse
import base64
from io import BytesIO
from pathlib import Path
from typing import NamedTuple
from dataclasses import dataclass, field
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import binary_erosion, binary_dilation, sobel
from scipy.optimize import curve_fit

STATIC_PATTERNS = re.compile(
    r"^(TerrainElevation|TerrainElevation3d|fuel_height_ground_level|InitialIgnitions)\b"
)
TIME_RE = re.compile(r"Time_(\d+)_s\.png$")
DEFAULT_SIGMA_THRESHOLD = 2.0
DEFAULT_BATCH_SIZE = 32

# Fire segmentation defaults (can be tuned via CLI)
DEFAULT_FIRE_THRESHOLD = 30  # intensity threshold for fire pixels
DEFAULT_FIRE_CHANNELS = "rg"  # which channels to check ('r', 'g', 'b', 'rg', 'rgb')

# Rendering-region masking defaults
DEFAULT_MASK_BORDER = True
DEFAULT_BORDER_TOP = 60      # Title region height (pixels)
DEFAULT_BORDER_BOTTOM = 60   # X-axis label region height
DEFAULT_BORDER_LEFT = 80     # Y-axis label region width
DEFAULT_BORDER_RIGHT = 100   # Colorbar region width
DEFAULT_ERODE_BOUNDARY = 2   # Pixels to erode from fire boundary

# Fire-metric verdict threshold defaults
DEFAULT_IOU_PASS = 0.95
DEFAULT_IOU_WARN = 0.85
DEFAULT_DRIFT_PASS = 5.0       # pixels
DEFAULT_DRIFT_WARN = 10.0
DEFAULT_AREA_DIFF_PASS = 0.05  # 5% relative
DEFAULT_AREA_DIFF_WARN = 0.15  # 15%

# ANSI colour helpers (disabled when stdout is not a terminal)
_USE_COLOR = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _red(text: str) -> str:
    return f"\033[91m{text}\033[0m" if _USE_COLOR else text


def _green(text: str) -> str:
    return f"\033[92m{text}\033[0m" if _USE_COLOR else text


def _yellow(text: str) -> str:
    return f"\033[93m{text}\033[0m" if _USE_COLOR else text


class _ColorFormatter(logging.Formatter):
    """Logging formatter that colours ERROR and CRITICAL messages red."""

    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt)

    def format(self, record):
        msg = super().format(record)
        if record.levelno >= logging.ERROR and _USE_COLOR:
            return f"\033[91m{msg}\033[0m"
        if record.levelno >= logging.WARNING and _USE_COLOR:
            return f"\033[93m{msg}\033[0m"
        return msg


_handler = logging.StreamHandler()
_handler.setFormatter(_ColorFormatter(
    fmt="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
))
logging.basicConfig(level=logging.INFO, handlers=[_handler])
log = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────────
#  Methodology Text
# ───────────────────────────────────────────────────────────────────────────

METHODOLOGY_TEXT_MD = """\
## Methodology

This report compares QUIC-Fire simulation outputs between two groups of runs
(Group A and Group B) using fire-region metrics computed on the PlotsFire
PNG renderings.

### Image Categorization

Each output image is classified by type:

| Type | Categories |
|------|-----------|
| **fire** | `perc_mass_burnt`, `bw_perc_mass_burnt`, `fuel_dens_Plane`, `wplume_Plane` |
| **wind** | `u_qu_*`, `v_qu_*`, `w_qu_*` |
| **emissions** | `co_emissions`, `pm_emissions` |
| **other** | Everything else |

**Verdicts are driven exclusively by fire-category images.**  Wind field
images use colormap auto-scaling that amplifies tiny float differences into
large pixel changes; they are reported for information only.

### Fire-Metric Verdicts

| Criterion | PASS | WARN | FAIL |
|-----------|------|------|------|
| **Fire IoU** (Intersection over Union) | ≥ 0.95 | ≥ 0.85 | < 0.85 |
| **Centroid Drift** (pixels) | ≤ 5.0 | ≤ 10.0 | > 10.0 |
| **Area Diff %** (relative) | ≤ 5% | ≤ 15% | > 15% |
| **Temporal** | No violations | Violations detected | — |

The overall project verdict is the **worst** of the sub-verdicts.  If no
fire-category images exist, the project verdict is **SKIP**.

### Perceptual Metrics (Informational)

| Metric | Description |
|--------|-------------|
| **SSIM** (Structural Similarity Index) | Measures luminance, contrast and structural similarity (Wang et al., 2004). Range: [-1, 1]; 1 = identical. Reported for reference; not used for verdicts. |
| **Histogram Correlation** | Pearson correlation of normalised colour histograms. Range: [-1, 1]; 1 = identical distributions. |

### Fire Region Metrics

| Metric | Description |
|--------|-------------|
| **Fire IoU** | Overlap of segmented fire regions. Range: [0, 1]; 1 = identical fire boundaries. |
| **Fire Area Diff** | Absolute difference in fire pixel count. Large values indicate spread-rate bugs. |
| **Centroid Drift** | Euclidean distance between fire region centroids. Indicates spatial displacement. |

### Comparison Design

For each matched project the tool constructs three categories of pair-wise
image comparisons:

* **Intra-A** — pairs drawn from runs within Group A (measures Group A's own
  run-to-run variability).
* **Intra-B** — pairs drawn from runs within Group B.
* **Cross** — pairs with one run from Group A and one from Group B.

### Temporal Monotonicity

Fire simulations should exhibit monotonic growth (fire area should not
decrease significantly over time).  The tool detects timesteps where fire
area decreases by more than 5%, which may indicate:
* Numerical instability
* Integration bugs
* State corruption between timesteps

### Flagged Images

Fire-category images whose average cross-group IoU falls below the IoU
threshold are flagged for manual review.  Up to 20 flagged images are
reported per project, sorted by ascending IoU (worst first).

### Difference Heatmaps

For flagged images, pixel-wise difference heatmaps are generated showing
where the two images differ most.  Brighter regions indicate larger errors,
enabling rapid localization of bugs.
"""

METHODOLOGY_TEXT_HTML = METHODOLOGY_TEXT_MD  # rendered via markdown section in HTML


# ───────────────────────────────────────────────────────────────────────────
#  Data Structures
# ───────────────────────────────────────────────────────────────────────────

class Metrics(NamedTuple):
    ssim_val: float
    hist_corr: float


class FireMetrics(NamedTuple):
    """Fire-region-specific metrics."""
    iou: float           # Intersection over Union of fire regions
    area_a: int          # Fire pixel count in image A
    area_b: int          # Fire pixel count in image B
    area_diff: int       # Absolute difference in fire area
    centroid_a: tuple[float, float]  # (y, x) centroid of fire in A
    centroid_b: tuple[float, float]  # (y, x) centroid of fire in B
    centroid_drift: float  # Euclidean distance between centroids


class MaskedMetrics(NamedTuple):
    """Metrics computed with rendering regions masked out."""
    masked_ssim: float       # SSIM computed on masked region
    mask_coverage: float     # Fraction of pixels included (0-1)


class EdgeMetrics(NamedTuple):
    """Edge and boundary comparison metrics."""
    edge_correlation: float
    edge_iou: float
    boundary_iou: float
    boundary_length_diff: float
    hausdorff_approx: float


@dataclass
class TemporalGradientAnalysis:
    """Detailed temporal error analysis."""
    linear_gradient: float          # Overall linear slope
    linear_r_squared: float         # Goodness of fit for linear model
    exponential_rate: float         # Exponential growth rate (if applicable)
    exponential_r_squared: float    # Goodness of fit for exponential model
    best_fit: str                   # "linear", "exponential", or "stable"
    early_phase_ssim: float         # Mean SSIM in first 1/3 of timesteps
    mid_phase_ssim: float           # Mean SSIM in middle 1/3
    late_phase_ssim: float          # Mean SSIM in last 1/3
    acceleration: float             # Second derivative (is error growth accelerating?)


@dataclass
class MultiscaleMetrics:
    """Multi-resolution analysis results."""
    scale_factors: list
    ssims: list
    dominant_scale: str  # "fine", "coarse", "uniform"
    fine_to_coarse_ratio: float
    scale_gradient: float


@dataclass
class ComparisonResult:
    project: str
    png_name: str
    category: str
    timestep: int
    run_a: str
    run_b: str
    platform_pair: str        # "intra_a", "intra_b", or "cross"
    metrics: Metrics
    fire_metrics: FireMetrics | None = None
    masked_metrics: MaskedMetrics | None = None
    edge_metrics: EdgeMetrics | None = None
    path_a: str = ""          # Path to image A (for heatmap generation)
    path_b: str = ""          # Path to image B (for heatmap generation)
    image_type: str = ""      # "fire", "wind", "emissions", or "other"


@dataclass
class TemporalViolation:
    """Records a timestep where fire area decreased (potential physics bug)."""
    timestep: int
    prev_timestep: int
    area_before: float
    area_after: float
    percent_decrease: float
    run_name: str


@dataclass
class ProjectSummary:
    project: str
    n_images: int = 0
    n_comparisons: int = 0
    intra_a_ssim_mean: float = 0.0
    intra_a_ssim_std: float = 0.0
    intra_b_ssim_mean: float = 0.0
    intra_b_ssim_std: float = 0.0
    cross_ssim_mean: float = 0.0
    cross_ssim_std: float = 0.0
    # Fire metrics
    cross_iou_mean: float = 0.0
    cross_iou_std: float = 0.0
    intra_iou_mean: float = 0.0
    cross_centroid_drift_mean: float = 0.0
    # Masked metrics
    cross_masked_ssim_mean: float = 0.0
    mask_coverage_mean: float = 0.0
    # Edge metrics
    cross_boundary_iou_mean: float = 0.0
    cross_hausdorff_mean: float = 0.0
    # Fire-category cross metrics (verdict-driving)
    fire_cross_iou_mean: float = 0.0
    fire_cross_iou_std: float = 0.0
    fire_cross_drift_mean: float = 0.0
    fire_cross_drift_std: float = 0.0
    fire_cross_area_diff_pct_mean: float = 0.0
    fire_cross_area_diff_pct_std: float = 0.0
    fire_n_cross_comparisons: int = 0
    # Temporal
    temporal_violations: list[TemporalViolation] = field(default_factory=list)
    ssim_temporal_gradient: float = 0.0  # Negative = similarity degrading over time
    temporal_gradient_analysis: TemporalGradientAnalysis | None = None
    # Verdict
    deviation_score: float = 0.0
    iou_verdict: str = "N/A"
    drift_verdict: str = "N/A"
    area_diff_verdict: str = "N/A"
    temporal_verdict: str = "N/A"
    verdict: str = "SKIP"
    flagged_images: list = field(default_factory=list)


@dataclass
class FlaggedImageData:
    """Extended data for flagged images including difference heatmap."""
    png_name: str
    avg_ssim: float
    avg_iou: float
    diff_heatmap_base64: str | None = None  # Base64-encoded PNG for HTML embedding
    multiscale_analysis: MultiscaleMetrics | None = None


@dataclass
class TaskInfo:
    """Metadata for a single comparison task (no image data)."""
    project: str
    png_name: str
    category: str
    timestep: int
    run_a: str
    run_b: str
    platform_pair: str
    path_a: str
    path_b: str
    image_type: str = ""      # "fire", "wind", "emissions", or "other"


# ───────────────────────────────────────────────────────────────────────────
#  Device Selection
# ───────────────────────────────────────────────────────────────────────────

def select_device() -> torch.device:
    """Select best available compute device: CUDA → MPS → CPU."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        log.info(f"Device: CUDA — {name}, {mem:.1f} GB")
        return dev
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
        log.info("Device: Apple MPS")
        return dev
    log.warning("No GPU available — falling back to CPU (will be slow)")
    return torch.device("cpu")


def _clear_device_cache(device: torch.device):
    """Clear GPU memory cache if applicable."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        # MPS does not expose a public cache-clear API in all PyTorch versions
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()


# ───────────────────────────────────────────────────────────────────────────
# 5.  Discovery & Matching
# ───────────────────────────────────────────────────────────────────────────

_STRIP_PREFIX_RE = re.compile(r"^(linux|mac|win)-", re.IGNORECASE)
_STRIP_SUFFIX_RE = re.compile(r"-\d+$")


def _extract_project_type(folder_name: str) -> str:
    """Normalise a run folder name to its project type.

    Example: ``linux-Canyon-1`` → ``Canyon``, ``Canyon-3`` → ``Canyon``.
    """
    name = _STRIP_PREFIX_RE.sub("", folder_name)
    name = _STRIP_SUFFIX_RE.sub("", name)
    return name


def resolve_plotsfire(d: Path) -> Path | None:
    """Return the PlotsFire PNG directory for *d*, or ``None``.

    Handles three layouts:
      1. *d*/PlotsFire/   — standard run directory
      2. *d* itself contains .png files — treated as the PNG dir
      3. None found
    """
    pf = d / "PlotsFire"
    if pf.is_dir():
        return pf
    # Maybe the path IS a PNG directory already
    if any(d.glob("*.png")):
        return d
    return None


def discover_group_projects(group_dir: Path) -> dict[str, list[Path]]:
    """Scan *group_dir* for run sub-directories containing PlotsFire/.

    Returns ``{project_type: [run_dir, ...]}`` sorted by name.
    """
    projects: dict[str, list[Path]] = defaultdict(list)
    if not group_dir.is_dir():
        return projects
    for child in sorted(group_dir.iterdir()):
        if not child.is_dir():
            continue
        if resolve_plotsfire(child) is not None:
            pt = _extract_project_type(child.name)
            projects[pt].append(child)
    return dict(projects)


def match_groups(
    proj_a: dict[str, list[Path]],
    proj_b: dict[str, list[Path]],
) -> list[str]:
    """Return sorted list of project types present in both groups."""
    return sorted(set(proj_a.keys()) & set(proj_b.keys()))


def find_common_pngs(png_dirs: list[Path]) -> list[str]:
    """Return sorted list of PNG names common to all *png_dirs*, excluding static frames."""
    if not png_dirs:
        return []
    sets = []
    for d in png_dirs:
        if d is not None and d.is_dir():
            sets.append({p.name for p in d.iterdir() if p.suffix == ".png"})
        else:
            sets.append(set())
    if not sets:
        return []
    common = sets[0]
    for s in sets[1:]:
        common &= s
    common = {n for n in common if TIME_RE.search(n)}
    return sorted(common)


def classify_png(name: str) -> tuple[str, int]:
    m = TIME_RE.search(name)
    timestep = int(m.group(1)) if m else -1
    cat = TIME_RE.sub("", name).rstrip("_").replace(".png", "")
    if cat == "":
        cat = name.replace(".png", "")
    return cat, timestep


_FIRE_RE = re.compile(r"(perc_mass_burnt|bw_perc_mass_burnt|fuel_dens_Plane|wplume_Plane)")
_WIND_RE = re.compile(r"(u_qu_|v_qu_|w_qu_)")
_EMISSIONS_RE = re.compile(r"(co_emissions|pm_emissions)")


def classify_image_type(category: str) -> str:
    """Classify a PNG category into fire/wind/emissions/other."""
    if _FIRE_RE.search(category):
        return "fire"
    if _WIND_RE.search(category):
        return "wind"
    if _EMISSIONS_RE.search(category):
        return "emissions"
    return "other"


# ───────────────────────────────────────────────────────────────────────────
#  Fire Segmentation
# ───────────────────────────────────────────────────────────────────────────

def segment_fire_region(
    img: np.ndarray,
    threshold: int = DEFAULT_FIRE_THRESHOLD,
    channels: str = DEFAULT_FIRE_CHANNELS
) -> np.ndarray:
    """Segment fire pixels based on intensity threshold.
    
    Args:
        img: RGB image as numpy array (H, W, 3)
        threshold: Intensity threshold for fire detection
        channels: Which channels to check - 'r', 'g', 'b', 'rg', 'rgb'
    
    Returns:
        Boolean mask of fire pixels (H, W)
    """
    if img.ndim != 3 or img.shape[2] != 3:
        return np.zeros(img.shape[:2], dtype=bool)
    
    # Build mask based on specified channels
    mask = np.zeros(img.shape[:2], dtype=bool)
    
    if 'r' in channels:
        mask |= (img[:, :, 0] > threshold)
    if 'g' in channels:
        mask |= (img[:, :, 1] > threshold)
    if 'b' in channels:
        mask |= (img[:, :, 2] > threshold)
    
    return mask


def compute_fire_area(mask: np.ndarray) -> int:
    """Count fire pixels in mask."""
    return int(mask.sum())


def compute_fire_centroid(mask: np.ndarray) -> tuple[float, float]:
    """Compute centroid of fire region.
    
    Returns:
        (y, x) centroid coordinates, or (0.0, 0.0) if no fire pixels
    """
    y_indices, x_indices = np.where(mask)
    if len(y_indices) == 0:
        return (0.0, 0.0)
    return (float(y_indices.mean()), float(x_indices.mean()))


def compute_fire_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute Intersection over Union of two fire masks.
    
    Returns:
        IoU in range [0, 1]. Returns 1.0 if both masks are empty.
    """
    intersection = (mask_a & mask_b).sum()
    union = (mask_a | mask_b).sum()
    if union == 0:
        return 1.0  # Both empty = identical
    return float(intersection) / float(union)


def compute_fire_metrics(
    img_a: np.ndarray,
    img_b: np.ndarray,
    threshold: int = DEFAULT_FIRE_THRESHOLD,
    channels: str = DEFAULT_FIRE_CHANNELS
) -> FireMetrics:
    """Compute all fire-region metrics for a pair of images."""
    mask_a = segment_fire_region(img_a, threshold, channels)
    mask_b = segment_fire_region(img_b, threshold, channels)
    
    area_a = compute_fire_area(mask_a)
    area_b = compute_fire_area(mask_b)
    centroid_a = compute_fire_centroid(mask_a)
    centroid_b = compute_fire_centroid(mask_b)
    
    # Centroid drift (Euclidean distance)
    if area_a > 0 and area_b > 0:
        drift = math.sqrt(
            (centroid_a[0] - centroid_b[0])**2 +
            (centroid_a[1] - centroid_b[1])**2
        )
    else:
        drift = 0.0  # Can't compute drift if one image has no fire
    
    return FireMetrics(
        iou=compute_fire_iou(mask_a, mask_b),
        area_a=area_a,
        area_b=area_b,
        area_diff=abs(area_a - area_b),
        centroid_a=centroid_a,
        centroid_b=centroid_b,
        centroid_drift=drift,
    )


# ───────────────────────────────────────────────────────────────────────────
#  Rendering-Region Masking
# ───────────────────────────────────────────────────────────────────────────

def create_rendering_mask(
    img_shape: tuple[int, int],
    border_top: int = DEFAULT_BORDER_TOP,
    border_bottom: int = DEFAULT_BORDER_BOTTOM,
    border_left: int = DEFAULT_BORDER_LEFT,
    border_right: int = DEFAULT_BORDER_RIGHT,
) -> np.ndarray:
    """Create a mask that excludes matplotlib rendering regions.

    Args:
        img_shape: (height, width) of the image
        border_*: Pixel width of each border region to exclude

    Returns:
        Boolean mask where True = include in comparison
    """
    h, w = img_shape
    mask = np.ones((h, w), dtype=bool)

    if border_top > 0 and border_top < h:
        mask[:border_top, :] = False
    if border_bottom > 0 and border_bottom < h:
        mask[-border_bottom:, :] = False
    if border_left > 0 and border_left < w:
        mask[:, :border_left] = False
    if border_right > 0 and border_right < w:
        mask[:, -border_right:] = False

    return mask


def create_interior_mask(
    fire_mask_a: np.ndarray,
    fire_mask_b: np.ndarray,
    erode_iterations: int = DEFAULT_ERODE_BOUNDARY,
) -> np.ndarray:
    """Create mask excluding boundary pixels where anti-aliasing differs.

    Args:
        fire_mask_a: Boolean fire region mask for image A
        fire_mask_b: Boolean fire region mask for image B
        erode_iterations: How many pixels to erode from boundaries

    Returns:
        Boolean mask of interior pixels safe for comparison
    """
    if erode_iterations <= 0:
        return fire_mask_a | fire_mask_b

    # Erode both masks to get interior regions
    interior_a = binary_erosion(fire_mask_a, iterations=erode_iterations)
    interior_b = binary_erosion(fire_mask_b, iterations=erode_iterations)

    # Also erode the "background" (non-fire) regions
    background_a = binary_erosion(~fire_mask_a, iterations=erode_iterations)
    background_b = binary_erosion(~fire_mask_b, iterations=erode_iterations)

    # Valid comparison pixels: interior fire OR interior background
    interior_fire = interior_a & interior_b
    interior_bg = background_a & background_b

    return interior_fire | interior_bg


def compute_masked_metrics(
    img_a: np.ndarray,
    img_b: np.ndarray,
    rendering_mask: np.ndarray | None = None,
    interior_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute pixel-level correlation with masking applied.

    Args:
        img_a, img_b: RGB images as numpy arrays (H, W, 3)
        rendering_mask: Mask excluding plot borders (True = include)
        interior_mask: Mask excluding anti-aliased boundaries (True = include)

    Returns:
        Dictionary with masked_ssim (approximated via correlation), mask_coverage
    """
    h, w = img_a.shape[:2]

    # Combine masks
    combined_mask = np.ones((h, w), dtype=bool)
    if rendering_mask is not None:
        combined_mask &= rendering_mask
    if interior_mask is not None:
        combined_mask &= interior_mask

    total_pixels = h * w
    masked_pixels = combined_mask.sum()
    coverage = masked_pixels / total_pixels if total_pixels > 0 else 0.0

    if masked_pixels == 0:
        return {
            "masked_ssim": 1.0,
            "mask_coverage": 0.0,
        }

    mask_3d = np.stack([combined_mask] * 3, axis=2)
    a_vals = img_a.astype(np.float32)[mask_3d]
    b_vals = img_b.astype(np.float32)[mask_3d]
    a_centered = a_vals - a_vals.mean()
    b_centered = b_vals - b_vals.mean()
    num = (a_centered * b_centered).sum()
    den = np.sqrt((a_centered ** 2).sum() * (b_centered ** 2).sum())
    corr = float(num / den) if den > 0 else 1.0

    return {
        "masked_ssim": corr,
        "mask_coverage": coverage,
    }


# ───────────────────────────────────────────────────────────────────────────
#  Edge Comparison
# ───────────────────────────────────────────────────────────────────────────

def compute_edge_metrics(
    img_a: np.ndarray,
    img_b: np.ndarray,
) -> dict[str, float]:
    """Compare edge structures between two images.

    Uses Sobel operators to extract edges, then compares:
    - Edge correlation: Pearson correlation of edge maps
    - Edge IoU: Overlap of thresholded edge pixels
    """
    gray_a = img_a.astype(np.float32).mean(axis=2)
    gray_b = img_b.astype(np.float32).mean(axis=2)

    def edge_magnitude(gray):
        sx = sobel(gray, axis=0)
        sy = sobel(gray, axis=1)
        return np.hypot(sx, sy)

    edges_a = edge_magnitude(gray_a)
    edges_b = edge_magnitude(gray_b)

    ea_flat = edges_a.flatten()
    eb_flat = edges_b.flatten()
    ea_centered = ea_flat - ea_flat.mean()
    eb_centered = eb_flat - eb_flat.mean()
    num = (ea_centered * eb_centered).sum()
    den = np.sqrt((ea_centered ** 2).sum() * (eb_centered ** 2).sum())
    edge_correlation = float(num / den) if den > 0 else 1.0

    threshold_a = edges_a.mean() + edges_a.std()
    threshold_b = edges_b.mean() + edges_b.std()
    threshold = (threshold_a + threshold_b) / 2

    edge_mask_a = edges_a > threshold
    edge_mask_b = edges_b > threshold

    intersection = (edge_mask_a & edge_mask_b).sum()
    union = (edge_mask_a | edge_mask_b).sum()
    edge_iou = float(intersection / union) if union > 0 else 1.0

    return {
        "edge_correlation": edge_correlation,
        "edge_iou": edge_iou,
    }


def compute_fire_boundary_metrics(
    img_a: np.ndarray,
    img_b: np.ndarray,
    fire_threshold: int = 30,
) -> dict[str, float]:
    """Compare fire boundary contours specifically.

    More targeted than general edge comparison - focuses on
    the perimeter of the fire region.
    """
    mask_a = (img_a[:, :, 0] > fire_threshold) | (img_a[:, :, 1] > fire_threshold)
    mask_b = (img_b[:, :, 0] > fire_threshold) | (img_b[:, :, 1] > fire_threshold)

    def get_boundary(mask):
        eroded = binary_erosion(mask)
        return mask & ~eroded

    boundary_a = get_boundary(mask_a)
    boundary_b = get_boundary(mask_b)

    length_a = boundary_a.sum()
    length_b = boundary_b.sum()
    boundary_length_diff = abs(length_a - length_b)

    intersection = (boundary_a & boundary_b).sum()
    union = (boundary_a | boundary_b).sum()
    boundary_iou = float(intersection / union) if union > 0 else 1.0

    def approx_hausdorff(boundary_from, boundary_to, max_dist=50):
        if boundary_from.sum() == 0 or boundary_to.sum() == 0:
            return 0.0
        dilated = boundary_to.copy()
        for dist in range(1, max_dist + 1):
            dilated = binary_dilation(dilated)
            if (boundary_from & dilated).sum() == boundary_from.sum():
                return float(dist)
        return float(max_dist)

    hausdorff_a_to_b = approx_hausdorff(boundary_a, boundary_b)
    hausdorff_b_to_a = approx_hausdorff(boundary_b, boundary_a)
    hausdorff_approx = max(hausdorff_a_to_b, hausdorff_b_to_a)

    return {
        "boundary_iou": boundary_iou,
        "boundary_length_diff": float(boundary_length_diff),
        "hausdorff_approx": hausdorff_approx,
    }


# ───────────────────────────────────────────────────────────────────────────
#  Multi-scale Analysis
# ───────────────────────────────────────────────────────────────────────────

def compute_multiscale_metrics(
    img_a: np.ndarray,
    img_b: np.ndarray,
    levels: int = 4,
    min_size: int = 32,
) -> dict[str, list[float]]:
    """Compute correlation at multiple resolutions."""
    pil_a = Image.fromarray(img_a)
    pil_b = Image.fromarray(img_b)

    scale_factors = []
    ssims = []

    current_a = pil_a
    current_b = pil_b
    scale = 1.0

    for level in range(levels):
        arr_a = np.array(current_a).astype(np.float32)
        arr_b = np.array(current_b).astype(np.float32)

        a_flat = arr_a.flatten()
        b_flat = arr_b.flatten()
        a_centered = a_flat - a_flat.mean()
        b_centered = b_flat - b_flat.mean()
        num = (a_centered * b_centered).sum()
        den = np.sqrt((a_centered ** 2).sum() * (b_centered ** 2).sum())
        corr = float(num / den) if den > 0 else 1.0
        ssims.append(corr)

        scale_factors.append(scale)

        new_w = current_a.width // 2
        new_h = current_a.height // 2

        if new_w < min_size or new_h < min_size:
            break

        current_a = current_a.resize((new_w, new_h), Image.BILINEAR)
        current_b = current_b.resize((new_w, new_h), Image.BILINEAR)
        scale *= 0.5

    return {
        "scale_factors": scale_factors,
        "ssims": ssims,
    }


def analyze_scale_pattern(multiscale: dict[str, list[float]]) -> dict:
    """Analyze the pattern of similarity across scales."""
    ssims = multiscale["ssims"]

    if len(ssims) < 2:
        return {
            "dominant_scale": "uniform",
            "fine_to_coarse_ratio": 1.0,
            "scale_gradient": 0.0,
        }

    # Use (1 - ssim) as error proxy for scale analysis
    fine_err = 1.0 - ssims[0]
    coarse_err = 1.0 - ssims[-1]

    ratio = fine_err / coarse_err if coarse_err > 0.001 else float('inf')

    try:
        errors = [1.0 - s for s in ssims]
        gradient = np.polyfit(range(len(errors)), errors, 1)[0]
    except (np.linalg.LinAlgError, ValueError):
        gradient = 0.0

    if ratio > 2.0:
        dominant_scale = "fine"
    elif ratio < 0.5:
        dominant_scale = "coarse"
    else:
        dominant_scale = "uniform"

    return {
        "dominant_scale": dominant_scale,
        "fine_to_coarse_ratio": float(ratio),
        "scale_gradient": float(gradient),
    }


# ───────────────────────────────────────────────────────────────────────────
#  Difference Heatmap Generation
# ───────────────────────────────────────────────────────────────────────────

def compute_difference_heatmap(
    img_a: np.ndarray,
    img_b: np.ndarray,
    normalize: bool = True
) -> np.ndarray:
    """Compute pixel-wise difference as a heatmap.
    
    Args:
        img_a, img_b: RGB images as numpy arrays
        normalize: If True, normalize to 0-255 range
    
    Returns:
        Single-channel heatmap (H, W) as uint8
    """
    # Compute absolute difference
    diff = np.abs(img_a.astype(np.float32) - img_b.astype(np.float32))
    
    # Take max across channels to highlight any channel difference
    heatmap = diff.max(axis=2)
    
    if normalize and heatmap.max() > 0:
        heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
    else:
        heatmap = heatmap.astype(np.uint8)
    
    return heatmap


def heatmap_to_colormap(heatmap: np.ndarray) -> np.ndarray:
    """Convert grayscale heatmap to a color-mapped RGB image.
    
    Uses a fire-like colormap: black → red → yellow → white
    """
    h, w = heatmap.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Normalize to 0-1
    normalized = heatmap.astype(np.float32) / 255.0
    
    # Fire colormap: black → red → yellow → white
    # 0.0-0.33: black to red
    # 0.33-0.66: red to yellow
    # 0.66-1.0: yellow to white
    
    mask1 = normalized <= 0.33
    mask2 = (normalized > 0.33) & (normalized <= 0.66)
    mask3 = normalized > 0.66
    
    # Black to red
    t1 = normalized[mask1] / 0.33
    rgb[mask1, 0] = (t1 * 255).astype(np.uint8)
    
    # Red to yellow
    t2 = (normalized[mask2] - 0.33) / 0.33
    rgb[mask2, 0] = 255
    rgb[mask2, 1] = (t2 * 255).astype(np.uint8)
    
    # Yellow to white
    t3 = (normalized[mask3] - 0.66) / 0.34
    rgb[mask3, 0] = 255
    rgb[mask3, 1] = 255
    rgb[mask3, 2] = (t3 * 255).astype(np.uint8)
    
    return rgb


def generate_difference_image(
    img_a: np.ndarray,
    img_b: np.ndarray,
    include_original: bool = True
) -> Image.Image:
    """Generate a comparison image showing originals and difference heatmap.
    
    Args:
        img_a, img_b: RGB images as numpy arrays
        include_original: If True, create side-by-side with originals
    
    Returns:
        PIL Image ready for saving
    """
    heatmap = compute_difference_heatmap(img_a, img_b)
    heatmap_rgb = heatmap_to_colormap(heatmap)
    
    if not include_original:
        return Image.fromarray(heatmap_rgb)
    
    # Create side-by-side comparison: [A | B | Diff]
    h = max(img_a.shape[0], img_b.shape[0])
    w_a, w_b = img_a.shape[1], img_b.shape[1]
    w_total = w_a + w_b + w_a  # Same width as A for heatmap
    
    combined = np.zeros((h, w_total, 3), dtype=np.uint8)
    combined[:img_a.shape[0], :w_a] = img_a
    combined[:img_b.shape[0], w_a:w_a+w_b] = img_b
    combined[:heatmap_rgb.shape[0], w_a+w_b:w_a+w_b+heatmap_rgb.shape[1]] = heatmap_rgb
    
    return Image.fromarray(combined)


def image_to_base64(img: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 string for HTML embedding."""
    buffer = BytesIO()
    img.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


# ───────────────────────────────────────────────────────────────────────────
#  GPU Kernels
# ───────────────────────────────────────────────────────────────────────────

def _fspecial_gauss_1d(size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g


def _create_ssim_kernel(window_size: int = 11, sigma: float = 1.5,
                         channels: int = 3) -> torch.Tensor:
    g1d = _fspecial_gauss_1d(window_size, sigma)
    g2d = g1d.unsqueeze(1) @ g1d.unsqueeze(0)
    kernel = g2d.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
    return kernel


def ssim_batch(img1: torch.Tensor, img2: torch.Tensor,
               kernel: torch.Tensor, window_size: int = 11,
               data_range: float = 255.0) -> torch.Tensor:
    C = img1.shape[1]
    pad = window_size // 2

    mu1 = F.conv2d(img1, kernel, padding=pad, groups=C)
    mu2 = F.conv2d(img2, kernel, padding=pad, groups=C)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, kernel, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, kernel, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel, padding=pad, groups=C) - mu1_mu2

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean(dim=(1, 2, 3))


def histogram_correlation_batch(
    img1: torch.Tensor, img2: torch.Tensor
) -> torch.Tensor:
    B, C, H, W = img1.shape
    results = torch.zeros(B, device=img1.device)

    for i in range(B):
        hists = []
        for im in (img1[i], img2[i]):
            ch_hists = []
            for c in range(C):
                try:
                    h = torch.histc(im[c].float(), bins=256, min=0, max=255)
                except RuntimeError:
                    # MPS fallback — histc may not be supported
                    vals = im[c].float().cpu()
                    h = torch.histc(vals, bins=256, min=0, max=255).to(img1.device)
                ch_hists.append(h)
            hists.append(torch.cat(ch_hists))

        h1, h2 = hists[0], hists[1]
        total1 = h1.sum()
        total2 = h2.sum()
        if total1 > 0:
            h1 = h1 / total1
        if total2 > 0:
            h2 = h2 / total2

        h1c = h1 - h1.mean()
        h2c = h2 - h2.mean()
        num = (h1c * h2c).sum()
        den = torch.sqrt((h1c ** 2).sum() * (h2c ** 2).sum())
        if den > 0:
            results[i] = num / den
        else:
            results[i] = 1.0 if torch.equal(h1, h2) else 0.0

    return results


# ───────────────────────────────────────────────────────────────────────────
#  Batch Comparison Engine
# ───────────────────────────────────────────────────────────────────────────

def _load_image_to_tensor(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def _load_pair(args):
    path_a, path_b = args
    return _load_image_to_tensor(path_a), _load_image_to_tensor(path_b)


def process_batch_gpu(
    batch_tasks: list[TaskInfo],
    device: torch.device,
    ssim_kernels: dict[tuple, torch.Tensor],
    io_pool: ThreadPoolExecutor,
    fire_threshold: int = DEFAULT_FIRE_THRESHOLD,
    fire_channels: str = DEFAULT_FIRE_CHANNELS,
    mask_rendering: bool = True,
    mask_boundaries: bool = True,
    border_top: int = DEFAULT_BORDER_TOP,
    border_bottom: int = DEFAULT_BORDER_BOTTOM,
    border_left: int = DEFAULT_BORDER_LEFT,
    border_right: int = DEFAULT_BORDER_RIGHT,
    erode_boundary: int = DEFAULT_ERODE_BOUNDARY,
    compute_edges: bool = True,
) -> list[ComparisonResult]:
    n = len(batch_tasks)

    # Parallel image loading
    load_args = [(t.path_a, t.path_b) for t in batch_tasks]
    pairs = list(io_pool.map(_load_pair, load_args))

    processed_a, processed_b, sizes = [], [], []
    raw_pairs = []  # Keep raw numpy arrays for fire metrics
    
    for arr_a, arr_b in pairs:
        h = min(arr_a.shape[0], arr_b.shape[0])
        w = min(arr_a.shape[1], arr_b.shape[1])
        cropped_a = np.array(arr_a[:h, :w, :])
        cropped_b = np.array(arr_b[:h, :w, :])
        processed_a.append(cropped_a)
        processed_b.append(cropped_b)
        raw_pairs.append((cropped_a, cropped_b))
        sizes.append((h, w))

    max_h = max(s[0] for s in sizes)
    max_w = max(s[1] for s in sizes)

    batch_a = torch.zeros(n, 3, max_h, max_w, dtype=torch.float32)
    batch_b = torch.zeros(n, 3, max_h, max_w, dtype=torch.float32)
    pixel_counts = torch.zeros(n, dtype=torch.float32)

    for i in range(n):
        h, w = sizes[i]
        t_a = torch.from_numpy(processed_a[i]).permute(2, 0, 1).float()
        t_b = torch.from_numpy(processed_b[i]).permute(2, 0, 1).float()
        batch_a[i, :, :h, :w] = t_a
        batch_b[i, :, :h, :w] = t_b
        pixel_counts[i] = h * w * 3

    del processed_a, processed_b, pairs

    batch_a = batch_a.to(device, non_blocking=True)
    batch_b = batch_b.to(device, non_blocking=True)
    pixel_counts = pixel_counts.to(device, non_blocking=True)

    # Histogram correlation
    hist_vals = histogram_correlation_batch(batch_a, batch_b)

    # SSIM — sub-batched by image size
    ssim_vals = torch.zeros(n, device=device)
    size_groups = defaultdict(list)
    for i in range(n):
        size_groups[sizes[i]].append(i)

    SSIM_SUB_BATCH = 8

    for (h, w), indices in size_groups.items():
        win_size = min(11, h, w)
        if win_size < 3:
            for i in indices:
                ssim_vals[i] = 1.0
            continue

        key = (win_size, device)
        if key not in ssim_kernels:
            ssim_kernels[key] = _create_ssim_kernel(win_size, 1.5, 3).to(device)
        kernel = ssim_kernels[key]

        for sb_start in range(0, len(indices), SSIM_SUB_BATCH):
            sb_indices = indices[sb_start:sb_start + SSIM_SUB_BATCH]
            idx = torch.tensor(sb_indices, device=device, dtype=torch.long)
            sub_a = batch_a[idx, :, :h, :w].contiguous()
            sub_b = batch_b[idx, :, :h, :w].contiguous()
            sv = ssim_batch(sub_a, sub_b, kernel, win_size)
            ssim_vals[idx] = sv
            del sub_a, sub_b, sv
            _clear_device_cache(device)

    # Collect to CPU
    ssim_cpu = ssim_vals.cpu().numpy()
    hist_cpu = hist_vals.cpu().numpy()

    # Return GPU results + raw pairs for CPU post-processing
    return ssim_cpu, hist_cpu, raw_pairs


def _compute_pair_cpu_metrics(args: tuple) -> tuple[FireMetrics, MaskedMetrics, EdgeMetrics | None]:
    """Compute all CPU-bound metrics for a single image pair.

    Designed to run in a ThreadPoolExecutor — numpy/scipy release the GIL
    so multiple pairs are processed in true parallel across CPU cores.
    """
    (arr_a, arr_b, fire_threshold, fire_channels,
     mask_rendering, mask_boundaries, border_top, border_bottom,
     border_left, border_right, erode_boundary, do_edges) = args

    # Fire metrics
    fm = compute_fire_metrics(arr_a, arr_b, fire_threshold, fire_channels)

    # Masked metrics
    h, w = arr_a.shape[:2]
    rendering_mask = create_rendering_mask(
        (h, w), border_top, border_bottom, border_left, border_right
    ) if mask_rendering else None

    if mask_boundaries and fm is not None:
        mask_a = segment_fire_region(arr_a, fire_threshold, fire_channels)
        mask_b = segment_fire_region(arr_b, fire_threshold, fire_channels)
        interior_mask = create_interior_mask(mask_a, mask_b, erode_boundary)
    else:
        interior_mask = None

    masked = compute_masked_metrics(arr_a, arr_b, rendering_mask, interior_mask)

    mm = MaskedMetrics(
        masked_ssim=masked["masked_ssim"],
        mask_coverage=masked["mask_coverage"],
    )

    # Edge metrics
    em = None
    if do_edges:
        edge_m = compute_edge_metrics(arr_a, arr_b)
        boundary_m = compute_fire_boundary_metrics(arr_a, arr_b, fire_threshold)
        em = EdgeMetrics(
            edge_correlation=edge_m["edge_correlation"],
            edge_iou=edge_m["edge_iou"],
            boundary_iou=boundary_m["boundary_iou"],
            boundary_length_diff=boundary_m["boundary_length_diff"],
            hausdorff_approx=boundary_m["hausdorff_approx"],
        )

    return fm, mm, em


def _assemble_results(
    batch_tasks: list[TaskInfo],
    ssim_cpu: np.ndarray,
    hist_cpu: np.ndarray,
    cpu_metrics: list[tuple[FireMetrics, MaskedMetrics, EdgeMetrics | None]],
) -> list[ComparisonResult]:
    """Combine GPU metric arrays and CPU metric tuples into ComparisonResults."""
    results = []
    for i, task in enumerate(batch_tasks):
        fm, mm, em = cpu_metrics[i]
        m = Metrics(
            ssim_val=float(ssim_cpu[i]),
            hist_corr=float(hist_cpu[i]),
        )
        results.append(ComparisonResult(
            project=task.project,
            png_name=task.png_name,
            category=task.category,
            timestep=task.timestep,
            run_a=task.run_a,
            run_b=task.run_b,
            platform_pair=task.platform_pair,
            metrics=m,
            fire_metrics=fm,
            masked_metrics=mm,
            edge_metrics=em,
            path_a=task.path_a,
            path_b=task.path_b,
            image_type=task.image_type,
        ))
    return results


# ───────────────────────────────────────────────────────────────────────────
#  Task Building 
# ───────────────────────────────────────────────────────────────────────────

def build_comparison_tasks(
    group_a_dir: Path,
    group_b_dir: Path,
    matched_projects: list[str],
    proj_a: dict[str, list[Path]],
    proj_b: dict[str, list[Path]],
) -> list[TaskInfo]:
    tasks: list[TaskInfo] = []
    for pt in matched_projects:
        runs_a = proj_a[pt]
        runs_b = proj_b[pt]

        # Resolve PlotsFire paths
        pf_a = [(r, resolve_plotsfire(r)) for r in runs_a]
        pf_b = [(r, resolve_plotsfire(r)) for r in runs_b]
        pf_a = [(r, p) for r, p in pf_a if p is not None]
        pf_b = [(r, p) for r, p in pf_b if p is not None]

        all_pf_dirs = [p for _, p in pf_a] + [p for _, p in pf_b]
        common_pngs = find_common_pngs(all_pf_dirs)
        if not common_pngs:
            log.info(f"  {pt}: no common PNGs, skipping")
            continue

        log.info(f"  {pt}: {len(common_pngs)} common PNGs, "
                 f"{len(pf_a)} group-A runs, {len(pf_b)} group-B runs")

        for png_name in common_pngs:
            cat, ts = classify_png(png_name)
            img_type = classify_image_type(cat)

            # Intra-A
            for (ra, pa), (rb, pb) in itertools.combinations(pf_a, 2):
                tasks.append(TaskInfo(
                    project=pt, png_name=png_name, category=cat,
                    timestep=ts, run_a=ra.name, run_b=rb.name,
                    platform_pair="intra_a",
                    path_a=str(pa / png_name), path_b=str(pb / png_name),
                    image_type=img_type,
                ))

            # Intra-B
            for (ra, pa), (rb, pb) in itertools.combinations(pf_b, 2):
                tasks.append(TaskInfo(
                    project=pt, png_name=png_name, category=cat,
                    timestep=ts, run_a=ra.name, run_b=rb.name,
                    platform_pair="intra_b",
                    path_a=str(pa / png_name), path_b=str(pb / png_name),
                    image_type=img_type,
                ))

            # Cross
            for ra, pa in pf_a:
                for rb, pb in pf_b:
                    tasks.append(TaskInfo(
                        project=pt, png_name=png_name, category=cat,
                        timestep=ts, run_a=ra.name, run_b=rb.name,
                        platform_pair="cross",
                        path_a=str(pa / png_name), path_b=str(pb / png_name),
                        image_type=img_type,
                    ))

    return tasks


# ───────────────────────────────────────────────────────────────────────────
#  Temporal Monotonicity Analysis
# ───────────────────────────────────────────────────────────────────────────

def analyze_temporal_monotonicity(
    results: list[ComparisonResult],
    project: str,
    tolerance: float = 0.05,  # 5% decrease tolerance
) -> list[TemporalViolation]:
    """Check for fire area decreases over time (physics violations).
    
    Args:
        results: Comparison results for a single project
        project: Project name
        tolerance: Fractional decrease that triggers a violation (0.05 = 5%)
    
    Returns:
        List of temporal violations found
    """
    violations = []
    
    # Group by run and collect fire areas per timestep
    # We use the average fire area from cross-group comparisons
    # This gives us fire area for each platform's runs
    
    # Build map: (run_name) -> {timestep: [fire_areas]}
    run_areas: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    
    for r in results:
        if r.timestep < 0 or r.fire_metrics is None:
            continue
        # Record areas for both runs in the comparison
        run_areas[r.run_a][r.timestep].append(r.fire_metrics.area_a)
        run_areas[r.run_b][r.timestep].append(r.fire_metrics.area_b)
    
    # Check each run for monotonicity violations
    for run_name, ts_areas in run_areas.items():
        if len(ts_areas) < 2:
            continue
        
        sorted_timesteps = sorted(ts_areas.keys())
        prev_area = None
        prev_ts = None
        
        for ts in sorted_timesteps:
            # Average area at this timestep (across all comparisons)
            avg_area = np.mean(ts_areas[ts])
            
            if prev_area is not None and avg_area < prev_area * (1 - tolerance):
                # Fire area decreased more than tolerance
                pct_decrease = (prev_area - avg_area) / prev_area * 100
                violations.append(TemporalViolation(
                    timestep=ts,
                    prev_timestep=prev_ts,
                    area_before=prev_area,
                    area_after=avg_area,
                    percent_decrease=pct_decrease,
                    run_name=run_name,
                ))
            
            prev_area = avg_area
            prev_ts = ts
    
    return violations


def compute_ssim_temporal_gradient(
    results: list[ComparisonResult],
) -> float:
    """Compute linear trend of SSIM over time.
    
    Negative gradient indicates similarity degrading over time (integration bugs).
    
    Returns:
        Slope of SSIM vs timestep linear fit, or 0.0 if insufficient data
    """
    cross_results = [r for r in results if r.platform_pair == "cross" and r.timestep >= 0]
    if len(cross_results) < 3:
        return 0.0
    
    # Group SSIM by timestep
    ts_ssims: dict[int, list[float]] = defaultdict(list)
    for r in cross_results:
        ts_ssims[r.timestep].append(r.metrics.ssim_val)
    
    if len(ts_ssims) < 2:
        return 0.0
    
    # Compute average SSIM per timestep
    timesteps = sorted(ts_ssims.keys())
    avg_ssims = [np.mean(ts_ssims[t]) for t in timesteps]
    
    # Linear fit
    try:
        slope = np.polyfit(timesteps, avg_ssims, 1)[0]
        return float(slope)
    except (np.linalg.LinAlgError, ValueError):
        return 0.0


def analyze_temporal_gradient_detailed(
    results: list[ComparisonResult],
) -> TemporalGradientAnalysis | None:
    """Perform detailed temporal gradient analysis.

    Fits both linear and exponential models to SSIM over time,
    segments the simulation into phases, and detects acceleration.
    """
    cross_results = [r for r in results if r.platform_pair == "cross" and r.timestep >= 0]
    if len(cross_results) < 6:
        return None

    ts_ssims: dict[int, list[float]] = defaultdict(list)
    for r in cross_results:
        ts_ssims[r.timestep].append(r.metrics.ssim_val)

    if len(ts_ssims) < 4:
        return None

    timesteps = np.array(sorted(ts_ssims.keys()))
    avg_ssims = np.array([np.mean(ts_ssims[t]) for t in timesteps])
    t_normalized = timesteps - timesteps[0]

    # Use (1 - SSIM) as error metric for fitting
    avg_errors = 1.0 - avg_ssims

    # Linear fit
    try:
        linear_coeffs = np.polyfit(t_normalized, avg_errors, 1)
        linear_gradient = float(linear_coeffs[0])
        linear_predicted = np.polyval(linear_coeffs, t_normalized)
        ss_res_linear = np.sum((avg_errors - linear_predicted) ** 2)
        ss_tot = np.sum((avg_errors - np.mean(avg_errors)) ** 2)
        linear_r_squared = 1 - (ss_res_linear / ss_tot) if ss_tot > 0 else 0.0
    except (np.linalg.LinAlgError, ValueError):
        linear_gradient = 0.0
        linear_r_squared = 0.0
        ss_tot = np.sum((avg_errors - np.mean(avg_errors)) ** 2)

    # Exponential fit
    def exp_func(t, a, b, c):
        return a * np.exp(b * t) + c

    try:
        p0 = [0.1, 0.001, avg_errors[0]]
        bounds = ([0, 0, 0], [np.inf, 1.0, np.inf])
        popt, _ = curve_fit(exp_func, t_normalized, avg_errors, p0=p0, bounds=bounds, maxfev=1000)
        exponential_rate = float(popt[1])
        exp_predicted = exp_func(t_normalized, *popt)
        ss_res_exp = np.sum((avg_errors - exp_predicted) ** 2)
        exponential_r_squared = 1 - (ss_res_exp / ss_tot) if ss_tot > 0 else 0.0
    except (RuntimeError, ValueError):
        exponential_rate = 0.0
        exponential_r_squared = 0.0

    if linear_r_squared < 0.5 and exponential_r_squared < 0.5:
        best_fit = "stable"
    elif exponential_r_squared > linear_r_squared + 0.1:
        best_fit = "exponential"
    else:
        best_fit = "linear"

    # Phase analysis
    n = len(timesteps)
    third = n // 3
    early_phase_ssim = float(np.mean(avg_ssims[:third])) if third > 0 else 0.0
    mid_phase_ssim = float(np.mean(avg_ssims[third:2*third])) if third > 0 else 0.0
    late_phase_ssim = float(np.mean(avg_ssims[2*third:])) if third > 0 else 0.0

    # Acceleration
    try:
        quad_coeffs = np.polyfit(t_normalized, avg_errors, 2)
        acceleration = float(2 * quad_coeffs[0])
    except (np.linalg.LinAlgError, ValueError):
        acceleration = 0.0

    return TemporalGradientAnalysis(
        linear_gradient=linear_gradient,
        linear_r_squared=float(linear_r_squared),
        exponential_rate=exponential_rate,
        exponential_r_squared=float(exponential_r_squared),
        best_fit=best_fit,
        early_phase_ssim=early_phase_ssim,
        mid_phase_ssim=mid_phase_ssim,
        late_phase_ssim=late_phase_ssim,
        acceleration=acceleration,
    )


# ───────────────────────────────────────────────────────────────────────────
#  Statistical Analysis
# ───────────────────────────────────────────────────────────────────────────

def get_comparison_ssim(r: ComparisonResult, use_masked: bool = True) -> float:
    """Get the appropriate SSIM value based on masking preference."""
    if use_masked and r.masked_metrics is not None:
        return r.masked_metrics.masked_ssim
    return r.metrics.ssim_val


def analyze_results(
    results: list[ComparisonResult],
    project_names: list[str],
    sigma_threshold: float,
    use_masked: bool = True,
    iou_threshold: float = DEFAULT_IOU_PASS,
    iou_warn_threshold: float = DEFAULT_IOU_WARN,
    drift_threshold: float = DEFAULT_DRIFT_PASS,
    drift_warn_threshold: float = DEFAULT_DRIFT_WARN,
    area_diff_threshold: float = DEFAULT_AREA_DIFF_PASS,
    area_diff_warn_threshold: float = DEFAULT_AREA_DIFF_WARN,
) -> list[ProjectSummary]:
    by_project = defaultdict(list)
    for r in results:
        by_project[r.project].append(r)

    summaries = []
    for pt in project_names:
        rs = by_project.get(pt, [])
        summary = ProjectSummary(project=pt)

        if not rs:
            summaries.append(summary)
            continue

        pngs = {r.png_name for r in rs}
        summary.n_images = len(pngs)
        summary.n_comparisons = len(rs)

        intra_a = [r for r in rs if r.platform_pair == "intra_a"]
        intra_b = [r for r in rs if r.platform_pair == "intra_b"]
        cross = [r for r in rs if r.platform_pair == "cross"]

        def safe_stats(vals):
            if not vals:
                return 0.0, 0.0
            return float(np.mean(vals)), float(np.std(vals))

        a_ssims = [get_comparison_ssim(r, use_masked) for r in intra_a]
        b_ssims = [get_comparison_ssim(r, use_masked) for r in intra_b]
        cross_ssims = [get_comparison_ssim(r, use_masked) for r in cross]

        summary.intra_a_ssim_mean, summary.intra_a_ssim_std = safe_stats(a_ssims)
        summary.intra_b_ssim_mean, summary.intra_b_ssim_std = safe_stats(b_ssims)
        summary.cross_ssim_mean, summary.cross_ssim_std = safe_stats(cross_ssims)

        # All-image fire metrics (for informational display)
        cross_ious = [r.fire_metrics.iou for r in cross if r.fire_metrics is not None]
        intra_ious = ([r.fire_metrics.iou for r in intra_a if r.fire_metrics is not None] +
                      [r.fire_metrics.iou for r in intra_b if r.fire_metrics is not None])
        cross_drifts = [r.fire_metrics.centroid_drift for r in cross if r.fire_metrics is not None]

        summary.cross_iou_mean, summary.cross_iou_std = safe_stats(cross_ious)
        summary.intra_iou_mean = safe_stats(intra_ious)[0]
        summary.cross_centroid_drift_mean = safe_stats(cross_drifts)[0]

        # Fire-category cross metrics (verdict-driving)
        fire_cross = [r for r in cross if r.image_type == "fire" and r.fire_metrics is not None]
        fire_cross_ious = [r.fire_metrics.iou for r in fire_cross]
        fire_cross_drifts = [r.fire_metrics.centroid_drift for r in fire_cross]
        fire_cross_area_diff_pcts = [
            abs(r.fire_metrics.area_diff) / max(r.fire_metrics.area_a, r.fire_metrics.area_b, 1)
            for r in fire_cross
        ]
        summary.fire_cross_iou_mean, summary.fire_cross_iou_std = safe_stats(fire_cross_ious)
        summary.fire_cross_drift_mean, summary.fire_cross_drift_std = safe_stats(fire_cross_drifts)
        summary.fire_cross_area_diff_pct_mean, summary.fire_cross_area_diff_pct_std = safe_stats(fire_cross_area_diff_pcts)
        summary.fire_n_cross_comparisons = len(fire_cross)

        # Masked metrics summary
        cross_masked_ssims = [r.masked_metrics.masked_ssim for r in cross if r.masked_metrics is not None]
        cross_coverages = [r.masked_metrics.mask_coverage for r in cross if r.masked_metrics is not None]
        summary.cross_masked_ssim_mean = safe_stats(cross_masked_ssims)[0]
        summary.mask_coverage_mean = safe_stats(cross_coverages)[0]

        # Edge metrics summary
        cross_boundary_ious = [r.edge_metrics.boundary_iou for r in cross if r.edge_metrics is not None]
        cross_hausdorffs = [r.edge_metrics.hausdorff_approx for r in cross if r.edge_metrics is not None]
        summary.cross_boundary_iou_mean = safe_stats(cross_boundary_ious)[0]
        summary.cross_hausdorff_mean = safe_stats(cross_hausdorffs)[0]

        # Temporal analysis
        summary.temporal_violations = analyze_temporal_monotonicity(rs, pt)
        summary.ssim_temporal_gradient = compute_ssim_temporal_gradient(rs)
        summary.temporal_gradient_analysis = analyze_temporal_gradient_detailed(rs)

        # Deviation score (SSIM-based, informational only)
        intra_all_ssims = a_ssims + b_ssims
        if intra_all_ssims:
            intra_mean, intra_std = safe_stats(intra_all_ssims)
            if intra_std > 0:
                summary.deviation_score = (intra_mean - summary.cross_ssim_mean) / intra_std
            elif summary.cross_ssim_mean < intra_mean:
                summary.deviation_score = float("inf")
            else:
                summary.deviation_score = 0.0
        else:
            summary.deviation_score = 1.0 - summary.cross_ssim_mean

        # Per-criterion fire-metric verdicts
        if fire_cross:
            # IoU verdict (higher is better)
            if summary.fire_cross_iou_mean >= iou_threshold:
                summary.iou_verdict = "PASS"
            elif summary.fire_cross_iou_mean >= iou_warn_threshold:
                summary.iou_verdict = "WARN"
            else:
                summary.iou_verdict = "FAIL"

            # Drift verdict (lower is better)
            if summary.fire_cross_drift_mean <= drift_threshold:
                summary.drift_verdict = "PASS"
            elif summary.fire_cross_drift_mean <= drift_warn_threshold:
                summary.drift_verdict = "WARN"
            else:
                summary.drift_verdict = "FAIL"

            # Area diff verdict (lower is better)
            if summary.fire_cross_area_diff_pct_mean <= area_diff_threshold:
                summary.area_diff_verdict = "PASS"
            elif summary.fire_cross_area_diff_pct_mean <= area_diff_warn_threshold:
                summary.area_diff_verdict = "WARN"
            else:
                summary.area_diff_verdict = "FAIL"
        # else: all remain "N/A"

        # Temporal verdict
        if summary.temporal_violations:
            summary.temporal_verdict = "WARN"
        else:
            summary.temporal_verdict = "PASS"

        # Overall verdict = worst of sub-verdicts (ignoring N/A)
        sub_verdicts = [summary.iou_verdict, summary.drift_verdict,
                        summary.area_diff_verdict, summary.temporal_verdict]
        active = [v for v in sub_verdicts if v != "N/A"]
        if not active:
            summary.verdict = "SKIP"
        elif any(v == "FAIL" for v in active):
            summary.verdict = "FAIL"
        elif any(v == "WARN" for v in active):
            summary.verdict = "WARN"
        else:
            summary.verdict = "PASS"

        # Flagged images (fire-category IoU below threshold)
        cross_by_png = defaultdict(list)
        for r in cross:
            cross_by_png[r.png_name].append(r)

        flagged = []
        for png_name, crs in cross_by_png.items():
            # Only flag fire-category images
            if not any(r.image_type == "fire" for r in crs):
                continue
            fire_crs = [r for r in crs if r.fire_metrics is not None]
            if not fire_crs:
                continue
            avg_iou = float(np.mean([r.fire_metrics.iou for r in fire_crs]))
            avg_ssim = float(np.mean([r.metrics.ssim_val for r in crs]))
            if avg_iou < iou_threshold:
                flagged.append((png_name, avg_ssim, avg_iou))

        flagged.sort(key=lambda x: x[2])  # Ascending IoU (worst first)
        summary.flagged_images = flagged[:20]
        summaries.append(summary)

    return summaries


# ───────────────────────────────────────────────────────────────────────────
#  Difference Heatmap Generation for Flagged Images
# ───────────────────────────────────────────────────────────────────────────

def generate_flagged_heatmaps(
    summaries: list[ProjectSummary],
    results: list[ComparisonResult],
    output_dir: Path,
    max_heatmaps: int = 10,
    multiscale_levels: int = 4,
) -> dict[str, dict[str, FlaggedImageData]]:
    """Generate difference heatmaps for top flagged images.
    
    Returns:
        {project: {png_name: FlaggedImageData}}
    """
    heatmap_dir = output_dir / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    
    result_map: dict[str, dict[str, FlaggedImageData]] = {}
    
    # Index results by project and png_name for quick lookup
    by_project_png: dict[str, dict[str, list[ComparisonResult]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        if r.platform_pair == "cross":
            by_project_png[r.project][r.png_name].append(r)
    
    for summary in summaries:
        if not summary.flagged_images:
            continue
        
        result_map[summary.project] = {}
        
        for i, (png_name, avg_ssim, avg_iou) in enumerate(summary.flagged_images[:max_heatmaps]):
            cross_results = by_project_png[summary.project][png_name]
            if not cross_results:
                continue
            
            # Use first cross-group pair for heatmap
            r = cross_results[0]
            
            try:
                img_a = _load_image_to_tensor(r.path_a)
                img_b = _load_image_to_tensor(r.path_b)
                
                # Crop to same size
                h = min(img_a.shape[0], img_b.shape[0])
                w = min(img_a.shape[1], img_b.shape[1])
                img_a = img_a[:h, :w]
                img_b = img_b[:h, :w]
                
                # Generate comparison image
                diff_img = generate_difference_image(img_a, img_b, include_original=True)
                
                # Save to file
                safe_name = re.sub(r'[^\w\-.]', '_', png_name)
                heatmap_path = heatmap_dir / f"{summary.project}_{safe_name}_diff.png"
                diff_img.save(heatmap_path)
                
                # Also generate base64 for HTML embedding
                base64_data = image_to_base64(diff_img)

                # Multi-scale analysis for flagged images
                ms_data = compute_multiscale_metrics(img_a, img_b, levels=multiscale_levels)
                ms_pattern = analyze_scale_pattern(ms_data)
                ms_metrics = MultiscaleMetrics(
                    scale_factors=ms_data["scale_factors"],
                    ssims=ms_data["ssims"],
                    dominant_scale=ms_pattern["dominant_scale"],
                    fine_to_coarse_ratio=ms_pattern["fine_to_coarse_ratio"],
                    scale_gradient=ms_pattern["scale_gradient"],
                )

                result_map[summary.project][png_name] = FlaggedImageData(
                    png_name=png_name,
                    avg_ssim=avg_ssim,
                    avg_iou=avg_iou,
                    diff_heatmap_base64=base64_data,
                    multiscale_analysis=ms_metrics,
                )
                
            except Exception as e:
                log.warning(f"Failed to generate heatmap for {png_name}: {e}")
                result_map[summary.project][png_name] = FlaggedImageData(
                    png_name=png_name,
                    avg_ssim=avg_ssim,
                    avg_iou=avg_iou,
                    diff_heatmap_base64=None,
                )
    
    return result_map


# ───────────────────────────────────────────────────────────────────────────
#  Report Generators
# ───────────────────────────────────────────────────────────────────────────

# --- Console ---

def _verdict_color(v: str) -> str:
    if v == "FAIL":
        return _red(v)
    elif v == "WARN":
        return _yellow(v)
    elif v == "PASS":
        return _green(v)
    return v


def print_console_summary(summaries: list[ProjectSummary],
                          label_a: str, label_b: str):
    print("\n" + "=" * 180)
    print("CROSS-GROUP COMPARISON SUMMARY  (verdicts driven by fire-category metrics)")
    print("=" * 180)
    header = (
        f"{'Project':<35} {'Verdict':<6} "
        f"{'FireIoU':>8} {'Drift':>6} {'AreaD%':>7} "
        f"{'IoU':>4} {'Drft':>4} {'Area':>4} {'Temp':>4} "
        f"{'SSIM(info)':>11} {'Flagged':>7} {'Temporal':>8}"
    )
    print(header)
    print("-" * 180)
    for s in summaries:
        if s.verdict == "SKIP":
            print(f"{s.project:<35} {'SKIP':<6}")
            continue

        temporal_str = f"{len(s.temporal_violations)}" if s.temporal_violations else "OK"
        if s.temporal_violations:
            temporal_str = _yellow(temporal_str)

        print(
            f"{s.project:<35} {_verdict_color(s.verdict):<6} "
            f"{s.fire_cross_iou_mean:>8.4f} {s.fire_cross_drift_mean:>6.1f} "
            f"{s.fire_cross_area_diff_pct_mean*100:>6.1f}% "
            f"{_verdict_color(s.iou_verdict):>4} "
            f"{_verdict_color(s.drift_verdict):>4} "
            f"{_verdict_color(s.area_diff_verdict):>4} "
            f"{_verdict_color(s.temporal_verdict):>4} "
            f"{s.cross_ssim_mean:>11.6f} "
            f"{len(s.flagged_images):>7d} {temporal_str:>8}"
        )
    print("=" * 180)

    # Temporal violations detail
    for s in summaries:
        if s.temporal_violations:
            print(f"\n  {_yellow('TEMPORAL VIOLATIONS')} in {s.project}:")
            for v in s.temporal_violations[:5]:
                print(f"    Run {v.run_name}: t={v.prev_timestep}->{v.timestep}s, "
                      f"area {v.area_before:.0f}->{v.area_after:.0f} "
                      f"({_red(f'-{v.percent_decrease:.1f}%')})")
            if len(s.temporal_violations) > 5:
                print(f"    ... and {len(s.temporal_violations) - 5} more violations")

    # SSIM temporal gradient warnings
    for s in summaries:
        if s.ssim_temporal_gradient < -0.0001:
            print(f"\n  {_yellow('SSIM DEGRADATION')} in {s.project}: "
                  f"gradient = {s.ssim_temporal_gradient:.6f} SSIM/timestep")

    for s in summaries:
        if s.flagged_images:
            print(f"\n  {s.project}: top flagged fire images (IoU below threshold):")
            for name, ssim_val, iou in s.flagged_images[:10]:
                print(f"    {name}: IoU = {iou:.4f}, SSIM = {ssim_val:.6f}")


# --- CSV ---

def write_csv(results: list[ComparisonResult], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "project", "png_name", "category", "image_type", "timestep",
            "run_a", "run_b", "pair_type",
            "ssim", "hist_corr",
            "fire_iou", "fire_area_a", "fire_area_b", "fire_area_diff",
            "centroid_drift",
            "masked_ssim", "mask_coverage",
            "edge_correlation", "edge_iou",
            "boundary_iou", "boundary_length_diff", "hausdorff_approx",
        ])
        for r in results:
            fm = r.fire_metrics
            mm = r.masked_metrics
            em = r.edge_metrics
            w.writerow([
                r.project, r.png_name, r.category, r.image_type, r.timestep,
                r.run_a, r.run_b, r.platform_pair,
                f"{r.metrics.ssim_val:.6f}",
                f"{r.metrics.hist_corr:.6f}",
                f"{fm.iou:.6f}" if fm else "",
                f"{fm.area_a}" if fm else "",
                f"{fm.area_b}" if fm else "",
                f"{fm.area_diff}" if fm else "",
                f"{fm.centroid_drift:.2f}" if fm else "",
                f"{mm.masked_ssim:.6f}" if mm else "",
                f"{mm.mask_coverage:.4f}" if mm else "",
                f"{em.edge_correlation:.6f}" if em else "",
                f"{em.edge_iou:.6f}" if em else "",
                f"{em.boundary_iou:.6f}" if em else "",
                f"{em.boundary_length_diff:.0f}" if em else "",
                f"{em.hausdorff_approx:.1f}" if em else "",
            ])
    log.info(f"CSV written: {out_path}")


# --- HTML ---

def _esc(s):
    return html_mod.escape(str(s))


def write_html_report(
    summaries: list[ProjectSummary],
    results: list[ComparisonResult],
    out_path: Path,
    label_a: str,
    label_b: str,
    heatmap_data: dict[str, dict[str, FlaggedImageData]] | None = None,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    by_project = defaultdict(list)
    for r in results:
        by_project[r.project].append(r)

    parts = []
    parts.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>QUICFire Comparison Report</title>
<style>
body {{ font-family: 'Segoe UI', Tahoma, sans-serif; margin: 20px; background: #f5f5f5; }}
h1 {{ color: #333; }}
h2 {{ color: #555; border-bottom: 2px solid #ddd; padding-bottom: 5px; }}
h3 {{ color: #666; }}
h4 {{ color: #777; margin-top: 20px; }}
table {{ border-collapse: collapse; margin: 10px 0; }}
th, td {{ border: 1px solid #ccc; padding: 6px 12px; text-align: right; }}
th {{ background: #e8e8e8; }}
td:first-child, th:first-child {{ text-align: left; }}
.pass {{ color: #2a7; font-weight: bold; }}
.warn {{ color: #c80; font-weight: bold; }}
.fail {{ color: #c22; font-weight: bold; }}
.skip {{ color: #888; }}
.detail {{ margin: 15px 0 30px 0; padding: 15px; background: #fff; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
.flagged-table td {{ font-size: 0.9em; }}
.metric-bar {{ display: inline-block; height: 12px; background: #4a90d9; border-radius: 2px; }}
.methodology {{ background: #fff; padding: 20px; border-radius: 6px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
.methodology h2 {{ margin-top: 0; }}
.methodology table {{ font-size: 0.9em; }}
.heatmap-container {{ margin: 10px 0; }}
.heatmap-img {{ max-width: 100%; height: auto; border: 1px solid #ccc; }}
.temporal-warning {{ background: #fff3cd; border: 1px solid #ffc107; padding: 10px; border-radius: 4px; margin: 10px 0; }}
.temporal-warning h4 {{ color: #856404; margin-top: 0; }}
.gradient-warning {{ background: #f8d7da; border: 1px solid #f5c6cb; padding: 10px; border-radius: 4px; margin: 10px 0; }}
</style>
</head>
<body>
<h1>QUICFire Comparison Report</h1>
<p>Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
<p>Comparing <strong>{_esc(label_a)}</strong> vs <strong>{_esc(label_b)}</strong>. GPU-accelerated with PyTorch.</p>

<div class="methodology">
<h2>Methodology</h2>
<p>This report compares QUICFire simulation outputs between two groups of runs.
Verdicts are driven by <strong>fire-category metrics</strong> (IoU, centroid drift, area diff).
SSIM is reported for information only.</p>

<h3>Image Categorization</h3>
<table>
<tr><th>Type</th><th>Categories</th></tr>
<tr><td>fire</td><td><code>perc_mass_burnt</code>, <code>bw_perc_mass_burnt</code>, <code>fuel_dens_Plane</code>, <code>wplume_Plane</code></td></tr>
<tr><td>wind</td><td><code>u_qu_*</code>, <code>v_qu_*</code>, <code>w_qu_*</code></td></tr>
<tr><td>emissions</td><td><code>co_emissions</code>, <code>pm_emissions</code></td></tr>
<tr><td>other</td><td>Everything else</td></tr>
</table>

<h3>Fire-Metric Verdict Thresholds</h3>
<table>
<tr><th>Criterion</th><th>PASS</th><th>WARN</th><th>FAIL</th></tr>
<tr><td>Fire IoU</td><td>&ge; 0.95</td><td>&ge; 0.85</td><td>&lt; 0.85</td></tr>
<tr><td>Centroid Drift (px)</td><td>&le; 5.0</td><td>&le; 10.0</td><td>&gt; 10.0</td></tr>
<tr><td>Area Diff %</td><td>&le; 5%</td><td>&le; 15%</td><td>&gt; 15%</td></tr>
<tr><td>Temporal</td><td>No violations</td><td>Violations</td><td>&mdash;</td></tr>
</table>
<p>Overall verdict = worst of sub-verdicts. If no fire images exist, verdict = SKIP.</p>

<h3>Comparison Design</h3>
<p><strong>Intra-{_esc(label_a)}</strong> &mdash; pairs within {_esc(label_a)} (run-to-run variability).<br>
<strong>Intra-{_esc(label_b)}</strong> &mdash; pairs within {_esc(label_b)}.<br>
<strong>Cross</strong> &mdash; one run from each group.</p>

<h3>Temporal Monotonicity</h3>
<p>Fire area should not decrease significantly over time. Violations may indicate numerical instability or integration bugs.</p>

<h3>Difference Heatmaps</h3>
<p>For flagged images, pixel-wise difference heatmaps show where images differ most. Color scale: black (no difference) &rarr; red &rarr; yellow &rarr; white (maximum difference).</p>
</div>

<h2>Summary</h2>
<table>
<tr>
  <th>Project</th><th>Verdict</th>
  <th>Fire IoU</th><th>Drift (px)</th><th>AreaDiff%</th>
  <th>IoU</th><th>Drift</th><th>Area</th><th>Temp</th>
  <th>SSIM (info)</th>
  <th># Flagged</th><th># Comparisons</th>
</tr>
""")

    for s in summaries:
        vc = s.verdict.lower()
        if s.verdict == "SKIP":
            parts.append(f'<tr><td>{_esc(s.project)}</td>'
                         f'<td class="skip">SKIP</td>'
                         f'<td colspan="10">No PNGs</td></tr>\n')
            continue

        parts.append(
            f'<tr><td><a href="#{_esc(s.project)}">{_esc(s.project)}</a></td>'
            f'<td class="{vc}">{s.verdict}</td>'
            f'<td>{s.fire_cross_iou_mean:.4f}</td>'
            f'<td>{s.fire_cross_drift_mean:.1f}</td>'
            f'<td>{s.fire_cross_area_diff_pct_mean*100:.1f}%</td>'
            f'<td class="{s.iou_verdict.lower()}">{s.iou_verdict}</td>'
            f'<td class="{s.drift_verdict.lower()}">{s.drift_verdict}</td>'
            f'<td class="{s.area_diff_verdict.lower()}">{s.area_diff_verdict}</td>'
            f'<td class="{s.temporal_verdict.lower()}">{s.temporal_verdict}</td>'
            f'<td>{s.cross_ssim_mean:.6f}</td>'
            f'<td>{len(s.flagged_images)}</td>'
            f'<td>{s.n_comparisons}</td></tr>\n'
        )

    parts.append("</table>\n")

    # Project details
    parts.append("<h2>Project Details</h2>\n")
    for s in summaries:
        if s.verdict == "SKIP":
            continue
        rs = by_project.get(s.project, [])
        parts.append(f'<div class="detail" id="{_esc(s.project)}">\n')
        parts.append(f'<h3>{_esc(s.project)} — '
                     f'<span class="{s.verdict.lower()}">{s.verdict}</span></h3>\n')
        parts.append(f'<p>Images: {s.n_images} | Comparisons: {s.n_comparisons} | '
                     f'Fire comparisons: {s.fire_n_cross_comparisons}</p>\n')

        # Temporal violations warning
        if s.temporal_violations:
            parts.append('<div class="temporal-warning">\n')
            parts.append(f'<h4>⚠️ Temporal Monotonicity Violations ({len(s.temporal_violations)} detected)</h4>\n')
            parts.append('<p>Fire area decreased unexpectedly at the following timesteps:</p>\n')
            parts.append('<table><tr><th>Run</th><th>Timestep</th><th>Area Before</th><th>Area After</th><th>Decrease</th></tr>\n')
            for v in s.temporal_violations[:10]:
                parts.append(f'<tr><td>{_esc(v.run_name)}</td>'
                             f'<td>{v.prev_timestep}→{v.timestep}s</td>'
                             f'<td>{v.area_before:.0f}</td>'
                             f'<td>{v.area_after:.0f}</td>'
                             f'<td class="fail">-{v.percent_decrease:.1f}%</td></tr>\n')
            parts.append('</table>\n')
            if len(s.temporal_violations) > 10:
                parts.append(f'<p>... and {len(s.temporal_violations) - 10} more violations</p>\n')
            parts.append('</div>\n')

        # SSIM gradient warning
        if s.ssim_temporal_gradient < -0.0001:
            parts.append('<div class="gradient-warning">\n')
            parts.append(f'<h4>⚠️ Similarity Degradation Detected</h4>\n')
            parts.append(f'<p>SSIM decreases over time (gradient: {s.ssim_temporal_gradient:.6f} SSIM/timestep). '
                         f'This may indicate numerical integration bugs that compound over simulation time.</p>\n')
            parts.append('</div>\n')

        # Temporal gradient detailed analysis
        if s.temporal_gradient_analysis and s.temporal_gradient_analysis.best_fit == "exponential":
            tga = s.temporal_gradient_analysis
            parts.append('<div class="gradient-warning">\n')
            parts.append(f'<h4>⚠️ Exponential Error Growth Detected</h4>\n')
            parts.append(f'<p>Errors grow exponentially (rate: {tga.exponential_rate:.4f}). '
                         f'R&sup2; = {tga.exponential_r_squared:.3f}</p>\n')
            parts.append(f'<p>Phase SSIM: Early={tga.early_phase_ssim:.6f}, '
                         f'Mid={tga.mid_phase_ssim:.6f}, '
                         f'Late={tga.late_phase_ssim:.6f}</p>\n')
            parts.append('</div>\n')

        parts.append(f"""<table>
<tr><th>Metric</th><th>Value</th><th>Verdict</th></tr>
""")
        parts.append(f'<tr><td><strong>Fire IoU (cross, fire-category)</strong></td>'
                     f'<td>{s.fire_cross_iou_mean:.4f} (&plusmn;{s.fire_cross_iou_std:.4f})</td>'
                     f'<td class="{s.iou_verdict.lower()}">{s.iou_verdict}</td></tr>\n')
        parts.append(f'<tr><td><strong>Centroid Drift (cross, fire-category)</strong></td>'
                     f'<td>{s.fire_cross_drift_mean:.2f} px (&plusmn;{s.fire_cross_drift_std:.2f})</td>'
                     f'<td class="{s.drift_verdict.lower()}">{s.drift_verdict}</td></tr>\n')
        parts.append(f'<tr><td><strong>Area Diff % (cross, fire-category)</strong></td>'
                     f'<td>{s.fire_cross_area_diff_pct_mean*100:.1f}% (&plusmn;{s.fire_cross_area_diff_pct_std*100:.1f}%)</td>'
                     f'<td class="{s.area_diff_verdict.lower()}">{s.area_diff_verdict}</td></tr>\n')
        parts.append(f'<tr><td><strong>Temporal</strong></td>'
                     f'<td>{len(s.temporal_violations)} violations</td>'
                     f'<td class="{s.temporal_verdict.lower()}">{s.temporal_verdict}</td></tr>\n')
        parts.append(f'<tr><td>SSIM (all, informational)</td>'
                     f'<td>{s.cross_ssim_mean:.6f} (&plusmn;{s.cross_ssim_std:.6f})</td>'
                     f'<td>&mdash;</td></tr>\n')
        parts.append(f'<tr><td>Fire IoU (all cross)</td>'
                     f'<td>{s.cross_iou_mean:.4f} (&plusmn;{s.cross_iou_std:.4f})</td>'
                     f'<td>&mdash;</td></tr>\n')
        parts.append(f'<tr><td>Masked SSIM (informational)</td>'
                     f'<td>{s.cross_masked_ssim_mean:.6f} (coverage: {s.mask_coverage_mean:.2f})</td>'
                     f'<td>&mdash;</td></tr>\n')
        parts.append("</table>\n")

        # Time-series
        cross_rs = [r for r in rs if r.platform_pair == "cross" and r.timestep >= 0]
        if cross_rs:
            ts_ssim = defaultdict(list)
            ts_iou = defaultdict(list)
            for r in cross_rs:
                ts_ssim[r.timestep].append(r.metrics.ssim_val)
                if r.fire_metrics:
                    ts_iou[r.timestep].append(r.fire_metrics.iou)
            sorted_ts = sorted(ts_ssim.keys())
            if len(sorted_ts) > 1:
                parts.append("<h4>Cross-Group Metrics Over Time</h4>\n")
                parts.append("<table><tr><th>Timestep (s)</th><th>Avg SSIM</th>"
                             "<th>Min SSIM</th><th>Avg IoU</th><th>Visual</th></tr>\n")
                min_overall = min(np.mean(ts_ssim[t]) for t in sorted_ts)
                for t in sorted_ts:
                    vals = ts_ssim[t]
                    avg = np.mean(vals)
                    mn = min(vals)
                    avg_iou = np.mean(ts_iou.get(t, [1.0]))
                    bar_w = int(200 * avg) if avg > 0 else 0
                    parts.append(
                        f'<tr><td>{t}</td><td>{avg:.6f}</td><td>{mn:.6f}</td>'
                        f'<td>{avg_iou:.4f}</td>'
                        f'<td><span class="metric-bar" style="width:{bar_w}px">'
                        f'</span></td></tr>\n'
                    )
                parts.append("</table>\n")

        # Flagged images with heatmaps (ENHANCED)
        if s.flagged_images:
            parts.append("<h4>Flagged Images (Fire IoU below threshold)</h4>\n")
            parts.append('<table class="flagged-table">'
                         '<tr><th>Image</th><th>Avg SSIM</th><th>Avg IoU</th>'
                         '<th>Scale Analysis</th><th>Heatmap</th></tr>\n')

            project_heatmaps = heatmap_data.get(s.project, {}) if heatmap_data else {}

            for name, ssim_val, iou in s.flagged_images:
                heatmap_cell = ""
                scale_cell = ""
                if name in project_heatmaps:
                    flagged_data = project_heatmaps[name]
                    if flagged_data.diff_heatmap_base64:
                        heatmap_cell = (f'<div class="heatmap-container">'
                                        f'<img class="heatmap-img" '
                                        f'src="data:image/png;base64,{flagged_data.diff_heatmap_base64}" '
                                        f'alt="Difference heatmap for {_esc(name)}">'
                                        f'</div>')
                    if flagged_data.multiscale_analysis:
                        ma = flagged_data.multiscale_analysis
                        if ma.dominant_scale == "fine":
                            interpretation = "Fine-grained (rendering/anti-aliasing)"
                        elif ma.dominant_scale == "coarse":
                            interpretation = "Structural (algorithm/computation)"
                        else:
                            interpretation = "Uniform (systematic)"
                        scale_cell = (f'{interpretation}<br>'
                                      f'<small>Ratio: {ma.fine_to_coarse_ratio:.2f}</small>')

                parts.append(f'<tr><td>{_esc(name)}</td>'
                             f'<td>{ssim_val:.6f}</td>'
                             f'<td>{iou:.4f}</td>'
                             f'<td>{scale_cell}</td>'
                             f'<td>{heatmap_cell}</td></tr>\n')
            parts.append("</table>\n")

        # Per-category
        categories = sorted({r.category for r in rs})
        if len(categories) > 1:
            parts.append("<h4>Per-Category Summary</h4>\n")
            parts.append("<table><tr><th>Category</th><th>Type</th>"
                         "<th>Cross SSIM</th>"
                         "<th>Cross IoU</th></tr>\n")
            for cat in categories:
                cat_rs = [r for r in rs if r.category == cat]
                img_type = cat_rs[0].image_type if cat_rs else "other"
                cross_s = [r.metrics.ssim_val for r in cat_rs
                           if r.platform_pair == "cross" and r.metrics.ssim_val >= 0]
                cross_iou = [r.fire_metrics.iou for r in cat_rs
                             if r.platform_pair == "cross" and r.fire_metrics]
                cs_m = np.mean(cross_s) if cross_s else 0.0
                ci_m = np.mean(cross_iou) if cross_iou else 0.0
                parts.append(
                    f'<tr><td>{_esc(cat)}</td><td>{_esc(img_type)}</td>'
                    f'<td>{cs_m:.6f}</td>'
                    f'<td>{ci_m:.4f}</td></tr>\n'
                )
            parts.append("</table>\n")

        parts.append("</div>\n")

    parts.append("</body></html>")

    with open(out_path, "w") as f:
        f.write("".join(parts))
    log.info(f"HTML report written: {out_path}")


# --- Markdown ---

def write_markdown_report(
    summaries: list[ProjectSummary],
    results: list[ComparisonResult],
    out_path: Path,
    label_a: str,
    label_b: str,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    by_project = defaultdict(list)
    for r in results:
        by_project[r.project].append(r)

    lines: list[str] = []
    lines.append(f"# QUICFire Comparison Report\n")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"Comparing **{label_a}** vs **{label_b}**. GPU-accelerated with PyTorch.\n")
    lines.append(METHODOLOGY_TEXT_MD)
    lines.append("")

    # Summary table
    lines.append("## Summary\n")
    lines.append(f"| Project | Verdict | Fire IoU | Drift (px) | AreaDiff% | "
                 f"IoU | Drift | Area | Temp | SSIM (info) | Flagged | Comparisons |")
    lines.append("|---------|---------|----------|------------|-----------|" + "-----|" * 4 + "-------------|---------|-------------|")
    for s in summaries:
        if s.verdict == "SKIP":
            lines.append(f"| {s.project} | SKIP | | | | | | | | | | |")
            continue
        lines.append(
            f"| {s.project} | **{s.verdict}** | "
            f"{s.fire_cross_iou_mean:.4f} | {s.fire_cross_drift_mean:.1f} | "
            f"{s.fire_cross_area_diff_pct_mean*100:.1f}% | "
            f"{s.iou_verdict} | {s.drift_verdict} | {s.area_diff_verdict} | {s.temporal_verdict} | "
            f"{s.cross_ssim_mean:.6f} | {len(s.flagged_images)} | {s.n_comparisons} |"
        )
    lines.append("")

    # Project details
    lines.append("## Project Details\n")
    for s in summaries:
        if s.verdict == "SKIP":
            continue
        rs = by_project.get(s.project, [])
        lines.append(f"### {s.project} — {s.verdict}\n")
        lines.append(f"Images: {s.n_images} | Comparisons: {s.n_comparisons} | "
                     f"Fire comparisons: {s.fire_n_cross_comparisons}\n")

        # Temporal violations
        if s.temporal_violations:
            lines.append("#### Temporal Monotonicity Violations\n")
            lines.append("| Run | Timestep | Area Before | Area After | Decrease |")
            lines.append("|-----|----------|-------------|------------|----------|")
            for v in s.temporal_violations[:10]:
                lines.append(f"| {v.run_name} | {v.prev_timestep}->{v.timestep}s | "
                             f"{v.area_before:.0f} | {v.area_after:.0f} | -{v.percent_decrease:.1f}% |")
            lines.append("")

        # SSIM gradient warning
        if s.ssim_temporal_gradient < -0.0001:
            lines.append(f"#### Similarity Degradation\n")
            lines.append(f"SSIM gradient: {s.ssim_temporal_gradient:.6f} SSIM/timestep\n")

        lines.append("| Metric | Value | Verdict |")
        lines.append("|--------|-------|---------|")
        lines.append(f"| **Fire IoU (fire-category)** | {s.fire_cross_iou_mean:.4f} (+/-{s.fire_cross_iou_std:.4f}) | {s.iou_verdict} |")
        lines.append(f"| **Centroid Drift (fire-category)** | {s.fire_cross_drift_mean:.2f} px (+/-{s.fire_cross_drift_std:.2f}) | {s.drift_verdict} |")
        lines.append(f"| **Area Diff % (fire-category)** | {s.fire_cross_area_diff_pct_mean*100:.1f}% (+/-{s.fire_cross_area_diff_pct_std*100:.1f}%) | {s.area_diff_verdict} |")
        lines.append(f"| **Temporal** | {len(s.temporal_violations)} violations | {s.temporal_verdict} |")
        lines.append(f"| SSIM (all, info) | {s.cross_ssim_mean:.6f} (+/-{s.cross_ssim_std:.6f}) | -- |")
        lines.append(f"| Fire IoU (all cross) | {s.cross_iou_mean:.4f} (+/-{s.cross_iou_std:.4f}) | -- |")
        lines.append("")

        # Time-series
        cross_rs = [r for r in rs if r.platform_pair == "cross" and r.timestep >= 0]
        if cross_rs:
            ts_ssim = defaultdict(list)
            ts_iou = defaultdict(list)
            for r in cross_rs:
                ts_ssim[r.timestep].append(r.metrics.ssim_val)
                if r.fire_metrics:
                    ts_iou[r.timestep].append(r.fire_metrics.iou)
            sorted_ts = sorted(ts_ssim.keys())
            if len(sorted_ts) > 1:
                lines.append("#### Cross-Group Metrics Over Time\n")
                lines.append("| Timestep (s) | Avg SSIM | Min SSIM | Avg IoU |")
                lines.append("|-------------|----------|----------|---------|")
                for t in sorted_ts:
                    vals = ts_ssim[t]
                    avg = np.mean(vals)
                    mn = min(vals)
                    avg_iou = np.mean(ts_iou.get(t, [1.0]))
                    lines.append(f"| {t} | {avg:.6f} | {mn:.6f} | {avg_iou:.4f} |")
                lines.append("")

        # Flagged images
        if s.flagged_images:
            lines.append("#### Flagged Images (Fire IoU below threshold)\n")
            lines.append("| Image | Avg IoU | Avg SSIM |")
            lines.append("|-------|---------|----------|")
            for name, ssim_val, iou in s.flagged_images:
                lines.append(f"| {name} | {iou:.4f} | {ssim_val:.6f} |")
            lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    log.info(f"Markdown report written: {out_path}")


# ───────────────────────────────────────────────────────────────────────────
#  Setup Check
# ───────────────────────────────────────────────────────────────────────────

def run_check_setup(
    group_a_dir: Path | None,
    group_b_dir: Path | None,
    project_a_dir: Path | None,
    project_b_dir: Path | None,
):
    """Print a diagnostic report about system and data readiness, then exit."""
    print("=" * 70)
    print("QF-COMPARE SETUP CHECK (Enhanced Version)")
    print("=" * 70)

    # System info
    print("\n--- System ---")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CPU cores:       {os.cpu_count()}")
    if torch.cuda.is_available():
        print(f"  GPU:             {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  GPU VRAM:        {mem:.1f} GB")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("  GPU:             Apple MPS (Metal Performance Shaders)")
    else:
        print("  GPU:             None (CPU only)")

    print("\n--- Enhanced Features ---")
    print("  ✓ Difference heatmaps for flagged images")
    print("  ✓ Fire region segmentation and IoU metrics")
    print("  ✓ Temporal monotonicity violation detection")
    print("  ✓ SSIM temporal gradient analysis")
    print("  ✓ Rendering-region masking (border/colorbar/title)")
    print("  ✓ Interior-only comparison (anti-aliased boundary erosion)")
    print("  ✓ Edge & boundary comparison (Sobel, Hausdorff)")
    print("  ✓ Multi-scale analysis (pyramid downsampling)")
    try:
        import scipy
        print(f"  ✓ scipy {scipy.__version__} available")
    except ImportError:
        print("  ✗ scipy NOT installed (required for new features)")

    # Data info
    if group_a_dir and group_b_dir:
        print(f"\n--- Group A: {group_a_dir} ---")
        proj_a = discover_group_projects(group_a_dir)
        for pt, runs in sorted(proj_a.items()):
            run_names = [r.name for r in runs]
            print(f"  {pt}: {len(runs)} runs  {run_names}")

        print(f"\n--- Group B: {group_b_dir} ---")
        proj_b = discover_group_projects(group_b_dir)
        for pt, runs in sorted(proj_b.items()):
            run_names = [r.name for r in runs]
            print(f"  {pt}: {len(runs)} runs  {run_names}")

        matched = match_groups(proj_a, proj_b)
        unmatched_a = set(proj_a.keys()) - set(matched)
        unmatched_b = set(proj_b.keys()) - set(matched)

        print(f"\n--- Matching ---")
        print(f"  Matched project types: {len(matched)}")
        for pt in matched:
            runs_a = proj_a[pt]
            runs_b = proj_b[pt]
            all_pf = [resolve_plotsfire(r) for r in runs_a + runs_b]
            all_pf = [p for p in all_pf if p is not None]
            common = find_common_pngs(all_pf)
            n_intra_a = len(list(itertools.combinations(runs_a, 2)))
            n_intra_b = len(list(itertools.combinations(runs_b, 2)))
            n_cross = len(runs_a) * len(runs_b)
            n_pairs = n_intra_a + n_intra_b + n_cross
            print(f"  {pt}: {len(common)} PNGs, "
                  f"{len(runs_a)}A + {len(runs_b)}B runs, "
                  f"{n_pairs} pairs/image, "
                  f"~{n_pairs * len(common)} total comparisons")

        if unmatched_a:
            print(f"  Unmatched in A: {sorted(unmatched_a)}")
        if unmatched_b:
            print(f"  Unmatched in B: {sorted(unmatched_b)}")

        total_comparisons = 0
        for pt in matched:
            runs_a = proj_a[pt]
            runs_b = proj_b[pt]
            all_pf = [resolve_plotsfire(r) for r in runs_a + runs_b]
            all_pf = [p for p in all_pf if p is not None]
            common = find_common_pngs(all_pf)
            n_intra_a = len(list(itertools.combinations(runs_a, 2)))
            n_intra_b = len(list(itertools.combinations(runs_b, 2)))
            n_cross = len(runs_a) * len(runs_b)
            total_comparisons += (n_intra_a + n_intra_b + n_cross) * len(common)

        print(f"\n  Total estimated comparisons: {total_comparisons}")

    elif project_a_dir and project_b_dir:
        print(f"\n--- Project A: {project_a_dir} ---")
        pf_a = resolve_plotsfire(project_a_dir)
        if pf_a:
            pngs_a = sorted(p.name for p in pf_a.iterdir() if p.suffix == ".png" and TIME_RE.search(p.name))
            print(f"  PlotsFire: {pf_a}  ({len(pngs_a)} timed PNGs)")
        else:
            print("  PlotsFire: NOT FOUND")

        print(f"\n--- Project B: {project_b_dir} ---")
        pf_b = resolve_plotsfire(project_b_dir)
        if pf_b:
            pngs_b = sorted(p.name for p in pf_b.iterdir() if p.suffix == ".png" and TIME_RE.search(p.name))
            print(f"  PlotsFire: {pf_b}  ({len(pngs_b)} timed PNGs)")
        else:
            print("  PlotsFire: NOT FOUND")

        if pf_a and pf_b:
            common = find_common_pngs([pf_a, pf_b])
            print(f"\n  Common PNGs: {len(common)}")
            print(f"  Comparisons: {len(common)}  (1 pair per image)")

    print("\n" + "=" * 70)


# ───────────────────────────────────────────────────────────────────────────
#  CLI Parsing
# ───────────────────────────────────────────────────────────────────────────

ALL_OUTPUT_FORMATS = {"html", "md", "csv"}


def _build_help_epilog() -> str:
    lines = [
        "output formats:",
        "  html   — rich HTML report with methodology, summary, per-project details",
        "  md     — Markdown report (same structure as HTML)",
        "  csv    — raw comparison data (one row per pair)",
        "  all    — shorthand for html,md,csv",
        "",
        "enhanced features (v2.1):",
        "  - Difference heatmaps for flagged images",
        "  - Fire region segmentation and IoU metrics",
        "  - Temporal monotonicity violation detection",
        "  - Rendering-region masking (default on)",
        "  - Edge & boundary comparison (Sobel, Hausdorff)",
        "  - Multi-scale error analysis",
        "",
        "examples:",
        "  %(prog)s --group-a projects/Mac --group-b projects/Linux",
        "  %(prog)s --project-a Canyon-1 --project-b linux-Canyon-1 --output csv,html",
        "  %(prog)s --group-a projects/A --group-b projects/B --check-setup",
        "  %(prog)s --group-a projects/A --group-b projects/B --output all --output-dir results/",
        "  %(prog)s --group-a projects/A --group-b projects/B --fire-threshold 50",
    ]
    # GPU capability note
    if torch.cuda.is_available():
        lines.insert(0, f"gpu: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        lines.insert(0, "gpu: Apple MPS")
    else:
        lines.insert(0, "gpu: none (CPU mode)")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qf-compare",
        description="General-purpose QUICFire simulation output comparison tool (GPU-accelerated, enhanced v2.1)",
        epilog=_build_help_epilog(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    group_mode = parser.add_argument_group("group mode")
    group_mode.add_argument("--group-a", type=Path, metavar="DIR",
                            help="Directory containing Group A run folders")
    group_mode.add_argument("--group-b", type=Path, metavar="DIR",
                            help="Directory containing Group B run folders")

    project_mode = parser.add_argument_group("project mode")
    project_mode.add_argument("--project-a", type=Path, metavar="DIR",
                              help="Single run directory for project A")
    project_mode.add_argument("--project-b", type=Path, metavar="DIR",
                              help="Single run directory for project B")

    parser.add_argument("--check-setup", action="store_true",
                        help="Print setup diagnostics and exit (no comparison)")
    parser.add_argument("--output", type=str, default="html",
                        help="Comma-separated output formats: html,md,csv,all (default: html)")
    parser.add_argument("--output-dir", type=Path, default=Path.cwd(),
                        help="Directory for output files (default: cwd)")
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA_THRESHOLD,
                        help=f"SSIM deviation threshold in sigma, informational only (default: {DEFAULT_SIGMA_THRESHOLD})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"GPU batch size (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--io-workers", type=int, default=8,
                        help="IO thread pool size (default: 8)")

    # New fire segmentation options
    fire_group = parser.add_argument_group("fire segmentation")
    fire_group.add_argument("--fire-threshold", type=int, default=DEFAULT_FIRE_THRESHOLD,
                            help=f"Intensity threshold for fire pixel detection (default: {DEFAULT_FIRE_THRESHOLD})")
    fire_group.add_argument("--fire-channels", type=str, default=DEFAULT_FIRE_CHANNELS,
                            help=f"Channels to check for fire: r, g, b, rg, rgb (default: {DEFAULT_FIRE_CHANNELS})")
    fire_group.add_argument("--no-heatmaps", action="store_true",
                            help="Disable generation of difference heatmaps")

    # Rendering mask options
    mask_group = parser.add_argument_group("rendering mask options")
    mask_group.add_argument("--no-mask-rendering", action="store_true",
                            help="Disable masking of plot borders (axes, colorbar, title)")
    mask_group.add_argument("--no-mask-boundaries", action="store_true",
                            help="Disable masking of anti-aliased fire boundaries")
    mask_group.add_argument("--border-top", type=int, default=DEFAULT_BORDER_TOP,
                            help=f"Top border mask height in pixels (default: {DEFAULT_BORDER_TOP})")
    mask_group.add_argument("--border-bottom", type=int, default=DEFAULT_BORDER_BOTTOM,
                            help=f"Bottom border mask height (default: {DEFAULT_BORDER_BOTTOM})")
    mask_group.add_argument("--border-left", type=int, default=DEFAULT_BORDER_LEFT,
                            help=f"Left border mask width (default: {DEFAULT_BORDER_LEFT})")
    mask_group.add_argument("--border-right", type=int, default=DEFAULT_BORDER_RIGHT,
                            help=f"Right border mask width (default: {DEFAULT_BORDER_RIGHT})")
    mask_group.add_argument("--erode-boundary", type=int, default=DEFAULT_ERODE_BOUNDARY,
                            help=f"Pixels to erode from fire boundary (default: {DEFAULT_ERODE_BOUNDARY})")

    # Edge comparison options
    edge_group = parser.add_argument_group("edge comparison options")
    edge_group.add_argument("--no-edge-metrics", action="store_true",
                            help="Disable edge/boundary comparison (faster)")

    # Multi-scale analysis options
    scale_group = parser.add_argument_group("multi-scale analysis options")
    scale_group.add_argument("--multiscale-levels", type=int, default=4,
                             help="Number of pyramid levels for multi-scale analysis (default: 4)")

    # Fire-metric verdict thresholds
    verdict_group = parser.add_argument_group("fire-metric verdict thresholds")
    verdict_group.add_argument("--iou-threshold", type=float, default=DEFAULT_IOU_PASS,
                               help=f"IoU PASS threshold (default: {DEFAULT_IOU_PASS})")
    verdict_group.add_argument("--iou-warn", type=float, default=DEFAULT_IOU_WARN,
                               help=f"IoU WARN threshold (default: {DEFAULT_IOU_WARN})")
    verdict_group.add_argument("--drift-threshold", type=float, default=DEFAULT_DRIFT_PASS,
                               help=f"Centroid drift PASS threshold in pixels (default: {DEFAULT_DRIFT_PASS})")
    verdict_group.add_argument("--drift-warn", type=float, default=DEFAULT_DRIFT_WARN,
                               help=f"Centroid drift WARN threshold in pixels (default: {DEFAULT_DRIFT_WARN})")
    verdict_group.add_argument("--area-diff-threshold", type=float, default=DEFAULT_AREA_DIFF_PASS,
                               help=f"Area diff PASS threshold as fraction (default: {DEFAULT_AREA_DIFF_PASS})")
    verdict_group.add_argument("--area-diff-warn", type=float, default=DEFAULT_AREA_DIFF_WARN,
                               help=f"Area diff WARN threshold as fraction (default: {DEFAULT_AREA_DIFF_WARN})")

    return parser


def _parse_output_formats(raw: str) -> set[str]:
    tokens = {t.strip().lower() for t in raw.split(",")}
    if "all" in tokens:
        return ALL_OUTPUT_FORMATS.copy()
    unknown = tokens - ALL_OUTPUT_FORMATS
    if unknown:
        log.warning(f"Unknown output formats ignored: {unknown}")
    return tokens & ALL_OUTPUT_FORMATS


# ───────────────────────────────────────────────────────────────────────────
#  Main Orchestrator
# ───────────────────────────────────────────────────────────────────────────

def main():
    parser = build_parser()
    args = parser.parse_args()

    # --- Validate args ---
    has_group = args.group_a or args.group_b
    has_project = args.project_a or args.project_b

    if has_group and has_project:
        parser.error("Cannot mix --group-a/--group-b with --project-a/--project-b")
    if not has_group and not has_project:
        parser.error("Provide either --group-a/--group-b or --project-a/--project-b")

    if has_group and not (args.group_a and args.group_b):
        parser.error("Both --group-a and --group-b are required")
    if has_project and not (args.project_a and args.project_b):
        parser.error("Both --project-a and --project-b are required")

    # --- Check-setup mode ---
    if args.check_setup:
        run_check_setup(args.group_a, args.group_b, args.project_a, args.project_b)
        sys.exit(0)

    output_formats = _parse_output_formats(args.output)
    out_dir = args.output_dir
    batch_size = args.batch_size

    device = select_device()

    # --- Determine mode and discover ---
    if has_group:
        group_a_dir = args.group_a.resolve()
        group_b_dir = args.group_b.resolve()
        label_a = group_a_dir.name
        label_b = group_b_dir.name

        log.info(f"Group A: {group_a_dir} (label: {label_a})")
        log.info(f"Group B: {group_b_dir} (label: {label_b})")

        proj_a = discover_group_projects(group_a_dir)
        proj_b = discover_group_projects(group_b_dir)
        matched = match_groups(proj_a, proj_b)

        if not matched:
            log.error("No matching project types found between groups.")
            sys.exit(1)

        log.info(f"Matched {len(matched)} project types: {matched}")
        for pt in matched:
            log.info(f"  {pt}: {len(proj_a[pt])} A runs, {len(proj_b[pt])} B runs")

        log.info("Building comparison tasks...")
        tasks = build_comparison_tasks(group_a_dir, group_b_dir, matched, proj_a, proj_b)
        project_names = matched

    else:  # project mode
        pa = args.project_a.resolve()
        pb = args.project_b.resolve()
        label_a = pa.name
        label_b = pb.name

        log.info(f"Project A: {pa}")
        log.info(f"Project B: {pb}")

        # Detect whether each path is a single run or contains multiple runs
        proj_a = _detect_project_runs(pa)
        proj_b = _detect_project_runs(pb)
        pt_name = "project"
        matched = [pt_name]

        proj_a_map = {pt_name: proj_a}
        proj_b_map = {pt_name: proj_b}

        # Use a dummy parent; paths resolved inside build_comparison_tasks
        tasks = build_comparison_tasks(pa.parent, pb.parent, matched, proj_a_map, proj_b_map)
        project_names = matched

    if not tasks:
        log.warning("No comparison tasks found.")
        sys.exit(1)

    log.info(f"Total comparison tasks: {len(tasks)}")
    log.info(f"Sigma threshold: {args.sigma}")
    log.info(f"Batch size: {batch_size}")
    log.info(f"IO workers: {args.io_workers}")
    log.info(f"Fire threshold: {args.fire_threshold}")
    log.info(f"Fire channels: {args.fire_channels}")
    log.info(f"Mask rendering: {not args.no_mask_rendering}")
    log.info(f"Mask boundaries: {not args.no_mask_boundaries}")
    log.info(f"Edge metrics: {not args.no_edge_metrics}")
    log.info(f"Verdict thresholds: IoU>={args.iou_threshold}, "
             f"drift<={args.drift_threshold}px, area_diff<={args.area_diff_threshold*100:.0f}%")

    # --- Process (pipelined: GPU batch N+1 overlaps with CPU metrics for batch N) ---
    log.info(f"Running {len(tasks)} comparisons in batches of {batch_size}...")
    t0 = time.time()

    all_results = []
    ssim_kernels = {}
    n_batches = (len(tasks) + batch_size - 1) // batch_size
    cpu_workers = max(args.io_workers, os.cpu_count() or 4)

    # Shared CPU args that don't change per pair
    cpu_shared = (
        args.fire_threshold, args.fire_channels,
        not args.no_mask_rendering, not args.no_mask_boundaries,
        args.border_top, args.border_bottom,
        args.border_left, args.border_right,
        args.erode_boundary, not args.no_edge_metrics,
    )

    with (
        ThreadPoolExecutor(max_workers=args.io_workers) as io_pool,
        ThreadPoolExecutor(max_workers=cpu_workers) as cpu_pool,
    ):
        # Pipeline state: previous batch's pending CPU work
        prev_pending: tuple | None = None
        # prev_pending = (cpu_futures_list, batch_tasks, ssim, hist)

        def _collect_pending(pending):
            """Block until previous batch's CPU metrics finish, assemble results."""
            futures, btasks, ssim_c, hist_c = pending
            cpu_metrics = [f.result() for f in futures]
            return _assemble_results(btasks, ssim_c, hist_c, cpu_metrics)

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(tasks))
            batch_tasks_cur = tasks[start:end]

            # GPU phase for current batch (uses io_pool for image loading)
            gpu_arrays = process_batch_gpu(
                batch_tasks_cur, device, ssim_kernels, io_pool,
                fire_threshold=args.fire_threshold,
                fire_channels=args.fire_channels,
                mask_rendering=not args.no_mask_rendering,
                mask_boundaries=not args.no_mask_boundaries,
                border_top=args.border_top,
                border_bottom=args.border_bottom,
                border_left=args.border_left,
                border_right=args.border_right,
                erode_boundary=args.erode_boundary,
                compute_edges=not args.no_edge_metrics,
            )

            # Collect previous batch's CPU results (blocks only if CPU is slower than GPU)
            if prev_pending is not None:
                all_results.extend(_collect_pending(prev_pending))

            # Submit current batch's CPU work as individual futures (one per pair)
            # Each runs in its own thread — numpy/scipy release the GIL
            ssim_cpu, hist_cpu, raw_pairs = gpu_arrays
            cpu_futures = [
                cpu_pool.submit(
                    _compute_pair_cpu_metrics,
                    (arr_a, arr_b) + cpu_shared,
                )
                for i, (arr_a, arr_b) in enumerate(raw_pairs)
            ]
            prev_pending = (cpu_futures, batch_tasks_cur,
                            ssim_cpu, hist_cpu)

            if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
                elapsed = time.time() - t0
                done = end
                rate = done / elapsed if elapsed > 0 else 0
                log.info(
                    f"  Batch {batch_idx+1}/{n_batches} — "
                    f"{done}/{len(tasks)} done ({rate:.0f} pairs/sec)"
                )

        # Collect final batch's CPU results
        if prev_pending is not None:
            all_results.extend(_collect_pending(prev_pending))

    elapsed = time.time() - t0
    log.info(f"Completed {len(all_results)} comparisons in {elapsed:.1f}s "
             f"({len(all_results)/elapsed:.0f} comparisons/sec)")

    # --- Analyse ---
    log.info("Analyzing results...")
    summaries = analyze_results(
        all_results, project_names, args.sigma,
        use_masked=not args.no_mask_rendering,
        iou_threshold=args.iou_threshold,
        iou_warn_threshold=args.iou_warn,
        drift_threshold=args.drift_threshold,
        drift_warn_threshold=args.drift_warn,
        area_diff_threshold=args.area_diff_threshold,
        area_diff_warn_threshold=args.area_diff_warn,
    )

    # --- Generate heatmaps for flagged images ---
    heatmap_data = None
    if "html" in output_formats and not args.no_heatmaps:
        log.info("Generating difference heatmaps for flagged images...")
        heatmap_data = generate_flagged_heatmaps(
            summaries, all_results, out_dir,
            multiscale_levels=args.multiscale_levels,
        )

    # --- Reports ---
    print_console_summary(summaries, label_a, label_b)

    out_dir.mkdir(parents=True, exist_ok=True)

    if "csv" in output_formats:
        write_csv(all_results, out_dir / "comparison_results.csv")
    if "html" in output_formats:
        write_html_report(summaries, all_results, out_dir / "report.html", label_a, label_b, heatmap_data)
    if "md" in output_formats:
        write_markdown_report(summaries, all_results, out_dir / "report.md", label_a, label_b)

    # --- Exit code ---
    if any(s.verdict == "FAIL" for s in summaries):
        log.warning("One or more projects FAILED comparison.")
        sys.exit(2)
    elif any(s.verdict == "WARN" for s in summaries):
        log.info("Some projects have warnings. Review report for details.")
    elif any(s.temporal_violations for s in summaries):
        log.warning("Temporal monotonicity violations detected. Review report for details.")
    else:
        log.info("All projects PASSED comparison.")


def _detect_project_runs(path: Path) -> list[Path]:
    """Auto-detect whether *path* is a single run dir or contains multiple runs.

    Returns a list of run directories (each having a PlotsFire/ or PNG dir).
    """
    # If path itself has PlotsFire/ or PNGs, treat as single run
    if resolve_plotsfire(path) is not None:
        return [path]
    # Otherwise scan for sub-directories that are runs
    runs = []
    for child in sorted(path.iterdir()):
        if child.is_dir() and resolve_plotsfire(child) is not None:
            runs.append(child)
    if not runs:
        log.warning(f"No PlotsFire data found in {path}")
    return runs


if __name__ == "__main__":
    main()
