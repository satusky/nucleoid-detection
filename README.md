# Nucleoid Detection
Python implementation of https://github.com/jkfindeisen/nucleoid_detection_and_kinetics_2021

>Christian Brüser, Jan Keller-Findeisen, and Stefan Jakobs. "The TFAM to mtDNA ratio defines inner-cellular nucleoid populations with distinct activity levels"
https://doi.org/10.1016/j.celrep.2021.110000

Nucleoid detection and replication kinetics analysis for STED microscopy images.



## Features

- **Nucleus detection** — Interactive threshold-based segmentation with real-time parameter tuning
- **Spot detection** — Local maxima detection with 2D Gaussian fitting (symmetric and rotated/asymmetric)
- **Colocalization** — Spatial colocalization of DNA and EdU signals to identify EdU-positive nucleoids
- **Kinetic modeling** — Fit single- and two-component replication models to time-series data with bootstrap confidence intervals
- **Image processing** — FFT-based smoothing (Gaussian, disk, ring, square kernels), masked smoothing, background subtraction

## Installation

Requires Python 3.12+.

```bash
pip install .
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Usage

The package includes example scripts that demonstrate the full analysis workflow using data in the `data/` directory.

### 1. Detect nucleus

```bash
python example_detect_nucleus.py
```

Interactively segment the cell nucleus from combined DNA/EdU STED images. Use arrow keys to adjust threshold and smoothing, then save the binary mask as a TIFF file.

### 2. Detect nucleoid spots

```bash
python example_detect_nucleoids.py
```

Detect DNA and EdU spots via interactive thresholding, then fit 2D Gaussians to each detected spot for sub-pixel localization. Results are saved as `.npz` files.

### 3. Colocalize DNA and EdU spots

```bash
python example_colocalize_DNA_EdU_spots.py
```

Compute pairwise distances between fitted DNA and EdU spot centers to determine which nucleoids are EdU-positive (actively replicating).

### 4. HDF5 mask subtraction and spot detection (notebook)

The Jupyter notebook [`h5_mask_subtraction.ipynb`](h5_mask_subtraction.ipynb) provides a step-by-step walkthrough for processing HDF5 microscopy data:

1. **Load and inspect** multi-dimensional HDF5 datasets (channels, z-slices, frames)
2. **Create a binary mask** using Otsu thresholding to segment mitochondria
3. **Smooth the masked image** at multiple scales using `img_smooth_mask`
4. **Detect the nucleus** from the smoothed background image using `img_detect_nucleus`
5. **Detect and fit nucleoid spots** — find local maxima in the detection image and fit 2D Gaussians for sub-pixel localization

Each step includes inline visualizations. Update the `filepath` variable in the first cell to point to your own `.h5` file.

### 5. Fit replication kinetics

```bash
python fit_timeseries_edu_positive_nucleoids.py
```

Fit kinetic models to time-series data of EdU-positive nucleoid fractions:
- **Model A**: Single population — `P(t) = 1 - exp(-2αt)`
- **Model B**: Two populations (fast/slow) — `P(t) = 1 - γ·exp(-2α_s·t) - (1-γ)·exp(-2α_f·t)`

## Package modules

| Module | Description |
|---|---|
| `loaders` | TIFF and HDF5 file loading with Streamlit upload widget |
| `img_detect_nucleus` | Threshold-based nucleus detection with hole filling and convex hull |
| `interactive_detect_nucleus` | Interactive matplotlib UI for nucleus parameter tuning |
| `spot_utils` | Image preparation, background subtraction, and spot filtering |
| `omex_local_max` | Local maxima/minima detection in 2D arrays |
| `fit_2d_gaussian_symmetric` | Symmetric 2D Gaussian fitting |
| `fit_2d_gaussian_rotated` | Rotated asymmetric 2D Gaussian fitting |
| `fit_nucleoids` | Combined symmetric + rotated Gaussian fitting pipeline |
| `omex_nearest_neighbour` | Nearest neighbor distance computation |
| `img_smooth` | FFT-based image smoothing with multiple kernel types |
| `img_smooth_mask` | Masked image smoothing |
| `img_fourier_grid` | Fourier grid generation for FFT operations |

## License

MIT
