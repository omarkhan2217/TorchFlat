# TorchFlat API Reference

## CLI Commands

### `torchflat umi_detrend` - Detrend light curves

```bash
torchflat umi_detrend --input /path/to/fits/ [options]
```

**Input/Output:**

| Flag | Default | Description |
|------|---------|-------------|
| `--input, -i` | required | Directory with FITS files, or a single FITS file |
| `--output, -o` | `<input>/../torchflat_output` | Output directory |
| `--output-format` | `fits` | Output format: `fits` (FITS table) or `npz` (numpy) |
| `--n-stars, -n` | 0 (all) | Max stars to process |

**Mission/Data Format:**

| Flag | Default | Description |
|------|---------|-------------|
| `--mission` | `tess` | Mission preset: `tess`, `kepler`, or `k2` |
| `--col-time` | from preset | Override time column name |
| `--col-flux` | from preset | Override flux column name |
| `--col-quality` | from preset | Override quality column name |

**UMI Algorithm:**

| Flag | Default | Description |
|------|---------|-------------|
| `--asymmetry` | `2.0` | Dip penalty: 2.0 (quiet stars), 1.5 (mixed), 1.0 (variable stars) |
| `--window-length` | `0.5` | Sliding window width in days |
| `--cval` | `5.0` | Rejection threshold in scale units |
| `--n-iter` | `5` | Number of bisquare iterations |
| `--bias-correct` | off | Subtract known asymmetry bias from detrended flux |

**Performance:**

| Flag | Default | Description |
|------|---------|-------------|
| `--device` | `cuda` | Device: `cuda` or `cpu` |
| `--max-batch` | auto | Override batch size (auto-detected from VRAM) |
| `--vram-budget` | auto | VRAM budget in GB |

**Pipeline:**

| Flag | Default | Description |
|------|---------|-------------|
| `--clip-sigma` | `5.0` | Sigma clipping threshold for Track A |
| `--skip-track-b` | off | Skip Track B (FFT highpass anomaly detection) |
| `--cutoff-days` | `5.0` | Track B FFT highpass cutoff period in days |
| `--window-scales` | `256:128,512:256,2048:512,8192:2048` | Window scales as `size:stride,...` |

---

### `torchflat benchmark` - Speed benchmark

```bash
torchflat benchmark --input /path/to/fits/ [--n-stars 500]
```

Same flags as `umi_detrend` except `--output` and `--output-format`.

---

### `torchflat plot` - Visualize detrending

```bash
torchflat plot --fits star.fits [--save output.png]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--fits, -f` | required | Path to a single FITS file |
| `--mission` | `tess` | Mission preset |
| `--asymmetry` | `2.0` | Dip penalty factor |
| `--device` | `cuda` | Device |
| `--save, -s` | show | Save plot to file instead of displaying |

Shows 3 panels: raw flux + trend, detrended flux, residuals in ppm.

---

## Python API

### `torchflat.umi_detrend`

The core UMI detrending function.

```python
from torchflat import umi_detrend

detrended, trend = umi_detrend(
    flux,               # [B, L] tensor - flux values
    time,               # [B, L] tensor - timestamps
    valid_mask,         # [B, L] bool tensor - True = valid point
    segment_id,         # [B, L] int32 tensor - segment labels
    window_length_days=0.5,
    n_iter=5,
    cval=5.0,
    min_segment_points=50,
    dtype=torch.float32,
    asymmetry=2.0,      # 2.0=best accuracy, 1.5=mixed, 1.0=variable stars
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `flux` | `Tensor [B, L]` | required | Flux values (PDCSAP_FLUX) |
| `time` | `Tensor [B, L]` | required | Timestamps |
| `valid_mask` | `Tensor [B, L]` | required | Boolean mask (True = valid) |
| `segment_id` | `Tensor [B, L]` | required | Segment labels from gap detection |
| `window_length_days` | `float` | `0.5` | Window width in days (0.5 = 12 hours) |
| `n_iter` | `int` | `5` | Asymmetric bisquare iterations |
| `cval` | `float` | `5.0` | Rejection threshold in scale units |
| `min_segment_points` | `int` | `50` | Minimum valid points per window |
| `dtype` | `torch.dtype` | `float32` | Computation precision |
| `asymmetry` | `float` | `2.0` | Dip penalty. 2.0 for quiet stars, 1.0 for variable stars. |

**Returns:**
- `detrended` - `[B, L]` tensor, `flux / trend`. NaN where trend is invalid.
- `trend` - `[B, L]` tensor, estimated stellar trend. NaN at edges.

---

### `torchflat.preprocess_sector`

Full pipeline: Track A (detrending + windows) + Track B (FFT anomaly detection).

```python
import torchflat

results, skipped = torchflat.preprocess_sector(
    star_data,          # list of dicts with time, pdcsap_flux, sap_flux, quality
    device="cuda",
    skip_track_b=False,
    progress_callback=None,
)
```

**Input format:** Each dict in `star_data` must have:
- `time` - numpy float64 array
- `pdcsap_flux` - numpy float32 array
- `sap_flux` - numpy float32 array (for Track B)
- `quality` - numpy int32 array (quality flags)

**Returns:**
- `results` - list of per-star dicts with keys:
  - `trend` - estimated trend (numpy array)
  - `detrended` - flux/trend (numpy array)
  - `windows_256`, `windows_2048`, etc. - extracted transit search windows
  - `track_b_curve` - FFT highpass anomaly curve (if Track B enabled)
- `skipped` - list of `{index, reason, details}` dicts

---

### `torchflat.umi.UMI_BIAS_PPM`

Lookup table for asymmetry-induced bias correction:

```python
from torchflat.umi import UMI_BIAS_PPM
# {1.0: -2, 1.5: -209, 2.0: -451, 2.5: -687, 3.0: -896}
```

---

## Mission Presets

| Mission | Time column | Flux column | Raw flux | Quality column | Bitmask |
|---------|------------|-------------|----------|----------------|---------|
| `tess` | TIME | PDCSAP_FLUX | SAP_FLUX | QUALITY | 3455 |
| `kepler` | TIME | PDCSAP_FLUX | SAP_FLUX | SAP_QUALITY | 8191 |
| `k2` | TIME | PDCSAP_FLUX | SAP_FLUX | SAP_QUALITY | 8191 |

Custom columns override any preset: `--col-time BJD --col-flux FLUX --col-quality FLAGS`

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TORCHFLAT_NO_KERNEL` | `0` | Set to `1` to disable HIP/CUDA kernel (uses torch.sort fallback, 6x slower) |
