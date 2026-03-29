# TorchFlat API Reference

## CLI

```bash
# Detrend FITS files
torchflat umi_detrend --input /path/to/fits/ --output /path/to/results/

# Speed benchmark
torchflat benchmark --input /path/to/fits/ --n-stars 500
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input, -i` | required | Directory containing FITS files |
| `--output, -o` | `<input>/../torchflat_output` | Output directory |
| `--n-stars, -n` | 0 (all) | Max stars to process |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--skip-track-b` | false | Skip FFT highpass (Track B) |

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
    window_length_days=0.5,  # sliding window width in days
    n_iter=5,           # number of asymmetric bisquare iterations
    cval=5.0,           # rejection threshold in MAD units
    min_segment_points=50,   # min valid points for a valid trend
    dtype=torch.float32,     # computation precision
    asymmetry=1.5,      # dip penalty factor (1.0 = standard biweight)
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `flux` | `Tensor [B, L]` | required | Flux values (PDCSAP_FLUX) |
| `time` | `Tensor [B, L]` | required | Timestamps (BTJD) |
| `valid_mask` | `Tensor [B, L]` | required | Boolean mask (True = valid) |
| `segment_id` | `Tensor [B, L]` | required | Segment labels from gap detection |
| `window_length_days` | `float` | `0.5` | Window width in days (0.5 = 12 hours) |
| `n_iter` | `int` | `5` | Asymmetric bisquare iterations |
| `cval` | `float` | `5.0` | Rejection threshold in MAD units |
| `min_segment_points` | `int` | `50` | Minimum valid points per window |
| `dtype` | `torch.dtype` | `float32` | Computation precision |
| `asymmetry` | `float` | `1.5` | Dip penalty. 1.0 = symmetric biweight. >1 penalizes downward dips more, preserving transit depth. Validated via train/test split on 500 stars. |

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
    progress_callback=None,  # callable(done, total) for progress
)
```

**Input format:** Each dict in `star_data` must have:
- `time` - numpy float64 array (BTJD timestamps)
- `pdcsap_flux` - numpy float32 array
- `sap_flux` - numpy float32 array (for Track B)
- `quality` - numpy int32 array (TESS quality flags)

**Returns:**
- `results` - list of per-star dicts with keys:
  - `trend` - estimated trend (numpy array)
  - `detrended` - flux/trend (numpy array)
  - `windows_256`, `windows_2048`, etc. - extracted transit search windows
  - `track_b_curve` - FFT highpass anomaly curve (if Track B enabled)
- `skipped` - list of `{index, reason, details}` dicts

---

### `torchflat.preprocess_track_a`

Track A only (detrending + window extraction, no FFT).

```python
results, skipped = torchflat.preprocess_track_a(
    times, fluxes, qualities,   # lists of numpy arrays, one per star
    device="cuda",
    window_scales=[(256, 128), (2048, 512)],
    window_length_days=0.5,
    progress_callback=None,
)
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TORCHFLAT_NO_KERNEL` | `0` | Set to `1` to disable HIP/CUDA kernel (uses torch.sort fallback) |
