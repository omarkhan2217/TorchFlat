# Validation Results

All results generated on AMD Radeon RX 9060 XT (16 GB VRAM).

## Reproducibility

- **Commit:** `ebd7cba` (v0.9.1 base) + Phase 1-3 fixes (pending commit)
- **GPU:** AMD Radeon RX 9060 XT, 16 GB VRAM, ROCm 7.2
- **CUDA tested:** NVIDIA GTX 1650 Ti, CUDA 12.8
- **CPU:** 12-core, 64 GB RAM, Windows 11
- **Python:** 3.13, PyTorch 2.9.1+rocmsdk20260116
- **Random seeds:** All injection tests use `numpy.random.default_rng(seed)` with seeds documented per script (typically 42 or 12345)

### Data Sources (all from MAST)
- **TESS:** Sectors 1, 6, 7, 8, 9, 10, 11, 12 (PDCSAP FITS from https://archive.stsci.edu/missions/tess/)
- **Kepler:** Quarters 2, 5, 9, 17 long-cadence (76,243 files from https://archive.stsci.edu/missions/kepler/lightcurves/)
- **K2:** Campaign 5 long-cadence (21,008 stars from https://archive.stsci.edu/missions/k2/)

### Reproducing Results
```bash
pip install torchflat[test]
# or from source: pip install -e ".[dev]"

# Injection recovery (1000 TESS stars, 9 methods)
python benchmarks/compare_methods.py

# Known planet recovery (304 planets)
python benchmarks/validate_known_planets.py

# Kepler multi-quarter (Q2, Q5, Q9, Q17)
python benchmarks/validate_kepler_quarters.py

# Asymmetry parameter sweep (10,000 stars)
python benchmarks/validate_asymmetry.py

# Multi-sector TESS (sectors 6, 7, 12)
python benchmarks/validate_multisector.py
```

- **Scripts:** `benchmarks/*.py` (each script is self-contained, loads data from `data/` directory)

---

## Current Results (use these)

### Accuracy
- `asymmetry_10k_stars.json` - 10,000 TESS stars, 5 asymmetry values x 7 depths. Definitive.
- `asymmetry_validation_2k.json` - 2000/2000 train/test split, optimal asymmetry=2.0
- `multisector_validation_2k.json` - 2000 stars x 3 TESS sectors (6, 7, 12)
- `injection_grid_umi.json` - UMI, 343 configs (10 periods x 5 durations x 7 depths), 500 stars
- `known_planet_recovery_all.json` - 304 confirmed planets (96 TESS + 208 Kepler), UMI wins 53%
- `multi_mission.json` - Kepler (1000 stars) and K2 (1000 stars) validation
- `kepler_multi_quarter.json` - Kepler Q2/Q5/Q9/Q17, 1000 stars each, 5 depths
- `method_comparison_1k.json` - 1000 stars, UMI vs 8 methods (first 2 depths)

### Analysis
- `method_comparison.md` - UMI vs 8 detrending methods, 1000 stars, all depths
- `known_planet_summary.md` - 304-planet recovery summary table
- `kepler_asymmetry_sweep.md` - Optimal asymmetry by mission (TESS=2.0, Kepler=3.0)
- `theoretical_analysis.md` - Influence function, breakdown point, Hampel classification
- `breakdown_point.json` - Monte Carlo breakdown point (200 trials)
- `systematic_analysis.json` - Normal stars, EBs, variable stars, flare preservation
- `gpu_tools_comparison.md` - Positioning vs wotan, lightkurve, jaxoplanet, etc.
- `speedup_comparison.md` - 154/sec full pipeline, 37x vs wotan

### Examples
- `example_tess.png` - 3-panel plot of TESS star detrending
- `example_kepler.png` - 3-panel plot of Kepler star detrending

---

## Old Results (superseded, kept for reference)

- `asymmetry_validation.json` - OLD: 500 stars. See asymmetry_validation_2k.json
- `multisector_validation.json` - OLD: 500 stars. See multisector_validation_2k.json
- `known_planet_recovery.json` - OLD: 42 TESS planets only. See known_planet_recovery_all.json
- `known_planet_recovery_4methods.json` - OLD: 41 TESS planets. See known_planet_recovery_all.json
- `injection_grid.json` - OLD: 100 stars with wotan. See injection_grid_umi.json
