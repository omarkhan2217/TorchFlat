# Validation Results

All results generated on AMD Radeon RX 9060 XT (16 GB VRAM), TESS sector 6/7/12 data.

## Reproducibility

- GPU: AMD Radeon RX 9060 XT, 16 GB VRAM, ROCm 7.2
- CPU: 12-core, 64 GB RAM, Windows 11
- Python: 3.12, PyTorch 2.x (ROCm build)
- TESS data: sectors 6, 7, 12 from MAST (19,618 / 19,995 / 19,989 stars)
- Kepler data: Quarter 5 long-cadence (13,673 stars)
- K2 data: Campaign 5 long-cadence (21,008 stars)
- Scripts: benchmarks/*.py (each results file lists the script that generated it)

## Accuracy Validation
- `asymmetry_validation.json` - Train/test split (500/500 stars), optimal asymmetry=1.5
- `asymmetry_validation_2k.json` - Train/test split (2000/2000 stars), optimal asymmetry=2.0
- `asymmetry_10k_stars.json` - 10,000 stars, 5 asymmetry values x 7 depths
- `multisector_validation.json` - 500 stars x 3 sectors (6, 7, 12)
- `multisector_validation_2k.json` - 2000 stars x 3 sectors
- `injection_grid.json` - UMI vs wotan, 90 configs (10 periods x 3 durations x 3 depths), 100 stars
- `injection_grid_umi.json` - UMI only, 343 configs (10 periods x 5 durations x 7 depths), 500 stars
- `known_planet_recovery.json` - 42 confirmed TESS planets, depth recovery vs wotan
- `multi_mission.json` - Kepler (1000 stars) and K2 (1000 stars) validation

## Algorithm Analysis
- `method_comparison.md` - UMI vs 8 detrending methods (accuracy + speed)
- `theoretical_analysis.md` - Influence function, breakdown point, Hampel classification
- `breakdown_point.json` - Monte Carlo breakdown point (200 trials, 5 depths, 9 fractions)
- `systematic_analysis.json` - Normal stars, EBs, variable stars, flare preservation
- `gpu_tools_comparison.md` - Positioning vs wotan, lightkurve, jaxoplanet, etc.

## Speed
- `speedup_comparison.md` - Full pipeline and detrend-only vs wotan (all numbers)

## Examples
- `example_tess.png` - 3-panel plot of TESS star detrending
- `example_kepler.png` - 3-panel plot of Kepler star detrending
