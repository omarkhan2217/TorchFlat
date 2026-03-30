# Comparison with Other GPU/Exoplanet Tools

## Summary

| Tool | Detrends light curves? | Same task as TorchFlat? | GPU? | Relationship |
|------|----------------------|------------------------|------|-------------|
| **wotan** | Yes (biweight, 15+ methods) | Yes | No (CPU) | Direct competitor |
| **lightkurve** | Basic (SavGol only) | Partially | No | Complementary |
| **stella** | No (flare detection CNN) | No | TensorFlow | Unrelated task |
| **jaxoplanet** | No (transit model fitting) | No | Yes (JAX) | Different pipeline stage |
| **batman** | No (transit model generation) | No | No (OpenMP) | Different pipeline stage |
| **juliet** | GP joint fitting | Different approach | No | Characterization, not survey |
| **pycheops** | CHEOPS-specific decorrelation | No | No | Different mission |

## Positioning

TorchFlat is the only tool that performs GPU-accelerated robust location
estimation (biweight-family) detrending for bulk light curve preprocessing.

- **wotan** (Hippke et al. 2019) is the only direct competitor - same
  algorithm family, CPU-only. TorchFlat's UMI algorithm extends wotan's
  biweight with asymmetric weights and upper-RMS scale, running 33x faster on GPU.

- **jaxoplanet** (Foreman-Mackey et al.) is the only other GPU-accelerated
  exoplanet tool, but it performs transit model evaluation and gradient
  computation (fitting stage), not survey-scale detrending (preprocessing
  stage). It demonstrates the value of GPU acceleration in the field.

- **lightkurve** (Lightkurve Collaboration 2018) is a general-purpose
  light curve analysis framework. Its built-in `flatten()` uses
  Savitzky-Golay, not biweight. For robust detrending, lightkurve users
  are directed to wotan. TorchFlat can consume lightkurve-downloaded data.

- **stella** (Feinstein et al. 2020), **batman** (Kreidberg 2015),
  **juliet** (Espinoza et al. 2019), and **pycheops** (Maxted et al. 2022)
  all solve different problems (flare detection, transit modeling, joint
  fitting, CHEOPS systematics) at different pipeline stages.

## Paper paragraph

"Several tools exist for exoplanet light curve analysis, but none
provide GPU-accelerated robust detrending for survey-scale preprocessing.
wotan (Hippke et al. 2019) offers 15+ detrending methods including the
Tukey biweight used here, but runs on CPU only. lightkurve (Lightkurve
Collaboration 2018) provides Savitzky-Golay flattening but not biweight
detrending. jaxoplanet demonstrates GPU acceleration for transit model
evaluation via JAX, but operates at the fitting stage rather than
preprocessing. batman (Kreidberg 2015), juliet (Espinoza et al. 2019),
and stella (Feinstein et al. 2020) address transit modeling, joint
fitting, and flare detection respectively. TorchFlat fills the gap:
GPU-native robust detrending with a novel asymmetric weight function,
targeting the preprocessing bottleneck in transit survey pipelines."
