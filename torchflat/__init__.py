"""TorchFlat: GPU-native photometric preprocessing for exoplanet transit searches."""

__version__ = "0.5.0"

from torchflat.biweight import biweight_detrend
from torchflat.pipeline import preprocess_sector, preprocess_sector_hybrid, preprocess_track_a, preprocess_track_b
from torchflat.windows import DEFAULT_WINDOW_SCALES

__all__ = [
    "biweight_detrend",
    "preprocess_sector",
    "preprocess_sector_hybrid",
    "preprocess_track_a",
    "preprocess_track_b",
    "DEFAULT_WINDOW_SCALES",
]
