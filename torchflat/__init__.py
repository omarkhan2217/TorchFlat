"""TorchFlat: GPU-native photometric preprocessing for exoplanet transit searches."""

__version__ = "0.9.1"

from torchflat.umi import umi_detrend
from torchflat.pipeline import preprocess_sector, preprocess_track_a, preprocess_track_b
from torchflat.windows import DEFAULT_WINDOW_SCALES

__all__ = [
    "umi_detrend",
    "preprocess_sector",
    "preprocess_track_a",
    "preprocess_track_b",
    "DEFAULT_WINDOW_SCALES",
]
