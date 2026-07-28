"""Configuration shared by extraction and evaluation."""

from dataclasses import asdict, dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class PreprocessConfig:
    """Paper-compatible preprocessing for ZuCo sentence EEG."""

    source_hz: int = 500
    target_hz: int = 200
    bandpass_hz: Tuple[float, float] = (0.1, 75.0)
    notch_hz: Optional[float] = 50.0
    notch_quality: float = 30.0
    drop_channel_indices: Tuple[int, ...] = (104,)
    max_nonfinite_fraction: float = 0.20
    standardize_per_recording: bool = False

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class EvaluationConfig:
    """Small-data evaluation settings."""

    seeds: Tuple[int, ...] = (42, 52, 62)
    n_splits: int = 5
    inner_splits: int = 3
    c_values: Tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    bootstrap_samples: int = 2000

    def to_dict(self):
        return asdict(self)

