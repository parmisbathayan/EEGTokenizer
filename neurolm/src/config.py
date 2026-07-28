"""Configuration for the frozen NeuroLM-to-ZuCo transfer test."""

from dataclasses import asdict, dataclass
from typing import Optional, Tuple


UPSTREAM_REPOSITORY = "https://github.com/935963004/NeuroLM.git"
UPSTREAM_COMMIT = "0cda9876d8ce6ee07ed0c43eee5e9a6f5c24b177"
CHECKPOINT_REPOSITORY = "Weibang/NeuroLM"
CHECKPOINT_FILENAME = "checkpoints/NeuroLM-B.pt"
CHECKPOINT_SIZE_BYTES = 2_380_000_000  # Display-only estimate; verify after download.


@dataclass(frozen=True)
class PreprocessConfig:
    """Paper-compatible preprocessing for ZuCo sentence EEG."""

    source_hz: int = 500
    target_hz: int = 200
    expected_channels: int = 105
    min_duration_seconds: float = 1.0
    bandpass_hz: Tuple[float, float] = (0.1, 75.0)
    notch_hz: Optional[float] = 50.0
    notch_quality: float = 30.0
    drop_channel_indices: Tuple[int, ...] = (104,)
    max_nonfinite_fraction: float = 0.20
    paper_input_divisor: float = 100.0

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class EncoderConfig:
    """Published NeuroLM-B neural encoder dimensions."""

    block_size: int = 1024
    patch_samples: int = 200
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    bias: bool = False
    dropout: float = 0.0
    in_chans: int = 1
    out_chans: int = 16

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class EvaluationConfig:
    """Locked small-data evaluation settings."""

    seeds: Tuple[int, ...] = (42, 52, 62)
    n_splits: int = 5
    inner_splits: int = 3
    c_values: Tuple[float, ...] = (0.001, 0.01, 0.1, 1.0, 10.0)
    bootstrap_samples: int = 2000
    bootstrap_ci: float = 0.9833
    minimum_delta: float = 0.015
    minimum_positive_seeds: int = 2

    def to_dict(self):
        return asdict(self)
