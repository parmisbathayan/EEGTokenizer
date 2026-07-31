"""Configuration shared by the EEGText data-audit commands."""

from dataclasses import asdict, dataclass
from typing import Optional


ZUCO_OSF_NODES = {
    "1.0": "q3zws",
    "2.0": "2urht",
}


@dataclass(frozen=True)
class AuditConfig:
    """Validation rules recorded alongside every corpus manifest."""

    dataset: str = "zuco"
    release: str = "1.0"
    task: str = "SR"
    pattern: str = "results*_SR.mat"
    source_hz: float = 500.0
    expected_channels: int = 105
    minimum_samples: int = 500
    maximum_nonfinite_fraction: float = 0.20
    recursive: bool = True
    labels_csv: Optional[str] = None

    def __post_init__(self):
        if not self.dataset.strip() or not self.release.strip() or not self.task.strip():
            raise ValueError("dataset, release, and task must be non-empty")
        if self.source_hz <= 0:
            raise ValueError("source_hz must be positive")
        if self.expected_channels < 1 or self.minimum_samples < 1:
            raise ValueError("channel and sample requirements must be positive")
        if not 0.0 <= self.maximum_nonfinite_fraction <= 1.0:
            raise ValueError("maximum_nonfinite_fraction must lie in [0, 1]")

    def to_dict(self):
        return asdict(self)
