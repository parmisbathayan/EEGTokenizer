"""Preprocessing for frozen TFM token extraction."""

from math import gcd

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, resample_poly, sosfiltfilt

from .config import PreprocessConfig


def _repair_nonfinite(eeg, max_fraction):
    eeg = np.asarray(eeg, dtype=np.float64).copy()
    for channel in range(eeg.shape[0]):
        row = eeg[channel]
        finite = np.isfinite(row)
        bad_fraction = 1.0 - float(finite.mean())
        if bad_fraction > max_fraction:
            raise ValueError(
                f"channel {channel} has {bad_fraction:.1%} non-finite samples; "
                f"limit is {max_fraction:.1%}"
            )
        if not finite.all():
            good = np.flatnonzero(finite)
            if not len(good):
                raise ValueError(f"channel {channel} has no finite samples")
            bad = np.flatnonzero(~finite)
            row[bad] = np.interp(bad, good, row[good])
    return eeg


def preprocess_eeg(eeg, config=PreprocessConfig()):
    """Return finite float32 EEG in ``channels x time`` at the target rate."""

    eeg = np.asarray(eeg)
    if eeg.ndim != 2:
        raise ValueError(f"expected channels x time EEG, got {eeg.shape}")
    if eeg.shape[0] != config.expected_channels:
        if eeg.shape[1] == config.expected_channels:
            raise ValueError(f"EEG appears transposed: {eeg.shape}")
        raise ValueError(
            f"unexpected_channels: expected {config.expected_channels}, got {eeg.shape[0]}"
        )
    minimum_samples = int(np.ceil(config.source_hz * config.min_duration_seconds))
    if eeg.shape[1] < minimum_samples:
        raise ValueError(
            f"too_short: expected at least {minimum_samples} samples "
            f"({config.min_duration_seconds:g}s at {config.source_hz} Hz), got {eeg.shape[1]}"
        )

    drop = sorted(set(config.drop_channel_indices), reverse=True)
    for index in drop:
        if index < 0 or index >= eeg.shape[0]:
            raise IndexError(f"cannot drop channel {index} from shape {eeg.shape}")
        eeg = np.delete(eeg, index, axis=0)

    eeg = _repair_nonfinite(eeg, config.max_nonfinite_fraction)
    divisor = gcd(config.source_hz, config.target_hz)
    eeg = resample_poly(
        eeg,
        up=config.target_hz // divisor,
        down=config.source_hz // divisor,
        axis=-1,
    )

    nyquist = config.target_hz / 2.0
    low, high = config.bandpass_hz
    if not 0 < low < high < nyquist:
        raise ValueError(f"invalid bandpass {config.bandpass_hz} for {config.target_hz} Hz")
    sos = butter(4, (low / nyquist, high / nyquist), btype="bandpass", output="sos")
    eeg = sosfiltfilt(sos, eeg, axis=-1)

    if config.notch_hz is not None:
        if not 0 < config.notch_hz < nyquist:
            raise ValueError(f"invalid notch {config.notch_hz} for {config.target_hz} Hz")
        b, a = iirnotch(config.notch_hz / nyquist, config.notch_quality)
        eeg = filtfilt(b, a, eeg, axis=-1)

    if config.standardize_per_recording:
        center = np.median(eeg, axis=-1, keepdims=True)
        scale = np.median(np.abs(eeg - center), axis=-1, keepdims=True) * 1.4826
        scale[scale < 1e-6] = 1.0
        eeg = (eeg - center) / scale

    if eeg.shape[-1] < config.target_hz:
        raise ValueError("recording is shorter than one second after resampling")
    if not np.isfinite(eeg).all():
        raise ValueError("preprocessing produced non-finite values")
    return eeg.astype(np.float32, copy=False)
