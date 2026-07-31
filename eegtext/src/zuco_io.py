"""Read sentence text and raw EEG from ZuCo MATLAB exports."""

from dataclasses import dataclass
from pathlib import Path
import re
import warnings

import numpy as np


@dataclass
class SentenceRecord:
    subject: str
    ordinal: int
    text: str
    eeg: object
    source_file: str


def subject_from_path(path, task=None):
    """Extract the participant code without assuming one particular task."""

    stem = Path(path).stem
    match = re.search(r"results(.+?)_(?:SR|NR|TSR)(?:$|_)", stem, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r"(?:gip|results)[_-]([A-Za-z0-9]+)[_-]", stem)
    if match:
        return match.group(1)
    if task:
        stem = re.sub(rf"[_-]{re.escape(str(task))}.*$", "", stem, flags=re.IGNORECASE)
    return stem


def orient_eeg(array, expected_channels=105):
    """Orient an EEG matrix as channels x samples without losing short trials."""

    if array is None:
        return None
    array = np.squeeze(np.asarray(array, dtype=np.float64))
    if array.ndim != 2 or min(array.shape) < 2:
        return None
    if array.shape[0] == expected_channels:
        return array
    if array.shape[1] == expected_channels:
        return array.T
    if array.shape[0] > array.shape[1]:
        array = array.T
    if array.shape[0] > 256:
        warnings.warn(f"unexpected EEG shape after orientation: {array.shape}")
    return array


def _decode_hdf5_text(handle, reference):
    if not reference:
        return ""
    value = np.asarray(handle[reference])
    if value.dtype.kind in {"S", "U"}:
        return "".join(value.astype(str).flatten()).strip()
    codes = value.flatten()
    return "".join(chr(int(code)) for code in codes if int(code) > 0).strip()


def _iter_hdf5(path, expected_channels):
    import h5py

    with h5py.File(path, "r") as handle:
        if "sentenceData" not in handle:
            raise KeyError(f"sentenceData is missing from {path}")
        data = handle["sentenceData"]
        if "content" not in data or "rawData" not in data:
            raise KeyError(f"sentenceData in {path} lacks content or rawData")
        content_refs = np.asarray(data["content"]).flatten()
        raw_refs = np.asarray(data["rawData"]).flatten()
        if len(content_refs) != len(raw_refs):
            raise ValueError(f"content/rawData length mismatch in {path}")
        for content_ref, raw_ref in zip(content_refs, raw_refs):
            content = _decode_hdf5_text(handle, content_ref)
            raw = None
            if raw_ref:
                raw = orient_eeg(handle[raw_ref], expected_channels)
            yield content, raw


def _iter_scipy(path, expected_channels):
    from scipy.io import loadmat

    data = loadmat(path, struct_as_record=False, squeeze_me=True)
    if "sentenceData" not in data:
        raise KeyError(f"sentenceData is missing from {path}")
    for sentence in np.atleast_1d(data["sentenceData"]):
        content = str(getattr(sentence, "content", "") or "").strip()
        raw = getattr(sentence, "rawData", None)
        if raw is not None and np.size(raw):
            raw = orient_eeg(raw, expected_channels)
        else:
            raw = None
        yield content, raw


def iter_sentence_values(path, expected_channels=105):
    """Yield `(text, eeg)` from either MATLAB storage format."""

    try:
        import h5py

        if h5py.is_hdf5(path):
            yield from _iter_hdf5(path, expected_channels)
            return
    except ImportError:
        pass
    yield from _iter_scipy(path, expected_channels)


def iter_subject_records(path, task=None, expected_channels=105):
    subject = subject_from_path(path, task)
    for ordinal, (text, eeg) in enumerate(
        iter_sentence_values(path, expected_channels), start=1
    ):
        yield SentenceRecord(subject, ordinal, text, eeg, str(Path(path).resolve()))


def find_subject_files(raw_dir, pattern, recursive=True):
    root = Path(raw_dir)
    if not root.exists():
        raise FileNotFoundError(f"ZuCo directory does not exist: {root}")
    iterator = root.rglob(pattern) if recursive else root.glob(pattern)
    files = sorted(path for path in iterator if path.is_file())
    if not files:
        raise FileNotFoundError(f"no files matching {pattern!r} under {root}")
    return files
