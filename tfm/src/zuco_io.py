"""Stream raw sentence EEG from ZuCo Task 1 MATLAB files."""

from dataclasses import dataclass
from pathlib import Path
import re
import warnings

import numpy as np

from .labels import label_lookup, normalize_text


@dataclass
class Recording:
    subject: str
    sentence_id: int
    label: int
    text: str
    eeg: np.ndarray


def subject_from_path(path):
    match = re.search(r"results([A-Za-z0-9]+)_SR", str(path))
    if match:
        return match.group(1)
    return Path(path).stem


def orient_eeg(array):
    array = np.squeeze(np.asarray(array, dtype=np.float64))
    if array.ndim != 2 or min(array.shape) < 2:
        return None
    if array.shape[0] > array.shape[1]:
        array = array.T
    if array.shape[0] > 256:
        warnings.warn(f"unexpected EEG shape {array.shape}; check orientation")
    return array


def _decode_hdf5_text(handle, reference):
    codes = np.asarray(handle[reference]).flatten()
    return "".join(chr(int(code)) for code in codes if int(code) > 0).strip()


def _iter_hdf5(path):
    import h5py

    with h5py.File(path, "r") as handle:
        if "sentenceData" not in handle:
            raise KeyError(f"sentenceData is missing from {path}")
        data = handle["sentenceData"]
        content_refs = np.asarray(data["content"]).flatten()
        raw_refs = np.asarray(data["rawData"]).flatten()
        for content_ref, raw_ref in zip(content_refs, raw_refs):
            content = _decode_hdf5_text(handle, content_ref)
            raw = orient_eeg(np.asarray(handle[raw_ref], dtype=np.float64)) if raw_ref else None
            yield content, raw


def _iter_scipy(path):
    from scipy.io import loadmat

    data = loadmat(path, struct_as_record=False, squeeze_me=True)
    if "sentenceData" not in data:
        raise KeyError(f"sentenceData is missing from {path}")
    for sentence in np.atleast_1d(data["sentenceData"]):
        content = str(getattr(sentence, "content", "") or "").strip()
        raw = getattr(sentence, "rawData", None)
        raw = orient_eeg(raw) if raw is not None and np.size(raw) else None
        yield content, raw


def iter_subject_sentences(path):
    try:
        import h5py

        if h5py.is_hdf5(path):
            yield from _iter_hdf5(path)
            return
    except ImportError:
        pass
    yield from _iter_scipy(path)


def find_subject_files(raw_dir, pattern="results*_SR.mat"):
    files = sorted(Path(raw_dir).glob(pattern))
    if not files:
        raise FileNotFoundError(f"no ZuCo files matching {pattern!r} in {raw_dir}")
    return files


def iter_zuco_recordings(raw_dir, labels_csv, pattern="results*_SR.mat"):
    lookup = label_lookup(labels_csv)
    for path in find_subject_files(raw_dir, pattern):
        subject = subject_from_path(path)
        for content, raw in iter_subject_sentences(path):
            match = lookup.get(normalize_text(content))
            if match is None or raw is None:
                continue
            sentence_id, label = match
            yield Recording(subject, sentence_id, label, content, raw)


def inspect_zuco(raw_dir, labels_csv, pattern="results*_SR.mat"):
    files = find_subject_files(raw_dir, pattern)
    lookup = label_lookup(labels_csv)
    rows = []
    for path in files:
        matched = raw_count = total = 0
        shapes = []
        for content, raw in iter_subject_sentences(path):
            total += 1
            matched += normalize_text(content) in lookup
            if raw is not None:
                raw_count += 1
                shapes.append(tuple(raw.shape))
        rows.append(
            {
                "subject": subject_from_path(path),
                "file": str(path),
                "sentences": total,
                "matched_labels": matched,
                "with_raw_eeg": raw_count,
                "channel_counts": sorted({shape[0] for shape in shapes}),
                "min_samples": min((shape[1] for shape in shapes), default=None),
                "max_samples": max((shape[1] for shape in shapes), default=None),
            }
        )
    return rows

