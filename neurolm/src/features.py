"""Aggregate reader-level frozen NeuroLM features into sentence features."""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def feature_files(cache_dir):
    return sorted(Path(cache_dir).glob("*/sentence_*.npz"))


def _mean_pairwise_cosine(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return np.nan
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    valid = norms[:, 0] > 0
    values = values[valid] / norms[valid]
    if len(values) < 2:
        return np.nan
    similarities = values @ values.T
    upper = similarities[np.triu_indices(len(values), k=1)]
    return float(upper.mean())


def build_sentence_features(cache_dir):
    grouped = defaultdict(list)
    labels = {}
    record_rows = []
    paths = feature_files(cache_dir)
    if not paths:
        raise FileNotFoundError(f"no NeuroLM feature caches found below {cache_dir}")

    feature_dim = None
    for path in paths:
        with np.load(path, allow_pickle=False) as cached:
            feature = np.asarray(cached["feature"], dtype=np.float32)
            sentence_id = int(cached["sentence_id"])
            label = int(cached["label"])
            subject = str(cached["subject"])
            seconds = int(cached["seconds"])
            patches = int(cached["patches"])
        if feature.ndim != 1 or not feature.size or not np.isfinite(feature).all():
            raise ValueError(f"invalid feature in {path}: shape={feature.shape}")
        if feature_dim is None:
            feature_dim = feature.size
        elif feature.size != feature_dim:
            raise ValueError(f"feature dimension mismatch in {path}")
        if sentence_id in labels and labels[sentence_id] != label:
            raise ValueError(f"conflicting labels for sentence {sentence_id}")
        labels[sentence_id] = label
        grouped[sentence_id].append(feature)
        record_rows.append(
            {
                "subject": subject,
                "sentence_id": sentence_id,
                "label": label,
                "seconds": seconds,
                "patches": patches,
                "feature_norm": float(np.linalg.norm(feature)),
            }
        )

    sentence_ids = np.asarray(sorted(grouped), dtype=np.int64)
    X = np.stack([np.mean(grouped[key], axis=0) for key in sentence_ids]).astype(np.float32)
    y = np.asarray([labels[key] for key in sentence_ids], dtype=np.int64)
    metadata = pd.DataFrame(
        {
            "sentence_id": sentence_ids,
            "label": y,
            "n_subjects": [len(grouped[key]) for key in sentence_ids],
            "reader_cosine": [_mean_pairwise_cosine(grouped[key]) for key in sentence_ids],
        }
    )
    records = pd.DataFrame(record_rows)
    diagnostics = {
        "n_recordings": len(paths),
        "n_sentences": len(sentence_ids),
        "feature_dim": int(feature_dim),
        "nonconstant_dimensions": int(np.count_nonzero(X.std(axis=0) > 1e-8)),
        "median_reader_cosine": float(metadata["reader_cosine"].median()),
        "records": records,
    }
    return X, y, metadata, diagnostics
