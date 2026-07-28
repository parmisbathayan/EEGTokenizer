"""Turn cached discrete tokens into simple sentence-level features."""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def token_files(cache_dir):
    return sorted(Path(cache_dir).glob("*/sentence_*.npz"))


def build_sentence_histograms(cache_dir, codebook_size=8192):
    """Average L1-normalized subject histograms for every sentence."""

    grouped = defaultdict(list)
    labels = {}
    token_totals = np.zeros(codebook_size, dtype=np.int64)
    record_rows = []
    paths = token_files(cache_dir)
    if not paths:
        raise FileNotFoundError(f"no token caches found below {cache_dir}")

    for path in paths:
        with np.load(path, allow_pickle=False) as cached:
            tokens = np.asarray(cached["tokens"], dtype=np.int64)
            sentence_id = int(cached["sentence_id"])
            label = int(cached["label"])
            subject = str(cached["subject"])
        if tokens.ndim != 2 or tokens.size == 0:
            raise ValueError(f"invalid token shape {tokens.shape} in {path}")
        if tokens.min() < 0 or tokens.max() >= codebook_size:
            raise ValueError(f"out-of-range token in {path}")
        counts = np.bincount(tokens.ravel(), minlength=codebook_size)
        token_totals += counts
        grouped[sentence_id].append(counts / counts.sum())
        if sentence_id in labels and labels[sentence_id] != label:
            raise ValueError(f"conflicting labels for sentence {sentence_id}")
        labels[sentence_id] = label
        probabilities = counts[counts > 0] / counts.sum()
        record_rows.append(
            {
                "subject": subject,
                "sentence_id": sentence_id,
                "label": label,
                "n_tokens": int(tokens.size),
                "unique_tokens": int(np.count_nonzero(counts)),
                "token_perplexity": float(np.exp(-(probabilities * np.log(probabilities)).sum())),
                "top_token_share": float(counts.max() / counts.sum()),
            }
        )

    sentence_ids = np.asarray(sorted(grouped), dtype=np.int64)
    X = np.stack([np.mean(grouped[sentence_id], axis=0) for sentence_id in sentence_ids])
    y = np.asarray([labels[sentence_id] for sentence_id in sentence_ids], dtype=np.int64)
    metadata = pd.DataFrame(
        {
            "sentence_id": sentence_ids,
            "label": y,
            "n_subjects": [len(grouped[sentence_id]) for sentence_id in sentence_ids],
        }
    )
    used = int(np.count_nonzero(token_totals))
    probabilities = token_totals[token_totals > 0] / token_totals.sum()
    diagnostics = {
        "n_recordings": len(paths),
        "n_sentences": len(sentence_ids),
        "codebook_size": codebook_size,
        "used_tokens": used,
        "codebook_coverage": used / codebook_size,
        "global_token_perplexity": float(
            np.exp(-(probabilities * np.log(probabilities)).sum())
        ),
        "top_token_share": float(token_totals.max() / token_totals.sum()),
        "records": pd.DataFrame(record_rows),
    }
    return X.astype(np.float32), y, metadata, diagnostics

