"""Cached token extraction from raw ZuCo recordings."""

import hashlib
import json
from pathlib import Path
import traceback

import numpy as np

from .config import PreprocessConfig
from .preprocess import preprocess_eeg
from .zuco_io import iter_zuco_recordings


def _config_hash(config):
    payload = json.dumps(config.to_dict(), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def cache_path(cache_dir, subject, sentence_id):
    return Path(cache_dir) / str(subject) / f"sentence_{int(sentence_id):04d}.npz"


def extract_token_cache(
    raw_dir,
    labels_csv,
    cache_dir,
    tokenizer,
    preprocess_config=PreprocessConfig(),
    overwrite=False,
    limit=None,
):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    report = {"written": 0, "reused": 0, "failed": 0, "failures": []}
    config_hash = _config_hash(preprocess_config)

    for index, recording in enumerate(iter_zuco_recordings(raw_dir, labels_csv)):
        if limit is not None and index >= limit:
            break
        output = cache_path(cache_dir, recording.subject, recording.sentence_id)
        if output.exists() and not overwrite:
            report["reused"] += 1
            continue
        try:
            eeg = preprocess_eeg(recording.eeg, preprocess_config)
            tokens = tokenizer.tokenize(eeg)
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix(".tmp.npz")
            np.savez_compressed(
                temporary,
                tokens=tokens,
                subject=np.asarray(recording.subject),
                sentence_id=np.int64(recording.sentence_id),
                label=np.int64(recording.label),
                n_channels=np.int64(tokens.shape[0]),
                n_tokens_per_channel=np.int64(tokens.shape[1]),
                preprocess_hash=np.asarray(config_hash),
            )
            temporary.replace(output)
            report["written"] += 1
        except Exception as error:  # continue long Colab jobs while preserving evidence
            report["failed"] += 1
            report["failures"].append(
                {
                    "subject": recording.subject,
                    "sentence_id": recording.sentence_id,
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                }
            )

    manifest = {
        "preprocessing": preprocess_config.to_dict(),
        "preprocess_hash": config_hash,
        "tokenizer": {
            "repo_dir": getattr(tokenizer, "repo_dir", None),
            "checkpoint_path": getattr(tokenizer, "checkpoint_path", None),
            "codebook_size": getattr(tokenizer, "codebook_size", None),
        },
        "checkpoint_load": getattr(tokenizer, "load_report", None),
        "report": report,
    }
    with (cache_dir / "extraction_manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest
