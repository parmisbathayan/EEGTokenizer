"""Resumable frozen NeuroLM feature extraction from raw ZuCo recordings."""

import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np

from .config import PreprocessConfig
from .preprocess import preprocess_eeg
from .zuco_io import iter_zuco_recordings


def _config_hash(preprocess_config, feature_version, channel_ids, zuco_indices):
    payload = {
        "preprocessing": preprocess_config.to_dict(),
        "feature_version": feature_version,
        "channel_ids": [int(value) for value in channel_ids],
        "zuco_indices": [int(value) for value in zuco_indices],
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def cache_path(cache_dir, subject, sentence_id):
    return Path(cache_dir) / str(subject) / f"sentence_{int(sentence_id):04d}.npz"


def extract_feature_cache(
    raw_dir,
    labels_csv,
    cache_dir,
    encoder,
    preprocess_config=PreprocessConfig(),
    overwrite=False,
    limit=None,
    progress_every=25,
):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    signature = _config_hash(
        preprocess_config,
        encoder.feature_version,
        encoder.channel_ids,
        encoder.zuco_indices,
    )
    report = {"written": 0, "reused": 0, "failed": 0, "failures": []}
    started = time.perf_counter()

    def save_runtime_status(processed, stage="extracting"):
        payload = {
            "stage": stage,
            "processed": int(processed),
            "written": report["written"],
            "reused": report["reused"],
            "failed": report["failed"],
            "elapsed_minutes": (time.perf_counter() - started) / 60,
            "signature": signature,
        }
        temporary = cache_dir / "runtime_status.tmp.json"
        temporary.write_text(json.dumps(payload, indent=2))
        temporary.replace(cache_dir / "runtime_status.json")

    save_runtime_status(0)

    for index, recording in enumerate(iter_zuco_recordings(raw_dir, labels_csv)):
        if limit is not None and index >= limit:
            break
        output = cache_path(cache_dir, recording.subject, recording.sentence_id)
        reuse = False
        if output.exists() and not overwrite:
            try:
                with np.load(output, allow_pickle=False) as cached:
                    reuse = str(cached["signature"]) == signature
            except Exception:
                reuse = False
        if reuse:
            report["reused"] += 1
        else:
            try:
                eeg = preprocess_eeg(recording.eeg, preprocess_config)
                feature, details = encoder.encode_recording(eeg)
                output.parent.mkdir(parents=True, exist_ok=True)
                temporary = output.with_suffix(".tmp.npz")
                np.savez_compressed(
                    temporary,
                    feature=feature.astype(np.float16),
                    subject=np.asarray(recording.subject),
                    sentence_id=np.int64(recording.sentence_id),
                    label=np.int64(recording.label),
                    channels=np.int64(details["channels"]),
                    seconds=np.int64(details["seconds"]),
                    patches=np.int64(details["patches"]),
                    feature_norm=np.float32(details["feature_norm"]),
                    signature=np.asarray(signature),
                    feature_version=np.asarray(encoder.feature_version),
                )
                temporary.replace(output)
                report["written"] += 1
            except Exception as error:
                report["failed"] += 1
                report["failures"].append(
                    {
                        "subject": recording.subject,
                        "sentence_id": int(recording.sentence_id),
                        "error": repr(error),
                        "traceback": traceback.format_exc(),
                    }
                )
        processed = index + 1
        if progress_every and processed % progress_every == 0:
            save_runtime_status(processed)
            minutes = (time.perf_counter() - started) / 60
            print(
                f"Processed {processed} recordings in {minutes:.1f} min "
                f"(written={report['written']}, reused={report['reused']}, "
                f"failed={report['failed']})",
                flush=True,
            )

    manifest = {
        "signature": signature,
        "preprocessing": preprocess_config.to_dict(),
        "feature_version": encoder.feature_version,
        "channel_ids": [int(value) for value in encoder.channel_ids],
        "zuco_indices": [int(value) for value in encoder.zuco_indices],
        "checkpoint_path": encoder.checkpoint_path,
        "upstream_repo_dir": encoder.repo_dir,
        "checkpoint_load": encoder.load_report,
        "report": report,
    }
    temporary = cache_dir / "extraction_manifest.tmp.json"
    temporary.write_text(json.dumps(manifest, indent=2))
    temporary.replace(cache_dir / "extraction_manifest.json")
    save_runtime_status(report["written"] + report["reused"] + report["failed"], "complete")
    return manifest
