"""Resumable structured NeuroLM feature cache for version 3."""

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd

from .raw_cache import _load_subject_pack


STRUCTURED_CACHE_FORMAT_VERSION = 1
STRUCTURED_FEATURE_VERSION = "neurolm_b_channel_mean_and_time_mean_v3"


@dataclass(frozen=True)
class StructuredRecord:
    subject: str
    sentence_id: int
    label: int
    channel_tokens: np.ndarray
    time_tokens: np.ndarray


def _string_scalar(value):
    value = np.asarray(value)
    return str(value.item() if value.ndim == 0 else value.reshape(-1)[0])


def _signature(raw_signature, encoder):
    checkpoint = Path(encoder.checkpoint_path)
    checkpoint_stat = checkpoint.stat()
    payload = {
        "format_version": STRUCTURED_CACHE_FORMAT_VERSION,
        "feature_version": STRUCTURED_FEATURE_VERSION,
        "raw_cache_signature": raw_signature,
        "encoder_config": encoder.config.to_dict(),
        "channel_ids": [int(value) for value in encoder.channel_ids],
        "zuco_indices": [int(value) for value in encoder.zuco_indices],
        "checkpoint_name": checkpoint.name,
        "checkpoint_bytes": checkpoint_stat.st_size,
        "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
        "storage_dtype": "float16",
    }
    encoded = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest(), payload


def _pack_is_current(path, signature, raw_pack_path):
    try:
        raw_stat = Path(raw_pack_path).stat()
        with np.load(path, allow_pickle=False) as cached:
            return (
                int(cached["format_version"]) == STRUCTURED_CACHE_FORMAT_VERSION
                and _string_scalar(cached["signature"]) == signature
                and int(cached["source_raw_pack_bytes"]) == raw_stat.st_size
                and int(cached["source_raw_pack_mtime_ns"]) == raw_stat.st_mtime_ns
                and len(cached["sentence_ids"]) == len(cached["seconds"])
            )
    except Exception:
        return False


def _write_subject_pack(
    path,
    subject,
    rows,
    signature,
    raw_pack_path,
    expected_channels,
    embedding_size,
):
    if not rows:
        raise ValueError(f"subject {subject} produced no structured features")
    channel_values = np.stack([row[2] for row in rows]).astype(np.float16, copy=False)
    seconds = np.asarray([row[3].shape[0] for row in rows], dtype=np.int32)
    sizes = seconds.astype(np.int64) * embedding_size
    time_offsets = np.concatenate(
        [np.zeros(1, dtype=np.int64), np.cumsum(sizes, dtype=np.int64)]
    )
    time_values = np.concatenate(
        [np.asarray(row[3], dtype=np.float16).reshape(-1) for row in rows]
    )
    if channel_values.shape[1:] != (expected_channels, embedding_size):
        raise ValueError(f"unexpected channel-token shape {channel_values.shape}")
    raw_stat = Path(raw_pack_path).stat()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        format_version=np.int64(STRUCTURED_CACHE_FORMAT_VERSION),
        feature_version=np.asarray(STRUCTURED_FEATURE_VERSION),
        signature=np.asarray(signature),
        subject=np.asarray(subject),
        channels=np.int64(expected_channels),
        embedding_size=np.int64(embedding_size),
        source_raw_pack_bytes=np.int64(raw_stat.st_size),
        source_raw_pack_mtime_ns=np.int64(raw_stat.st_mtime_ns),
        sentence_ids=np.asarray([row[0] for row in rows], dtype=np.int64),
        labels=np.asarray([row[1] for row in rows], dtype=np.int64),
        seconds=seconds,
        channel_values=channel_values,
        time_values=time_values,
        time_offsets=time_offsets,
    )
    temporary.replace(path)


def extract_structured_subject_packs(
    raw_pack_dir,
    output_dir,
    encoder,
    overwrite=False,
    progress_every=25,
):
    """Encode raw subject packs and save channel/time summaries per reader."""

    raw_pack_dir = Path(raw_pack_dir)
    raw_manifest_path = raw_pack_dir / "cache_manifest.json"
    if not raw_manifest_path.exists():
        raise FileNotFoundError(
            f"V2 raw cache manifest is missing: {raw_manifest_path}; finish V2 Cell 3"
        )
    raw_manifest = json.loads(raw_manifest_path.read_text())
    raw_signature = str(raw_manifest["signature"])
    raw_paths = sorted(raw_pack_dir.glob("*.npz"))
    if not raw_paths:
        raise FileNotFoundError(f"no V2 raw subject packs found in {raw_pack_dir}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    signature, signature_payload = _signature(raw_signature, encoder)
    expected_channels = len(encoder.channel_ids)
    embedding_size = encoder.config.n_embd
    report = {
        "signature": signature,
        "subjects_written": 0,
        "subjects_reused": 0,
        "recordings_written": 0,
        "failed": 0,
        "failures": [],
    }
    started = time.perf_counter()

    def write_status(stage, subject=None):
        payload = {
            "stage": stage,
            "subject": subject,
            "elapsed_minutes": (time.perf_counter() - started) / 60,
            **{key: value for key, value in report.items() if key != "failures"},
            "failure_count": len(report["failures"]),
        }
        temporary = output_dir / "runtime_status.tmp.json"
        temporary.write_text(json.dumps(payload, indent=2))
        temporary.replace(output_dir / "runtime_status.json")

    write_status("starting")
    for raw_path in raw_paths:
        output = output_dir / raw_path.name
        if output.exists() and not overwrite and _pack_is_current(
            output, signature, raw_path
        ):
            report["subjects_reused"] += 1
            print(f"Reused structured NeuroLM pack: {raw_path.stem}", flush=True)
            write_status("extracting", raw_path.stem)
            continue
        raw_records, _, _ = _load_subject_pack(
            raw_path, expected_signature=raw_signature
        )
        rows = []
        for position, record in enumerate(raw_records, start=1):
            try:
                embeddings, details = encoder.encode_recording_tokens(record.eeg)
                if embeddings.shape != (
                    details["seconds"],
                    expected_channels,
                    embedding_size,
                ):
                    raise ValueError(
                        f"unexpected encoder-token shape {embeddings.shape}"
                    )
                channel_tokens = embeddings.mean(axis=0)
                time_tokens = embeddings.mean(axis=1)
                if not (
                    np.isfinite(channel_tokens).all()
                    and np.isfinite(time_tokens).all()
                ):
                    raise ValueError("structured reduction produced non-finite values")
                rows.append(
                    (
                        int(record.sentence_id),
                        int(record.label),
                        channel_tokens,
                        time_tokens,
                    )
                )
            except Exception as error:
                report["failed"] += 1
                report["failures"].append(
                    {
                        "subject": record.subject,
                        "sentence_id": int(record.sentence_id),
                        "error": repr(error),
                        "traceback": traceback.format_exc(),
                    }
                )
            if progress_every and position % progress_every == 0:
                print(
                    f"{raw_path.stem}: encoded {position}/{len(raw_records)} "
                    f"(kept={len(rows)}, failed={report['failed']})",
                    flush=True,
                )
                write_status("extracting", raw_path.stem)
        _write_subject_pack(
            output,
            raw_path.stem,
            rows,
            signature,
            raw_path,
            expected_channels,
            embedding_size,
        )
        report["subjects_written"] += 1
        report["recordings_written"] += len(rows)
        print(
            f"Saved structured NeuroLM pack: {raw_path.stem} ({len(rows)} recordings)",
            flush=True,
        )
        write_status("extracting", raw_path.stem)

    manifest = {
        "format_version": STRUCTURED_CACHE_FORMAT_VERSION,
        "feature_version": STRUCTURED_FEATURE_VERSION,
        "signature": signature,
        "signature_payload": signature_payload,
        "encoder_load_report": encoder.load_report,
        "report": report,
    }
    temporary = output_dir / "cache_manifest.tmp.json"
    temporary.write_text(json.dumps(manifest, indent=2))
    temporary.replace(output_dir / "cache_manifest.json")
    write_status("complete")
    return manifest


def _load_structured_pack(path, expected_signature):
    path = Path(path)
    with np.load(path, allow_pickle=False) as cached:
        signature = _string_scalar(cached["signature"])
        if signature != expected_signature:
            raise ValueError(f"stale structured cache signature in {path}")
        subject = _string_scalar(cached["subject"])
        channels = int(cached["channels"])
        embedding_size = int(cached["embedding_size"])
        sentence_ids = np.asarray(cached["sentence_ids"], dtype=np.int64)
        labels = np.asarray(cached["labels"], dtype=np.int64)
        seconds = np.asarray(cached["seconds"], dtype=np.int64)
        channel_values = np.asarray(cached["channel_values"], dtype=np.float16)
        time_values = np.asarray(cached["time_values"], dtype=np.float16)
        offsets = np.asarray(cached["time_offsets"], dtype=np.int64)
    count = len(sentence_ids)
    if not (len(labels) == len(seconds) == count):
        raise ValueError(f"inconsistent structured metadata in {path}")
    if channel_values.shape != (count, channels, embedding_size):
        raise ValueError(f"invalid channel values in {path}: {channel_values.shape}")
    if len(offsets) != count + 1 or offsets[-1] != len(time_values):
        raise ValueError(f"invalid time-token offsets in {path}")
    records = []
    for index in range(count):
        start, stop = int(offsets[index]), int(offsets[index + 1])
        if stop - start != int(seconds[index]) * embedding_size:
            raise ValueError(f"invalid time-token size for row {index} in {path}")
        time_tokens = time_values[start:stop].reshape(int(seconds[index]), embedding_size)
        records.append(
            StructuredRecord(
                subject=subject,
                sentence_id=int(sentence_ids[index]),
                label=int(labels[index]),
                channel_tokens=channel_values[index],
                time_tokens=time_tokens,
            )
        )
    return records, channel_values, time_values


def load_structured_records(cache_dir):
    """Load compact V3 packs and return records, metadata, and diagnostics."""

    cache_dir = Path(cache_dir)
    manifest_path = cache_dir / "cache_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"structured cache manifest is missing from {cache_dir}")
    manifest = json.loads(manifest_path.read_text())
    signature = str(manifest["signature"])
    paths = sorted(cache_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no structured subject packs found in {cache_dir}")
    records = []
    backing = []
    fingerprint = hashlib.sha256()
    for position, path in enumerate(paths, start=1):
        subject_records, channel_values, time_values = _load_structured_pack(
            path, signature
        )
        records.extend(subject_records)
        backing.extend([channel_values, time_values])
        fingerprint.update(path.name.encode())
        fingerprint.update(np.asarray(channel_values).view(np.uint8))
        fingerprint.update(np.asarray(time_values).view(np.uint8))
        for record in subject_records:
            fingerprint.update(
                f"{record.subject}|{record.sentence_id}|{record.label}|"
                f"{record.channel_tokens.shape}|{record.time_tokens.shape}\n".encode()
            )
        print(
            f"Loaded structured pack {position}/{len(paths)}: {path.stem} "
            f"({len(subject_records)} recordings)",
            flush=True,
        )

    labels_by_sentence = {}
    for record in records:
        existing = labels_by_sentence.setdefault(record.sentence_id, record.label)
        if existing != record.label:
            raise ValueError(f"conflicting label for sentence {record.sentence_id}")
    counts = Counter(record.sentence_id for record in records)
    metadata = pd.DataFrame(
        [
            {
                "subject": record.subject,
                "sentence_id": record.sentence_id,
                "label": record.label,
                "channels": record.channel_tokens.shape[0],
                "seconds": record.time_tokens.shape[0],
                "embedding_size": record.channel_tokens.shape[1],
            }
            for record in records
        ]
    )
    report = {
        "n_subject_packs": len(paths),
        "n_recordings": len(records),
        "n_sentences": len(labels_by_sentence),
        "n_subjects": len({record.subject for record in records}),
        "minimum_readers_per_sentence": min(counts.values()),
        "maximum_readers_per_sentence": max(counts.values()),
        "minimum_seconds": int(metadata["seconds"].min()),
        "maximum_seconds": int(metadata["seconds"].max()),
        "channels": int(metadata["channels"].iloc[0]),
        "embedding_size": int(metadata["embedding_size"].iloc[0]),
        "memory_bytes_float16": int(sum(array.nbytes for array in backing)),
        "cache_signature": signature,
        "dataset_fingerprint": fingerprint.hexdigest(),
    }
    return records, metadata, report
