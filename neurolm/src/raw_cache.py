"""Resumable subject-packed raw EEG cache for the bounded V2 EEGNet test."""

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd

from .config import PreprocessConfig
from .labels import label_lookup, normalize_text
from .zuco_io import find_subject_files, iter_subject_sentences, subject_from_path


RAW_CACHE_FORMAT_VERSION = 1


@dataclass(frozen=True)
class RawEEGRecord:
    """One preprocessed reader/sentence recording held as channels x samples."""

    subject: str
    sentence_id: int
    label: int
    eeg: np.ndarray


def _sha256_file(path, chunk_size=2**20):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_signature(config, labels_sha256=None):
    payload = {
        "format_version": RAW_CACHE_FORMAT_VERSION,
        "preprocessing": config.to_dict(),
        "storage_dtype": "float16",
        "labels_sha256": labels_sha256,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _string_scalar(value):
    value = np.asarray(value)
    return str(value.item() if value.ndim == 0 else value.reshape(-1)[0])


def _json_equivalent(left, right):
    """Compare config payloads after JSON normalizes tuples into lists."""

    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def _pack_is_current(path, signature, source_path=None):
    try:
        with np.load(path, allow_pickle=False) as cached:
            current = (
                int(cached["format_version"]) == RAW_CACHE_FORMAT_VERSION
                and _string_scalar(cached["signature"]) == signature
                and len(cached["sentence_ids"]) == len(cached["sample_lengths"])
                and len(cached["offsets"]) == len(cached["sentence_ids"]) + 1
            )
            if source_path is not None:
                stat = Path(source_path).stat()
                current = current and int(cached["source_size_bytes"]) == stat.st_size
                current = current and int(cached["source_mtime_ns"]) == stat.st_mtime_ns
            return current
    except Exception:
        return False


def _write_subject_pack(
    path,
    subject,
    recordings,
    signature,
    config,
    source_path=None,
):
    if not recordings:
        raise ValueError(f"subject {subject} has no usable labelled recordings")
    lengths = np.asarray([eeg.shape[1] for _, _, eeg in recordings], dtype=np.int32)
    sizes = lengths.astype(np.int64) * int(config.expected_channels - len(config.drop_channel_indices))
    offsets = np.concatenate(
        [np.zeros(1, dtype=np.int64), np.cumsum(sizes, dtype=np.int64)]
    )
    values = np.concatenate(
        [np.asarray(eeg, dtype=np.float16).reshape(-1) for _, _, eeg in recordings]
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    source_stat = Path(source_path).stat() if source_path is not None else None
    np.savez_compressed(
        temporary,
        format_version=np.int64(RAW_CACHE_FORMAT_VERSION),
        signature=np.asarray(signature),
        subject=np.asarray(subject),
        channels=np.int64(config.expected_channels - len(config.drop_channel_indices)),
        sample_rate_hz=np.int64(config.target_hz),
        source_size_bytes=np.int64(source_stat.st_size if source_stat else -1),
        source_mtime_ns=np.int64(source_stat.st_mtime_ns if source_stat else -1),
        eeg_values=values,
        offsets=offsets,
        sample_lengths=lengths,
        sentence_ids=np.asarray([row[0] for row in recordings], dtype=np.int64),
        labels=np.asarray([row[1] for row in recordings], dtype=np.int64),
    )
    temporary.replace(path)


def build_raw_subject_packs(
    raw_dir,
    labels_csv,
    pack_dir,
    preprocess_config=PreprocessConfig(),
    overwrite=False,
    progress_every=50,
):
    """Preprocess ZuCo once and atomically save one reusable pack per subject."""

    from .preprocess import preprocess_eeg

    pack_dir = Path(pack_dir)
    pack_dir.mkdir(parents=True, exist_ok=True)
    lookup = label_lookup(labels_csv)
    labels_sha256 = _sha256_file(labels_csv)
    signature = _cache_signature(preprocess_config, labels_sha256=labels_sha256)
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
        temporary = pack_dir / "runtime_status.tmp.json"
        temporary.write_text(json.dumps(payload, indent=2))
        temporary.replace(pack_dir / "runtime_status.json")

    write_status("starting")
    for path in find_subject_files(raw_dir):
        subject = subject_from_path(path)
        output = pack_dir / f"{subject}.npz"
        if output.exists() and not overwrite and _pack_is_current(
            output, signature, source_path=path
        ):
            report["subjects_reused"] += 1
            write_status("packing", subject)
            print(f"Reused raw EEG pack: {subject}", flush=True)
            continue

        recordings = []
        for position, (content, raw) in enumerate(iter_subject_sentences(path), start=1):
            match = lookup.get(normalize_text(content))
            if match is None or raw is None:
                continue
            sentence_id, label = match
            try:
                eeg = preprocess_eeg(raw, preprocess_config)
                recordings.append((int(sentence_id), int(label), eeg))
            except Exception as error:
                report["failed"] += 1
                report["failures"].append(
                    {
                        "subject": subject,
                        "sentence_id": int(sentence_id),
                        "error": repr(error),
                        "traceback": traceback.format_exc(),
                    }
                )
            if progress_every and position % progress_every == 0:
                print(
                    f"{subject}: inspected {position} rows, kept {len(recordings)}",
                    flush=True,
                )
                write_status("packing", subject)
        _write_subject_pack(
            output,
            subject,
            recordings,
            signature,
            preprocess_config,
            source_path=path,
        )
        report["subjects_written"] += 1
        report["recordings_written"] += len(recordings)
        print(f"Saved raw EEG pack: {subject} ({len(recordings)} recordings)", flush=True)
        write_status("packing", subject)

    manifest = {
        "format_version": RAW_CACHE_FORMAT_VERSION,
        "signature": signature,
        "labels_sha256": labels_sha256,
        "preprocessing": preprocess_config.to_dict(),
        "report": report,
    }
    temporary = pack_dir / "cache_manifest.tmp.json"
    temporary.write_text(json.dumps(manifest, indent=2))
    temporary.replace(pack_dir / "cache_manifest.json")
    write_status("complete")
    return manifest


def _load_subject_pack(path, expected_signature=None):
    path = Path(path)
    with np.load(path, allow_pickle=False) as cached:
        signature = _string_scalar(cached["signature"])
        if expected_signature is not None and signature != expected_signature:
            raise ValueError(f"stale raw cache signature in {path}")
        subject = _string_scalar(cached["subject"])
        channels = int(cached["channels"])
        values = np.asarray(cached["eeg_values"], dtype=np.float16)
        offsets = np.asarray(cached["offsets"], dtype=np.int64)
        lengths = np.asarray(cached["sample_lengths"], dtype=np.int64)
        sentence_ids = np.asarray(cached["sentence_ids"], dtype=np.int64)
        labels = np.asarray(cached["labels"], dtype=np.int64)
    if not (len(lengths) == len(sentence_ids) == len(labels)):
        raise ValueError(f"inconsistent record metadata in {path}")
    if len(offsets) != len(lengths) + 1 or offsets[-1] != len(values):
        raise ValueError(f"invalid packed offsets in {path}")
    records = []
    for index, (sentence_id, label, samples) in enumerate(
        zip(sentence_ids, labels, lengths)
    ):
        start, stop = int(offsets[index]), int(offsets[index + 1])
        expected = channels * int(samples)
        if stop - start != expected:
            raise ValueError(f"invalid packed EEG size for row {index} in {path}")
        eeg = values[start:stop].reshape(channels, int(samples))
        records.append(RawEEGRecord(subject, int(sentence_id), int(label), eeg))
    return records, signature, values


def load_raw_records(pack_dir, preprocess_config=PreprocessConfig()):
    """Load all subject packs into compact shared float16 arrays."""

    paths = sorted(Path(pack_dir).glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no raw EEG subject packs found in {pack_dir}")
    manifest_path = Path(pack_dir) / "cache_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"raw cache manifest is missing from {pack_dir}")
    manifest = json.loads(manifest_path.read_text())
    if not _json_equivalent(
        manifest.get("preprocessing"), preprocess_config.to_dict()
    ):
        raise ValueError("raw cache preprocessing does not match the requested configuration")
    expected_signature = str(manifest["signature"])
    records = []
    backing_arrays = []
    fingerprints = hashlib.sha256()
    for position, path in enumerate(paths, start=1):
        subject_records, signature, backing = _load_subject_pack(
            path, expected_signature=expected_signature
        )
        records.extend(subject_records)
        backing_arrays.append(backing)
        fingerprints.update(path.name.encode())
        fingerprints.update(signature.encode())
        fingerprints.update(np.asarray(backing).view(np.uint8))
        for record in subject_records:
            fingerprints.update(
                f"{record.subject}|{record.sentence_id}|{record.label}|{record.eeg.shape}\n".encode()
            )
        print(
            f"Loaded subject pack {position}/{len(paths)}: {path.stem} "
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
                "channels": record.eeg.shape[0],
                "samples": record.eeg.shape[1],
                "seconds": record.eeg.shape[1] / preprocess_config.target_hz,
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
        "minimum_seconds": float(metadata["seconds"].min()),
        "maximum_seconds": float(metadata["seconds"].max()),
        "memory_bytes_float16": int(sum(array.nbytes for array in backing_arrays)),
        "cache_signature": expected_signature,
        "dataset_fingerprint": fingerprints.hexdigest(),
    }
    return records, metadata, report
