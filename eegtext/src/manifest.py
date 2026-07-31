"""Build auditable recording manifests before any model sees the corpus."""

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import csv
import json
import os
from pathlib import Path
import statistics

import numpy as np

from .labels import load_label_lookup
from .text import normalize_text, text_hash
from .zuco_io import SentenceRecord, find_subject_files, iter_subject_records


@dataclass(frozen=True)
class ManifestRow:
    recording_id: str
    dataset: str
    release: str
    task: str
    subject: str
    sentence_ordinal: int
    sentence_id: object
    text: str
    normalized_text: str
    text_hash: str
    sentence_group_id: str
    sentiment_label: object
    source_file: str
    channels: object
    samples: object
    source_hz: float
    duration_seconds: object
    nonfinite_fraction: object
    usable: bool
    exclusion_reason: str


def _row_from_record(record, config, label_lookup):
    normalized = normalize_text(record.text)
    digest = text_hash(record.text)
    reasons = []
    channels = samples = duration = nonfinite_fraction = None
    if not normalized:
        reasons.append("empty_text")
    if record.eeg is None:
        reasons.append("missing_raw_eeg")
    else:
        channels, samples = map(int, record.eeg.shape)
        duration = float(samples / config.source_hz)
        nonfinite_fraction = float(1.0 - np.isfinite(record.eeg).mean())
        if channels != config.expected_channels:
            reasons.append("unexpected_channels")
        if samples < config.minimum_samples:
            reasons.append("too_short")
        if nonfinite_fraction > config.maximum_nonfinite_fraction:
            reasons.append("too_many_nonfinite")
    sentence_id = sentiment_label = None
    label = label_lookup.get(normalized)
    if label is not None:
        sentence_id, sentiment_label = label
    recording_id = (
        f"{config.dataset}:{config.release}:{config.task}:"
        f"{record.subject}:{record.ordinal:04d}"
    )
    return ManifestRow(
        recording_id=recording_id,
        dataset=config.dataset,
        release=config.release,
        task=config.task,
        subject=record.subject,
        sentence_ordinal=int(record.ordinal),
        sentence_id=sentence_id,
        text=str(record.text),
        normalized_text=normalized,
        text_hash=digest,
        sentence_group_id=digest,
        sentiment_label=sentiment_label,
        source_file=record.source_file,
        channels=channels,
        samples=samples,
        source_hz=float(config.source_hz),
        duration_seconds=duration,
        nonfinite_fraction=nonfinite_fraction,
        usable=not reasons,
        exclusion_reason=";".join(reasons),
    )


def build_manifest(raw_dir, config):
    """Return manifest rows and file errors without hiding failed subjects."""

    label_lookup = load_label_lookup(config.labels_csv)
    rows = []
    for path in find_subject_files(raw_dir, config.pattern, config.recursive):
        try:
            records = iter_subject_records(path, config.task, config.expected_channels)
            rows.extend(_row_from_record(record, config, label_lookup) for record in records)
        except Exception as error:
            subject = path.stem
            record = SentenceRecord(subject, 0, "", None, str(path.resolve()))
            row = asdict(_row_from_record(record, config, label_lookup))
            row["recording_id"] = (
                f"{config.dataset}:{config.release}:{config.task}:{subject}:file_error"
            )
            row["exclusion_reason"] = f"file_error:{type(error).__name__}:{error}"
            rows.append(ManifestRow(**row))
    return rows


def summarize_manifest(rows):
    usable = [row for row in rows if row.usable]
    exclusions = Counter(
        reason
        for row in rows
        for reason in row.exclusion_reason.split(";")
        if reason
    )
    groups = defaultdict(list)
    for row in rows:
        if row.normalized_text:
            groups[row.sentence_group_id].append(row)
    cross_context = 0
    for group_rows in groups.values():
        contexts = {(row.dataset, row.release, row.task) for row in group_rows}
        cross_context += len(contexts) > 1
    durations = [row.duration_seconds for row in usable if row.duration_seconds is not None]
    labels = Counter(str(row.sentiment_label) for row in usable if row.sentiment_label is not None)
    subjects = Counter(row.subject for row in rows)
    return {
        "recordings": len(rows),
        "usable_recordings": len(usable),
        "excluded_recordings": len(rows) - len(usable),
        "subjects": len(subjects),
        "records_per_subject": dict(sorted(subjects.items())),
        "unique_text_groups": len(groups),
        "cross_context_text_groups": int(cross_context),
        "labelled_usable_recordings": int(sum(labels.values())),
        "sentiment_label_counts": dict(sorted(labels.items())),
        "exclusion_counts": dict(sorted(exclusions.items())),
        "duration_seconds": {
            "minimum": min(durations) if durations else None,
            "median": statistics.median(durations) if durations else None,
            "maximum": max(durations) if durations else None,
        },
    }


def _atomic_write_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _write_rows_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(ManifestRow.__dataclass_fields__)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    os.replace(temporary, path)


def write_audit(rows, config, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_rows_csv(rows, output_dir / "recordings.csv")
    summary = summarize_manifest(rows)
    _atomic_write_text(
        output_dir / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    _atomic_write_text(
        output_dir / "audit_config.json",
        json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n",
    )
    return summary


def audit_zuco(raw_dir, config, output_dir):
    rows = build_manifest(raw_dir, config)
    return write_audit(rows, config, output_dir)


def _optional_number(value, converter):
    if value is None or str(value).strip() == "":
        return None
    return converter(value)


def read_manifest(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected = set(ManifestRow.__dataclass_fields__)
        if set(reader.fieldnames or []) != expected:
            raise ValueError(f"manifest schema mismatch in {path}")
        for value in reader:
            value["sentence_ordinal"] = int(value["sentence_ordinal"])
            value["sentence_id"] = _optional_number(value["sentence_id"], int)
            value["sentiment_label"] = _optional_number(value["sentiment_label"], int)
            value["channels"] = _optional_number(value["channels"], int)
            value["samples"] = _optional_number(value["samples"], int)
            value["source_hz"] = float(value["source_hz"])
            value["duration_seconds"] = _optional_number(
                value["duration_seconds"], float
            )
            value["nonfinite_fraction"] = _optional_number(
                value["nonfinite_fraction"], float
            )
            value["usable"] = value["usable"].strip().casefold() == "true"
            rows.append(ManifestRow(**value))
    return rows


def combine_manifests(manifest_paths, output_dir):
    """Merge task audits and recompute duplicate counts across all sources."""

    paths = [Path(path) for path in manifest_paths]
    if not paths:
        raise ValueError("at least one manifest is required")
    rows = [row for path in paths for row in read_manifest(path)]
    identifiers = [row.recording_id for row in rows]
    duplicates = [key for key, count in Counter(identifiers).items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate recording IDs across manifests: {duplicates[:5]}")
    rows.sort(
        key=lambda row: (
            row.dataset,
            row.release,
            row.task,
            row.subject,
            row.sentence_ordinal,
        )
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_rows_csv(rows, output_dir / "recordings.csv")
    summary = summarize_manifest(rows)
    _atomic_write_text(
        output_dir / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    _atomic_write_text(
        output_dir / "source_manifests.json",
        json.dumps([str(path.resolve()) for path in paths], indent=2) + "\n",
    )
    return summary
