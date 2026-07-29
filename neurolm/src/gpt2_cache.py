"""Frozen full-NeuroLM/GPT-2 representation cache for bounded version 4."""

from collections import Counter, OrderedDict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import pandas as pd

from .channels import NEUROLM_CHANNELS
from .raw_cache import _load_subject_pack


GPT2_CACHE_FORMAT_VERSION = 1
GPT2_FEATURE_VERSION = "neurolm_b_gpt2_prompt_hidden_v4"
FIXED_PROMPT = (
    "Classify the sentiment represented by this EEG recording. "
    "Answer with one word: negative, neutral, or positive. Sentiment:"
)
LABEL_VERBALIZERS = OrderedDict(((-1, " negative"), (0, " neutral"), (1, " positive")))


@dataclass(frozen=True)
class GPT2Record:
    subject: str
    sentence_id: int
    label: int
    hidden: np.ndarray


def _string_scalar(value):
    value = np.asarray(value)
    return str(value.item() if value.ndim == 0 else value.reshape(-1)[0])


def selected_seconds(total_seconds, maximum_seconds=3):
    """Choose deterministic start/middle/end coverage without label information."""

    total_seconds = int(total_seconds)
    maximum_seconds = int(maximum_seconds)
    if total_seconds < 1 or maximum_seconds < 1:
        raise ValueError("second counts must be positive")
    if total_seconds <= maximum_seconds:
        return np.arange(total_seconds, dtype=np.int64)
    values = np.linspace(0, total_seconds - 1, maximum_seconds)
    indices = np.rint(values).astype(np.int64)
    if len(np.unique(indices)) != maximum_seconds:
        raise RuntimeError("uniform second selection produced duplicates")
    return indices


def prepare_eeg_tokens(eeg, zuco_indices, channel_ids, maximum_seconds=3):
    """Return fixed-size time-major EEG patches and the official masks/IDs."""

    eeg = np.asarray(eeg, dtype=np.float32)
    zuco_indices = np.asarray(zuco_indices, dtype=np.int64)
    channel_ids = np.asarray(channel_ids, dtype=np.int64)
    if eeg.ndim != 2 or eeg.shape[0] != 104:
        raise ValueError(f"expected 104 x time EEG, got {eeg.shape}")
    if zuco_indices.shape != channel_ids.shape or not len(channel_ids):
        raise ValueError("channel IDs and ZuCo indices must be non-empty and aligned")
    seconds = eeg.shape[1] // 200
    chosen = selected_seconds(seconds, maximum_seconds)
    channel_count = len(channel_ids)
    maximum_patches = maximum_seconds * channel_count
    patches = np.zeros((maximum_patches, 200), dtype=np.float32)
    channels = np.full(
        maximum_patches, NEUROLM_CHANNELS.index("PAD"), dtype=np.int32
    )
    times = np.zeros(maximum_patches, dtype=np.int32)
    valid = np.zeros(maximum_patches, dtype=bool)
    mapped = eeg[zuco_indices]
    for position, second in enumerate(chosen):
        start = position * channel_count
        stop = start + channel_count
        sample_start = int(second) * 200
        patches[start:stop] = mapped[:, sample_start : sample_start + 200]
        channels[start:stop] = channel_ids
        times[start:stop] = int(second)
        valid[start:stop] = True
    return patches, channels, times, valid, chosen


class OfficialNeuroLMGPT2:
    """Load the complete official checkpoint and expose prompt-position states."""

    def __init__(
        self,
        repo_dir,
        checkpoint_path,
        channel_ids,
        zuco_indices,
        device="cuda",
        maximum_seconds=3,
        prompt=FIXED_PROMPT,
    ):
        import torch
        import tiktoken

        self.repo_dir = str(Path(repo_dir).resolve())
        self.checkpoint_path = str(Path(checkpoint_path).resolve())
        self.device = torch.device(device)
        self.channel_ids = np.asarray(channel_ids, dtype=np.int64)
        self.zuco_indices = np.asarray(zuco_indices, dtype=np.int64)
        self.maximum_seconds = int(maximum_seconds)
        self.prompt = str(prompt)
        if self.channel_ids.shape != self.zuco_indices.shape:
            raise ValueError("channel IDs and ZuCo indices must have equal shape")
        if self.maximum_seconds * len(self.channel_ids) >= 1024:
            raise ValueError("V4 EEG prefix leaves no room for the fixed instruction")
        if self.repo_dir not in sys.path:
            sys.path.insert(0, self.repo_dir)

        from model.model import GPTConfig
        from model.model_neurolm import NeuroLM

        load_kwargs = {"map_location": "cpu"}
        try:
            checkpoint = torch.load(
                self.checkpoint_path, mmap=True, weights_only=False, **load_kwargs
            )
        except TypeError:
            checkpoint = torch.load(self.checkpoint_path, **load_kwargs)
        if "model" not in checkpoint or "model_args" not in checkpoint:
            raise KeyError("official checkpoint must contain model and model_args")
        model_args = dict(checkpoint["model_args"])
        self.model = NeuroLM(GPTConfig(**model_args), init_from="scratch")
        state = OrderedDict()
        for key, value in checkpoint["model"].items():
            for prefix in ("_orig_mod.", "module."):
                if key.startswith(prefix):
                    key = key[len(prefix) :]
            state[key] = value
        incompatibility = self.model.load_state_dict(state, strict=False)
        self.load_report = {
            "selected_keys": len(state),
            "missing_keys": list(incompatibility.missing_keys),
            "unexpected_keys": list(incompatibility.unexpected_keys),
        }
        if self.load_report["missing_keys"] or self.load_report["unexpected_keys"]:
            raise RuntimeError(f"full NeuroLM-B checkpoint mismatch: {self.load_report}")
        del checkpoint, state
        self.model.eval().requires_grad_(False).to(self.device)
        if not hasattr(self.model, "GPT2"):
            raise AttributeError("official NeuroLM model has no GPT2 component")
        gpt2 = self.model.GPT2
        if not hasattr(gpt2, "transformer") or not hasattr(gpt2.transformer, "ln_f"):
            raise AttributeError("GPT2 final normalization layer is unavailable")
        if not hasattr(gpt2.transformer, "wte"):
            raise AttributeError("GPT2 token embedding table is unavailable")
        self.encoding = tiktoken.get_encoding("gpt2")
        self.prompt_ids = self.encoding.encode(self.prompt)
        if not self.prompt_ids:
            raise ValueError("fixed V4 prompt tokenized to an empty sequence")
        self.verbalizer_token_ids = []
        for verbalizer in LABEL_VERBALIZERS.values():
            token_ids = self.encoding.encode(verbalizer)
            if len(token_ids) != 1:
                raise ValueError(
                    f"V4 verbalizer must be one GPT-2 token: {verbalizer!r} -> {token_ids}"
                )
            self.verbalizer_token_ids.append(int(token_ids[0]))
        with torch.inference_mode():
            self.verbalizer_vectors = (
                gpt2.transformer.wte.weight[self.verbalizer_token_ids]
                .detach()
                .float()
                .cpu()
                .numpy()
            )
        self.embedding_size = int(self.verbalizer_vectors.shape[1])
        self.model_args = model_args

    def encode_batch(self, eeg_rows):
        """Encode preprocessed recordings into frozen final-prompt GPT-2 states."""

        import torch

        prepared = [
            prepare_eeg_tokens(
                eeg,
                self.zuco_indices,
                self.channel_ids,
                self.maximum_seconds,
            )
            for eeg in eeg_rows
        ]
        patches = torch.as_tensor(
            np.stack([row[0] for row in prepared]), device=self.device
        )
        channels = torch.as_tensor(
            np.stack([row[1] for row in prepared]), device=self.device
        )
        times = torch.as_tensor(np.stack([row[2] for row in prepared]), device=self.device)
        eeg_valid = torch.as_tensor(
            np.stack([row[3] for row in prepared]), device=self.device
        )
        batch_size, eeg_length = patches.shape[:2]
        end_token = int(self.encoding.eot_token)
        if end_token >= int(self.model.GPT2.config.vocab_size):
            raise ValueError("GPT-2 end token lies outside the checkpoint vocabulary")
        text_values = self.prompt_ids + [end_token]
        text = torch.as_tensor(text_values, dtype=torch.long, device=self.device)[None]
        text = text.repeat(batch_size, 1)
        text_targets = torch.full_like(text, -1)
        text_targets[:, len(self.prompt_ids) - 1] = end_token
        eeg_targets = torch.full(
            (batch_size, eeg_length),
            fill_value=-1-int(self.model.GPT2.config.vocab_size),
            dtype=torch.long,
            device=self.device,
        )
        total = eeg_length + text.shape[1]
        attention = torch.tril(
            torch.ones((batch_size, total, total), dtype=torch.bool, device=self.device)
        ).unsqueeze(1)
        channel_count = len(self.channel_ids)
        for batch_index, row in enumerate(prepared):
            chosen = row[4]
            for position in range(len(chosen)):
                start = position * channel_count
                stop = start + channel_count
                attention[batch_index, :, start:stop, start:stop] = True
            invalid_positions = torch.nonzero(
                ~eeg_valid[batch_index], as_tuple=False
            ).flatten()
            if invalid_positions.numel():
                attention[batch_index, :, :, invalid_positions] = False

        captured = []

        def save_hidden(_, __, output):
            captured.append(output[0] if isinstance(output, (tuple, list)) else output)

        hook = self.model.GPT2.transformer.ln_f.register_forward_hook(save_hidden)
        autocast_enabled = self.device.type == "cuda"
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        try:
            with torch.inference_mode(), torch.autocast(
                device_type=self.device.type, dtype=dtype, enabled=autocast_enabled
            ):
                self.model(
                    patches,
                    eeg_targets,
                    text,
                    text_targets,
                    channels,
                    times,
                    eeg_valid,
                    eeg_text_mask=attention,
                )
        finally:
            hook.remove()
        if len(captured) != 1:
            raise RuntimeError(f"expected one GPT2 hidden-state capture, got {len(captured)}")
        hidden = captured[0]
        expected_length = total
        if hidden.ndim != 3 or hidden.shape[:2] != (batch_size, expected_length):
            raise ValueError(f"unexpected GPT2 hidden shape {tuple(hidden.shape)}")
        prompt_position = eeg_length + len(self.prompt_ids) - 1
        values = hidden[:, prompt_position].detach().float().cpu().numpy()
        if values.shape != (batch_size, self.embedding_size):
            raise ValueError(f"unexpected prompt representation shape {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("GPT2 prompt representation contains non-finite values")
        return values


def _cache_signature(raw_signature, encoder):
    checkpoint = Path(encoder.checkpoint_path)
    stat = checkpoint.stat()
    payload = {
        "format_version": GPT2_CACHE_FORMAT_VERSION,
        "feature_version": GPT2_FEATURE_VERSION,
        "raw_cache_signature": raw_signature,
        "checkpoint_name": checkpoint.name,
        "checkpoint_bytes": stat.st_size,
        "checkpoint_mtime_ns": stat.st_mtime_ns,
        "channel_ids": [int(value) for value in encoder.channel_ids],
        "zuco_indices": [int(value) for value in encoder.zuco_indices],
        "maximum_seconds": encoder.maximum_seconds,
        "second_selection": "uniform_start_middle_end",
        "prompt": encoder.prompt,
        "prompt_ids": [int(value) for value in encoder.prompt_ids],
        "verbalizers": {str(key): value for key, value in LABEL_VERBALIZERS.items()},
        "verbalizer_token_ids": encoder.verbalizer_token_ids,
        "embedding_size": encoder.embedding_size,
        "storage_dtype": "float16",
    }
    signature = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return signature, payload


def _pack_is_current(path, signature, raw_pack_path):
    try:
        raw_stat = Path(raw_pack_path).stat()
        with np.load(path, allow_pickle=False) as cached:
            return (
                int(cached["format_version"]) == GPT2_CACHE_FORMAT_VERSION
                and _string_scalar(cached["signature"]) == signature
                and int(cached["source_raw_pack_bytes"]) == raw_stat.st_size
                and int(cached["source_raw_pack_mtime_ns"]) == raw_stat.st_mtime_ns
                and len(cached["sentence_ids"]) == len(cached["hidden_values"])
            )
    except Exception:
        return False


def _write_subject_pack(path, subject, rows, signature, raw_pack_path, embedding_size):
    if not rows:
        raise ValueError(f"subject {subject} produced no V4 representations")
    hidden = np.stack([row[2] for row in rows]).astype(np.float16, copy=False)
    if hidden.shape[1:] != (embedding_size,):
        raise ValueError(f"unexpected GPT2 hidden shape {hidden.shape}")
    stat = Path(raw_pack_path).stat()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        format_version=np.int64(GPT2_CACHE_FORMAT_VERSION),
        feature_version=np.asarray(GPT2_FEATURE_VERSION),
        signature=np.asarray(signature),
        subject=np.asarray(subject),
        embedding_size=np.int64(embedding_size),
        source_raw_pack_bytes=np.int64(stat.st_size),
        source_raw_pack_mtime_ns=np.int64(stat.st_mtime_ns),
        sentence_ids=np.asarray([row[0] for row in rows], dtype=np.int64),
        labels=np.asarray([row[1] for row in rows], dtype=np.int64),
        hidden_values=hidden,
    )
    temporary.replace(path)


def extract_gpt2_subject_packs(
    raw_pack_dir,
    output_dir,
    encoder,
    overwrite=False,
    batch_size=4,
):
    """Run the frozen full model once and save compact reader representations."""

    raw_pack_dir = Path(raw_pack_dir)
    raw_manifest_path = raw_pack_dir / "cache_manifest.json"
    if not raw_manifest_path.exists():
        raise FileNotFoundError("V2 raw cache manifest is missing; finish V2 Cell 3")
    raw_manifest = json.loads(raw_manifest_path.read_text())
    raw_signature = str(raw_manifest["signature"])
    raw_paths = sorted(raw_pack_dir.glob("*.npz"))
    if not raw_paths:
        raise FileNotFoundError(f"no raw subject packs found in {raw_pack_dir}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    signature, signature_payload = _cache_signature(raw_signature, encoder)
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
            print(f"Reused V4 GPT2 pack: {raw_path.stem}", flush=True)
            write_status("extracting", raw_path.stem)
            continue
        records, _, _ = _load_subject_pack(raw_path, expected_signature=raw_signature)
        rows = []
        for start in range(0, len(records), int(batch_size)):
            batch = records[start : start + int(batch_size)]
            try:
                hidden = encoder.encode_batch([record.eeg for record in batch])
                rows.extend(
                    (int(record.sentence_id), int(record.label), representation)
                    for record, representation in zip(batch, hidden)
                )
            except Exception as error:
                report["failed"] += len(batch)
                report["failures"].append(
                    {
                        "subject": raw_path.stem,
                        "sentence_ids": [int(record.sentence_id) for record in batch],
                        "error": repr(error),
                        "traceback": traceback.format_exc(),
                    }
                )
                raise
            print(
                f"{raw_path.stem}: encoded {min(start + len(batch), len(records))}/"
                f"{len(records)}",
                flush=True,
            )
            write_status("extracting", raw_path.stem)
        _write_subject_pack(
            output,
            raw_path.stem,
            rows,
            signature,
            raw_path,
            encoder.embedding_size,
        )
        report["subjects_written"] += 1
        report["recordings_written"] += len(rows)
        write_status("extracting", raw_path.stem)

    vector_path = output_dir / "verbalizer_vectors.npy"
    np.save(vector_path, encoder.verbalizer_vectors.astype(np.float32, copy=False))
    manifest = {
        "format_version": GPT2_CACHE_FORMAT_VERSION,
        "feature_version": GPT2_FEATURE_VERSION,
        "signature": signature,
        "signature_payload": signature_payload,
        "encoder_load_report": encoder.load_report,
        "model_args": encoder.model_args,
        "report": report,
    }
    temporary = output_dir / "cache_manifest.tmp.json"
    temporary.write_text(json.dumps(manifest, indent=2))
    temporary.replace(output_dir / "cache_manifest.json")
    write_status("complete")
    return manifest


def _load_gpt2_pack(path, expected_signature):
    with np.load(path, allow_pickle=False) as cached:
        signature = _string_scalar(cached["signature"])
        if signature != expected_signature:
            raise ValueError(f"stale V4 cache signature in {path}")
        subject = _string_scalar(cached["subject"])
        embedding_size = int(cached["embedding_size"])
        sentence_ids = np.asarray(cached["sentence_ids"], dtype=np.int64)
        labels = np.asarray(cached["labels"], dtype=np.int64)
        hidden = np.asarray(cached["hidden_values"], dtype=np.float16)
    if hidden.shape != (len(sentence_ids), embedding_size) or len(labels) != len(sentence_ids):
        raise ValueError(f"invalid V4 pack shape in {path}")
    records = [
        GPT2Record(subject, int(sentence_id), int(label), hidden[index])
        for index, (sentence_id, label) in enumerate(zip(sentence_ids, labels))
    ]
    return records, hidden


def load_gpt2_records(cache_dir):
    """Load V4 reader representations, fixed label vectors, and diagnostics."""

    cache_dir = Path(cache_dir)
    manifest_path = cache_dir / "cache_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"V4 cache manifest is missing from {cache_dir}")
    manifest = json.loads(manifest_path.read_text())
    signature = str(manifest["signature"])
    paths = sorted(cache_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no V4 subject packs found in {cache_dir}")
    vector_path = cache_dir / "verbalizer_vectors.npy"
    if not vector_path.exists():
        raise FileNotFoundError(f"V4 verbalizer vectors are missing: {vector_path}")
    vectors = np.load(vector_path, allow_pickle=False).astype(np.float32, copy=False)
    records = []
    backing = []
    fingerprint = hashlib.sha256()
    for position, path in enumerate(paths, start=1):
        subject_records, hidden = _load_gpt2_pack(path, signature)
        records.extend(subject_records)
        backing.append(hidden)
        fingerprint.update(path.name.encode())
        fingerprint.update(hidden.view(np.uint8))
        print(
            f"Loaded V4 pack {position}/{len(paths)}: {path.stem} "
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
                "embedding_size": record.hidden.shape[0],
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
        "embedding_size": int(metadata["embedding_size"].iloc[0]),
        "memory_bytes_float16": int(sum(array.nbytes for array in backing)),
        "cache_signature": signature,
        "dataset_fingerprint": fingerprint.hexdigest(),
        "prompt": manifest["signature_payload"]["prompt"],
        "verbalizers": manifest["signature_payload"]["verbalizers"],
        "selected_seconds_maximum": manifest["signature_payload"]["maximum_seconds"],
    }
    if vectors.shape != (3, report["embedding_size"]):
        raise ValueError(f"invalid verbalizer-vector shape {vectors.shape}")
    return records, vectors, metadata, report
