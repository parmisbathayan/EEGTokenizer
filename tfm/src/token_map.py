"""Frozen-codebook token-map classifier for the bounded TFM V2 experiment."""

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import copy
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd


LABELS = (-1, 0, 1)
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}
INDEX_TO_LABEL = {index: label for label, index in LABEL_TO_INDEX.items()}


@dataclass(frozen=True)
class TokenMapConfig:
    """Predeclared V2 architecture, training, and evaluation settings."""

    seeds: tuple = (42, 52, 62)
    n_splits: int = 5
    validation_fraction: float = 0.15
    codebook_size: int = 8192
    embedding_size: int = 64
    expected_channels: int = 104
    hidden_size: int = 48
    temporal_kernel_size: int = 3
    dropout: float = 0.30
    batch_size: int = 32
    max_epochs: int = 15
    patience: int = 3
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    gradient_clip: float = 1.0
    bootstrap_samples: int = 5000
    planned_versions: int = 3
    familywise_alpha: float = 0.05
    minimum_delta: float = 0.015
    minimum_positive_seeds: int = 2

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class TokenRecord:
    subject: str
    sentence_id: int
    label: int
    tokens: np.ndarray
    preprocess_hash: str
    source_path: str


def _string_value(value):
    value = np.asarray(value)
    return str(value.item() if value.ndim == 0 else value.reshape(-1)[0])


def load_token_records(cache_dir, config=TokenMapConfig()):
    """Load the small cached token arrays once so epochs do not reread Drive."""

    paths = sorted(Path(cache_dir).glob("*/sentence_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no token caches found below {cache_dir}")
    records = []
    labels_by_sentence = {}
    shape_counts = Counter()
    preprocess_hashes = Counter()
    fingerprint = hashlib.sha256()
    for path in paths:
        with np.load(path, allow_pickle=False) as cached:
            tokens = np.asarray(cached["tokens"], dtype=np.int64)
            subject = _string_value(cached["subject"])
            sentence_id = int(cached["sentence_id"])
            label = int(cached["label"])
            preprocess_hash = (
                _string_value(cached["preprocess_hash"])
                if "preprocess_hash" in cached
                else "missing"
            )
        if tokens.ndim != 2 or tokens.shape[0] != config.expected_channels:
            raise ValueError(
                f"invalid token shape {tokens.shape} in {path}; expected "
                f"{config.expected_channels} x time"
            )
        if tokens.shape[1] < 1:
            raise ValueError(f"empty token time axis in {path}")
        if tokens.min() < 0 or tokens.max() >= config.codebook_size:
            raise ValueError(f"token outside [0, {config.codebook_size}) in {path}")
        if label not in LABEL_TO_INDEX:
            raise ValueError(f"unexpected label {label} in {path}")
        if sentence_id in labels_by_sentence and labels_by_sentence[sentence_id] != label:
            raise ValueError(f"conflicting labels for sentence {sentence_id}")
        labels_by_sentence[sentence_id] = label
        shape_counts[tuple(tokens.shape)] += 1
        preprocess_hashes[preprocess_hash] += 1
        fingerprint.update(
            f"{subject}|{sentence_id}|{label}|{tokens.shape}|{preprocess_hash}\n".encode()
        )
        fingerprint.update(tokens.tobytes())
        records.append(
            TokenRecord(
                subject=subject,
                sentence_id=sentence_id,
                label=label,
                tokens=tokens,
                preprocess_hash=preprocess_hash,
                source_path=str(path),
            )
        )

    sentence_counts = Counter(record.sentence_id for record in records)
    report = {
        "n_recordings": len(records),
        "n_sentences": len(labels_by_sentence),
        "n_subjects": len({record.subject for record in records}),
        "minimum_readers_per_sentence": min(sentence_counts.values()),
        "maximum_readers_per_sentence": max(sentence_counts.values()),
        "minimum_tokens_per_channel": min(record.tokens.shape[1] for record in records),
        "maximum_tokens_per_channel": max(record.tokens.shape[1] for record in records),
        "token_shape_counts": {
            f"{channels}x{time}": count
            for (channels, time), count in sorted(shape_counts.items())
        },
        "preprocess_hash_counts": dict(sorted(preprocess_hashes.items())),
        "dataset_fingerprint": fingerprint.hexdigest(),
    }
    metadata = pd.DataFrame(
        [
            {
                "subject": record.subject,
                "sentence_id": record.sentence_id,
                "label": record.label,
                "n_channels": record.tokens.shape[0],
                "n_tokens_per_channel": record.tokens.shape[1],
                "preprocess_hash": record.preprocess_hash,
            }
            for record in records
        ]
    )
    return records, metadata, report


def _select_frozen_codebook(state, config, report):
    candidates = []
    for name, tensor in state.items():
        if tensor.ndim != 2:
            continue
        shape = tuple(tensor.shape)
        if shape not in {
            (config.codebook_size, config.embedding_size),
            (config.embedding_size, config.codebook_size),
        }:
            continue
        lowered = name.lower()
        score = 0
        score += 8 if "quant" in lowered else 0
        score += 8 if "codebook" in lowered else 0
        score += 5 if "embed" in lowered else 0
        score += 3 if "vq" in lowered else 0
        score -= 8 if "decoder" in lowered else 0
        score -= 3 if "position" in lowered or "pos_" in lowered else 0
        candidates.append((score, name, tensor))
    if not candidates:
        raise ValueError(
            "could not identify a 8192 x 64 codebook in the official checkpoint"
        )
    candidates.sort(key=lambda item: (-item[0], item[1]))
    if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
        names = [f"{name} (score={score})" for score, name, _ in candidates]
        raise ValueError(f"ambiguous codebook candidates: {names}")
    score, name, tensor = candidates[0]
    matrix = tensor.detach().cpu().float().contiguous()
    if tuple(matrix.shape) == (config.embedding_size, config.codebook_size):
        matrix = matrix.T.contiguous()
    digest = hashlib.sha256(matrix.numpy().tobytes()).hexdigest()
    report = {
        **report,
        "state_key": name,
        "selection_score": score,
        "shape": list(matrix.shape),
        "sha256": digest,
        "candidate_keys": [candidate_name for _, candidate_name, _ in candidates],
    }
    return matrix, report


def extract_frozen_codebook(tokenizer, config=TokenMapConfig()):
    """Extract the codebook from an already loaded official tokenizer."""

    return _select_frozen_codebook(
        tokenizer.model.state_dict(),
        config=config,
        report={
            "source": "loaded_official_tokenizer",
            "checkpoint_load": tokenizer.load_report,
            "checkpoint_path": tokenizer.checkpoint_path,
        },
    )


def extract_frozen_codebook_from_checkpoint(checkpoint_path, config=TokenMapConfig()):
    """Read only the frozen codebook tensor; no upstream model import is needed."""

    torch = _require_torch()
    from .official_tfm import _unwrap_state_dict

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = _unwrap_state_dict(checkpoint)
    return _select_frozen_codebook(
        state,
        config=config,
        report={
            "source": "checkpoint_state",
            "checkpoint_path": str(Path(checkpoint_path).resolve()),
            "checkpoint_state_key_count": len(state),
        },
    )


def _require_torch():
    try:
        import torch
    except ImportError as error:
        raise ImportError("PyTorch is supplied by the intended Colab runtime") from error
    return torch


def build_token_map_model(codebook, config=TokenMapConfig()):
    """Create the fixed small classifier while leaving the codebook frozen."""

    torch = _require_torch()
    nn = torch.nn

    class FrozenTokenMapClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            if tuple(codebook.shape) != (config.codebook_size, config.embedding_size):
                raise ValueError(
                    f"expected codebook {(config.codebook_size, config.embedding_size)}, "
                    f"got {tuple(codebook.shape)}"
                )
            self.embedding = nn.Embedding.from_pretrained(
                codebook.detach().float(), freeze=True
            )
            self.channel_embedding = nn.Parameter(
                torch.zeros(1, config.expected_channels, 1, config.embedding_size)
            )
            nn.init.trunc_normal_(self.channel_embedding, std=0.02)
            padding = config.temporal_kernel_size // 2
            self.temporal = nn.Sequential(
                nn.Conv1d(
                    config.embedding_size,
                    config.hidden_size,
                    kernel_size=config.temporal_kernel_size,
                    padding=padding,
                ),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Conv1d(
                    config.hidden_size,
                    config.hidden_size,
                    kernel_size=config.temporal_kernel_size,
                    padding=padding,
                ),
                nn.GELU(),
            )
            attention_hidden = max(8, config.hidden_size // 2)
            self.channel_attention = nn.Sequential(
                nn.Linear(config.hidden_size, attention_hidden),
                nn.Tanh(),
                nn.Linear(attention_hidden, 1),
            )
            self.classifier = nn.Sequential(
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_size, len(LABELS)),
            )

        def forward(self, token_ids, time_mask):
            if token_ids.ndim != 3:
                raise ValueError("token_ids must be batch x channels x time")
            if token_ids.shape[1] != config.expected_channels:
                raise ValueError(
                    f"expected {config.expected_channels} channels, got {token_ids.shape[1]}"
                )
            mask = time_mask[:, None, :, None].to(dtype=torch.float32)
            embedded = (self.embedding(token_ids) + self.channel_embedding) * mask
            batch, channels, time, width = embedded.shape
            temporal = embedded.permute(0, 1, 3, 2).reshape(
                batch * channels, width, time
            )
            encoded = self.temporal(temporal).reshape(
                batch, channels, config.hidden_size, time
            )
            temporal_mask = time_mask[:, None, None, :].to(encoded.dtype)
            encoded = encoded * temporal_mask
            denominator = temporal_mask.sum(dim=-1).clamp_min(1.0)
            per_channel = encoded.sum(dim=-1) / denominator
            attention = torch.softmax(self.channel_attention(per_channel), dim=1)
            pooled = (per_channel * attention).sum(dim=1)
            return self.classifier(pooled)

    return FrozenTokenMapClassifier()


class _TokenDataset:
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def collate_token_maps(batch):
    torch = _require_torch()
    max_time = max(example[0].tokens.shape[1] for example in batch)
    channels = batch[0][0].tokens.shape[0]
    tokens = torch.zeros((len(batch), channels, max_time), dtype=torch.long)
    time_mask = torch.zeros((len(batch), max_time), dtype=torch.bool)
    labels = torch.empty(len(batch), dtype=torch.long)
    weights = torch.empty(len(batch), dtype=torch.float32)
    sentence_ids = []
    subjects = []
    for row, (record, label, weight) in enumerate(batch):
        length = record.tokens.shape[1]
        tokens[row, :, :length] = torch.from_numpy(record.tokens.astype(np.int64, copy=False))
        time_mask[row, :length] = True
        labels[row] = LABEL_TO_INDEX[int(label)]
        weights[row] = float(weight)
        sentence_ids.append(record.sentence_id)
        subjects.append(record.subject)
    return {
        "tokens": tokens,
        "time_mask": time_mask,
        "labels": labels,
        "weights": weights,
        "sentence_ids": np.asarray(sentence_ids, dtype=np.int64),
        "subjects": subjects,
    }


def _sentence_table(records):
    labels = {}
    for record in records:
        if record.sentence_id in labels and labels[record.sentence_id] != record.label:
            raise ValueError(f"conflicting labels for sentence {record.sentence_id}")
        labels[record.sentence_id] = record.label
    sentence_ids = np.asarray(sorted(labels), dtype=np.int64)
    y = np.asarray([labels[sentence_id] for sentence_id in sentence_ids], dtype=np.int64)
    return sentence_ids, y


def _label_mapping(sentence_ids, labels, shuffled, rng):
    values = np.asarray(labels, dtype=np.int64)
    if shuffled:
        values = values[rng.permutation(len(values))]
    return {int(sentence_id): int(label) for sentence_id, label in zip(sentence_ids, values)}


def _record_examples(records, sentence_ids, label_mapping):
    selected = set(map(int, sentence_ids))
    counts = Counter(
        record.sentence_id for record in records if record.sentence_id in selected
    )
    sentence_labels = [label_mapping[int(sentence_id)] for sentence_id in sentence_ids]
    class_counts = Counter(sentence_labels)
    class_weights = {
        label: len(sentence_ids) / (len(LABELS) * count)
        for label, count in class_counts.items()
    }
    examples = []
    for record in records:
        if record.sentence_id not in selected:
            continue
        label = label_mapping[record.sentence_id]
        weight = class_weights[label] / counts[record.sentence_id]
        examples.append((record, label, weight))
    return examples


def _loader(examples, config, shuffle, seed):
    torch = _require_torch()
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return torch.utils.data.DataLoader(
        _TokenDataset(examples),
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_token_maps,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )


def _set_seed(seed):
    torch = _require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _aggregate_predictions(model, loader, device):
    torch = _require_torch()
    probability_sums = defaultdict(lambda: np.zeros(len(LABELS), dtype=np.float64))
    counts = Counter()
    labels = {}
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            logits = model(
                batch["tokens"].to(device, non_blocking=True),
                batch["time_mask"].to(device, non_blocking=True),
            )
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            batch_labels = batch["labels"].numpy()
            for sentence_id, label_index, probability in zip(
                batch["sentence_ids"], batch_labels, probabilities
            ):
                sentence_id = int(sentence_id)
                label = INDEX_TO_LABEL[int(label_index)]
                if sentence_id in labels and labels[sentence_id] != label:
                    raise ValueError(f"conflicting evaluation labels for sentence {sentence_id}")
                labels[sentence_id] = label
                probability_sums[sentence_id] += probability
                counts[sentence_id] += 1
    sentence_ids = np.asarray(sorted(probability_sums), dtype=np.int64)
    probabilities = np.stack(
        [probability_sums[sentence_id] / counts[sentence_id] for sentence_id in sentence_ids]
    )
    truth = np.asarray([labels[sentence_id] for sentence_id in sentence_ids], dtype=np.int64)
    return sentence_ids, truth, probabilities


def _train_one_model(
    records,
    codebook,
    outer_train_sentence_ids,
    outer_train_labels,
    config,
    device,
    seed,
    shuffled,
):
    torch = _require_torch()
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedShuffleSplit

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=config.validation_fraction,
        random_state=seed + 20_000,
    )
    train_position, validation_position = next(
        splitter.split(outer_train_sentence_ids, outer_train_labels)
    )
    train_sentence_ids = outer_train_sentence_ids[train_position]
    train_labels = outer_train_labels[train_position]
    validation_sentence_ids = outer_train_sentence_ids[validation_position]
    validation_labels = outer_train_labels[validation_position]
    rng = np.random.default_rng(seed + 30_000)
    train_mapping = _label_mapping(
        train_sentence_ids, train_labels, shuffled=shuffled, rng=rng
    )
    validation_mapping = _label_mapping(
        validation_sentence_ids, validation_labels, shuffled=shuffled, rng=rng
    )
    train_examples = _record_examples(records, train_sentence_ids, train_mapping)
    validation_examples = _record_examples(
        records, validation_sentence_ids, validation_mapping
    )
    train_loader = _loader(train_examples, config, shuffle=True, seed=seed)
    validation_loader = _loader(
        validation_examples, config, shuffle=False, seed=seed
    )

    _set_seed(seed)
    model = build_token_map_model(codebook, config=config).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    best_score = -np.inf
    best_epoch = None
    best_state = None
    stale_epochs = 0
    history = []
    for epoch in range(config.max_epochs):
        model.train()
        weighted_loss_sum = 0.0
        weight_sum = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                batch["tokens"].to(device, non_blocking=True),
                batch["time_mask"].to(device, non_blocking=True),
            )
            weights = batch["weights"].to(device, non_blocking=True)
            losses = torch.nn.functional.cross_entropy(
                logits,
                batch["labels"].to(device, non_blocking=True),
                reduction="none",
            )
            loss = (losses * weights).sum() / weights.sum().clamp_min(1e-8)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, config.gradient_clip)
            optimizer.step()
            weighted_loss_sum += float((losses.detach() * weights).sum().cpu())
            weight_sum += float(weights.sum().cpu())

        validation_ids, validation_truth, validation_probabilities = _aggregate_predictions(
            model, validation_loader, device
        )
        del validation_ids
        validation_predictions = np.asarray(
            [LABELS[index] for index in validation_probabilities.argmax(axis=1)],
            dtype=np.int64,
        )
        validation_macro_f1 = f1_score(
            validation_truth,
            validation_predictions,
            labels=list(LABELS),
            average="macro",
            zero_division=0,
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": weighted_loss_sum / max(weight_sum, 1e-8),
                "validation_macro_f1": float(validation_macro_f1),
            }
        )
        print(
            f"  epoch={epoch + 1:02d} "
            f"loss={history[-1]['train_loss']:.4f} "
            f"val_macro_f1={validation_macro_f1:.4f}",
            flush=True,
        )
        if validation_macro_f1 > best_score + 1e-6:
            best_score = float(validation_macro_f1)
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("training did not produce a finite validation score")
    model.load_state_dict(best_state)
    return model, history, best_epoch, best_score


def _metric_values(y_true, y_pred):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    per_class = f1_score(
        y_true,
        y_pred,
        labels=list(LABELS),
        average=None,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=list(LABELS),
                average="macro",
                zero_division=0,
            )
        ),
        **{
            f"f1_class_{label}": float(score)
            for label, score in zip(LABELS, per_class)
        },
    }


def _atomic_csv(frame, path):
    path = Path(path)
    temporary = path.with_suffix(".tmp.csv")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _read_partial(path, columns):
    path = Path(path)
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=columns)


def _write_json(payload, path):
    path = Path(path)
    temporary = path.with_suffix(".tmp.json")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2)
    temporary.replace(path)


def _bootstrap_delta(predictions, config, seed=2026):
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(seed)
    paired = []
    for run_seed in sorted(predictions["seed"].unique()):
        subset = predictions[predictions["seed"] == run_seed]
        aligned = subset[subset["setup"] == "token_map"].sort_values("sentence_id")
        shuffled = subset[subset["setup"] == "token_map_shuffled"].sort_values(
            "sentence_id"
        )
        if not np.array_equal(
            aligned["sentence_id"].to_numpy(), shuffled["sentence_id"].to_numpy()
        ):
            raise ValueError("aligned and shuffled predictions are not paired")
        paired.append(
            (
                aligned["label"].to_numpy(),
                aligned["prediction"].to_numpy(),
                shuffled["prediction"].to_numpy(),
            )
        )
    draws = []
    for _ in range(config.bootstrap_samples):
        seed_deltas = []
        for truth, aligned_prediction, shuffled_prediction in paired:
            indices = rng.integers(0, len(truth), size=len(truth))
            seed_deltas.append(
                f1_score(
                    truth[indices],
                    aligned_prediction[indices],
                    labels=list(LABELS),
                    average="macro",
                    zero_division=0,
                )
                - f1_score(
                    truth[indices],
                    shuffled_prediction[indices],
                    labels=list(LABELS),
                    average="macro",
                    zero_division=0,
                )
            )
        draws.append(float(np.mean(seed_deltas)))
    per_comparison_alpha = config.familywise_alpha / config.planned_versions
    lower_quantile = per_comparison_alpha / 2
    upper_quantile = 1 - lower_quantile
    return {
        "mean_delta": float(np.mean(draws)),
        "ci_level": float(1 - per_comparison_alpha),
        "ci_low": float(np.quantile(draws, lower_quantile)),
        "ci_high": float(np.quantile(draws, upper_quantile)),
        "bootstrap_samples": config.bootstrap_samples,
        "planned_versions": config.planned_versions,
        "familywise_alpha": config.familywise_alpha,
    }


def _gate_report(metrics, delta, config):
    aligned = metrics[metrics["setup"] == "token_map"]
    shuffled = metrics[metrics["setup"] == "token_map_shuffled"]
    majority = metrics[metrics["setup"] == "majority"]
    aligned_macro = float(aligned["macro_f1"].mean())
    shuffled_macro = float(shuffled["macro_f1"].mean())
    majority_macro = float(majority["macro_f1"].mean())
    observed_delta = aligned_macro - shuffled_macro
    seed_means = (
        metrics[metrics["setup"].isin(["token_map", "token_map_shuffled"])]
        .groupby(["seed", "setup"])["macro_f1"]
        .mean()
        .unstack("setup")
    )
    seed_deltas = {
        str(int(seed)): float(row["token_map"] - row["token_map_shuffled"])
        for seed, row in seed_means.iterrows()
    }
    positive_seeds = sum(value > 0 for value in seed_deltas.values())
    criteria = {
        "balanced_accuracy_above_chance": float(aligned["balanced_accuracy"].mean())
        > 1 / 3,
        "macro_f1_delta_at_least_minimum": observed_delta >= config.minimum_delta,
        "enough_positive_seeds": positive_seeds >= config.minimum_positive_seeds,
        "corrected_bootstrap_ci_low_above_zero": delta["ci_low"] > 0,
        "macro_f1_above_majority": aligned_macro > majority_macro,
    }
    passes = all(criteria.values())
    return {
        "aligned_macro_f1": aligned_macro,
        "shuffled_macro_f1": shuffled_macro,
        "majority_macro_f1": majority_macro,
        "aligned_balanced_accuracy": float(aligned["balanced_accuracy"].mean()),
        "chance_balanced_accuracy": 1 / 3,
        "observed_delta": observed_delta,
        "minimum_required_delta": config.minimum_delta,
        "seed_deltas": seed_deltas,
        "positive_seed_count": positive_seeds,
        "minimum_positive_seeds": config.minimum_positive_seeds,
        "bootstrap": delta,
        "criteria": criteria,
        "passes": passes,
        "decision": (
            "PASS — complete all three predefined versions before interpretation"
            if passes
            else "V2 FAIL — proceed only to predefined V3"
        ),
    }


def evaluate_token_map(
    records,
    codebook,
    output_dir,
    cache_report,
    codebook_report,
    config=TokenMapConfig(),
    device="cuda",
    resume=True,
):
    """Run sentence-grouped V2 evaluation with fold/setup-level resumption."""

    torch = _require_torch()
    from sklearn.model_selection import StratifiedKFold

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_device = torch.device(
        device if device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    if resolved_device.type != "cuda":
        raise RuntimeError("V2 is intended for a Colab GPU runtime")

    signature_payload = {
        "experiment": "tfm_v2_frozen_token_map",
        "config": config.to_dict(),
        "dataset_fingerprint": cache_report["dataset_fingerprint"],
        "codebook_sha256": codebook_report["sha256"],
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode()
    ).hexdigest()
    signature_path = output_dir / "run_signature.json"
    if signature_path.exists():
        previous = json.loads(signature_path.read_text())
        if previous.get("signature") != signature:
            raise ValueError(
                "existing V2 results use a different dataset/codebook/config; "
                "choose a new results directory"
            )
    else:
        _write_json({"signature": signature, **signature_payload}, signature_path)

    metric_columns = [
        "setup",
        "seed",
        "fold",
        "best_epoch",
        "validation_macro_f1",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "f1_class_-1",
        "f1_class_0",
        "f1_class_1",
    ]
    prediction_columns = [
        "setup",
        "seed",
        "fold",
        "sentence_id",
        "feature_sentence_id",
        "label",
        "prediction",
        "probability_-1",
        "probability_0",
        "probability_1",
    ]
    history_columns = [
        "setup",
        "seed",
        "fold",
        "epoch",
        "train_loss",
        "validation_macro_f1",
    ]
    metrics_path = output_dir / "fold_metrics_partial.csv"
    predictions_path = output_dir / "oof_predictions_partial.csv"
    history_path = output_dir / "training_history_partial.csv"
    metrics = _read_partial(metrics_path, metric_columns) if resume else pd.DataFrame(columns=metric_columns)
    predictions = (
        _read_partial(predictions_path, prediction_columns)
        if resume
        else pd.DataFrame(columns=prediction_columns)
    )
    history = _read_partial(history_path, history_columns) if resume else pd.DataFrame(columns=history_columns)
    completed = {
        (str(row.setup), int(row.seed), int(row.fold))
        for row in metrics.itertuples(index=False)
    }

    sentence_ids, y = _sentence_table(records)
    for run_seed in config.seeds:
        outer = StratifiedKFold(
            n_splits=config.n_splits,
            shuffle=True,
            random_state=run_seed,
        )
        for fold, (train_position, test_position) in enumerate(outer.split(sentence_ids, y)):
            outer_train_sentence_ids = sentence_ids[train_position]
            outer_train_labels = y[train_position]
            test_sentence_ids = sentence_ids[test_position]
            test_labels = y[test_position]
            test_mapping = {
                int(sentence_id): int(label)
                for sentence_id, label in zip(test_sentence_ids, test_labels)
            }
            test_examples = _record_examples(records, test_sentence_ids, test_mapping)
            test_loader = _loader(
                test_examples,
                config,
                shuffle=False,
                seed=run_seed * 100 + fold,
            )

            for setup, shuffled in (
                ("token_map", False),
                ("token_map_shuffled", True),
            ):
                key = (setup, int(run_seed), int(fold))
                if key in completed:
                    saved_predictions = predictions[
                        (predictions["setup"] == setup)
                        & (predictions["seed"].astype(int) == int(run_seed))
                        & (predictions["fold"].astype(int) == int(fold))
                    ]
                    if len(saved_predictions) == len(test_sentence_ids):
                        print(f"Reusing completed {setup}, seed={run_seed}, fold={fold}")
                        continue
                    print(f"Repairing incomplete {setup}, seed={run_seed}, fold={fold}")
                    metrics = metrics[
                        ~(
                            (metrics["setup"] == setup)
                            & (metrics["seed"].astype(int) == int(run_seed))
                            & (metrics["fold"].astype(int) == int(fold))
                        )
                    ].reset_index(drop=True)
                    completed.remove(key)
                predictions = predictions[
                    ~(
                        (predictions["setup"] == setup)
                        & (predictions["seed"].astype(int) == int(run_seed))
                        & (predictions["fold"].astype(int) == int(fold))
                    )
                ].reset_index(drop=True)
                history = history[
                    ~(
                        (history["setup"] == setup)
                        & (history["seed"].astype(int) == int(run_seed))
                        & (history["fold"].astype(int) == int(fold))
                    )
                ].reset_index(drop=True)
                model_seed = run_seed * 1_000 + fold
                print(f"Training {setup}, seed={run_seed}, fold={fold}", flush=True)
                model, run_history, best_epoch, best_score = _train_one_model(
                    records=records,
                    codebook=codebook,
                    outer_train_sentence_ids=outer_train_sentence_ids,
                    outer_train_labels=outer_train_labels,
                    config=config,
                    device=resolved_device,
                    seed=model_seed,
                    shuffled=shuffled,
                )
                evaluated_ids, truth, probabilities = _aggregate_predictions(
                    model, test_loader, resolved_device
                )
                if not np.array_equal(evaluated_ids, np.sort(test_sentence_ids)):
                    raise ValueError("test sentence aggregation changed the outer fold")
                feature_sentence_ids = evaluated_ids.copy()
                if shuffled:
                    rng = np.random.default_rng(run_seed * 100 + fold)
                    permutation = rng.permutation(len(probabilities))
                    probabilities = probabilities[permutation]
                    feature_sentence_ids = feature_sentence_ids[permutation]
                predicted_indices = probabilities.argmax(axis=1)
                predicted_labels = np.asarray(
                    [INDEX_TO_LABEL[int(index)] for index in predicted_indices],
                    dtype=np.int64,
                )
                run_prediction_rows = []
                for sentence_id, feature_sentence_id, label, prediction, probability in zip(
                    evaluated_ids,
                    feature_sentence_ids,
                    truth,
                    predicted_labels,
                    probabilities,
                ):
                    run_prediction_rows.append(
                        {
                            "setup": setup,
                            "seed": run_seed,
                            "fold": fold,
                            "sentence_id": int(sentence_id),
                            "feature_sentence_id": int(feature_sentence_id),
                            "label": int(label),
                            "prediction": int(prediction),
                            "probability_-1": float(probability[0]),
                            "probability_0": float(probability[1]),
                            "probability_1": float(probability[2]),
                        }
                    )
                run_history_rows = [
                    {"setup": setup, "seed": run_seed, "fold": fold, **row}
                    for row in run_history
                ]
                predictions = pd.concat(
                    [predictions, pd.DataFrame(run_prediction_rows)], ignore_index=True
                )
                history = pd.concat(
                    [history, pd.DataFrame(run_history_rows)], ignore_index=True
                )
                _atomic_csv(predictions, predictions_path)
                _atomic_csv(history, history_path)
                metric_row = {
                    "setup": setup,
                    "seed": run_seed,
                    "fold": fold,
                    "best_epoch": best_epoch,
                    "validation_macro_f1": best_score,
                    **_metric_values(truth, predicted_labels),
                }
                metrics = pd.concat(
                    [metrics, pd.DataFrame([metric_row])], ignore_index=True
                )
                _atomic_csv(metrics, metrics_path)
                completed.add(key)
                del model
                torch.cuda.empty_cache()

            majority_key = ("majority", int(run_seed), int(fold))
            saved_majority = predictions[
                (predictions["setup"] == "majority")
                & (predictions["seed"].astype(int) == int(run_seed))
                & (predictions["fold"].astype(int) == int(fold))
            ]
            if majority_key in completed and len(saved_majority) != len(test_sentence_ids):
                metrics = metrics[
                    ~(
                        (metrics["setup"] == "majority")
                        & (metrics["seed"].astype(int) == int(run_seed))
                        & (metrics["fold"].astype(int) == int(fold))
                    )
                ].reset_index(drop=True)
                completed.remove(majority_key)
            if majority_key not in completed:
                predictions = predictions[
                    ~(
                        (predictions["setup"] == "majority")
                        & (predictions["seed"].astype(int) == int(run_seed))
                        & (predictions["fold"].astype(int) == int(fold))
                    )
                ].reset_index(drop=True)
                majority_label = Counter(outer_train_labels).most_common(1)[0][0]
                majority_predictions = np.full(len(test_labels), majority_label, dtype=np.int64)
                majority_rows = [
                    {
                        "setup": "majority",
                        "seed": run_seed,
                        "fold": fold,
                        "sentence_id": int(sentence_id),
                        "feature_sentence_id": int(sentence_id),
                        "label": int(label),
                        "prediction": int(majority_label),
                        "probability_-1": float(majority_label == -1),
                        "probability_0": float(majority_label == 0),
                        "probability_1": float(majority_label == 1),
                    }
                    for sentence_id, label in zip(test_sentence_ids, test_labels)
                ]
                predictions = pd.concat(
                    [predictions, pd.DataFrame(majority_rows)], ignore_index=True
                )
                _atomic_csv(predictions, predictions_path)
                metric_row = {
                    "setup": "majority",
                    "seed": run_seed,
                    "fold": fold,
                    "best_epoch": np.nan,
                    "validation_macro_f1": np.nan,
                    **_metric_values(test_labels, majority_predictions),
                }
                metrics = pd.concat(
                    [metrics, pd.DataFrame([metric_row])], ignore_index=True
                )
                _atomic_csv(metrics, metrics_path)
                completed.add(majority_key)

    metrics = metrics.sort_values(["seed", "fold", "setup"]).reset_index(drop=True)
    predictions = predictions.sort_values(
        ["seed", "fold", "setup", "sentence_id"]
    ).reset_index(drop=True)
    history = history.sort_values(["seed", "fold", "setup", "epoch"]).reset_index(
        drop=True
    )
    summary = (
        metrics.groupby("setup")[["accuracy", "balanced_accuracy", "macro_f1"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    delta = _bootstrap_delta(predictions, config=config)
    gate = _gate_report(metrics, delta, config=config)
    _atomic_csv(metrics, output_dir / "fold_metrics.csv")
    _atomic_csv(predictions, output_dir / "oof_predictions.csv")
    _atomic_csv(history, output_dir / "training_history.csv")
    summary.to_csv(output_dir / "summary.csv", index=False)
    _write_json(config.to_dict(), output_dir / "evaluation_config.json")
    _write_json(cache_report, output_dir / "token_cache_report.json")
    _write_json(codebook_report, output_dir / "codebook_report.json")
    _write_json(delta, output_dir / "alignment_delta.json")
    _write_json(gate, output_dir / "viability_gate.json")
    return metrics, predictions, history, summary, delta, gate
