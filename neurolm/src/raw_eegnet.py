"""Bounded raw-EEG EEGNet screen for NeuroLM project version 2."""

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
ALIGNED = "raw_eegnet"
SHUFFLED = "raw_eegnet_shuffled"
TEMPORAL_SHUFFLE = "raw_eegnet_temporal_block_shuffle"
IMPLEMENTATION_VERSION = "raw-eegnet-v2.0"


@dataclass(frozen=True)
class RawEEGNetConfig:
    """Single predeclared architecture and evaluation recipe for V2."""

    seeds: tuple = (42, 52, 62)
    n_splits: int = 5
    validation_fraction: float = 0.15
    sample_rate_hz: int = 200
    expected_channels: int = 104
    window_seconds: float = 1.0
    temporal_block_samples: int = 10
    temporal_filters: int = 8
    depth_multiplier: int = 2
    temporal_kernel_samples: int = 63
    separable_kernel_samples: int = 15
    pool_one: int = 4
    pool_two: int = 8
    dropout: float = 0.50
    batch_size: int = 16
    max_epochs: int = 20
    patience: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    gradient_clip: float = 1.0
    normalization_clip: float = 8.0
    bootstrap_samples: int = 5000
    planned_versions: int = 3
    familywise_alpha: float = 0.05
    minimum_delta: float = 0.015
    minimum_positive_seeds: int = 2

    @property
    def window_samples(self):
        return int(round(self.sample_rate_hz * self.window_seconds))

    @property
    def corrected_ci(self):
        return 1.0 - self.familywise_alpha / self.planned_versions

    def to_dict(self):
        values = asdict(self)
        values["window_samples"] = self.window_samples
        values["corrected_ci"] = self.corrected_ci
        values["implementation_version"] = IMPLEMENTATION_VERSION
        return values


@dataclass(frozen=True)
class RawExample:
    record: object
    target_sentence_id: int
    target_label: int
    weight: float


def _require_torch():
    try:
        import torch
    except ImportError as error:
        raise ImportError("V2 training requires PyTorch; use the Colab notebook") from error
    return torch


def build_raw_eegnet(config=RawEEGNetConfig()):
    """Construct the one locked compact EEGNet-style architecture."""

    torch = _require_torch()
    nn = torch.nn
    spatial_filters = config.temporal_filters * config.depth_multiplier

    class RawEEGNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.temporal = nn.Sequential(
                nn.Conv2d(
                    1,
                    config.temporal_filters,
                    kernel_size=(1, config.temporal_kernel_samples),
                    padding="same",
                    bias=False,
                ),
                nn.BatchNorm2d(config.temporal_filters),
            )
            self.spatial = nn.Sequential(
                nn.Conv2d(
                    config.temporal_filters,
                    spatial_filters,
                    kernel_size=(config.expected_channels, 1),
                    groups=config.temporal_filters,
                    bias=False,
                ),
                nn.BatchNorm2d(spatial_filters),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(1, config.pool_one)),
                nn.Dropout(config.dropout),
            )
            self.separable = nn.Sequential(
                nn.Conv2d(
                    spatial_filters,
                    spatial_filters,
                    kernel_size=(1, config.separable_kernel_samples),
                    padding="same",
                    groups=spatial_filters,
                    bias=False,
                ),
                nn.Conv2d(
                    spatial_filters,
                    spatial_filters,
                    kernel_size=(1, 1),
                    bias=False,
                ),
                nn.BatchNorm2d(spatial_filters),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(1, config.pool_two)),
                nn.Dropout(config.dropout),
            )
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Linear(spatial_filters, len(LABELS))

        def forward(self, windows):
            if windows.ndim != 3:
                raise ValueError("windows must be batch x channels x samples")
            if windows.shape[1] != config.expected_channels:
                raise ValueError(
                    f"expected {config.expected_channels} channels, got {windows.shape[1]}"
                )
            values = self.temporal(windows.unsqueeze(1))
            values = self.spatial(values)
            values = self.separable(values)
            values = self.pool(values).flatten(1)
            return self.classifier(values)

    return RawEEGNet()


def sentence_table(records):
    labels = {}
    for record in records:
        if record.sentence_id in labels and labels[record.sentence_id] != record.label:
            raise ValueError(f"conflicting labels for sentence {record.sentence_id}")
        labels[record.sentence_id] = record.label
    sentence_ids = np.asarray(sorted(labels), dtype=np.int64)
    y = np.asarray([labels[value] for value in sentence_ids], dtype=np.int64)
    if set(np.unique(y)) != set(LABELS):
        raise ValueError(f"expected labels {LABELS}, got {sorted(np.unique(y))}")
    return sentence_ids, y


def _records_by_sentence(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[int(record.sentence_id)].append(record)
    return grouped


def make_bundle_examples(
    records,
    target_sentence_ids,
    target_labels,
    shuffled=False,
    seed=0,
):
    """Pair whole reader bundles with target sentences, optionally permuting bundles."""

    target_sentence_ids = np.asarray(target_sentence_ids, dtype=np.int64)
    target_labels = np.asarray(target_labels, dtype=np.int64)
    if len(target_sentence_ids) != len(target_labels):
        raise ValueError("target sentence IDs and labels must have equal length")
    grouped = _records_by_sentence(records)
    source_ids = target_sentence_ids.copy()
    if shuffled:
        source_ids = source_ids[np.random.default_rng(seed).permutation(len(source_ids))]
    class_counts = Counter(map(int, target_labels))
    class_weights = {
        label: len(target_labels) / (len(LABELS) * count)
        for label, count in class_counts.items()
    }
    examples = []
    for target_id, target_label, source_id in zip(
        target_sentence_ids, target_labels, source_ids
    ):
        source_records = grouped.get(int(source_id), [])
        if not source_records:
            raise ValueError(f"no recordings for source sentence {source_id}")
        reader_weight = class_weights[int(target_label)] / len(source_records)
        examples.extend(
            RawExample(
                record=record,
                target_sentence_id=int(target_id),
                target_label=int(target_label),
                weight=float(reader_weight),
            )
            for record in source_records
        )
    return examples


def channel_statistics(examples, expected_channels=104):
    """Compute leakage-safe channel moments from training examples only."""

    sums = np.zeros(expected_channels, dtype=np.float64)
    square_sums = np.zeros(expected_channels, dtype=np.float64)
    count = 0
    seen = set()
    for example in examples:
        identity = id(example.record)
        if identity in seen:
            continue
        seen.add(identity)
        eeg = np.asarray(example.record.eeg, dtype=np.float32)
        if eeg.ndim != 2 or eeg.shape[0] != expected_channels:
            raise ValueError(f"invalid raw EEG shape {eeg.shape}")
        sums += eeg.sum(axis=1, dtype=np.float64)
        square_sums += np.square(eeg, dtype=np.float32).sum(axis=1, dtype=np.float64)
        count += eeg.shape[1]
    if count < 2:
        raise ValueError("not enough training samples for channel normalization")
    mean = sums / count
    variance = np.maximum(square_sums / count - np.square(mean), 1e-8)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def window_recording(
    record,
    channel_mean,
    channel_std,
    config=RawEEGNetConfig(),
    temporal_shuffle=False,
    shuffle_seed=0,
):
    eeg = np.asarray(record.eeg, dtype=np.float32)
    if eeg.ndim != 2 or eeg.shape[0] != config.expected_channels:
        raise ValueError(f"invalid raw EEG shape {eeg.shape}")
    usable = eeg.shape[1] // config.window_samples * config.window_samples
    if usable < config.window_samples:
        raise ValueError("recording contains no complete V2 window")
    windows = eeg[:, :usable].reshape(
        config.expected_channels, -1, config.window_samples
    ).transpose(1, 0, 2)
    windows = (windows - channel_mean[None, :, None]) / channel_std[None, :, None]
    windows = np.clip(windows, -config.normalization_clip, config.normalization_clip)
    if temporal_shuffle:
        block = config.temporal_block_samples
        if config.window_samples % block:
            raise ValueError("temporal block size must divide the window length")
        blocks = config.window_samples // block
        stable = hashlib.sha256(
            f"{record.subject}|{record.sentence_id}|{shuffle_seed}".encode()
        ).digest()
        base_seed = int.from_bytes(stable[:8], "little")
        rng = np.random.default_rng(base_seed)
        reshaped = windows.reshape(len(windows), config.expected_channels, blocks, block)
        for index in range(len(reshaped)):
            reshaped[index] = reshaped[index][:, rng.permutation(blocks), :]
        windows = reshaped.reshape(
            len(windows), config.expected_channels, config.window_samples
        )
    return np.ascontiguousarray(windows, dtype=np.float32)


class RawEEGCollator:
    def __init__(
        self,
        channel_mean,
        channel_std,
        config=RawEEGNetConfig(),
        temporal_shuffle=False,
        shuffle_seed=0,
    ):
        self.channel_mean = np.asarray(channel_mean, dtype=np.float32)
        self.channel_std = np.asarray(channel_std, dtype=np.float32)
        self.config = config
        self.temporal_shuffle = temporal_shuffle
        self.shuffle_seed = shuffle_seed

    def __call__(self, batch):
        torch = _require_torch()
        arrays = []
        record_indices = []
        for record_index, example in enumerate(batch):
            values = window_recording(
                example.record,
                self.channel_mean,
                self.channel_std,
                self.config,
                temporal_shuffle=self.temporal_shuffle,
                shuffle_seed=self.shuffle_seed,
            )
            arrays.append(values)
            record_indices.extend([record_index] * len(values))
        return {
            "windows": torch.from_numpy(np.concatenate(arrays, axis=0)),
            "record_index": torch.as_tensor(record_indices, dtype=torch.long),
            "n_records": len(batch),
            "labels": torch.as_tensor(
                [LABEL_TO_INDEX[example.target_label] for example in batch],
                dtype=torch.long,
            ),
            "weights": torch.as_tensor(
                [example.weight for example in batch], dtype=torch.float32
            ),
            "sentence_ids": np.asarray(
                [example.target_sentence_id for example in batch], dtype=np.int64
            ),
        }


class _ExampleDataset:
    def __init__(self, examples):
        self.examples = list(examples)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def _loader(
    examples,
    mean,
    std,
    config,
    shuffle,
    seed,
    temporal_shuffle=False,
):
    torch = _require_torch()
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return torch.utils.data.DataLoader(
        _ExampleDataset(examples),
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=RawEEGCollator(
            mean,
            std,
            config,
            temporal_shuffle=temporal_shuffle,
            shuffle_seed=seed,
        ),
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
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _mean_record_logits(window_logits, record_index, n_records):
    torch = _require_torch()
    sums = torch.zeros(
        (n_records, window_logits.shape[1]),
        dtype=window_logits.dtype,
        device=window_logits.device,
    )
    sums.index_add_(0, record_index, window_logits)
    counts = torch.zeros(n_records, dtype=window_logits.dtype, device=window_logits.device)
    counts.index_add_(0, record_index, torch.ones_like(record_index, dtype=window_logits.dtype))
    return sums / counts[:, None].clamp_min(1.0)


def _aggregate_predictions(model, loader, device):
    torch = _require_torch()
    probability_sums = defaultdict(lambda: np.zeros(len(LABELS), dtype=np.float64))
    reader_counts = Counter()
    labels = {}
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            window_logits = model(batch["windows"].to(device, non_blocking=True))
            record_logits = _mean_record_logits(
                window_logits,
                batch["record_index"].to(device, non_blocking=True),
                batch["n_records"],
            )
            probabilities = torch.softmax(record_logits, dim=1).cpu().numpy()
            batch_labels = batch["labels"].numpy()
            for sentence_id, label_index, probability in zip(
                batch["sentence_ids"], batch_labels, probabilities
            ):
                sentence_id = int(sentence_id)
                label = INDEX_TO_LABEL[int(label_index)]
                if sentence_id in labels and labels[sentence_id] != label:
                    raise ValueError(f"conflicting labels for target sentence {sentence_id}")
                labels[sentence_id] = label
                probability_sums[sentence_id] += probability
                reader_counts[sentence_id] += 1
    sentence_ids = np.asarray(sorted(probability_sums), dtype=np.int64)
    probabilities = np.stack(
        [probability_sums[value] / reader_counts[value] for value in sentence_ids]
    )
    truth = np.asarray([labels[value] for value in sentence_ids], dtype=np.int64)
    return sentence_ids, truth, probabilities


def _train_one_model(
    records,
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

    split = StratifiedShuffleSplit(
        n_splits=1,
        test_size=config.validation_fraction,
        random_state=seed + 20_000,
    )
    train_position, validation_position = next(
        split.split(outer_train_sentence_ids, outer_train_labels)
    )
    train_ids = outer_train_sentence_ids[train_position]
    train_labels = outer_train_labels[train_position]
    validation_ids = outer_train_sentence_ids[validation_position]
    validation_labels = outer_train_labels[validation_position]
    train_examples = make_bundle_examples(
        records,
        train_ids,
        train_labels,
        shuffled=shuffled,
        seed=seed + 30_000,
    )
    validation_examples = make_bundle_examples(
        records,
        validation_ids,
        validation_labels,
        shuffled=shuffled,
        seed=seed + 40_000,
    )
    channel_mean, channel_std = channel_statistics(
        train_examples, expected_channels=config.expected_channels
    )
    train_loader = _loader(
        train_examples, channel_mean, channel_std, config, True, seed
    )
    validation_loader = _loader(
        validation_examples, channel_mean, channel_std, config, False, seed
    )

    _set_seed(seed)
    model = build_raw_eegnet(config).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
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
        weighted_loss = 0.0
        weight_total = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            window_logits = model(batch["windows"].to(device, non_blocking=True))
            record_logits = _mean_record_logits(
                window_logits,
                batch["record_index"].to(device, non_blocking=True),
                batch["n_records"],
            )
            labels = batch["labels"].to(device, non_blocking=True)
            weights = batch["weights"].to(device, non_blocking=True)
            losses = torch.nn.functional.cross_entropy(
                record_logits, labels, reduction="none"
            )
            loss = (losses * weights).sum() / weights.sum().clamp_min(1e-8)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip)
            optimizer.step()
            weighted_loss += float((losses.detach() * weights).sum().cpu())
            weight_total += float(weights.sum().cpu())

        _, validation_truth, validation_probabilities = _aggregate_predictions(
            model, validation_loader, device
        )
        validation_predictions = np.asarray(
            [LABELS[index] for index in validation_probabilities.argmax(axis=1)],
            dtype=np.int64,
        )
        score = f1_score(
            validation_truth,
            validation_predictions,
            labels=list(LABELS),
            average="macro",
            zero_division=0,
        )
        row = {
            "epoch": epoch + 1,
            "train_loss": weighted_loss / max(weight_total, 1e-8),
            "validation_macro_f1": float(score),
        }
        history.append(row)
        print(
            f"  epoch={row['epoch']:02d} loss={row['train_loss']:.4f} "
            f"val_macro_f1={score:.4f}",
            flush=True,
        )
        if np.isfinite(score) and score > best_score + 1e-6:
            best_score = float(score)
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("training produced no finite validation result")
    model.load_state_dict(best_state)
    return model, channel_mean, channel_std, history, best_epoch, best_score


def _metric_values(y_true, y_pred):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    per_class = f1_score(
        y_true, y_pred, labels=list(LABELS), average=None, zero_division=0
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


def _prediction_rows(setup, seed, fold, sentence_ids, truth, probabilities):
    predictions = np.asarray(
        [LABELS[index] for index in probabilities.argmax(axis=1)], dtype=np.int64
    )
    rows = []
    for sentence_id, label, prediction, probability in zip(
        sentence_ids, truth, predictions, probabilities
    ):
        rows.append(
            {
                "setup": setup,
                "seed": int(seed),
                "fold": int(fold),
                "sentence_id": int(sentence_id),
                "label": int(label),
                "prediction": int(prediction),
                "probability_negative": float(probability[0]),
                "probability_neutral": float(probability[1]),
                "probability_positive": float(probability[2]),
            }
        )
    return rows, predictions


def _atomic_csv(frame, path):
    path = Path(path)
    temporary = path.with_suffix(".tmp.csv")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _write_json(payload, path):
    path = Path(path)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def _read_csv(path, columns):
    path = Path(path)
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=columns)


def _drop_fold_rows(frame, setups, seed, fold):
    if frame.empty:
        return frame
    mask = (
        frame["setup"].isin(setups)
        & (frame["seed"].astype(int) == int(seed))
        & (frame["fold"].astype(int) == int(fold))
    )
    return frame.loc[~mask].copy()


def _fold_complete(metrics, predictions, setups, seed, fold, expected_sentences, marker):
    if not Path(marker).exists():
        return False
    for setup in setups:
        metric_count = len(
            metrics[
                (metrics["setup"] == setup)
                & (metrics["seed"].astype(int) == int(seed))
                & (metrics["fold"].astype(int) == int(fold))
            ]
        )
        prediction_count = len(
            predictions[
                (predictions["setup"] == setup)
                & (predictions["seed"].astype(int) == int(seed))
                & (predictions["fold"].astype(int) == int(fold))
            ]
        )
        if metric_count != 1 or prediction_count != expected_sentences:
            return False
    return True


def bootstrap_alignment_delta(predictions, config=RawEEGNetConfig(), seed=2026):
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(seed)
    paired = []
    seed_deltas = {}
    for run_seed in sorted(predictions["seed"].unique()):
        subset = predictions[predictions["seed"] == run_seed]
        aligned = subset[subset["setup"] == ALIGNED].sort_values("sentence_id")
        shuffled = subset[subset["setup"] == SHUFFLED].sort_values("sentence_id")
        if not np.array_equal(
            aligned["sentence_id"].to_numpy(), shuffled["sentence_id"].to_numpy()
        ):
            raise ValueError("aligned and shuffled V2 predictions are not paired")
        truth = aligned["label"].to_numpy(dtype=np.int64)
        aligned_prediction = aligned["prediction"].to_numpy(dtype=np.int64)
        shuffled_prediction = shuffled["prediction"].to_numpy(dtype=np.int64)
        delta = f1_score(
            truth,
            aligned_prediction,
            labels=list(LABELS),
            average="macro",
            zero_division=0,
        ) - f1_score(
            truth,
            shuffled_prediction,
            labels=list(LABELS),
            average="macro",
            zero_division=0,
        )
        seed_deltas[str(int(run_seed))] = float(delta)
        paired.append((truth, aligned_prediction, shuffled_prediction))
    draws = []
    for _ in range(config.bootstrap_samples):
        deltas = []
        for truth, aligned_prediction, shuffled_prediction in paired:
            indices = rng.integers(0, len(truth), size=len(truth))
            deltas.append(
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
        draws.append(float(np.mean(deltas)))
    alpha = config.familywise_alpha / config.planned_versions
    return {
        "observed_mean_seed_delta": float(np.mean(list(seed_deltas.values()))),
        "bootstrap_mean_delta": float(np.mean(draws)),
        "ci_level": float(1 - alpha),
        "ci_low": float(np.quantile(draws, alpha / 2)),
        "ci_high": float(np.quantile(draws, 1 - alpha / 2)),
        "seed_deltas": seed_deltas,
        "bootstrap_samples": int(config.bootstrap_samples),
        "planned_screen_versions": int(config.planned_versions),
        "familywise_alpha": float(config.familywise_alpha),
    }


def gate_report(metrics, delta, config=RawEEGNetConfig()):
    aligned = metrics[metrics["setup"] == ALIGNED]
    shuffled = metrics[metrics["setup"] == SHUFFLED]
    majority = metrics[metrics["setup"] == "majority"]
    aligned_macro = float(aligned["macro_f1"].mean())
    shuffled_macro = float(shuffled["macro_f1"].mean())
    majority_macro = float(majority["macro_f1"].mean())
    observed_delta = aligned_macro - shuffled_macro
    positive_seeds = sum(value > 0 for value in delta["seed_deltas"].values())
    criteria = {
        "balanced_accuracy_above_chance": float(aligned["balanced_accuracy"].mean())
        > 1 / 3,
        "macro_f1_above_majority": aligned_macro > majority_macro,
        "aligned_minus_shuffled_at_least_minimum": observed_delta
        >= config.minimum_delta,
        "enough_positive_seeds": positive_seeds >= config.minimum_positive_seeds,
        "corrected_bootstrap_ci_low_above_zero": delta["ci_low"] > 0,
    }
    passes = all(criteria.values())
    core_without_ci = all(
        value
        for key, value in criteria.items()
        if key != "corrected_bootstrap_ci_low_above_zero"
    )
    status = "green" if passes else "yellow" if core_without_ci else "red"
    decision = {
        "green": "GREEN — eligible for later tuning after the three-version screen",
        "yellow": "YELLOW — suggestive only; record without tuning",
        "red": "RED — no alignment-specific V2 evidence; do not tune",
    }[status]
    return {
        "aligned_macro_f1": aligned_macro,
        "shuffled_macro_f1": shuffled_macro,
        "majority_macro_f1": majority_macro,
        "aligned_balanced_accuracy": float(aligned["balanced_accuracy"].mean()),
        "chance_balanced_accuracy": 1 / 3,
        "observed_fold_mean_delta": observed_delta,
        "minimum_required_delta": config.minimum_delta,
        "positive_seeds": int(positive_seeds),
        "minimum_positive_seeds": config.minimum_positive_seeds,
        "bootstrap": delta,
        "criteria": criteria,
        "status": status,
        "passes": bool(passes),
        "decision": decision,
    }


def smoke_test_raw_eegnet(records, config=RawEEGNetConfig(), device="cuda"):
    torch = _require_torch()
    sentence_ids, y = sentence_table(records)
    selected_ids = sentence_ids[: min(4, len(sentence_ids))]
    selected_labels = y[: len(selected_ids)]
    examples = make_bundle_examples(records, selected_ids, selected_labels)
    mean, std = channel_statistics(examples, config.expected_channels)
    batch = RawEEGCollator(mean, std, config)(examples[: min(4, len(examples))])
    resolved = torch.device(device if torch.cuda.is_available() else "cpu")
    model = build_raw_eegnet(config).to(resolved).eval()
    with torch.inference_mode():
        logits = model(batch["windows"].to(resolved))
    return {
        "device": str(resolved),
        "input_windows": list(batch["windows"].shape),
        "window_logits": list(logits.shape),
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "channel_mean_finite": bool(np.isfinite(mean).all()),
        "channel_std_minimum": float(std.min()),
    }


def evaluate_raw_eegnet(
    records,
    output_dir,
    dataset_fingerprint,
    config=RawEEGNetConfig(),
    device="cuda",
):
    """Run/resume the complete three-seed sentence-grouped V2 evaluation."""

    torch = _require_torch()
    from sklearn.model_selection import StratifiedKFold

    output_dir = Path(output_dir)
    completion_dir = output_dir / "completed_folds"
    history_dir = output_dir / "histories"
    output_dir.mkdir(parents=True, exist_ok=True)
    completion_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)
    resolved_device = torch.device(device if torch.cuda.is_available() else "cpu")
    if resolved_device.type != "cuda":
        raise RuntimeError("V2 evaluation requires a Colab GPU runtime")

    signature_payload = {
        "implementation_version": IMPLEMENTATION_VERSION,
        "dataset_fingerprint": str(dataset_fingerprint),
        "config": config.to_dict(),
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode()
    ).hexdigest()
    signature_path = output_dir / "run_signature.json"
    if signature_path.exists():
        existing = json.loads(signature_path.read_text())
        if existing.get("signature") != signature:
            raise RuntimeError(
                "V2 result directory belongs to a different data/configuration signature"
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
        "label",
        "prediction",
        "probability_negative",
        "probability_neutral",
        "probability_positive",
    ]
    partial_metrics_path = output_dir / "partial_fold_metrics.csv"
    partial_predictions_path = output_dir / "partial_oof_predictions.csv"
    metrics = _read_csv(partial_metrics_path, metric_columns)
    predictions = _read_csv(partial_predictions_path, prediction_columns)
    sentence_ids, y = sentence_table(records)

    for seed in config.seeds:
        outer = StratifiedKFold(config.n_splits, shuffle=True, random_state=seed)
        for fold, (train_position, test_position) in enumerate(
            outer.split(sentence_ids, y)
        ):
            train_ids, train_labels = sentence_ids[train_position], y[train_position]
            test_ids, test_labels = sentence_ids[test_position], y[test_position]
            for setup, shuffled in ((ALIGNED, False), (SHUFFLED, True)):
                produced_setups = [setup, TEMPORAL_SHUFFLE] if not shuffled else [setup]
                marker = completion_dir / f"{setup}_seed{seed}_fold{fold}.json"
                if _fold_complete(
                    metrics,
                    predictions,
                    produced_setups,
                    seed,
                    fold,
                    len(test_ids),
                    marker,
                ):
                    print(f"Reused {setup} seed={seed} fold={fold}", flush=True)
                    continue
                metrics = _drop_fold_rows(metrics, produced_setups, seed, fold)
                predictions = _drop_fold_rows(
                    predictions, produced_setups, seed, fold
                )
                fit_seed = int(seed * 100 + fold * 10 + int(shuffled))
                print(f"Training {setup} seed={seed} fold={fold}", flush=True)
                model, mean, std, history, best_epoch, best_score = _train_one_model(
                    records,
                    train_ids,
                    train_labels,
                    config,
                    resolved_device,
                    fit_seed,
                    shuffled,
                )
                test_examples = make_bundle_examples(
                    records,
                    test_ids,
                    test_labels,
                    shuffled=shuffled,
                    seed=fit_seed + 50_000,
                )
                test_loader = _loader(
                    test_examples, mean, std, config, False, fit_seed + 60_000
                )
                evaluated_ids, truth, probabilities = _aggregate_predictions(
                    model, test_loader, resolved_device
                )
                rows, hard_predictions = _prediction_rows(
                    setup, seed, fold, evaluated_ids, truth, probabilities
                )
                metric_row = {
                    "setup": setup,
                    "seed": seed,
                    "fold": fold,
                    "best_epoch": best_epoch,
                    "validation_macro_f1": best_score,
                    **_metric_values(truth, hard_predictions),
                }
                new_metrics = [metric_row]
                new_predictions = rows

                if not shuffled:
                    temporal_loader = _loader(
                        test_examples,
                        mean,
                        std,
                        config,
                        False,
                        fit_seed + 70_000,
                        temporal_shuffle=True,
                    )
                    temporal_ids, temporal_truth, temporal_probabilities = (
                        _aggregate_predictions(model, temporal_loader, resolved_device)
                    )
                    temporal_rows, temporal_predictions = _prediction_rows(
                        TEMPORAL_SHUFFLE,
                        seed,
                        fold,
                        temporal_ids,
                        temporal_truth,
                        temporal_probabilities,
                    )
                    new_metrics.append(
                        {
                            "setup": TEMPORAL_SHUFFLE,
                            "seed": seed,
                            "fold": fold,
                            "best_epoch": best_epoch,
                            "validation_macro_f1": best_score,
                            **_metric_values(temporal_truth, temporal_predictions),
                        }
                    )
                    new_predictions.extend(temporal_rows)

                metrics = pd.concat(
                    [metrics, pd.DataFrame(new_metrics)], ignore_index=True
                )
                predictions = pd.concat(
                    [predictions, pd.DataFrame(new_predictions)], ignore_index=True
                )
                metrics = metrics[metric_columns]
                predictions = predictions[prediction_columns]
                _atomic_csv(metrics, partial_metrics_path)
                _atomic_csv(predictions, partial_predictions_path)
                _write_json(
                    {
                        "setup": setup,
                        "seed": seed,
                        "fold": fold,
                        "best_epoch": best_epoch,
                        "validation_macro_f1": best_score,
                        "history": history,
                    },
                    history_dir / f"{setup}_seed{seed}_fold{fold}.json",
                )
                _write_json(
                    {
                        "signature": signature,
                        "setup": setup,
                        "seed": seed,
                        "fold": fold,
                        "produced_setups": produced_setups,
                    },
                    marker,
                )
                del model
                torch.cuda.empty_cache()

            majority_marker = completion_dir / f"majority_seed{seed}_fold{fold}.json"
            if not _fold_complete(
                metrics,
                predictions,
                ["majority"],
                seed,
                fold,
                len(test_ids),
                majority_marker,
            ):
                metrics = _drop_fold_rows(metrics, ["majority"], seed, fold)
                predictions = _drop_fold_rows(
                    predictions, ["majority"], seed, fold
                )
                counts = Counter(map(int, train_labels))
                majority_label = counts.most_common(1)[0][0]
                hard_predictions = np.full(len(test_ids), majority_label, dtype=np.int64)
                probability = np.zeros((len(test_ids), len(LABELS)), dtype=np.float64)
                probability[:, LABEL_TO_INDEX[majority_label]] = 1.0
                rows, _ = _prediction_rows(
                    "majority", seed, fold, test_ids, test_labels, probability
                )
                metrics = pd.concat(
                    [
                        metrics,
                        pd.DataFrame(
                            [
                                {
                                    "setup": "majority",
                                    "seed": seed,
                                    "fold": fold,
                                    "best_epoch": np.nan,
                                    "validation_macro_f1": np.nan,
                                    **_metric_values(test_labels, hard_predictions),
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )[metric_columns]
                predictions = pd.concat(
                    [predictions, pd.DataFrame(rows)], ignore_index=True
                )[prediction_columns]
                _atomic_csv(metrics, partial_metrics_path)
                _atomic_csv(predictions, partial_predictions_path)
                _write_json(
                    {"signature": signature, "seed": seed, "fold": fold},
                    majority_marker,
                )

    metrics = metrics.sort_values(["seed", "fold", "setup"]).reset_index(drop=True)
    predictions = predictions.sort_values(
        ["setup", "seed", "sentence_id"]
    ).reset_index(drop=True)
    summary = (
        metrics.groupby("setup")[["accuracy", "balanced_accuracy", "macro_f1"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    delta = bootstrap_alignment_delta(predictions, config)
    gate = gate_report(metrics, delta, config)
    _atomic_csv(metrics, output_dir / "fold_metrics.csv")
    _atomic_csv(predictions, output_dir / "oof_predictions.csv")
    _atomic_csv(summary, output_dir / "summary.csv")
    _write_json(config.to_dict(), output_dir / "evaluation_config.json")
    _write_json(delta, output_dir / "alignment_delta.json")
    _write_json(gate, output_dir / "viability_gate.json")
    _write_json(
        {
            "stage": "complete",
            "signature": signature,
            "completed_metric_rows": len(metrics),
            "completed_prediction_rows": len(predictions),
            "status": gate["status"],
        },
        output_dir / "runtime_status.json",
    )
    return metrics, predictions, summary, delta, gate
