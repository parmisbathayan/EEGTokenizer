"""Final V4: supervised adaptation of the official MTP-pretrained TFM encoder."""

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import copy
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd

from .encoder_probe import _sha256_file, _unwrap_encoder_checkpoint
from .token_map import LABELS, LABEL_TO_INDEX


_CHECKPOINT_CACHE = {}


def _checkpoint_cache_entry(checkpoint_path):
    """Load and hash the small Drive checkpoint once per Colab process."""

    import torch

    path = Path(checkpoint_path).resolve()
    stat = path.stat()
    key = (str(path), int(stat.st_size), int(stat.st_mtime_ns))
    if key not in _CHECKPOINT_CACHE:
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu")
        _CHECKPOINT_CACHE.clear()
        _CHECKPOINT_CACHE[key] = {
            "state": _unwrap_encoder_checkpoint(checkpoint),
            "sha256": _sha256_file(path),
        }
    return _CHECKPOINT_CACHE[key]


@dataclass(frozen=True)
class FinetuneProbeConfig:
    """Locked architecture, optimization, evaluation, and gate settings."""

    implementation_version: int = 2
    seeds: tuple = (42, 52, 62)
    n_splits: int = 5
    validation_fraction: float = 0.20
    codebook_size: int = 8192
    embedding_size: int = 64
    expected_channels: int = 104
    channel_group_size: int = 16
    official_max_sequence_length: int = 2048
    batch_size: int = 8
    evaluation_batch_size: int = 16
    max_epochs: int = 12
    minimum_epochs: int = 4
    patience: int = 3
    encoder_learning_rate: float = 1e-5
    head_learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    gradient_clip: float = 1.0
    bootstrap_samples: int = 5000
    planned_versions: int = 4
    familywise_alpha: float = 0.05
    minimum_delta: float = 0.015
    minimum_positive_seeds: int = 2

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class SentenceExample:
    """A target sentence/label paired with one source sentence's reader records."""

    sentence_id: int
    label: int
    feature_sentence_id: int
    records: tuple


def _set_seed(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FinetunableOfficialTFMEncoder:
    """Official MTP encoder with a new head and all encoder weights trainable."""

    def __init__(
        self,
        repo_dir,
        checkpoint_path,
        config=FinetuneProbeConfig(),
        device="cuda",
        initialization_seed=0,
    ):
        import torch

        repo_dir = str(Path(repo_dir).resolve())
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)
        try:
            from models.tfm_token import get_tfm_token_classifier_64x4
        except ImportError as error:
            raise ImportError(
                "could not import the official TFM encoder; run Colab setup first"
            ) from error

        _set_seed(initialization_seed)
        self.torch = torch
        self.config = config
        self.repo_dir = repo_dir
        self.checkpoint_path = str(Path(checkpoint_path).resolve())
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = get_tfm_token_classifier_64x4(
            n_classes=len(LABELS),
            code_book_size=config.codebook_size,
            emb_size=config.embedding_size,
        )
        checkpoint_entry = _checkpoint_cache_entry(self.checkpoint_path)
        state = checkpoint_entry["state"]
        filtered = {
            name: value
            for name, value in state.items()
            if "classification_head" not in name
        }
        incompatible = self.model.load_state_dict(filtered, strict=False)
        non_head_missing = [
            name
            for name in incompatible.missing_keys
            if "classification_head" not in name
        ]
        if non_head_missing:
            raise ValueError(
                "MTP checkpoint is missing non-head encoder weights: "
                f"{non_head_missing}"
            )
        if incompatible.unexpected_keys:
            raise ValueError(
                f"unexpected MTP checkpoint keys: {incompatible.unexpected_keys}"
            )
        if not hasattr(self.model, "classification_head"):
            raise AttributeError("official encoder has no classification_head")
        group = config.channel_group_size
        group_sizes = [group] * (config.expected_channels // group)
        if config.expected_channels % group:
            group_sizes.append(config.expected_channels % group)
        self.group_sizes = tuple(group_sizes)
        group_mixer = torch.nn.Linear(len(group_sizes) * len(LABELS), len(LABELS))
        with torch.no_grad():
            group_mixer.weight.zero_()
            group_mixer.bias.zero_()
            for output_class in range(len(LABELS)):
                for group_index, group_size in enumerate(group_sizes):
                    input_index = group_index * len(LABELS) + output_class
                    group_mixer.weight[output_class, input_index] = (
                        group_size / config.expected_channels
                    )
        self.model.add_module("v4_group_mixer", group_mixer)
        nonfloating_parameters = []
        for name, parameter in self.model.named_parameters():
            is_differentiable = parameter.is_floating_point() or parameter.is_complex()
            parameter.requires_grad_(is_differentiable)
            if not is_differentiable:
                nonfloating_parameters.append(
                    {
                        "name": name,
                        "dtype": str(parameter.dtype),
                        "parameter_count": int(parameter.numel()),
                    }
                )
        self.model.to(self.device)

        head_parameters = []
        encoder_parameters = []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if "classification_head" in name or "v4_group_mixer" in name:
                head_parameters.append(parameter)
            else:
                encoder_parameters.append(parameter)
        if not head_parameters or not encoder_parameters:
            raise ValueError("could not separate encoder and classification-head parameters")
        head_ids = {id(parameter) for parameter in head_parameters}
        encoder_ids = {id(parameter) for parameter in encoder_parameters}
        trainable_ids = {
            id(parameter)
            for parameter in self.model.parameters()
            if parameter.requires_grad
        }
        if (head_ids & encoder_ids) or ((head_ids | encoder_ids) != trainable_ids):
            raise RuntimeError("V4 optimizer parameter groups overlap or omit parameters")
        self.encoder_parameters = encoder_parameters
        self.head_parameters = head_parameters
        total_parameters = sum(parameter.numel() for parameter in self.model.parameters())
        encoder_count = sum(parameter.numel() for parameter in encoder_parameters)
        head_count = sum(parameter.numel() for parameter in head_parameters)
        loaded = len(filtered) - len(incompatible.unexpected_keys)
        self.report = {
            "repo_dir": repo_dir,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": checkpoint_entry["sha256"],
            "checkpoint_state_key_count": len(state),
            "loaded_key_count": loaded,
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
            "excluded_classification_head_keys": sorted(set(state) - set(filtered)),
            "all_floating_parameters_trainable": all(
                parameter.requires_grad
                for parameter in self.model.parameters()
                if parameter.is_floating_point() or parameter.is_complex()
            ),
            "nonfloating_parameters": nonfloating_parameters,
            "total_parameter_count": int(total_parameters),
            "trainable_encoder_parameter_count": int(encoder_count),
            "trainable_head_parameter_count": int(head_count),
            "channel_group_sizes": list(self.group_sizes),
            "group_mixer_initialization": "channel-count-weighted class identity",
        }

    def train(self):
        self.model.train()
        return self

    def eval(self):
        self.model.eval()
        return self

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state):
        return self.model.load_state_dict(state, strict=True)

    def parameters(self):
        return self.model.parameters()

    def _forward_groups(self, token_groups):
        if token_groups.ndim != 3:
            raise ValueError(
                "expected batch x channels x time token groups, got "
                f"{tuple(token_groups.shape)}"
            )
        channels, time = token_groups.shape[1:]
        estimated_length = channels * time + channels
        if estimated_length > self.config.official_max_sequence_length:
            raise ValueError(
                f"group sequence estimate {estimated_length} exceeds official maximum "
                f"{self.config.official_max_sequence_length}"
            )
        logits = self.model(token_groups, num_ch=channels)
        if logits.ndim != 2 or logits.shape != (len(token_groups), len(LABELS)):
            raise ValueError(
                f"unexpected official encoder logits {tuple(logits.shape)}"
            )
        return logits

    def __call__(self, token_maps):
        torch = self.torch
        if token_maps.ndim != 3:
            raise ValueError(
                f"expected batch x channels x time token maps, got {token_maps.shape}"
            )
        batch, channels, time = token_maps.shape
        if channels != self.config.expected_channels:
            raise ValueError(
                f"expected {self.config.expected_channels} channels, got {channels}"
            )
        group = self.config.channel_group_size
        full_group_count = channels // group
        remainder = channels % group
        group_logits = []
        if full_group_count:
            full = token_maps[:, : full_group_count * group].reshape(
                batch * full_group_count,
                group,
                time,
            )
            full_logits = self._forward_groups(full).reshape(
                batch,
                full_group_count,
                len(LABELS),
            )
            group_logits.append(full_logits)
        if remainder:
            tail = token_maps[:, full_group_count * group :]
            tail_logits = self._forward_groups(tail)
            group_logits.append(tail_logits.unsqueeze(1))
        if not group_logits:
            raise ValueError("no channel groups were constructed")
        stacked_logits = torch.cat(group_logits, dim=1)
        if stacked_logits.shape[1] != len(self.group_sizes):
            raise ValueError(
                f"expected {len(self.group_sizes)} channel groups, got "
                f"{stacked_logits.shape[1]}"
            )
        logits = self.model.v4_group_mixer(stacked_logits.flatten(start_dim=1))
        if not torch.isfinite(logits).all():
            raise ValueError("official encoder produced non-finite logits")
        return logits


def _sentence_groups(records):
    groups = defaultdict(list)
    labels = {}
    for record in records:
        sentence_id = int(record.sentence_id)
        label = int(record.label)
        if sentence_id in labels and labels[sentence_id] != label:
            raise ValueError(f"conflicting labels for sentence {sentence_id}")
        labels[sentence_id] = label
        groups[sentence_id].append(record)
    for sentence_id, sentence_records in groups.items():
        subjects = [record.subject for record in sentence_records]
        if len(subjects) != len(set(subjects)):
            raise ValueError(f"duplicate subject recordings for sentence {sentence_id}")
        groups[sentence_id] = tuple(sorted(sentence_records, key=lambda item: item.subject))
    sentence_ids = np.asarray(sorted(groups), dtype=np.int64)
    y = np.asarray([labels[int(sentence_id)] for sentence_id in sentence_ids], dtype=np.int64)
    return dict(groups), sentence_ids, y


def _deranged_mapping(sentence_ids, rng):
    """Map each target to a different source within the same data split."""

    sentence_ids = np.asarray(sentence_ids, dtype=np.int64)
    if len(sentence_ids) < 2:
        raise ValueError("a shuffled split needs at least two sentences")
    order = sentence_ids[rng.permutation(len(sentence_ids))]
    sources = np.roll(order, 1)
    mapping = {
        int(target): int(source)
        for target, source in zip(order, sources)
    }
    if any(target == source for target, source in mapping.items()):
        raise RuntimeError("failed to construct a deranged sentence mapping")
    return mapping


def _examples_for_split(groups, sentence_ids, shuffled, rng):
    sentence_ids = np.asarray(sentence_ids, dtype=np.int64)
    if shuffled:
        source_mapping = _deranged_mapping(sentence_ids, rng)
    else:
        source_mapping = {
            int(sentence_id): int(sentence_id) for sentence_id in sentence_ids
        }
    examples = []
    for sentence_id in sentence_ids:
        target = int(sentence_id)
        source = source_mapping[target]
        target_records = groups[target]
        label = int(target_records[0].label)
        examples.append(
            SentenceExample(
                sentence_id=target,
                label=label,
                feature_sentence_id=source,
                records=groups[source],
            )
        )
    return examples


def _items_by_length(items, batch_size, rng=None):
    by_length = defaultdict(list)
    for example, record in items:
        by_length[record.tokens.shape[1]].append((example, record))
    batches = []
    for length in sorted(by_length):
        length_items = by_length[length]
        if rng is not None:
            rng.shuffle(length_items)
        batches.extend(
            length_items[start : start + batch_size]
            for start in range(0, len(length_items), batch_size)
        )
    if rng is not None:
        rng.shuffle(batches)
    return batches


def _class_weight_tensor(examples, device):
    import torch

    counts = Counter(example.label for example in examples)
    if set(counts) != set(LABELS):
        raise ValueError(f"training split is missing a class: {counts}")
    weights = [len(examples) / (len(LABELS) * counts[label]) for label in LABELS]
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def _batch_tensors(batch, device):
    import torch

    token_maps = np.stack([record.tokens for _, record in batch])
    if token_maps.dtype != np.uint16:
        token_maps = token_maps.astype(np.uint16, copy=False)
    tokens = torch.as_tensor(token_maps.astype(np.int64), device=device)
    labels = torch.as_tensor(
        [LABEL_TO_INDEX[example.label] for example, _ in batch],
        dtype=torch.long,
        device=device,
    )
    return tokens, labels


def _predict_examples(model, examples, device, batch_size):
    import torch

    probability_sums = defaultdict(lambda: np.zeros(len(LABELS), dtype=np.float64))
    reader_counts = Counter()
    labels = {}
    feature_sentence_ids = {}
    items = [
        (example, record)
        for example in examples
        for record in example.records
    ]
    model.eval()
    with torch.inference_mode():
        for batch in _items_by_length(items, batch_size):
            tokens, _ = _batch_tensors(batch, device)
            probabilities = torch.softmax(model(tokens), dim=1).cpu().numpy()
            for (example, _), probability in zip(batch, probabilities):
                sentence_id = example.sentence_id
                labels[sentence_id] = example.label
                feature_sentence_ids[sentence_id] = example.feature_sentence_id
                probability_sums[sentence_id] += probability
                reader_counts[sentence_id] += 1
    sentence_ids = np.asarray(sorted(probability_sums), dtype=np.int64)
    probabilities = np.stack(
        [
            probability_sums[int(sentence_id)] / reader_counts[int(sentence_id)]
            for sentence_id in sentence_ids
        ]
    )
    truth = np.asarray(
        [labels[int(sentence_id)] for sentence_id in sentence_ids],
        dtype=np.int64,
    )
    source_ids = np.asarray(
        [feature_sentence_ids[int(sentence_id)] for sentence_id in sentence_ids],
        dtype=np.int64,
    )
    return sentence_ids, source_ids, truth, probabilities


def _train_one_model(
    groups,
    train_sentence_ids,
    train_labels,
    repo_dir,
    checkpoint_path,
    config,
    device,
    seed,
    shuffled,
    status_callback=None,
):
    import torch
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedShuffleSplit

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=config.validation_fraction,
        random_state=seed + 20_000,
    )
    fit_position, validation_position = next(
        splitter.split(train_sentence_ids, train_labels)
    )
    fit_ids = train_sentence_ids[fit_position]
    validation_ids = train_sentence_ids[validation_position]
    mapping_rng = np.random.default_rng(seed + 30_000)
    fit_examples = _examples_for_split(
        groups, fit_ids, shuffled=shuffled, rng=mapping_rng
    )
    validation_examples = _examples_for_split(
        groups, validation_ids, shuffled=shuffled, rng=mapping_rng
    )

    _set_seed(seed)
    model = FinetunableOfficialTFMEncoder(
        repo_dir,
        checkpoint_path,
        config=config,
        device=device,
        initialization_seed=seed,
    )
    initial_encoder_parameters = [
        parameter.detach().cpu().clone()
        for parameter in model.encoder_parameters
    ]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.encoder_parameters,
                "lr": config.encoder_learning_rate,
            },
            {
                "params": model.head_parameters,
                "lr": config.head_learning_rate,
            },
        ],
        weight_decay=config.weight_decay,
    )
    class_weights = _class_weight_tensor(fit_examples, model.device)
    trainable = list(model.parameters())
    best_score = -np.inf
    best_epoch = None
    best_state = None
    stale_epochs = 0
    history = []
    sampling_rng = np.random.default_rng(seed + 40_000)
    for epoch in range(config.max_epochs):
        model.train()
        sampled_items = [
            (
                example,
                example.records[int(sampling_rng.integers(0, len(example.records)))],
            )
            for example in fit_examples
        ]
        weighted_loss_sum = 0.0
        item_count = 0
        for batch in _items_by_length(
            sampled_items,
            config.batch_size,
            rng=sampling_rng,
        ):
            optimizer.zero_grad(set_to_none=True)
            tokens, labels = _batch_tensors(batch, model.device)
            logits = model(tokens)
            loss = torch.nn.functional.cross_entropy(
                logits,
                labels,
                weight=class_weights,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, config.gradient_clip)
            optimizer.step()
            weighted_loss_sum += float(loss.detach().cpu()) * len(batch)
            item_count += len(batch)

        _, _, validation_truth, validation_probabilities = _predict_examples(
            model,
            validation_examples,
            model.device,
            config.evaluation_batch_size,
        )
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
        row = {
            "epoch": epoch + 1,
            "train_loss": weighted_loss_sum / max(item_count, 1),
            "validation_macro_f1": float(validation_macro_f1),
        }
        history.append(row)
        print(
            f"  epoch={epoch + 1:02d} loss={row['train_loss']:.4f} "
            f"val_macro_f1={validation_macro_f1:.4f}",
            flush=True,
        )
        if status_callback is not None:
            status_callback(
                "fit_epoch",
                epoch=epoch + 1,
                validation_macro_f1=float(validation_macro_f1),
            )
        if validation_macro_f1 > best_score + 1e-6:
            best_score = float(validation_macro_f1)
            best_epoch = epoch + 1
            best_state = copy.deepcopy(
                {name: value.detach().cpu() for name, value in model.state_dict().items()}
            )
            stale_epochs = 0
        else:
            stale_epochs += 1
        if (
            epoch + 1 >= config.minimum_epochs
            and stale_epochs >= config.patience
        ):
            break
    if best_state is None:
        raise RuntimeError("training did not produce a finite validation score")
    model.load_state_dict(best_state)
    model.model.to(model.device)
    encoder_update_squared = sum(
        float(
            ((parameter.detach().cpu() - initial) ** 2).sum()
        )
        for parameter, initial in zip(
            model.encoder_parameters,
            initial_encoder_parameters,
        )
    )
    encoder_update_l2 = math.sqrt(encoder_update_squared)
    if not np.isfinite(encoder_update_l2) or encoder_update_l2 <= 0:
        raise RuntimeError("V4 training did not update the pretrained encoder")
    return model, history, best_epoch, best_score, encoder_update_l2


def _metric_values(y_true, y_pred):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
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


def _atomic_json(payload, path):
    path = Path(path)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def _read_partial(path, columns):
    path = Path(path)
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=columns)


def _bootstrap_delta(predictions, config, seed=2026):
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(seed)
    paired = []
    for run_seed in sorted(predictions["seed"].unique()):
        subset = predictions[predictions["seed"] == run_seed]
        aligned = subset[subset["setup"] == "encoder_finetune"].sort_values(
            "sentence_id"
        )
        shuffled = subset[
            subset["setup"] == "encoder_finetune_shuffled"
        ].sort_values("sentence_id")
        if not np.array_equal(
            aligned["sentence_id"].to_numpy(),
            shuffled["sentence_id"].to_numpy(),
        ):
            raise ValueError("aligned and shuffled V4 predictions are not paired")
        if not np.array_equal(
            aligned["label"].to_numpy(dtype=np.int64),
            shuffled["label"].to_numpy(dtype=np.int64),
        ):
            raise ValueError("aligned and shuffled V4 labels are not paired")
        paired.append(
            (
                aligned["label"].to_numpy(dtype=np.int64),
                aligned["prediction"].to_numpy(dtype=np.int64),
                shuffled["prediction"].to_numpy(dtype=np.int64),
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
    return {
        "mean_delta": float(np.mean(draws)),
        "ci_level": float(1 - per_comparison_alpha),
        "ci_low": float(np.quantile(draws, per_comparison_alpha / 2)),
        "ci_high": float(np.quantile(draws, 1 - per_comparison_alpha / 2)),
        "bootstrap_samples": config.bootstrap_samples,
        "planned_versions": config.planned_versions,
        "familywise_alpha": config.familywise_alpha,
    }


def _gate_report(metrics, delta, config):
    aligned = metrics[metrics["setup"] == "encoder_finetune"]
    shuffled = metrics[metrics["setup"] == "encoder_finetune_shuffled"]
    majority = metrics[metrics["setup"] == "majority"]
    aligned_macro = float(aligned["macro_f1"].mean())
    shuffled_macro = float(shuffled["macro_f1"].mean())
    majority_macro = float(majority["macro_f1"].mean())
    observed_delta = aligned_macro - shuffled_macro
    seed_means = (
        metrics[
            metrics["setup"].isin(
                ["encoder_finetune", "encoder_finetune_shuffled"]
            )
        ]
        .groupby(["seed", "setup"])["macro_f1"]
        .mean()
        .unstack("setup")
    )
    seed_deltas = {
        str(int(seed)): float(
            row["encoder_finetune"] - row["encoder_finetune_shuffled"]
        )
        for seed, row in seed_means.iterrows()
    }
    positive_seeds = sum(value > 0 for value in seed_deltas.values())
    criteria = {
        "balanced_accuracy_above_chance": float(
            aligned["balanced_accuracy"].mean()
        )
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
            "V4 PASS — adapted TFM carries reproducible aligned signal"
            if passes
            else "V4 FAIL — stop the final bounded TFM transfer sequence"
        ),
    }


def evaluate_finetuned_encoder(
    records,
    repo_dir,
    checkpoint_path,
    output_dir,
    cache_report,
    source_revision,
    config=FinetuneProbeConfig(),
    device="cuda",
    resume=True,
    status_callback=None,
):
    """Run sentence-grouped V4 fine-tuning with fold/setup-level resumption."""

    import torch
    from sklearn.dummy import DummyClassifier
    from sklearn.model_selection import StratifiedKFold

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_device = torch.device(
        device if device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    if resolved_device.type != "cuda":
        raise RuntimeError("V4 requires a Colab GPU runtime")
    checkpoint_sha256 = _checkpoint_cache_entry(checkpoint_path)["sha256"]
    runtime_packages = {
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "pandas": str(pd.__version__),
        **{
            name: importlib.metadata.version(name)
            for name in (
                "scikit-learn",
                "einops",
                "linear-attention-transformer",
                "timm",
            )
        },
    }
    signature_payload = {
        "experiment": "tfm_v4_full_mtp_encoder_finetune",
        "config": config.to_dict(),
        "dataset_fingerprint": cache_report["dataset_fingerprint"],
        "encoder_checkpoint_sha256": checkpoint_sha256,
        "official_source_revision": source_revision,
        "runtime_packages": runtime_packages,
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode()
    ).hexdigest()
    signature_path = output_dir / "run_signature.json"
    if signature_path.exists():
        previous = json.loads(signature_path.read_text())
        if previous.get("signature") != signature:
            raise ValueError(
                "existing V4 results use different data/source/checkpoint/config; "
                "choose a new results directory"
            )
    else:
        _atomic_json({"signature": signature, **signature_payload}, signature_path)

    metric_columns = [
        "setup",
        "seed",
        "fold",
        "best_epoch",
        "validation_macro_f1",
        "encoder_update_l2",
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
    metrics = (
        _read_partial(metrics_path, metric_columns)
        if resume
        else pd.DataFrame(columns=metric_columns)
    )
    predictions = (
        _read_partial(predictions_path, prediction_columns)
        if resume
        else pd.DataFrame(columns=prediction_columns)
    )
    history = (
        _read_partial(history_path, history_columns)
        if resume
        else pd.DataFrame(columns=history_columns)
    )
    completed = {
        (str(row.setup), int(row.seed), int(row.fold))
        for row in metrics.itertuples(index=False)
    }
    groups, sentence_ids, y = _sentence_groups(records)
    encoder_report = None
    for run_seed in config.seeds:
        outer = StratifiedKFold(
            n_splits=config.n_splits,
            shuffle=True,
            random_state=run_seed,
        )
        for fold, (train_position, test_position) in enumerate(
            outer.split(sentence_ids, y)
        ):
            train_ids = sentence_ids[train_position]
            train_labels = y[train_position]
            test_ids = sentence_ids[test_position]
            test_labels = y[test_position]
            for setup, shuffled in (
                ("encoder_finetune", False),
                ("encoder_finetune_shuffled", True),
            ):
                key = (setup, int(run_seed), int(fold))
                saved_predictions = predictions[
                    (predictions["setup"] == setup)
                    & (predictions["seed"].astype(int) == int(run_seed))
                    & (predictions["fold"].astype(int) == int(fold))
                ]
                saved_history = history[
                    (history["setup"] == setup)
                    & (history["seed"].astype(int) == int(run_seed))
                    & (history["fold"].astype(int) == int(fold))
                ]
                if (
                    key in completed
                    and len(saved_predictions) == len(test_ids)
                    and len(saved_history) >= config.minimum_epochs
                ):
                    print(f"Reusing completed {setup}, seed={run_seed}, fold={fold}")
                    continue
                if key in completed:
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
                print(f"Training {setup}, seed={run_seed}, fold={fold}", flush=True)
                if status_callback is not None:
                    status_callback(
                        "fit_started",
                        setup=setup,
                        seed=int(run_seed),
                        fold=int(fold),
                    )

                def fit_status(stage, **details):
                    if status_callback is not None:
                        status_callback(
                            stage,
                            setup=setup,
                            seed=int(run_seed),
                            fold=int(fold),
                            **details,
                        )

                (
                    model,
                    fit_history,
                    best_epoch,
                    best_score,
                    encoder_update_l2,
                ) = _train_one_model(
                    groups,
                    train_ids,
                    train_labels,
                    repo_dir,
                    checkpoint_path,
                    config,
                    resolved_device,
                    seed=int(run_seed * 100 + fold),
                    shuffled=shuffled,
                    status_callback=fit_status,
                )
                if encoder_report is None:
                    encoder_report = dict(model.report)
                    encoder_report["official_source_revision"] = source_revision
                    _atomic_json(encoder_report, output_dir / "encoder_report.json")
                test_examples = _examples_for_split(
                    groups,
                    test_ids,
                    shuffled=shuffled,
                    rng=np.random.default_rng(run_seed * 1000 + fold + 50_000),
                )
                predicted_ids, source_ids, truth, probabilities = _predict_examples(
                    model,
                    test_examples,
                    resolved_device,
                    config.evaluation_batch_size,
                )
                if not np.array_equal(predicted_ids, np.sort(test_ids)):
                    raise ValueError("V4 test prediction IDs do not match the outer fold")
                if not np.array_equal(truth, test_labels[np.argsort(test_ids)]):
                    raise ValueError("V4 test prediction labels do not match the outer fold")
                if shuffled:
                    if np.any(source_ids == predicted_ids) or set(source_ids) != set(
                        predicted_ids
                    ):
                        raise ValueError("V4 shuffled test mapping is not a derangement")
                elif not np.array_equal(source_ids, predicted_ids):
                    raise ValueError("V4 aligned test mapping changed sentence identity")
                predicted = np.asarray(
                    [LABELS[index] for index in probabilities.argmax(axis=1)],
                    dtype=np.int64,
                )
                metric_row = {
                    "setup": setup,
                    "seed": int(run_seed),
                    "fold": int(fold),
                    "best_epoch": int(best_epoch),
                    "validation_macro_f1": float(best_score),
                    "encoder_update_l2": float(encoder_update_l2),
                    **_metric_values(truth, predicted),
                }
                metrics = pd.concat(
                    [metrics, pd.DataFrame([metric_row])],
                    ignore_index=True,
                )
                prediction_rows = []
                for sentence_id, source_id, label, prediction, probability in zip(
                    predicted_ids,
                    source_ids,
                    truth,
                    predicted,
                    probabilities,
                ):
                    prediction_rows.append(
                        {
                            "setup": setup,
                            "seed": int(run_seed),
                            "fold": int(fold),
                            "sentence_id": int(sentence_id),
                            "feature_sentence_id": int(source_id),
                            "label": int(label),
                            "prediction": int(prediction),
                            "probability_-1": float(probability[0]),
                            "probability_0": float(probability[1]),
                            "probability_1": float(probability[2]),
                        }
                    )
                predictions = pd.concat(
                    [predictions, pd.DataFrame(prediction_rows)],
                    ignore_index=True,
                )
                history_rows = [
                    {
                        "setup": setup,
                        "seed": int(run_seed),
                        "fold": int(fold),
                        **row,
                    }
                    for row in fit_history
                ]
                history = pd.concat(
                    [history, pd.DataFrame(history_rows)],
                    ignore_index=True,
                )
                metrics = metrics.sort_values(["seed", "fold", "setup"]).reset_index(
                    drop=True
                )
                predictions = predictions.sort_values(
                    ["seed", "fold", "setup", "sentence_id"]
                ).reset_index(drop=True)
                history = history.sort_values(
                    ["seed", "fold", "setup", "epoch"]
                ).reset_index(drop=True)
                _atomic_csv(metrics, metrics_path)
                _atomic_csv(predictions, predictions_path)
                _atomic_csv(history, history_path)
                completed.add(key)
                if status_callback is not None:
                    status_callback(
                        "fit_complete",
                        setup=setup,
                        seed=int(run_seed),
                        fold=int(fold),
                        macro_f1=float(metric_row["macro_f1"]),
                    )
                del model
                torch.cuda.empty_cache()

            majority_key = ("majority", int(run_seed), int(fold))
            saved_majority = predictions[
                (predictions["setup"] == "majority")
                & (predictions["seed"].astype(int) == int(run_seed))
                & (predictions["fold"].astype(int) == int(fold))
            ]
            if majority_key not in completed or len(saved_majority) != len(test_ids):
                metrics = metrics[
                    ~(
                        (metrics["setup"] == "majority")
                        & (metrics["seed"].astype(int) == int(run_seed))
                        & (metrics["fold"].astype(int) == int(fold))
                    )
                ].reset_index(drop=True)
                predictions = predictions[
                    ~(
                        (predictions["setup"] == "majority")
                        & (predictions["seed"].astype(int) == int(run_seed))
                        & (predictions["fold"].astype(int) == int(fold))
                    )
                ].reset_index(drop=True)
                dummy = DummyClassifier(strategy="prior").fit(
                    np.zeros((len(train_labels), 1)),
                    train_labels,
                )
                if not np.array_equal(dummy.classes_, np.asarray(LABELS)):
                    raise ValueError(f"unexpected dummy classes {dummy.classes_}")
                majority_prediction = dummy.predict(
                    np.zeros((len(test_labels), 1))
                ).astype(np.int64)
                majority_probabilities = dummy.predict_proba(
                    np.zeros((len(test_labels), 1))
                )
                metric_row = {
                    "setup": "majority",
                    "seed": int(run_seed),
                    "fold": int(fold),
                    "best_epoch": np.nan,
                    "validation_macro_f1": np.nan,
                    "encoder_update_l2": np.nan,
                    **_metric_values(test_labels, majority_prediction),
                }
                metrics = pd.concat(
                    [metrics, pd.DataFrame([metric_row])], ignore_index=True
                )
                majority_rows = [
                    {
                        "setup": "majority",
                        "seed": int(run_seed),
                        "fold": int(fold),
                        "sentence_id": int(sentence_id),
                        "feature_sentence_id": int(sentence_id),
                        "label": int(label),
                        "prediction": int(prediction),
                        "probability_-1": float(probability[0]),
                        "probability_0": float(probability[1]),
                        "probability_1": float(probability[2]),
                    }
                    for sentence_id, label, prediction, probability in zip(
                        test_ids,
                        test_labels,
                        majority_prediction,
                        majority_probabilities,
                    )
                ]
                predictions = pd.concat(
                    [predictions, pd.DataFrame(majority_rows)], ignore_index=True
                )
                metrics = metrics.sort_values(["seed", "fold", "setup"]).reset_index(
                    drop=True
                )
                predictions = predictions.sort_values(
                    ["seed", "fold", "setup", "sentence_id"]
                ).reset_index(drop=True)
                _atomic_csv(metrics, metrics_path)
                _atomic_csv(predictions, predictions_path)
                completed.add(majority_key)

    expected_metric_rows = len(config.seeds) * config.n_splits * 3
    if len(metrics) != expected_metric_rows:
        raise RuntimeError(
            f"expected {expected_metric_rows} completed V4 metric rows, got {len(metrics)}"
        )
    expected_prediction_rows = len(config.seeds) * len(sentence_ids) * 3
    if len(predictions) != expected_prediction_rows:
        raise RuntimeError(
            "expected "
            f"{expected_prediction_rows} completed V4 predictions, got {len(predictions)}"
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
    _atomic_csv(summary, output_dir / "summary.csv")
    _atomic_json(config.to_dict(), output_dir / "evaluation_config.json")
    _atomic_json(cache_report, output_dir / "token_cache_report.json")
    _atomic_json(delta, output_dir / "alignment_delta.json")
    _atomic_json(gate, output_dir / "viability_gate.json")
    return metrics, predictions, history, summary, delta, gate
