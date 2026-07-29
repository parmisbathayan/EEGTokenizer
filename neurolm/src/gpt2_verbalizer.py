"""Small GPT-2 verbalizer adapter and bounded V4 evaluation."""

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import copy
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd

from .raw_eegnet import (
    INDEX_TO_LABEL,
    LABELS,
    LABEL_TO_INDEX,
    _atomic_csv,
    _drop_fold_rows,
    _fold_complete,
    _metric_values,
    _prediction_rows,
    _read_csv,
    _write_json,
    make_bundle_examples,
    sentence_table,
)


ALIGNED = "neurolm_gpt2_verbalizer"
SHUFFLED = "neurolm_gpt2_verbalizer_shuffled"
IMPLEMENTATION_VERSION = "neurolm-gpt2-verbalizer-v4.0"


@dataclass(frozen=True)
class GPT2VerbalizerConfig:
    """One locked low-capacity adapter and evaluation recipe for V4."""

    seeds: tuple = (42, 52, 62)
    n_splits: int = 5
    validation_fraction: float = 0.15
    embedding_size: int = 768
    adapter_size: int = 32
    dropout: float = 0.25
    batch_size: int = 64
    max_epochs: int = 15
    patience: int = 3
    learning_rate: float = 5e-4
    weight_decay: float = 1e-2
    gradient_clip: float = 1.0
    bootstrap_samples: int = 5000
    planned_versions: int = 3
    familywise_alpha: float = 0.05
    minimum_delta: float = 0.015
    minimum_positive_seeds: int = 2

    @property
    def corrected_ci(self):
        return 1.0 - self.familywise_alpha / self.planned_versions

    def to_dict(self):
        values = asdict(self)
        values["corrected_ci"] = self.corrected_ci
        values["implementation_version"] = IMPLEMENTATION_VERSION
        return values


def _require_torch():
    try:
        import torch
    except ImportError as error:
        raise ImportError("V4 training requires PyTorch; use the Colab notebook") from error
    return torch


def build_gpt2_verbalizer(verbalizer_vectors, config=GPT2VerbalizerConfig()):
    """Build a residual bottleneck scored by fixed GPT-2 label embeddings."""

    torch = _require_torch()
    nn = torch.nn
    vectors = np.asarray(verbalizer_vectors, dtype=np.float32)
    if vectors.shape != (len(LABELS), config.embedding_size):
        raise ValueError(f"expected verbalizer vectors (3, {config.embedding_size}), got {vectors.shape}")

    class GPT2VerbalizerAdapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.down = nn.Linear(config.embedding_size, config.adapter_size, bias=False)
            self.up = nn.Linear(config.adapter_size, config.embedding_size, bias=False)
            self.dropout = nn.Dropout(config.dropout)
            self.bias = nn.Parameter(torch.zeros(len(LABELS)))
            self.register_buffer(
                "verbalizer_vectors", torch.as_tensor(vectors, dtype=torch.float32)
            )
            nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.up.weight)

        def forward(self, hidden):
            if hidden.ndim != 2 or hidden.shape[1] != config.embedding_size:
                raise ValueError(
                    f"expected batch x {config.embedding_size} GPT2 states, got {tuple(hidden.shape)}"
                )
            adapted = hidden + self.up(
                self.dropout(torch.nn.functional.gelu(self.down(hidden)))
            )
            return adapted @ self.verbalizer_vectors.T + self.bias

    return GPT2VerbalizerAdapter()


class GPT2Collator:
    def __init__(self, config=GPT2VerbalizerConfig()):
        self.config = config

    def __call__(self, batch):
        torch = _require_torch()
        hidden = np.stack(
            [np.asarray(example.record.hidden, dtype=np.float32) for example in batch]
        )
        if hidden.shape[1:] != (self.config.embedding_size,):
            raise ValueError(f"invalid V4 hidden-state shape {hidden.shape}")
        return {
            "hidden": torch.from_numpy(hidden),
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


def _loader(examples, config, shuffle, seed):
    torch = _require_torch()
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return torch.utils.data.DataLoader(
        _ExampleDataset(examples),
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=GPT2Collator(config),
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


def _aggregate_predictions(model, loader, device):
    torch = _require_torch()
    probability_sums = defaultdict(lambda: np.zeros(len(LABELS), dtype=np.float64))
    reader_counts = Counter()
    labels = {}
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            logits = model(batch["hidden"].to(device, non_blocking=True))
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            batch_labels = batch["labels"].numpy()
            for sentence_id, label_index, probability in zip(
                batch["sentence_ids"], batch_labels, probabilities
            ):
                sentence_id = int(sentence_id)
                label = INDEX_TO_LABEL[int(label_index)]
                if sentence_id in labels and labels[sentence_id] != label:
                    raise ValueError(f"conflicting label for sentence {sentence_id}")
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
    verbalizer_vectors,
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
    train_examples = make_bundle_examples(
        records,
        outer_train_sentence_ids[train_position],
        outer_train_labels[train_position],
        shuffled=shuffled,
        seed=seed + 30_000,
    )
    validation_examples = make_bundle_examples(
        records,
        outer_train_sentence_ids[validation_position],
        outer_train_labels[validation_position],
        shuffled=shuffled,
        seed=seed + 40_000,
    )
    train_loader = _loader(train_examples, config, True, seed)
    validation_loader = _loader(validation_examples, config, False, seed)
    _set_seed(seed)
    model = build_gpt2_verbalizer(verbalizer_vectors, config).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=config.learning_rate, weight_decay=config.weight_decay
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
            logits = model(batch["hidden"].to(device, non_blocking=True))
            labels = batch["labels"].to(device, non_blocking=True)
            weights = batch["weights"].to(device, non_blocking=True)
            losses = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
            loss = (losses * weights).sum() / weights.sum().clamp_min(1e-8)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip)
            optimizer.step()
            weighted_loss += float((losses.detach() * weights).sum().cpu())
            weight_total += float(weights.sum().cpu())
        _, truth, probabilities = _aggregate_predictions(
            model, validation_loader, device
        )
        prediction = np.asarray(
            [LABELS[index] for index in probabilities.argmax(axis=1)], dtype=np.int64
        )
        score = f1_score(
            truth,
            prediction,
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
        raise RuntimeError("V4 training produced no finite validation result")
    model.load_state_dict(best_state)
    return model, history, best_epoch, best_score


def bootstrap_alignment_delta(predictions, config=GPT2VerbalizerConfig(), seed=2026):
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
            raise ValueError("aligned and shuffled V4 predictions are not paired")
        truth = aligned["label"].to_numpy(dtype=np.int64)
        aligned_prediction = aligned["prediction"].to_numpy(dtype=np.int64)
        shuffled_prediction = shuffled["prediction"].to_numpy(dtype=np.int64)
        seed_deltas[str(int(run_seed))] = float(
            f1_score(
                truth,
                aligned_prediction,
                labels=list(LABELS),
                average="macro",
                zero_division=0,
            )
            - f1_score(
                truth,
                shuffled_prediction,
                labels=list(LABELS),
                average="macro",
                zero_division=0,
            )
        )
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


def gate_report(metrics, delta, config=GPT2VerbalizerConfig()):
    aligned = metrics[metrics["setup"] == ALIGNED]
    shuffled = metrics[metrics["setup"] == SHUFFLED]
    majority = metrics[metrics["setup"] == "majority"]
    aligned_macro = float(aligned["macro_f1"].mean())
    shuffled_macro = float(shuffled["macro_f1"].mean())
    majority_macro = float(majority["macro_f1"].mean())
    observed_delta = aligned_macro - shuffled_macro
    positive_seeds = sum(value > 0 for value in delta["seed_deltas"].values())
    criteria = {
        "balanced_accuracy_above_chance": float(aligned["balanced_accuracy"].mean()) > 1 / 3,
        "macro_f1_above_majority": aligned_macro > majority_macro,
        "aligned_minus_shuffled_at_least_minimum": observed_delta >= config.minimum_delta,
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
        "green": "GREEN — eligible for a separately locked confirmation",
        "yellow": "YELLOW — suggestive only; record without tuning",
        "red": "RED — no alignment-specific V4 evidence; stop the broad screen",
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


def smoke_test_gpt2_verbalizer(
    records,
    verbalizer_vectors,
    config=GPT2VerbalizerConfig(),
    device="cuda",
):
    torch = _require_torch()
    sentence_ids, y = sentence_table(records)
    selected = sentence_ids[: min(4, len(sentence_ids))]
    examples = make_bundle_examples(records, selected, y[: len(selected)])
    batch = GPT2Collator(config)(examples[: min(4, len(examples))])
    resolved = torch.device(device if torch.cuda.is_available() else "cpu")
    model = build_gpt2_verbalizer(verbalizer_vectors, config).to(resolved).eval()
    with torch.inference_mode():
        logits = model(batch["hidden"].to(resolved))
    return {
        "device": str(resolved),
        "input": list(batch["hidden"].shape),
        "logits": list(logits.shape),
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "fixed_verbalizer_parameters": int(np.asarray(verbalizer_vectors).size),
    }


def evaluate_gpt2_verbalizer(
    records,
    verbalizer_vectors,
    output_dir,
    dataset_fingerprint,
    config=GPT2VerbalizerConfig(),
    device="cuda",
):
    """Run/resume aligned and split-local shuffled V4 evaluation."""

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
        raise RuntimeError("V4 evaluation requires a Colab GPU runtime")

    vector_hash = hashlib.sha256(
        np.asarray(verbalizer_vectors, dtype=np.float32).tobytes()
    ).hexdigest()
    signature_payload = {
        "implementation_version": IMPLEMENTATION_VERSION,
        "dataset_fingerprint": str(dataset_fingerprint),
        "verbalizer_vector_sha256": vector_hash,
        "config": config.to_dict(),
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode()
    ).hexdigest()
    signature_path = output_dir / "run_signature.json"
    if signature_path.exists():
        existing = json.loads(signature_path.read_text())
        if existing.get("signature") != signature:
            raise RuntimeError("V4 result directory belongs to another configuration")
    else:
        _write_json({"signature": signature, **signature_payload}, signature_path)

    metric_columns = [
        "setup", "seed", "fold", "best_epoch", "validation_macro_f1",
        "accuracy", "balanced_accuracy", "macro_f1", "f1_class_-1",
        "f1_class_0", "f1_class_1",
    ]
    prediction_columns = [
        "setup", "seed", "fold", "sentence_id", "label", "prediction",
        "probability_negative", "probability_neutral", "probability_positive",
    ]
    partial_metrics_path = output_dir / "partial_fold_metrics.csv"
    partial_predictions_path = output_dir / "partial_oof_predictions.csv"
    metrics = _read_csv(partial_metrics_path, metric_columns)
    predictions = _read_csv(partial_predictions_path, prediction_columns)
    sentence_ids, y = sentence_table(records)

    for seed in config.seeds:
        outer = StratifiedKFold(config.n_splits, shuffle=True, random_state=seed)
        for fold, (train_position, test_position) in enumerate(outer.split(sentence_ids, y)):
            train_ids, train_labels = sentence_ids[train_position], y[train_position]
            test_ids, test_labels = sentence_ids[test_position], y[test_position]
            for setup, shuffled in ((ALIGNED, False), (SHUFFLED, True)):
                marker = completion_dir / f"{setup}_seed{seed}_fold{fold}.json"
                if _fold_complete(
                    metrics, predictions, [setup], seed, fold, len(test_ids), marker
                ):
                    print(f"Reused {setup} seed={seed} fold={fold}", flush=True)
                    continue
                metrics = _drop_fold_rows(metrics, [setup], seed, fold)
                predictions = _drop_fold_rows(predictions, [setup], seed, fold)
                fit_seed = int(seed * 100 + fold * 10 + int(shuffled))
                print(f"Training {setup} seed={seed} fold={fold}", flush=True)
                model, history, best_epoch, best_score = _train_one_model(
                    records,
                    verbalizer_vectors,
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
                test_loader = _loader(test_examples, config, False, fit_seed + 60_000)
                evaluated_ids, truth, probabilities = _aggregate_predictions(
                    model, test_loader, resolved_device
                )
                rows, hard_predictions = _prediction_rows(
                    setup, seed, fold, evaluated_ids, truth, probabilities
                )
                metrics = pd.concat(
                    [
                        metrics,
                        pd.DataFrame(
                            [
                                {
                                    "setup": setup,
                                    "seed": seed,
                                    "fold": fold,
                                    "best_epoch": best_epoch,
                                    "validation_macro_f1": best_score,
                                    **_metric_values(truth, hard_predictions),
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
                    {"signature": signature, "setup": setup, "seed": seed, "fold": fold},
                    marker,
                )
                del model
                torch.cuda.empty_cache()

            majority_marker = completion_dir / f"majority_seed{seed}_fold{fold}.json"
            if not _fold_complete(
                metrics, predictions, ["majority"], seed, fold, len(test_ids), majority_marker
            ):
                metrics = _drop_fold_rows(metrics, ["majority"], seed, fold)
                predictions = _drop_fold_rows(predictions, ["majority"], seed, fold)
                majority_label = Counter(map(int, train_labels)).most_common(1)[0][0]
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
