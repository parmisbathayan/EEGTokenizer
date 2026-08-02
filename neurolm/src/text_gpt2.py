"""Frozen GPT-2 text-only reference for ZuCo sentiment labels."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .labels import load_labels


MODEL_NAME = "openai-community/gpt2"
MODEL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
ALIGNED = "gpt2_text_probe"
SHUFFLED = "gpt2_text_probe_shuffled"
IMPLEMENTATION_VERSION = "gpt2-text-reference-v1.0"


@dataclass(frozen=True)
class TextGPT2Config:
    """Locked extraction and nested-CV settings for the text reference."""

    seeds: tuple = (42, 52, 62)
    n_splits: int = 5
    inner_splits: int = 3
    c_values: tuple = (0.001, 0.01, 0.1, 1.0, 10.0)
    max_length: int = 128
    batch_size: int = 32
    bootstrap_samples: int = 5000
    bootstrap_confidence: float = 0.95

    def to_dict(self):
        return {**asdict(self), "implementation_version": IMPLEMENTATION_VERSION}


def sentence_fingerprint(table):
    """Hash the semantic text/label table rather than its filesystem metadata."""

    required = ["sentence_id", "sentence", "sentiment_label"]
    missing = set(required) - set(table.columns)
    if missing:
        raise ValueError(f"text table is missing columns: {sorted(missing)}")
    rows = [
        {
            "sentence_id": int(row.sentence_id),
            "sentence": str(row.sentence),
            "sentiment_label": int(row.sentiment_label),
        }
        for row in table[required].sort_values("sentence_id").itertuples(index=False)
    ]
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_table(table):
    table = table[["sentence_id", "sentence", "sentiment_label"]].copy()
    table["sentence_id"] = table["sentence_id"].astype(int)
    table["sentiment_label"] = table["sentiment_label"].astype(int)
    table["sentence"] = table["sentence"].astype(str)
    if table["sentence_id"].duplicated().any():
        raise ValueError("sentence_id must be unique for the text-only reference")
    if table["sentence"].str.strip().eq("").any():
        raise ValueError("sentence text must be non-empty")
    observed_labels = set(table["sentiment_label"].unique())
    if observed_labels != {-1, 0, 1}:
        raise ValueError(
            f"expected ZuCo sentiment labels -1, 0, 1; got {sorted(observed_labels)}"
        )
    return table.sort_values("sentence_id").reset_index(drop=True)


def load_text_encoder(cache_dir, device="cuda"):
    """Load the pinned standard 124M-parameter GPT-2 encoder."""

    try:
        from transformers import GPT2Model, GPT2TokenizerFast
    except ImportError as error:
        raise ImportError("GPT-2 extraction requires transformers; use the Colab notebook") from error

    tokenizer = GPT2TokenizerFast.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        cache_dir=str(cache_dir),
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2Model.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        cache_dir=str(cache_dir),
        use_safetensors=True,
    )
    model.eval().requires_grad_(False).to(device)
    return tokenizer, model


def encode_text_table(table, tokenizer, model, device="cuda", config=TextGPT2Config()):
    """Encode each unique sentence as its final non-padding GPT-2 hidden state."""

    import torch

    table = _validate_table(table)
    texts = table["sentence"].tolist()
    raw = tokenizer(texts, add_special_tokens=True, truncation=False)
    token_lengths = np.asarray([len(row) for row in raw["input_ids"]], dtype=np.int64)
    if np.any(token_lengths == 0):
        raise ValueError("GPT-2 produced an empty token sequence")
    features = []
    autocast_enabled = str(device).startswith("cuda") and torch.cuda.is_available()
    dtype = torch.bfloat16 if autocast_enabled and torch.cuda.is_bf16_supported() else torch.float16
    for start in range(0, len(texts), config.batch_size):
        batch = tokenizer(
            texts[start : start + config.batch_size],
            padding=True,
            truncation=True,
            max_length=config.max_length,
            return_tensors="pt",
        )
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.inference_mode(), torch.autocast(
            device_type="cuda" if autocast_enabled else "cpu",
            dtype=dtype,
            enabled=autocast_enabled,
        ):
            output = model(**batch, use_cache=False)
        last_positions = batch["attention_mask"].sum(dim=1) - 1
        row_indices = torch.arange(len(last_positions), device=last_positions.device)
        hidden = output.last_hidden_state[row_indices, last_positions]
        features.append(hidden.detach().float().cpu().numpy())

    X = np.concatenate(features, axis=0).astype(np.float32, copy=False)
    if X.shape != (len(table), int(model.config.n_embd)):
        raise ValueError(f"unexpected GPT-2 feature shape {X.shape}")
    if not np.isfinite(X).all():
        raise ValueError("GPT-2 text features contain non-finite values")
    report = {
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "pooling": "final_non_padding_token_last_hidden_state",
        "sentences": int(len(table)),
        "embedding_size": int(X.shape[1]),
        "max_length": int(config.max_length),
        "maximum_observed_tokens": int(token_lengths.max()),
        "truncated_sentences": int(np.sum(token_lengths > config.max_length)),
        "dataset_fingerprint": sentence_fingerprint(table),
    }
    return (
        X,
        table["sentiment_label"].to_numpy(dtype=np.int64),
        table["sentence_id"].to_numpy(dtype=np.int64),
        report,
    )


def save_feature_cache(path, X, y, sentence_ids, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            X=np.asarray(X, dtype=np.float32),
            y=np.asarray(y, dtype=np.int64),
            sentence_ids=np.asarray(sentence_ids, dtype=np.int64),
            report_json=np.asarray(json.dumps(report, sort_keys=True)),
        )
    temporary.replace(path)


def load_feature_cache(path, expected_fingerprint=None, max_length=None):
    with np.load(path, allow_pickle=False) as cached:
        X = cached["X"].astype(np.float32)
        y = cached["y"].astype(np.int64)
        sentence_ids = cached["sentence_ids"].astype(np.int64)
        report = json.loads(str(cached["report_json"].item()))
    if expected_fingerprint and report.get("dataset_fingerprint") != expected_fingerprint:
        raise ValueError("cached GPT-2 features were built from a different text/label table")
    if report.get("model_name") != MODEL_NAME or report.get("model_revision") != MODEL_REVISION:
        raise ValueError("cached text features were built with a different GPT-2 checkpoint")
    if report.get("pooling") != "final_non_padding_token_last_hidden_state":
        raise ValueError("cached text features used a different pooling rule")
    if max_length is not None and int(report.get("max_length", -1)) != int(max_length):
        raise ValueError("cached GPT-2 features used a different maximum token length")
    if not (len(X) == len(y) == len(sentence_ids)):
        raise ValueError("cached GPT-2 feature arrays have inconsistent lengths")
    return X, y, sentence_ids, report


def _metric_values(y_true, predictions, labels):
    per_class = f1_score(
        y_true, predictions, labels=labels, average=None, zero_division=0
    )
    values = {
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
        "macro_f1": f1_score(
            y_true, predictions, labels=labels, average="macro", zero_division=0
        ),
    }
    values.update(
        {f"f1_class_{int(label)}": float(score) for label, score in zip(labels, per_class)}
    )
    return values


def _fit_probe(X, y, config, seed):
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=4000,
                    random_state=seed,
                    solver="liblinear",
                ),
            ),
        ]
    )
    inner = StratifiedKFold(
        n_splits=config.inner_splits, shuffle=True, random_state=seed + 10_000
    )
    search = GridSearchCV(
        pipeline,
        {"classifier__C": config.c_values},
        scoring="f1_macro",
        cv=inner,
        n_jobs=-1,
        refit=True,
    )
    return search.fit(X, y)


def _bootstrap_delta(predictions, config, seed=2026):
    labels = np.sort(predictions["label"].unique())
    rng = np.random.default_rng(seed)
    paired = []
    seed_deltas = {}
    for run_seed in sorted(predictions["seed"].unique()):
        subset = predictions[predictions["seed"] == run_seed]
        aligned = subset[subset["setup"] == ALIGNED].sort_values("sentence_id")
        shuffled = subset[subset["setup"] == SHUFFLED].sort_values("sentence_id")
        if not np.array_equal(aligned["sentence_id"].values, shuffled["sentence_id"].values):
            raise ValueError("text and shuffled predictions are not paired")
        if not np.array_equal(aligned["label"].values, shuffled["label"].values):
            raise ValueError("text and shuffled predictions have different truth labels")
        truth = aligned["label"].to_numpy()
        text_prediction = aligned["prediction"].to_numpy()
        shuffled_prediction = shuffled["prediction"].to_numpy()
        delta = f1_score(
            truth, text_prediction, labels=labels, average="macro", zero_division=0
        ) - f1_score(
            truth, shuffled_prediction, labels=labels, average="macro", zero_division=0
        )
        seed_deltas[str(int(run_seed))] = float(delta)
        paired.append((truth, text_prediction, shuffled_prediction))

    draws = []
    for _ in range(config.bootstrap_samples):
        values = []
        for truth, text_prediction, shuffled_prediction in paired:
            indices = rng.integers(0, len(truth), size=len(truth))
            values.append(
                f1_score(
                    truth[indices],
                    text_prediction[indices],
                    labels=labels,
                    average="macro",
                    zero_division=0,
                )
                - f1_score(
                    truth[indices],
                    shuffled_prediction[indices],
                    labels=labels,
                    average="macro",
                    zero_division=0,
                )
            )
        draws.append(float(np.mean(values)))
    alpha = 1.0 - config.bootstrap_confidence
    return {
        "comparison": f"{ALIGNED}_minus_{SHUFFLED}",
        "observed_mean_seed_delta": float(np.mean(list(seed_deltas.values()))),
        "ci_level": float(config.bootstrap_confidence),
        "ci_low": float(np.quantile(draws, alpha / 2)),
        "ci_high": float(np.quantile(draws, 1 - alpha / 2)),
        "seed_deltas": seed_deltas,
        "bootstrap_samples": int(config.bootstrap_samples),
    }


def evaluate_text_features(
    X,
    y,
    sentence_ids,
    output_dir,
    extraction_report,
    config=TextGPT2Config(),
):
    """Evaluate frozen GPT-2 features with nested sentence-level CV."""

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    sentence_ids = np.asarray(sentence_ids, dtype=np.int64)
    if X.ndim != 2 or not (len(X) == len(y) == len(sentence_ids)):
        raise ValueError("X must be 2D and all input arrays must have equal length")
    if len(np.unique(sentence_ids)) != len(sentence_ids):
        raise ValueError("text-only evaluation requires one row per unique sentence")
    labels = np.sort(np.unique(y))
    metric_rows = []
    prediction_rows = []

    for seed in config.seeds:
        outer = StratifiedKFold(config.n_splits, shuffle=True, random_state=seed)
        for fold, (train_index, test_index) in enumerate(outer.split(X, y)):
            rng = np.random.default_rng(seed * 100 + fold)
            setups = {
                ALIGNED: (X[train_index], X[test_index]),
                SHUFFLED: (
                    X[train_index][rng.permutation(len(train_index))],
                    X[test_index][rng.permutation(len(test_index))],
                ),
            }
            for setup, (X_train, X_test) in setups.items():
                fitted = _fit_probe(X_train, y[train_index], config, seed + fold)
                prediction = fitted.predict(X_test)
                metric_rows.append(
                    {
                        "setup": setup,
                        "seed": int(seed),
                        "fold": int(fold),
                        "best_C": float(fitted.best_params_["classifier__C"]),
                        **_metric_values(y[test_index], prediction, labels),
                    }
                )
                for position, value in zip(test_index, prediction):
                    prediction_rows.append(
                        {
                            "setup": setup,
                            "seed": int(seed),
                            "fold": int(fold),
                            "sentence_id": int(sentence_ids[position]),
                            "label": int(y[position]),
                            "prediction": int(value),
                        }
                    )

            dummy = DummyClassifier(strategy="prior").fit(X[train_index], y[train_index])
            prediction = dummy.predict(X[test_index])
            metric_rows.append(
                {
                    "setup": "majority",
                    "seed": int(seed),
                    "fold": int(fold),
                    "best_C": np.nan,
                    **_metric_values(y[test_index], prediction, labels),
                }
            )
            for position, value in zip(test_index, prediction):
                prediction_rows.append(
                    {
                        "setup": "majority",
                        "seed": int(seed),
                        "fold": int(fold),
                        "sentence_id": int(sentence_ids[position]),
                        "label": int(y[position]),
                        "prediction": int(value),
                    }
                )

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    summary = metrics.groupby("setup").agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_std=("balanced_accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
    ).reset_index()
    delta = _bootstrap_delta(predictions, config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    predictions.to_csv(output_dir / "oof_predictions.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "evaluation_config.json").write_text(
        json.dumps(config.to_dict(), indent=2)
    )
    (output_dir / "extraction_report.json").write_text(
        json.dumps(extraction_report, indent=2)
    )
    (output_dir / "text_vs_shuffled_bootstrap.json").write_text(
        json.dumps(delta, indent=2)
    )
    return metrics, predictions, summary, delta


def run_text_reference(
    labels_csv,
    feature_cache,
    model_cache_dir,
    output_dir,
    device="cuda",
    config=TextGPT2Config(),
    force_extract=False,
):
    """Extract or reuse frozen features, then run the complete reference."""

    table = _validate_table(load_labels(labels_csv))
    fingerprint = sentence_fingerprint(table)
    feature_cache = Path(feature_cache)
    if feature_cache.exists() and not force_extract:
        X, y, sentence_ids, report = load_feature_cache(
            feature_cache,
            expected_fingerprint=fingerprint,
            max_length=config.max_length,
        )
    else:
        tokenizer, model = load_text_encoder(model_cache_dir, device=device)
        X, y, sentence_ids, report = encode_text_table(
            table, tokenizer, model, device=device, config=config
        )
        save_feature_cache(feature_cache, X, y, sentence_ids, report)
    return evaluate_text_features(
        X, y, sentence_ids, output_dir, report, config=config
    )
