"""Sentence-level evaluation for frozen NeuroLM-B EEG features."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import EvaluationConfig


ALIGNED = "neurolm_frozen_probe"
SHUFFLED = "neurolm_frozen_probe_shuffled"


def _metrics(y_true, y_pred, labels):
    per_class = f1_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        ),
    }
    result.update({f"f1_class_{label}": score for label, score in zip(labels, per_class)})
    return result


def _fit_probe(X, y, config, seed):
    estimator = Pipeline(
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
        estimator,
        {"classifier__C": config.c_values},
        scoring="f1_macro",
        cv=inner,
        n_jobs=-1,
        refit=True,
    )
    return search.fit(X, y)


def evaluate_features(X, y, sentence_ids, output_dir, config=EvaluationConfig()):
    """Nested CV for aligned, split-local shuffled, and majority setups."""

    X = np.asarray(X)
    y = np.asarray(y)
    sentence_ids = np.asarray(sentence_ids)
    if X.ndim != 2 or not (len(X) == len(y) == len(sentence_ids)):
        raise ValueError("X must be 2D and X, y, sentence_ids must have equal length")
    if len(np.unique(sentence_ids)) != len(sentence_ids):
        raise ValueError("sentence_ids must be unique after reader aggregation")
    labels = np.sort(np.unique(y))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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
                predictions = fitted.predict(X_test)
                metric_rows.append(
                    {
                        "setup": setup,
                        "seed": seed,
                        "fold": fold,
                        "best_C": fitted.best_params_["classifier__C"],
                        **_metrics(y[test_index], predictions, labels),
                    }
                )
                for position, prediction in zip(test_index, predictions):
                    prediction_rows.append(
                        {
                            "setup": setup,
                            "seed": seed,
                            "fold": fold,
                            "sentence_id": int(sentence_ids[position]),
                            "label": int(y[position]),
                            "prediction": int(prediction),
                        }
                    )

            dummy = DummyClassifier(strategy="prior").fit(X[train_index], y[train_index])
            predictions = dummy.predict(X[test_index])
            metric_rows.append(
                {
                    "setup": "majority",
                    "seed": seed,
                    "fold": fold,
                    "best_C": np.nan,
                    **_metrics(y[test_index], predictions, labels),
                }
            )
            for position, prediction in zip(test_index, predictions):
                prediction_rows.append(
                    {
                        "setup": "majority",
                        "seed": seed,
                        "fold": fold,
                        "sentence_id": int(sentence_ids[position]),
                        "label": int(y[position]),
                        "prediction": int(prediction),
                    }
                )

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    summary = (
        metrics.groupby("setup")[["accuracy", "balanced_accuracy", "macro_f1"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    predictions.to_csv(output_dir / "oof_predictions.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "evaluation_config.json").write_text(
        json.dumps(config.to_dict(), indent=2)
    )
    return metrics, predictions, summary


def _paired_seed_arrays(predictions):
    by_seed = []
    for run_seed in sorted(predictions["seed"].unique()):
        subset = predictions[predictions["seed"] == run_seed]
        aligned = subset[subset["setup"] == ALIGNED].sort_values("sentence_id")
        shuffled = subset[subset["setup"] == SHUFFLED].sort_values("sentence_id")
        if not np.array_equal(aligned["sentence_id"].values, shuffled["sentence_id"].values):
            raise ValueError("aligned and shuffled predictions are not paired")
        if not np.array_equal(aligned["label"].values, shuffled["label"].values):
            raise ValueError("aligned and shuffled labels differ")
        by_seed.append(
            (
                int(run_seed),
                aligned["label"].to_numpy(),
                aligned["prediction"].to_numpy(),
                shuffled["prediction"].to_numpy(),
            )
        )
    if not by_seed:
        raise ValueError("predictions contain no paired NeuroLM setups")
    return by_seed


def bootstrap_alignment_delta(
    predictions, samples=2000, seed=2026, confidence=0.9833
):
    """Paired sentence bootstrap of aligned-minus-shuffled macro-F1."""

    if not 0 < confidence < 1:
        raise ValueError("confidence must lie inside (0, 1)")
    rng = np.random.default_rng(seed)
    by_seed = _paired_seed_arrays(predictions)
    all_labels = np.sort(predictions["label"].unique())
    seed_deltas = {}
    for run_seed, truth, aligned, shuffled in by_seed:
        seed_deltas[str(run_seed)] = float(
            f1_score(truth, aligned, labels=all_labels, average="macro", zero_division=0)
            - f1_score(truth, shuffled, labels=all_labels, average="macro", zero_division=0)
        )
    draws = []
    for _ in range(samples):
        deltas = []
        for _, truth, aligned, shuffled in by_seed:
            indices = rng.integers(0, len(truth), size=len(truth))
            deltas.append(
                f1_score(
                    truth[indices], aligned[indices], labels=all_labels,
                    average="macro", zero_division=0
                )
                - f1_score(
                    truth[indices], shuffled[indices], labels=all_labels,
                    average="macro", zero_division=0
                )
            )
        draws.append(float(np.mean(deltas)))
    alpha = 1.0 - confidence
    return {
        "observed_mean_seed_delta": float(np.mean(list(seed_deltas.values()))),
        "bootstrap_mean_delta": float(np.mean(draws)),
        "ci_level": confidence,
        "ci_low": float(np.quantile(draws, alpha / 2)),
        "ci_high": float(np.quantile(draws, 1 - alpha / 2)),
        "seed_deltas": seed_deltas,
        "bootstrap_samples": int(samples),
    }


def viability_gate(metrics, delta, config=EvaluationConfig()):
    aligned = float(metrics.loc[metrics["setup"] == ALIGNED, "macro_f1"].mean())
    shuffled = float(metrics.loc[metrics["setup"] == SHUFFLED, "macro_f1"].mean())
    positive_seeds = sum(value > 0 for value in delta["seed_deltas"].values())
    passes = (
        aligned - shuffled >= config.minimum_delta
        and positive_seeds >= config.minimum_positive_seeds
        and delta["ci_low"] > 0
    )
    return {
        "aligned_macro_f1": aligned,
        "shuffled_macro_f1": shuffled,
        "observed_fold_mean_delta": aligned - shuffled,
        "minimum_required_delta": config.minimum_delta,
        "positive_seeds": int(positive_seeds),
        "minimum_positive_seeds": config.minimum_positive_seeds,
        "bootstrap": delta,
        "passes": bool(passes),
        "decision": (
            "PASS — consider the predeclared structured sequence probe"
            if passes
            else "STOP — frozen NeuroLM transfer is not alignment-specific"
        ),
    }
