"""Strict sentence-level evaluation for frozen TFM token histograms."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from .config import EvaluationConfig


def _metrics(y_true, y_pred):
    labels = np.unique(y_true)
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    result.update({f"f1_class_{label}": score for label, score in zip(labels, per_class)})
    return result


def _fit_histogram_classifier(X, y, config, seed):
    estimator = Pipeline(
        [
            ("tfidf", TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True)),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=3000,
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


def evaluate_histograms(X, y, sentence_ids, output_dir, config=EvaluationConfig()):
    """Evaluate aligned tokens, split-local shuffling, and a majority baseline."""

    X = np.asarray(X)
    y = np.asarray(y)
    sentence_ids = np.asarray(sentence_ids)
    if not (len(X) == len(y) == len(sentence_ids)):
        raise ValueError("X, y, and sentence_ids must have equal length")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    prediction_rows = []

    for seed in config.seeds:
        outer = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=seed)
        for fold, (train_index, test_index) in enumerate(outer.split(X, y)):
            rng = np.random.default_rng(seed * 100 + fold)
            setups = {
                "tfm_histogram": (X[train_index], X[test_index]),
                "tfm_histogram_shuffled": (
                    X[train_index][rng.permutation(len(train_index))],
                    X[test_index][rng.permutation(len(test_index))],
                ),
            }
            for setup, (X_train, X_test) in setups.items():
                fitted = _fit_histogram_classifier(
                    X_train, y[train_index], config=config, seed=seed + fold
                )
                predictions = fitted.predict(X_test)
                scores = _metrics(y[test_index], predictions)
                metric_rows.append(
                    {
                        "setup": setup,
                        "seed": seed,
                        "fold": fold,
                        "best_C": fitted.best_params_["classifier__C"],
                        **scores,
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
            scores = _metrics(y[test_index], predictions)
            metric_rows.append(
                {"setup": "majority", "seed": seed, "fold": fold, "best_C": np.nan, **scores}
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
    with (output_dir / "evaluation_config.json").open("w") as handle:
        json.dump(config.to_dict(), handle, indent=2)
    return metrics, predictions, summary


def bootstrap_alignment_delta(predictions, samples=2000, seed=2026):
    """Bootstrap the paired macro-F1 delta between aligned and shuffled EEG."""

    required = {"tfm_histogram", "tfm_histogram_shuffled"}
    if not required.issubset(set(predictions["setup"])):
        raise ValueError("predictions must contain aligned and shuffled setups")
    rng = np.random.default_rng(seed)
    by_seed = []
    for run_seed in sorted(predictions["seed"].unique()):
        subset = predictions[predictions["seed"] == run_seed]
        aligned = subset[subset["setup"] == "tfm_histogram"].sort_values("sentence_id")
        shuffled = subset[subset["setup"] == "tfm_histogram_shuffled"].sort_values(
            "sentence_id"
        )
        if not np.array_equal(aligned["sentence_id"].values, shuffled["sentence_id"].values):
            raise ValueError("aligned and shuffled predictions are not paired")
        by_seed.append(
            (
                aligned["label"].to_numpy(),
                aligned["prediction"].to_numpy(),
                shuffled["prediction"].to_numpy(),
            )
        )
    draws = []
    for _ in range(samples):
        seed_deltas = []
        for truth, aligned_prediction, shuffled_prediction in by_seed:
            indices = rng.integers(0, len(truth), size=len(truth))
            seed_deltas.append(
                f1_score(truth[indices], aligned_prediction[indices], average="macro", zero_division=0)
                - f1_score(
                    truth[indices], shuffled_prediction[indices], average="macro", zero_division=0
                )
            )
        draws.append(np.mean(seed_deltas))
    return {
        "mean_delta": float(np.mean(draws)),
        "ci_95_low": float(np.quantile(draws, 0.025)),
        "ci_95_high": float(np.quantile(draws, 0.975)),
        "bootstrap_samples": samples,
    }

