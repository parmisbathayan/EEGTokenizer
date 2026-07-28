"""Frozen official TFM-Encoder features and the bounded V3 linear probe."""

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from .token_map import LABELS, TokenRecord


@dataclass(frozen=True)
class EncoderProbeConfig:
    """Predeclared extraction, linear-probe, and gate settings for V3."""

    seeds: tuple = (42, 52, 62)
    n_splits: int = 5
    inner_splits: int = 3
    c_values: tuple = (0.01, 0.1, 1.0, 10.0)
    codebook_size: int = 8192
    embedding_size: int = 64
    expected_channels: int = 104
    channel_group_size: int = 16
    extraction_batch_size: int = 8
    official_max_sequence_length: int = 2048
    bootstrap_samples: int = 5000
    planned_versions: int = 3
    familywise_alpha: float = 0.05
    minimum_delta: float = 0.015
    minimum_positive_seeds: int = 2

    def to_dict(self):
        return asdict(self)


def _sha256_file(path, chunk_size=2**20):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _unwrap_encoder_checkpoint(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError("encoder checkpoint must contain a state dictionary")
    for key in ("state_dict", "model_state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            checkpoint = value
            break
    state = {}
    for name, value in checkpoint.items():
        clean = name
        changed = True
        while changed:
            changed = False
            for prefix in ("tfm_token.", "module."):
                if clean.startswith(prefix):
                    clean = clean[len(prefix) :]
                    changed = True
        state[clean] = value
    return state


class OfficialFrozenTFMEncoder:
    """Official MTP-pretrained encoder with its classification head bypassed."""

    def __init__(
        self,
        repo_dir,
        checkpoint_path,
        config=EncoderProbeConfig(),
        device="cuda",
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
        try:
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        state = _unwrap_encoder_checkpoint(checkpoint)
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
        matched = len(self.model.state_dict()) - len(incompatible.missing_keys)
        if matched <= 0:
            raise ValueError("MTP checkpoint did not match the official encoder")
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.to(self.device).eval()
        if not hasattr(self.model, "classification_head"):
            raise AttributeError("official encoder has no classification_head to bypass")
        self._captured = None
        self._hook = self.model.classification_head.register_forward_pre_hook(
            self._capture_head_input
        )
        self.report = {
            "repo_dir": repo_dir,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": _sha256_file(checkpoint_path),
            "checkpoint_state_key_count": len(state),
            "loaded_key_count": matched,
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
            "excluded_classification_head_keys": sorted(set(state) - set(filtered)),
            "all_parameters_frozen": all(
                not parameter.requires_grad for parameter in self.model.parameters()
            ),
        }

    def _capture_head_input(self, module, inputs):
        del module
        if not inputs:
            raise RuntimeError("classification head received no positional input")
        self._captured = inputs[0].detach()

    def _encode_groups(self, token_groups):
        torch = self.torch
        token_groups = np.asarray(token_groups)
        if token_groups.ndim != 3:
            raise ValueError(
                f"expected batch x channels x time token groups, got {token_groups.shape}"
            )
        channels, time = token_groups.shape[1:]
        estimated_length = channels * time + channels
        if estimated_length > self.config.official_max_sequence_length:
            raise ValueError(
                f"group sequence estimate {estimated_length} exceeds official maximum "
                f"{self.config.official_max_sequence_length}"
            )
        self._captured = None
        with torch.inference_mode():
            values = torch.as_tensor(
                token_groups,
                dtype=torch.long,
                device=self.device,
            )
            self.model(values, num_ch=channels)
        if self._captured is None:
            raise RuntimeError("classification-head hook did not capture encoder features")
        features = self._captured
        if features.ndim != 2 or features.shape[0] != len(token_groups):
            raise ValueError(
                "unexpected official encoder feature shape "
                f"{tuple(features.shape)}; expected batch x feature"
            )
        if features.shape[1] != self.config.embedding_size:
            raise ValueError(
                f"expected {self.config.embedding_size} encoder features, got "
                f"{features.shape[1]}"
            )
        return features.float().cpu().numpy()

    def encode(self, token_maps):
        """Encode equal-length 104-channel maps using fixed 16-channel windows."""

        token_maps = np.asarray(token_maps)
        if token_maps.ndim != 3:
            raise ValueError(
                f"expected batch x channels x time token maps, got {token_maps.shape}"
            )
        batch, channels, time = token_maps.shape
        if channels != self.config.expected_channels:
            raise ValueError(
                f"expected {self.config.expected_channels} channels, got {channels}"
            )
        if token_maps.min() < 0 or token_maps.max() >= self.config.codebook_size:
            raise ValueError("token IDs fall outside the configured codebook")
        group = self.config.channel_group_size
        full_group_count = channels // group
        remainder = channels % group
        weighted_sum = None
        if full_group_count:
            full = token_maps[:, : full_group_count * group].reshape(
                batch * full_group_count,
                group,
                time,
            )
            full_features = self._encode_groups(full).reshape(
                batch,
                full_group_count,
                -1,
            )
            weighted_sum = full_features.sum(axis=1) * group
        if remainder:
            tail = token_maps[:, full_group_count * group :]
            tail_features = self._encode_groups(tail)
            contribution = tail_features * remainder
            weighted_sum = contribution if weighted_sum is None else weighted_sum + contribution
        features = weighted_sum / channels
        if not np.isfinite(features).all():
            raise ValueError("official encoder produced non-finite features")
        return features.astype(np.float32, copy=False)


def _feature_cache_key(config, dataset_fingerprint, encoder_sha256):
    payload = {
        "format_version": 1,
        "config": {
            "codebook_size": config.codebook_size,
            "embedding_size": config.embedding_size,
            "expected_channels": config.expected_channels,
            "channel_group_size": config.channel_group_size,
            "official_max_sequence_length": config.official_max_sequence_length,
        },
        "dataset_fingerprint": dataset_fingerprint,
        "encoder_sha256": encoder_sha256,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _string_scalar(value):
    value = np.asarray(value)
    return str(value.item() if value.ndim == 0 else value.reshape(-1)[0])


def _cached_subject_is_current(path, subject, count, cache_key):
    try:
        with np.load(path, allow_pickle=False) as cached:
            return (
                int(cached["format_version"]) == 1
                and _string_scalar(cached["subject"]) == subject
                and len(cached["sentence_ids"]) == count
                and cached["features"].shape[0] == count
                and _string_scalar(cached["cache_key"]) == cache_key
            )
    except (OSError, KeyError, ValueError):
        return False


def _extract_subject_features(records, encoder, config):
    features = [None] * len(records)
    by_length = defaultdict(list)
    for index, record in enumerate(records):
        by_length[record.tokens.shape[1]].append(index)
    completed = 0
    next_report = 50
    for length in sorted(by_length):
        indices = by_length[length]
        for start in range(0, len(indices), config.extraction_batch_size):
            positions = indices[start : start + config.extraction_batch_size]
            token_maps = np.stack([records[index].tokens for index in positions])
            encoded = encoder.encode(token_maps)
            for index, feature in zip(positions, encoded):
                features[index] = feature
            completed += len(positions)
            if completed >= next_report or completed == len(records):
                print(
                    f"  encoded {completed}/{len(records)} recordings",
                    flush=True,
                )
                while next_report <= completed:
                    next_report += 50
    if any(feature is None for feature in features):
        raise RuntimeError("feature extraction left incomplete subject rows")
    return np.stack(features).astype(np.float32, copy=False)


def extract_or_load_encoder_features(
    records,
    encoder,
    cache_dir,
    dataset_fingerprint,
    config=EncoderProbeConfig(),
):
    """Persist one atomic feature shard per subject and return all record rows."""

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    encoder_sha256 = encoder.report["checkpoint_sha256"]
    cache_key = _feature_cache_key(config, dataset_fingerprint, encoder_sha256)
    by_subject = defaultdict(list)
    for record in records:
        by_subject[record.subject].append(record)
    feature_blocks = []
    metadata_rows = []
    for subject in sorted(by_subject):
        subject_records = sorted(
            by_subject[subject],
            key=lambda record: record.sentence_id,
        )
        path = cache_dir / f"{subject}.npz"
        if _cached_subject_is_current(
            path,
            subject,
            len(subject_records),
            cache_key,
        ):
            print(f"Reusing V3 encoder features for {subject}", flush=True)
        else:
            print(f"Encoding V3 features for {subject}", flush=True)
            features = _extract_subject_features(subject_records, encoder, config)
            temporary = path.with_suffix(".tmp.npz")
            np.savez_compressed(
                temporary,
                format_version=np.int64(1),
                cache_key=np.asarray(cache_key),
                subject=np.asarray(subject),
                sentence_ids=np.asarray(
                    [record.sentence_id for record in subject_records],
                    dtype=np.int64,
                ),
                labels=np.asarray(
                    [record.label for record in subject_records],
                    dtype=np.int64,
                ),
                features=features,
            )
            temporary.replace(path)
            print(f"Saved V3 subject features: {path}", flush=True)
        with np.load(path, allow_pickle=False) as cached:
            sentence_ids = np.asarray(cached["sentence_ids"], dtype=np.int64)
            labels = np.asarray(cached["labels"], dtype=np.int64)
            features = np.asarray(cached["features"], dtype=np.float32)
        if features.shape != (len(subject_records), config.embedding_size):
            raise ValueError(f"invalid cached V3 feature shape {features.shape} in {path}")
        if not np.isfinite(features).all():
            raise ValueError(f"non-finite cached V3 feature in {path}")
        feature_blocks.append(features)
        metadata_rows.extend(
            {
                "subject": subject,
                "sentence_id": int(sentence_id),
                "label": int(label),
                "feature_cache": str(path),
            }
            for sentence_id, label in zip(sentence_ids, labels)
        )
    record_features = np.concatenate(feature_blocks, axis=0)
    metadata = pd.DataFrame(metadata_rows)
    fingerprint = hashlib.sha256()
    for row, feature in zip(metadata.itertuples(index=False), record_features):
        fingerprint.update(f"{row.subject}|{row.sentence_id}|{row.label}\n".encode())
        fingerprint.update(feature.tobytes())
    report = {
        "n_recordings": int(len(metadata)),
        "n_sentences": int(metadata["sentence_id"].nunique()),
        "n_subjects": int(metadata["subject"].nunique()),
        "feature_dimension": int(record_features.shape[1]),
        "feature_cache_key": cache_key,
        "feature_fingerprint": fingerprint.hexdigest(),
        "encoder_checkpoint_sha256": encoder_sha256,
        "channel_group_size": config.channel_group_size,
        "channel_group_aggregation": "channel-count-weighted mean",
        "feature_norm_mean": float(np.linalg.norm(record_features, axis=1).mean()),
        "feature_norm_std": float(np.linalg.norm(record_features, axis=1).std()),
        "zero_variance_dimensions": int(
            np.count_nonzero(record_features.std(axis=0) < 1e-12)
        ),
    }
    return record_features, metadata, report


def build_sentence_features(record_features, record_metadata):
    """Mean-pool frozen reader features into one independent sentence row."""

    record_features = np.asarray(record_features, dtype=np.float32)
    if len(record_features) != len(record_metadata):
        raise ValueError("record features and metadata must have equal length")
    grouped = defaultdict(list)
    labels = {}
    for index, row in enumerate(record_metadata.itertuples(index=False)):
        sentence_id = int(row.sentence_id)
        label = int(row.label)
        if sentence_id in labels and labels[sentence_id] != label:
            raise ValueError(f"conflicting labels for sentence {sentence_id}")
        labels[sentence_id] = label
        grouped[sentence_id].append(record_features[index])
    sentence_ids = np.asarray(sorted(grouped), dtype=np.int64)
    features = np.stack(
        [np.mean(grouped[sentence_id], axis=0) for sentence_id in sentence_ids]
    ).astype(np.float32)
    y = np.asarray([labels[sentence_id] for sentence_id in sentence_ids], dtype=np.int64)
    metadata = pd.DataFrame(
        {
            "sentence_id": sentence_ids,
            "label": y,
            "n_readers": [len(grouped[sentence_id]) for sentence_id in sentence_ids],
        }
    )
    return features, y, metadata


def _classification_metrics(y_true, y_pred):
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
            f"f1_class_{label}": float(value)
            for label, value in zip(LABELS, per_class)
        },
    }


def _fit_linear_probe(X, y, config, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    estimator = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=seed,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    inner = StratifiedKFold(
        n_splits=config.inner_splits,
        shuffle=True,
        random_state=seed + 10_000,
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


def _bootstrap_delta(predictions, config, seed=2026):
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(seed)
    paired = []
    for run_seed in sorted(predictions["seed"].unique()):
        subset = predictions[predictions["seed"] == run_seed]
        aligned = subset[subset["setup"] == "encoder_probe"].sort_values("sentence_id")
        shuffled = subset[
            subset["setup"] == "encoder_probe_shuffled"
        ].sort_values("sentence_id")
        if not np.array_equal(
            aligned["sentence_id"].to_numpy(),
            shuffled["sentence_id"].to_numpy(),
        ):
            raise ValueError("aligned and shuffled V3 predictions are not paired")
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
    aligned = metrics[metrics["setup"] == "encoder_probe"]
    shuffled = metrics[metrics["setup"] == "encoder_probe_shuffled"]
    majority = metrics[metrics["setup"] == "majority"]
    aligned_macro = float(aligned["macro_f1"].mean())
    shuffled_macro = float(shuffled["macro_f1"].mean())
    majority_macro = float(majority["macro_f1"].mean())
    observed_delta = aligned_macro - shuffled_macro
    seed_means = (
        metrics[metrics["setup"].isin(["encoder_probe", "encoder_probe_shuffled"])]
        .groupby(["seed", "setup"])["macro_f1"]
        .mean()
        .unstack("setup")
    )
    seed_deltas = {
        str(int(seed)): float(row["encoder_probe"] - row["encoder_probe_shuffled"])
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
            "V3 PASS — frozen TFM encoder carries reproducible aligned signal"
            if passes
            else "V3 FAIL — stop the bounded TFM transfer sequence"
        ),
    }


def evaluate_encoder_probe(
    X,
    y,
    sentence_ids,
    output_dir,
    cache_report,
    encoder_report,
    feature_report,
    config=EncoderProbeConfig(),
):
    """Evaluate frozen sentence features with nested linear probes and controls."""

    from sklearn.dummy import DummyClassifier
    from sklearn.model_selection import StratifiedKFold

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    sentence_ids = np.asarray(sentence_ids, dtype=np.int64)
    if X.ndim != 2 or X.shape[1] != config.embedding_size:
        raise ValueError(
            f"expected sentence features N x {config.embedding_size}, got {X.shape}"
        )
    if not (len(X) == len(y) == len(sentence_ids)):
        raise ValueError("X, y, and sentence IDs must have equal length")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    signature_payload = {
        "experiment": "tfm_v3_frozen_mtp_encoder_linear_probe",
        "config": config.to_dict(),
        "dataset_fingerprint": cache_report["dataset_fingerprint"],
        "encoder_sha256": encoder_report["checkpoint_sha256"],
        "feature_fingerprint": feature_report["feature_fingerprint"],
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode()
    ).hexdigest()
    signature_path = output_dir / "run_signature.json"
    if signature_path.exists():
        previous = json.loads(signature_path.read_text())
        if previous.get("signature") != signature:
            raise ValueError(
                "existing V3 results use different data/encoder/config; "
                "choose a new results directory"
            )
    else:
        _atomic_json({"signature": signature, **signature_payload}, signature_path)

    metric_rows = []
    prediction_rows = []
    for run_seed in config.seeds:
        outer = StratifiedKFold(
            n_splits=config.n_splits,
            shuffle=True,
            random_state=run_seed,
        )
        for fold, (train_index, test_index) in enumerate(outer.split(X, y)):
            rng = np.random.default_rng(run_seed * 100 + fold)
            train_permutation = rng.permutation(len(train_index))
            test_permutation = rng.permutation(len(test_index))
            setups = {
                "encoder_probe": (
                    X[train_index],
                    X[test_index],
                    sentence_ids[test_index],
                ),
                "encoder_probe_shuffled": (
                    X[train_index][train_permutation],
                    X[test_index][test_permutation],
                    sentence_ids[test_index][test_permutation],
                ),
            }
            for setup, (X_train, X_test, feature_sentence_ids) in setups.items():
                print(f"Fitting {setup}, seed={run_seed}, fold={fold}", flush=True)
                fitted = _fit_linear_probe(
                    X_train,
                    y[train_index],
                    config=config,
                    seed=run_seed + fold,
                )
                predicted = fitted.predict(X_test).astype(np.int64)
                probabilities = fitted.predict_proba(X_test)
                if not np.array_equal(fitted.classes_, np.asarray(LABELS)):
                    raise ValueError(f"unexpected classifier classes {fitted.classes_}")
                metric_rows.append(
                    {
                        "setup": setup,
                        "seed": run_seed,
                        "fold": fold,
                        "best_C": fitted.best_params_["classifier__C"],
                        **_classification_metrics(y[test_index], predicted),
                    }
                )
                for position, feature_sentence_id, prediction, probability in zip(
                    test_index,
                    feature_sentence_ids,
                    predicted,
                    probabilities,
                ):
                    prediction_rows.append(
                        {
                            "setup": setup,
                            "seed": run_seed,
                            "fold": fold,
                            "sentence_id": int(sentence_ids[position]),
                            "feature_sentence_id": int(feature_sentence_id),
                            "label": int(y[position]),
                            "prediction": int(prediction),
                            "probability_-1": float(probability[0]),
                            "probability_0": float(probability[1]),
                            "probability_1": float(probability[2]),
                        }
                    )

            dummy = DummyClassifier(strategy="prior").fit(X[train_index], y[train_index])
            predicted = dummy.predict(X[test_index]).astype(np.int64)
            probabilities = dummy.predict_proba(X[test_index])
            if not np.array_equal(dummy.classes_, np.asarray(LABELS)):
                raise ValueError(f"unexpected dummy classifier classes {dummy.classes_}")
            metric_rows.append(
                {
                    "setup": "majority",
                    "seed": run_seed,
                    "fold": fold,
                    "best_C": np.nan,
                    **_classification_metrics(y[test_index], predicted),
                }
            )
            for position, prediction, probability in zip(
                test_index,
                predicted,
                probabilities,
            ):
                prediction_rows.append(
                    {
                        "setup": "majority",
                        "seed": run_seed,
                        "fold": fold,
                        "sentence_id": int(sentence_ids[position]),
                        "feature_sentence_id": int(sentence_ids[position]),
                        "label": int(y[position]),
                        "prediction": int(prediction),
                        "probability_-1": float(probability[0]),
                        "probability_0": float(probability[1]),
                        "probability_1": float(probability[2]),
                    }
                )

    metrics = pd.DataFrame(metric_rows).sort_values(
        ["seed", "fold", "setup"]
    ).reset_index(drop=True)
    predictions = pd.DataFrame(prediction_rows).sort_values(
        ["seed", "fold", "setup", "sentence_id"]
    ).reset_index(drop=True)
    summary = (
        metrics.groupby("setup")[["accuracy", "balanced_accuracy", "macro_f1"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    delta = _bootstrap_delta(predictions, config=config)
    gate = _gate_report(metrics, delta, config=config)
    _atomic_csv(metrics, output_dir / "fold_metrics.csv")
    _atomic_csv(predictions, output_dir / "oof_predictions.csv")
    _atomic_csv(summary, output_dir / "summary.csv")
    _atomic_json(config.to_dict(), output_dir / "evaluation_config.json")
    _atomic_json(cache_report, output_dir / "token_cache_report.json")
    _atomic_json(encoder_report, output_dir / "encoder_report.json")
    _atomic_json(feature_report, output_dir / "feature_report.json")
    _atomic_json(delta, output_dir / "alignment_delta.json")
    _atomic_json(gate, output_dir / "viability_gate.json")
    return metrics, predictions, summary, delta, gate
