"""Partially unfrozen NeuroLM/GPT-2 evaluation for EEG-only V5.

The neural tokenizer, GPT-2 embeddings, and lower transformer blocks remain
frozen.  The final GPT-2 blocks, final normalization, and the small V4
verbalizer adapter are optimized from raw preprocessed EEG.
"""

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd

from .gpt2_cache import prepare_eeg_tokens
from .gpt2_verbalizer import build_gpt2_verbalizer
from .raw_eegnet import (
    INDEX_TO_LABEL,
    LABELS,
    LABEL_TO_INDEX,
    RawExample,
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


ALIGNED = "neurolm_partial_finetune"
SHUFFLED = "neurolm_partial_finetune_shuffled"
IMPLEMENTATION_VERSION = "neurolm-partial-finetune-v5.0"


@dataclass(frozen=True)
class PartialFinetuneConfig:
    """One resource-bounded, predeclared V5 fine-tuning recipe."""

    seeds: tuple = (42, 52, 62)
    n_splits: int = 5
    validation_fraction: float = 0.15
    maximum_seconds: int = 3
    top_gpt2_blocks: int = 2
    embedding_size: int = 768
    adapter_size: int = 32
    dropout: float = 0.25
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_epochs: int = 6
    patience: int = 2
    transformer_learning_rate: float = 1e-5
    adapter_learning_rate: float = 5e-4
    weight_decay: float = 1e-2
    gradient_clip: float = 1.0
    bootstrap_samples: int = 5000
    confidence: float = 0.95
    minimum_delta: float = 0.015
    minimum_positive_seeds: int = 2

    def to_dict(self):
        values = asdict(self)
        values["implementation_version"] = IMPLEMENTATION_VERSION
        return values


def _require_torch():
    try:
        import torch
    except ImportError as error:
        raise ImportError("V5 training requires PyTorch; use the Colab notebook") from error
    return torch


try:
    from torch.nn import Module as _TorchModule
except ImportError:
    class _TorchModule:
        """Import-only fallback; construction still requires Colab's PyTorch."""

        pass


def selected_trainable_blocks(total_blocks, top_blocks):
    """Return the zero-based indices of the final blocks selected for V5."""

    total_blocks = int(total_blocks)
    top_blocks = int(top_blocks)
    if total_blocks < 1:
        raise ValueError("GPT-2 must expose at least one transformer block")
    if not 1 <= top_blocks <= total_blocks:
        raise ValueError(
            f"top_gpt2_blocks must lie in [1, {total_blocks}], got {top_blocks}"
        )
    return tuple(range(total_blocks - top_blocks, total_blocks))


def one_reader_per_sentence(examples, seed):
    """Choose one reader from each fixed sentence bundle and restore its full weight."""

    grouped = defaultdict(list)
    for example in examples:
        grouped[int(example.target_sentence_id)].append(example)
    rng = np.random.default_rng(int(seed))
    sampled = []
    for sentence_id in sorted(grouped):
        rows = grouped[sentence_id]
        chosen = rows[int(rng.integers(0, len(rows)))]
        sampled.append(
            RawExample(
                record=chosen.record,
                target_sentence_id=chosen.target_sentence_id,
                target_label=chosen.target_label,
                weight=float(sum(row.weight for row in rows)),
            )
        )
    return sampled


class PartialFinetuneCollator:
    """Turn reader recordings into the fixed three-second NeuroLM prefix."""

    def __init__(self, zuco_indices, channel_ids, config=PartialFinetuneConfig()):
        self.zuco_indices = np.asarray(zuco_indices, dtype=np.int64)
        self.channel_ids = np.asarray(channel_ids, dtype=np.int64)
        self.config = config
        if self.zuco_indices.shape != self.channel_ids.shape or not len(self.channel_ids):
            raise ValueError("channel IDs and ZuCo indices must be non-empty and aligned")

    def __call__(self, batch):
        torch = _require_torch()
        prepared = [
            prepare_eeg_tokens(
                example.record.eeg,
                self.zuco_indices,
                self.channel_ids,
                maximum_seconds=self.config.maximum_seconds,
            )
            for example in batch
        ]
        return {
            "patches": torch.from_numpy(np.stack([row[0] for row in prepared])),
            "channels": torch.from_numpy(np.stack([row[1] for row in prepared])),
            "times": torch.from_numpy(np.stack([row[2] for row in prepared])),
            "valid": torch.from_numpy(np.stack([row[3] for row in prepared])),
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


def _loader(examples, model, config, shuffle, seed):
    torch = _require_torch()
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return torch.utils.data.DataLoader(
        _ExampleDataset(examples),
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=PartialFinetuneCollator(
            model.zuco_indices, model.channel_ids, config
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


class PartiallyUnfrozenNeuroLM(_TorchModule):
    """Reuse one loaded official model and reset its trainable subset per fit."""

    def __init__(self, official_encoder, config=PartialFinetuneConfig()):
        torch = _require_torch()
        super().__init__()
        self.config = config
        self.neurolm = official_encoder.model
        self.channel_ids = np.asarray(official_encoder.channel_ids, dtype=np.int64)
        self.zuco_indices = np.asarray(official_encoder.zuco_indices, dtype=np.int64)
        self.prompt_ids = tuple(int(value) for value in official_encoder.prompt_ids)
        self.end_token = int(official_encoder.encoding.eot_token)
        self.embedding_size = int(official_encoder.embedding_size)
        self.verbalizer_vectors = np.asarray(
            official_encoder.verbalizer_vectors, dtype=np.float32
        ).copy()
        if self.embedding_size != config.embedding_size:
            raise ValueError(
                f"checkpoint embedding size {self.embedding_size} != {config.embedding_size}"
            )
        if official_encoder.maximum_seconds != config.maximum_seconds:
            raise ValueError("loaded model and V5 maximum_seconds do not match")
        if config.maximum_seconds * len(self.channel_ids) + len(self.prompt_ids) + 1 > 1024:
            raise ValueError("V5 EEG and instruction exceed GPT-2's context length")

        self.neurolm.requires_grad_(False)
        transformer = self.neurolm.GPT2.transformer
        if not hasattr(transformer, "h") or not hasattr(transformer, "ln_f"):
            raise AttributeError("official GPT-2 transformer blocks/final norm unavailable")
        self.gpt2_blocks = transformer.h
        self.trainable_block_indices = selected_trainable_blocks(
            len(self.gpt2_blocks), config.top_gpt2_blocks
        )
        for index in self.trainable_block_indices:
            self.gpt2_blocks[index].requires_grad_(True)
        transformer.ln_f.requires_grad_(True)
        self.final_norm = transformer.ln_f
        self.verbalizer = build_gpt2_verbalizer(self.verbalizer_vectors, config)
        self.to(official_encoder.device)
        self.runtime_device = official_encoder.device

        self._pretrained_transformer_state = {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.named_parameters()
            if parameter.requires_grad and name.startswith("neurolm.")
        }
        self.reset_for_fit(0)
        self.report = self._build_report()

    def _build_report(self):
        tokenizer_trainable = sum(
            parameter.numel()
            for parameter in self.neurolm.tokenizer.parameters()
            if parameter.requires_grad
        )
        transformer_count = sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if parameter.requires_grad and name.startswith("neurolm.")
        )
        adapter_count = sum(
            parameter.numel()
            for parameter in self.verbalizer.parameters()
            if parameter.requires_grad
        )
        return {
            "gpt2_block_count": len(self.gpt2_blocks),
            "trainable_gpt2_block_indices": list(self.trainable_block_indices),
            "trainable_transformer_parameters": int(transformer_count),
            "trainable_adapter_parameters": int(adapter_count),
            "total_trainable_parameters": int(transformer_count + adapter_count),
            "trainable_tokenizer_parameters": int(tokenizer_trainable),
            "maximum_seconds": self.config.maximum_seconds,
            "mapped_channels": len(self.channel_ids),
        }

    def trainable_parameters(self):
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def transformer_parameters(self):
        return [
            parameter
            for name, parameter in self.named_parameters()
            if parameter.requires_grad and name.startswith("neurolm.")
        ]

    def adapter_parameters(self):
        return list(self.verbalizer.parameters())

    def trainable_state(self):
        return {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }

    def load_trainable_state(self, state):
        torch = _require_torch()
        parameters = dict(self.named_parameters())
        expected = {name for name, parameter in parameters.items() if parameter.requires_grad}
        if set(state) != expected:
            raise ValueError("V5 trainable-state keys do not match the model")
        with torch.no_grad():
            for name, value in state.items():
                parameters[name].copy_(value.to(parameters[name].device))

    def reset_for_fit(self, seed):
        torch = _require_torch()
        parameters = dict(self.named_parameters())
        with torch.no_grad():
            for name, value in self._pretrained_transformer_state.items():
                parameters[name].copy_(value.to(parameters[name].device))
        _set_seed(int(seed))
        with torch.no_grad():
            torch.nn.init.normal_(self.verbalizer.down.weight, mean=0.0, std=0.02)
            torch.nn.init.zeros_(self.verbalizer.up.weight)
            torch.nn.init.zeros_(self.verbalizer.bias)
        self.zero_grad(set_to_none=True)
        return self

    def train(self, mode=True):
        super().train(mode)
        # Frozen components stay deterministic; only the selected top stays in train mode.
        self.neurolm.eval()
        for index in self.trainable_block_indices:
            self.gpt2_blocks[index].train(mode)
        self.final_norm.train(mode)
        self.verbalizer.train(mode)
        return self

    def forward(self, patches, channels, times, valid):
        torch = _require_torch()
        if patches.ndim != 3 or patches.shape[-1] != 200:
            raise ValueError(f"expected batch x EEG-tokens x 200, got {tuple(patches.shape)}")
        batch_size, eeg_length = patches.shape[:2]
        if channels.shape != (batch_size, eeg_length):
            raise ValueError("V5 channel tensor does not match EEG patches")
        if times.shape != channels.shape or valid.shape != channels.shape:
            raise ValueError("V5 time/valid tensors do not match EEG patches")

        text_values = list(self.prompt_ids) + [self.end_token]
        text = torch.as_tensor(
            text_values, dtype=torch.long, device=patches.device
        )[None].repeat(batch_size, 1)
        text_targets = torch.full_like(text, -1)
        text_targets[:, len(self.prompt_ids) - 1] = self.end_token
        vocab_size = int(self.neurolm.GPT2.config.vocab_size)
        eeg_targets = torch.full(
            (batch_size, eeg_length),
            fill_value=-1 - vocab_size,
            dtype=torch.long,
            device=patches.device,
        )
        total = eeg_length + text.shape[1]
        attention = torch.tril(
            torch.ones((batch_size, total, total), dtype=torch.bool, device=patches.device)
        ).unsqueeze(1)
        channel_count = len(self.channel_ids)
        for batch_index in range(batch_size):
            valid_count = int(valid[batch_index].sum().item())
            valid_seconds = valid_count // channel_count
            for position in range(valid_seconds):
                start = position * channel_count
                stop = start + channel_count
                attention[batch_index, :, start:stop, start:stop] = True
            invalid_positions = torch.nonzero(
                ~valid[batch_index], as_tuple=False
            ).flatten()
            if invalid_positions.numel():
                attention[batch_index, :, :, invalid_positions] = False

        captured = []

        def save_hidden(_, __, output):
            captured.append(output[0] if isinstance(output, (tuple, list)) else output)

        hook = self.final_norm.register_forward_hook(save_hidden)
        try:
            self.neurolm(
                patches,
                eeg_targets,
                text,
                text_targets,
                channels,
                times,
                valid,
                eeg_text_mask=attention,
            )
        finally:
            hook.remove()
        if len(captured) != 1:
            raise RuntimeError(f"expected one GPT-2 hidden capture, got {len(captured)}")
        hidden = captured[0]
        prompt_position = eeg_length + len(self.prompt_ids) - 1
        prompt_hidden = hidden[:, prompt_position]
        if prompt_hidden.shape != (batch_size, self.embedding_size):
            raise ValueError(f"unexpected V5 prompt state {tuple(prompt_hidden.shape)}")
        return self.verbalizer(prompt_hidden)


def _move_batch(batch, device):
    return {
        name: batch[name].to(device, non_blocking=True)
        for name in ("patches", "channels", "times", "valid")
    }


def _aggregate_predictions(model, loader):
    torch = _require_torch()
    probability_sums = defaultdict(lambda: np.zeros(len(LABELS), dtype=np.float64))
    reader_counts = Counter()
    labels = {}
    model.eval()
    autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with torch.inference_mode():
        for batch in loader:
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                logits = model(**_move_batch(batch, model.runtime_device))
            probabilities = torch.softmax(logits.float(), dim=1).cpu().numpy()
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
    model,
    records,
    outer_train_sentence_ids,
    outer_train_labels,
    config,
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
    train_bundles = make_bundle_examples(
        records,
        outer_train_sentence_ids[train_position],
        outer_train_labels[train_position],
        shuffled=shuffled,
        seed=seed + 30_000,
    )
    validation_bundles = make_bundle_examples(
        records,
        outer_train_sentence_ids[validation_position],
        outer_train_labels[validation_position],
        shuffled=shuffled,
        seed=seed + 40_000,
    )
    validation_examples = one_reader_per_sentence(
        validation_bundles, seed=seed + 45_000
    )
    validation_loader = _loader(
        validation_examples, model, config, False, seed + 46_000
    )

    model.reset_for_fit(seed)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.transformer_parameters(),
                "lr": config.transformer_learning_rate,
            },
            {
                "params": model.adapter_parameters(),
                "lr": config.adapter_learning_rate,
            },
        ],
        weight_decay=config.weight_decay,
    )
    use_scaler = not torch.cuda.is_bf16_supported()
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    parameters = model.trainable_parameters()
    best_score = -np.inf
    best_epoch = None
    best_state = None
    stale_epochs = 0
    history = []

    for epoch in range(config.max_epochs):
        train_examples = one_reader_per_sentence(
            train_bundles, seed=seed + 50_000 + epoch
        )
        train_loader = _loader(
            train_examples, model, config, True, seed + 60_000 + epoch
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        weighted_loss = 0.0
        weight_total = 0.0
        for step, batch in enumerate(train_loader, start=1):
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                logits = model(**_move_batch(batch, model.runtime_device))
                labels = batch["labels"].to(model.runtime_device, non_blocking=True)
                weights = batch["weights"].to(model.runtime_device, non_blocking=True)
                losses = torch.nn.functional.cross_entropy(
                    logits.float(), labels, reduction="none"
                )
                micro_loss = (losses * weights).sum() / weights.sum().clamp_min(1e-8)
                loss = micro_loss / config.gradient_accumulation_steps
            scaler.scale(loss).backward()
            last_step = step == len(train_loader)
            if step % config.gradient_accumulation_steps == 0 or last_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            weighted_loss += float((losses.detach() * weights).sum().cpu())
            weight_total += float(weights.sum().cpu())

        _, truth, probabilities = _aggregate_predictions(model, validation_loader)
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
            best_state = model.trainable_state()
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("V5 training produced no finite validation result")
    model.load_trainable_state(best_state)
    return history, best_epoch, best_score


def bootstrap_alignment_delta(predictions, config=PartialFinetuneConfig(), seed=2026):
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
            raise ValueError("aligned and shuffled V5 predictions are not paired")
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
    alpha = 1.0 - config.confidence
    return {
        "observed_mean_seed_delta": float(np.mean(list(seed_deltas.values()))),
        "bootstrap_mean_delta": float(np.mean(draws)),
        "ci_level": float(config.confidence),
        "ci_low": float(np.quantile(draws, alpha / 2)),
        "ci_high": float(np.quantile(draws, 1 - alpha / 2)),
        "seed_deltas": seed_deltas,
        "bootstrap_samples": int(config.bootstrap_samples),
    }


def gate_report(metrics, delta, config=PartialFinetuneConfig()):
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
        "bootstrap_ci_low_above_zero": delta["ci_low"] > 0,
    }
    passes = all(criteria.values())
    core_without_ci = all(
        value for key, value in criteria.items() if key != "bootstrap_ci_low_above_zero"
    )
    status = "green" if passes else "yellow" if core_without_ci else "red"
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
        "decision": {
            "green": "GREEN — partially unfrozen NeuroLM shows repeatable EEG-specific value",
            "yellow": "YELLOW — suggestive only; do not expand fine-tuning yet",
            "red": "RED — no reliable V5 EEG-specific value; stop unfreezing",
        }[status],
    }


def smoke_test_partial_finetune(model, records, config=PartialFinetuneConfig()):
    torch = _require_torch()
    sentence_ids, labels = sentence_table(records)
    examples = make_bundle_examples(records, sentence_ids[:2], labels[:2])
    examples = one_reader_per_sentence(examples, seed=0)
    batch = PartialFinetuneCollator(
        model.zuco_indices, model.channel_ids, config
    )(examples)
    model.reset_for_fit(0).train()
    autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with torch.autocast(device_type="cuda", dtype=autocast_dtype):
        logits = model(**_move_batch(batch, model.runtime_device))
        loss = torch.nn.functional.cross_entropy(
            logits.float(), batch["labels"].to(model.runtime_device)
        )
    loss.backward()
    gradient_parameters = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    frozen_gradients = [
        name
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad and parameter.grad is not None
    ]
    if not gradient_parameters or frozen_gradients:
        raise RuntimeError("V5 gradient smoke test failed its freeze boundary")
    report = {
        **model.report,
        "input_shape": list(batch["patches"].shape),
        "logit_shape": list(logits.shape),
        "loss": float(loss.detach().cpu()),
        "gradient_tensor_count": len(gradient_parameters),
        "frozen_gradient_tensor_count": len(frozen_gradients),
    }
    model.reset_for_fit(0).eval()
    torch.cuda.empty_cache()
    return report


def evaluate_partial_finetune(
    model,
    records,
    output_dir,
    dataset_fingerprint,
    checkpoint_fingerprint,
    config=PartialFinetuneConfig(),
):
    """Run/resume the complete V5 aligned-versus-shuffled evaluation."""

    torch = _require_torch()
    from sklearn.model_selection import StratifiedKFold

    output_dir = Path(output_dir)
    completion_dir = output_dir / "completed_folds"
    history_dir = output_dir / "histories"
    output_dir.mkdir(parents=True, exist_ok=True)
    completion_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)
    if model.runtime_device.type != "cuda":
        raise RuntimeError("V5 evaluation requires a Colab GPU runtime")

    signature_payload = {
        "implementation_version": IMPLEMENTATION_VERSION,
        "dataset_fingerprint": str(dataset_fingerprint),
        "checkpoint_fingerprint": str(checkpoint_fingerprint),
        "channel_ids": [int(value) for value in model.channel_ids],
        "zuco_indices": [int(value) for value in model.zuco_indices],
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
                "V5 result directory belongs to a different data/configuration signature"
            )
    else:
        _write_json({"signature": signature, **signature_payload}, signature_path)

    metric_columns = [
        "setup", "seed", "fold", "best_epoch", "validation_macro_f1",
        "accuracy", "balanced_accuracy", "macro_f1",
        "f1_class_-1", "f1_class_0", "f1_class_1",
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
                history, best_epoch, best_score = _train_one_model(
                    model,
                    records,
                    train_ids,
                    train_labels,
                    config,
                    fit_seed,
                    shuffled,
                )
                test_examples = make_bundle_examples(
                    records,
                    test_ids,
                    test_labels,
                    shuffled=shuffled,
                    seed=fit_seed + 70_000,
                )
                test_loader = _loader(
                    test_examples, model, config, False, fit_seed + 80_000
                )
                evaluated_ids, truth, probabilities = _aggregate_predictions(
                    model, test_loader
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
                metrics = pd.concat(
                    [metrics, pd.DataFrame([metric_row])], ignore_index=True
                )[metric_columns]
                predictions = pd.concat(
                    [predictions, pd.DataFrame(rows)], ignore_index=True
                )[prediction_columns]
                _atomic_csv(metrics, partial_metrics_path)
                _atomic_csv(predictions, partial_predictions_path)
                _atomic_csv(
                    pd.DataFrame(history),
                    history_dir / f"{setup}_seed{seed}_fold{fold}.csv",
                )
                _write_json(
                    {
                        "setup": setup,
                        "seed": int(seed),
                        "fold": int(fold),
                        "best_epoch": int(best_epoch),
                        "validation_macro_f1": float(best_score),
                        "signature": signature,
                    },
                    marker,
                )
                _write_json(
                    {
                        "stage": "evaluating",
                        "last_completed": {
                            "setup": setup, "seed": int(seed), "fold": int(fold)
                        },
                        "completed_neural_fits": int(
                            len(metrics[metrics["setup"].isin([ALIGNED, SHUFFLED])])
                        ),
                        "total_neural_fits": int(2 * len(config.seeds) * config.n_splits),
                    },
                    output_dir / "runtime_status.json",
                )
                model.reset_for_fit(0).eval()
                torch.cuda.empty_cache()

            majority_mask = (
                (metrics["setup"] == "majority")
                & (metrics["seed"].astype(int) == int(seed))
                & (metrics["fold"].astype(int) == int(fold))
            ) if not metrics.empty else np.asarray([], dtype=bool)
            if not bool(np.any(majority_mask)):
                majority_label = Counter(map(int, train_labels)).most_common(1)[0][0]
                hard = np.full(len(test_ids), majority_label, dtype=np.int64)
                probabilities = np.zeros((len(test_ids), len(LABELS)), dtype=np.float64)
                probabilities[:, LABEL_TO_INDEX[majority_label]] = 1.0
                rows, _ = _prediction_rows(
                    "majority", seed, fold, test_ids, test_labels, probabilities
                )
                metrics = pd.concat(
                    [metrics, pd.DataFrame([{
                        "setup": "majority",
                        "seed": seed,
                        "fold": fold,
                        "best_epoch": np.nan,
                        "validation_macro_f1": np.nan,
                        **_metric_values(test_labels, hard),
                    }])], ignore_index=True
                )[metric_columns]
                predictions = pd.concat(
                    [predictions, pd.DataFrame(rows)], ignore_index=True
                )[prediction_columns]
                _atomic_csv(metrics, partial_metrics_path)
                _atomic_csv(predictions, partial_predictions_path)

    metrics = metrics.sort_values(["seed", "fold", "setup"]).reset_index(drop=True)
    predictions = predictions.sort_values(
        ["seed", "fold", "setup", "sentence_id"]
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
    _write_json(model.report, output_dir / "trainability_report.json")
    _write_json(delta, output_dir / "paired_bootstrap.json")
    _write_json(gate, output_dir / "viability_gate.json")
    _write_json(
        {
            "stage": "complete",
            "signature": signature,
            "completed_neural_fits": int(2 * len(config.seeds) * config.n_splits),
            "status": gate["status"],
        },
        output_dir / "runtime_status.json",
    )
    return metrics, predictions, summary, delta, gate
