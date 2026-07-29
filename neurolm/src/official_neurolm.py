"""Load only the frozen neural encoder from the official NeuroLM-B checkpoint."""

from collections import OrderedDict
from pathlib import Path
import sys

import numpy as np

from .config import EncoderConfig
from .channels import NEUROLM_CHANNELS


def select_tokenizer_state(state_dict):
    """Extract NeuroLM's tokenizer subtree without materializing GPT-2."""

    selected = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod.") :]
        if key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("tokenizer."):
            selected[key[len("tokenizer.") :]] = value
    if not selected:
        raise ValueError("checkpoint has no tokenizer.* weights")
    return selected


class OfficialNeuroLMEncoder:
    """Frozen NeuroLM-B encoder plus a fixed EEG-only pooling rule."""

    feature_version = "neurolm_b_mean_std_temporal_slope_v1"

    def __init__(
        self,
        repo_dir,
        checkpoint_path,
        channel_ids,
        zuco_indices=None,
        device="cuda",
        config=EncoderConfig(),
    ):
        import torch

        self.repo_dir = str(Path(repo_dir).resolve())
        self.checkpoint_path = str(Path(checkpoint_path).resolve())
        self.device = torch.device(device)
        self.config = config
        self.channel_ids = np.asarray(channel_ids, dtype=np.int64)
        if self.channel_ids.ndim != 1 or not len(self.channel_ids):
            raise ValueError(f"expected a non-empty channel-ID vector, got {self.channel_ids.shape}")
        if zuco_indices is None:
            if len(self.channel_ids) != 104:
                raise ValueError("zuco_indices are required when fewer than 104 channels are used")
            zuco_indices = np.arange(104)
        self.zuco_indices = np.asarray(zuco_indices, dtype=np.int64)
        if self.zuco_indices.shape != self.channel_ids.shape:
            raise ValueError("zuco_indices and channel_ids must have equal shape")
        if len(np.unique(self.zuco_indices)) != len(self.zuco_indices):
            raise ValueError("zuco_indices must be unique")
        if self.zuco_indices.min() < 0 or self.zuco_indices.max() >= 104:
            raise ValueError("zuco_indices must lie inside [0, 104)")
        if self.channel_ids.min() < 0 or self.channel_ids.max() >= 256:
            raise ValueError("channel IDs must fit NeuroLM's 256-entry position table")
        if self.repo_dir not in sys.path:
            sys.path.insert(0, self.repo_dir)

        from model.model_neural_transformer import NTConfig, NeuralTransformer

        encoder_args = dict(
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_embd=config.n_embd,
            block_size=config.block_size,
            bias=config.bias,
            dropout=config.dropout,
            num_classes=0,
            in_chans=config.in_chans,
            out_chans=config.out_chans,
        )
        self.model = NeuralTransformer(NTConfig(**encoder_args))
        load_kwargs = {"map_location": "cpu"}
        # mmap limits peak host RAM on current PyTorch; older versions simply omit it.
        try:
            checkpoint = torch.load(
                self.checkpoint_path, mmap=True, weights_only=False, **load_kwargs
            )
        except TypeError:
            checkpoint = torch.load(self.checkpoint_path, **load_kwargs)
        if "model" not in checkpoint:
            raise KeyError("official checkpoint is missing its model state")
        tokenizer_state = select_tokenizer_state(checkpoint["model"])
        incompatibility = self.model.load_state_dict(tokenizer_state, strict=False)
        self.load_report = {
            "selected_keys": len(tokenizer_state),
            "missing_keys": list(incompatibility.missing_keys),
            "unexpected_keys": list(incompatibility.unexpected_keys),
        }
        if self.load_report["missing_keys"] or self.load_report["unexpected_keys"]:
            raise RuntimeError(f"NeuroLM-B encoder checkpoint mismatch: {self.load_report}")
        del checkpoint, tokenizer_state
        self.model.eval().requires_grad_(False).to(self.device)

    def _encode_block(self, patches, channel_ids, time_ids):
        import torch

        n_patches = len(patches)
        if not 0 < n_patches <= self.config.block_size:
            raise ValueError(f"invalid patch count {n_patches}")
        x = torch.zeros(
            (1, self.config.block_size, self.config.patch_samples),
            dtype=torch.float32,
            device=self.device,
        )
        chans = torch.full(
            (1, self.config.block_size),
            fill_value=NEUROLM_CHANNELS.index("PAD"),
            dtype=torch.int32,
            device=self.device,
        )
        times = torch.zeros_like(chans)
        valid = torch.zeros((1, self.config.block_size), dtype=torch.bool, device=self.device)
        x[0, :n_patches] = torch.as_tensor(patches, device=self.device)
        chans[0, :n_patches] = torch.as_tensor(channel_ids, dtype=torch.int32, device=self.device)
        times[0, :n_patches] = torch.as_tensor(time_ids, dtype=torch.int32, device=self.device)
        valid[0, :n_patches] = True
        attention_mask = valid.unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        autocast_enabled = self.device.type == "cuda"
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type, dtype=dtype, enabled=autocast_enabled
        ):
            encoded = self.model(
                x, chans, times, attention_mask, return_all_tokens=True
            )
        if isinstance(encoded, (tuple, list)):
            encoded = encoded[0]
        if encoded.ndim != 3 or encoded.shape[1] < n_patches:
            raise ValueError(f"unexpected encoder output shape {tuple(encoded.shape)}")
        return encoded[0, :n_patches].float().cpu().numpy()

    @staticmethod
    def pool_embeddings(embeddings_by_second):
        """Mean, standard deviation, and coarse temporal slope; no learned weights."""

        values = np.asarray(embeddings_by_second, dtype=np.float32)
        if values.ndim != 3 or values.shape[0] < 1:
            raise ValueError(f"expected seconds x channels x embedding, got {values.shape}")
        flattened = values.reshape(-1, values.shape[-1])
        mean = flattened.mean(axis=0)
        std = flattened.std(axis=0)
        per_second = values.mean(axis=1)
        if len(per_second) == 1:
            slope = np.zeros_like(mean)
        else:
            time = np.linspace(-1.0, 1.0, len(per_second), dtype=np.float32)
            denominator = float(np.square(time).sum())
            slope = (time[:, None] * per_second).sum(axis=0) / denominator
        feature = np.concatenate((mean, std, slope)).astype(np.float32, copy=False)
        if not np.isfinite(feature).all():
            raise ValueError("pooling produced non-finite features")
        return feature

    def encode_recording_tokens(self, eeg):
        """Return frozen embeddings as seconds x mapped channels x embedding."""

        eeg = np.asarray(eeg, dtype=np.float32)
        if eeg.ndim != 2 or eeg.shape[0] != 104:
            raise ValueError(f"expected 104 x time EEG, got {eeg.shape}")
        eeg = eeg[self.zuco_indices]
        n_channels = len(self.channel_ids)
        seconds = eeg.shape[1] // self.config.patch_samples
        if seconds < 1:
            raise ValueError("recording has no complete one-second patch")
        eeg = eeg[:, : seconds * self.config.patch_samples]
        max_seconds = self.config.block_size // n_channels
        chunks = []
        for start in range(0, seconds, max_seconds):
            stop = min(start + max_seconds, seconds)
            window = eeg[:, start * self.config.patch_samples : stop * self.config.patch_samples]
            patches = window.reshape(n_channels, stop - start, self.config.patch_samples)
            patches = patches.transpose(1, 0, 2).reshape(-1, self.config.patch_samples)
            channel_ids = np.tile(self.channel_ids, stop - start)
            time_ids = np.repeat(np.arange(stop - start, dtype=np.int32), n_channels)
            encoded = self._encode_block(patches, channel_ids, time_ids)
            chunks.append(encoded.reshape(stop - start, n_channels, -1))
        embeddings = np.concatenate(chunks, axis=0)
        details = {
            "seconds": int(seconds),
            "channels": int(n_channels),
            "patches": int(seconds * n_channels),
            "embedding_dim": int(embeddings.shape[-1]),
        }
        return embeddings, details

    def encode_recording(self, eeg):
        """Encode channels x time EEG and return the unchanged V1 pooled vector."""

        embeddings, details = self.encode_recording_tokens(eeg)
        feature = self.pool_embeddings(embeddings)
        return feature, {
            **details,
            "feature_dim": int(feature.size),
            "feature_norm": float(np.linalg.norm(feature)),
        }
