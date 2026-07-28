"""Small bridge to the authors' official TFM tokenizer implementation."""

from pathlib import Path
import sys

import numpy as np


def _tfm_stft(eeg, sampling_rate):
    """Compute the paper's STFT without importing upstream's heavy utility module.

    The official utility module also imports training/evaluation packages such as
    ``pyhealth``. Tokenizer inference only needs this operation: a one-second
    Hann window, 50% overlap, magnitude output, no centering, and one-sided FFT.
    """

    import torch

    if eeg.ndim != 3:
        raise ValueError(f"expected batch x channels x time EEG, got {tuple(eeg.shape)}")
    batch_size, n_channels, n_samples = eeg.shape
    flattened = eeg.reshape(batch_size * n_channels, n_samples)
    window = torch.hann_window(
        sampling_rate,
        periodic=True,
        dtype=eeg.dtype,
        device=eeg.device,
    )
    spectral = torch.stft(
        flattened,
        n_fft=sampling_rate,
        hop_length=sampling_rate // 2,
        win_length=sampling_rate,
        window=window,
        center=False,
        normalized=False,
        onesided=True,
        return_complex=True,
    ).abs()
    return spectral.reshape(
        batch_size,
        n_channels,
        spectral.shape[-2],
        spectral.shape[-1],
    )


def discover_checkpoint(repo_dir):
    """Choose a tokenizer checkpoint, preferring multi-dataset pretraining."""

    root = Path(repo_dir)
    candidates = []
    for suffix in ("*.pt", "*.pth", "*.ckpt"):
        candidates.extend(root.rglob(suffix))
    candidates = [path for path in candidates if path.stat().st_size > 1024]
    candidates = [
        path
        for path in candidates
        if any(key in path.name.lower() for key in ("vq", "tokenizer", "token"))
        and not any(key in path.name.lower() for key in ("encoder", "classifier", "finetun"))
    ]
    if not candidates:
        raise FileNotFoundError(
            "no materialized tokenizer checkpoint found; run the notebook's Git LFS cell"
        )

    def rank(path):
        name = str(path).lower()
        return (
            "multiple_dataset" not in name and "multiple-dataset" not in name,
            "pretrain" not in name,
            len(name),
        )

    return sorted(candidates, key=rank)[0]


def _unwrap_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must contain a state dictionary")
    for key in ("state_dict", "model_state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            checkpoint = value
            break
    prefixes = ("vqvae.", "model.vqvae.", "module.")
    state = {}
    for key, value in checkpoint.items():
        clean = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if clean.startswith(prefix):
                    clean = clean[len(prefix) :]
                    changed = True
        state[clean] = value
    return state


class OfficialTFMTokenizer:
    """Frozen tokenizer with the same STFT call used by upstream inference."""

    def __init__(
        self,
        repo_dir,
        checkpoint_path,
        device="cuda",
        codebook_size=8192,
        embedding_size=64,
        sampling_rate=200,
    ):
        import torch

        repo_dir = str(Path(repo_dir).resolve())
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)
        try:
            from models.tfm_token import get_tfm_tokenizer_2x2x8
        except ImportError as error:
            raise ImportError(
                "could not import the official TFM model; verify the Colab setup dependencies"
            ) from error

        self.torch = torch
        self.repo_dir = repo_dir
        self.checkpoint_path = str(Path(checkpoint_path).resolve())
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.sampling_rate = sampling_rate
        self.codebook_size = codebook_size
        self.model = get_tfm_tokenizer_2x2x8(
            code_book_size=codebook_size,
            emb_size=embedding_size,
        )
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        state = _unwrap_state_dict(checkpoint)
        incompatible = self.model.load_state_dict(state, strict=False)
        matched = len(self.model.state_dict()) - len(incompatible.missing_keys)
        if matched <= 0:
            raise ValueError("checkpoint did not match any TFM tokenizer parameters")
        self.load_report = {
            "matched_keys": matched,
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
        }
        self.model.to(self.device).eval()

    def tokenize(self, eeg, channel_batch_size=16):
        """Tokenize one ``channels x time`` recording into ``channels x tokens``."""

        torch = self.torch
        eeg = np.asarray(eeg, dtype=np.float32)
        if eeg.ndim != 2:
            raise ValueError(f"expected channels x time EEG, got {eeg.shape}")
        outputs = []
        with torch.inference_mode():
            for start in range(0, eeg.shape[0], channel_batch_size):
                temporal = torch.from_numpy(eeg[start : start + channel_batch_size]).to(
                    self.device
                )
                batched = temporal.unsqueeze(0)
                spectral = _tfm_stft(batched, sampling_rate=self.sampling_rate)
                spectral = spectral.reshape(-1, spectral.shape[-2], spectral.shape[-1])
                _, tokens, _ = self.model.tokenize(spectral, temporal)
                outputs.append(tokens.detach().cpu().to(torch.int64))
        token_ids = torch.cat(outputs, dim=0).numpy()
        if token_ids.min(initial=0) < 0 or token_ids.max(initial=0) >= self.codebook_size:
            raise ValueError("token IDs fall outside the configured codebook")
        return token_ids.astype(np.uint16, copy=False)
