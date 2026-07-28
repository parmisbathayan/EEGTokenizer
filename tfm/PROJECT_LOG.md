# TFM transfer project log

This log is chronological. It records motivation, implementation decisions,
problems, and observed results. New entries should be appended at the bottom.

## 2026-07-28 — Define the first pretrained-tokenizer experiment

### Starting point

Earlier ZuCo experiments used 2,496 sentence-level classical EEG features. The
best text+EEG model did not show an alignment-specific gain: aligned, shuffled,
noise, and zero-EEG controls performed almost identically. The next experiment
therefore needs a genuinely different EEG representation rather than another
fusion head over the same summaries.

The TFM paper learns discrete time-frequency motifs directly from raw EEG and
publishes pretrained tokenizer weights. Its single-channel operation is a good
structural match for ZuCo's 105-channel montage, even though its pretraining data
is clinical rather than natural reading EEG.

### Scope decision

- Start with ZuCo Task 1 only: 400 labelled English sentiment sentences read by
  12 subjects.
- Use sentence-level `rawData` at 500 Hz.
- Freeze the official multi-dataset TFM tokenizer.
- Remove the flat Cz reference channel at index 104.
- Match the paper's 200 Hz rate and STFT path.
- Cache tokens in Google Drive so extraction is resumable.
- Test token histograms with logistic regression before using the pretrained
  downstream Transformer or any text modality.
- Keep sentence IDs as the cross-validation unit.
- Compare aligned EEG against split-local shuffled EEG and a majority baseline.

### Implementation

Created a self-contained `tfm/` experiment folder matching the structure of the
other thesis codebases:

- `src/zuco_io.py` streams both classic and HDF5 MATLAB variants and matches
  sentences to the established labels CSV.
- `src/preprocess.py` validates orientation, removes Cz, interpolates small NaN
  gaps, resamples to 200 Hz, and applies the paper-compatible filters.
- `src/official_tfm.py` imports the official model, loads a frozen checkpoint,
  and uses the upstream `get_stft_torch` function.
- `src/extraction.py` writes one compressed token cache per subject/sentence and
  preserves errors in an extraction manifest instead of losing a long Colab run.
  Cache writes are atomic, so a Colab disconnect cannot leave a partial file that
  is mistaken for a completed recording.
- `src/features.py` creates equal-weight subject histograms and token-collapse
  diagnostics.
- `src/evaluation.py` performs nested stratified cross-validation with multiple
  seeds and saves out-of-fold predictions.
- `notebooks/tfm_zuco_colab.ipynb` is the minimal runnable interface.

The notebook shallow-clones upstream with Git LFS smudging disabled, then
materializes only the selected tokenizer checkpoint. It prints the checkpoint's
exact LFS size before the cloud download. No upstream code, weight, raw EEG,
cache, or result is committed here.

### Status

Code preparation is complete. No scientific result has been recorded yet; the
first full extraction and evaluation still need to run in Colab.

### First-run acceptance checks

- All 12 subject files are found and close to 400 sentences match the labels.
- A smoke-test recording produces a two-dimensional `channel x token` array.
- Token IDs are inside `[0, 8192)`.
- The checkpoint load report has a nonzero matched-key count and no suspiciously
  large missing-key set.
- Codebook use is not collapsed to a handful of tokens.
- Aligned performance is evaluated against the shuffled control before any text
  fusion is attempted.

## 2026-07-28 — Align paths with the existing Drive organization

Inspected the actual Google Drive thesis folders and the current multimodal
project convention. Updated the Colab notebook to use:

- raw data and labels from `MyDrive/Thesis/Data`;
- reusable token caches under
  `MyDrive/Thesis/CachedArtifacts/eeg_tokenizer/tfm/tokens_v1`;
- run outputs under
  `MyDrive/Thesis/Results/eeg_tokenizer/tfm/tfm_histogram_v1`.

The `eeg_tokenizer/tfm` nesting keeps TFM artifacts separate while leaving room
for a sibling folder for the other tokenizer paper.

## 2026-07-28 — Remove an unnecessary `pyhealth` inference dependency

The first Colab smoke test showed that importing the upstream general-purpose
`utils.utils` module also imported `pyhealth`, even though frozen tokenizer
inference only needed its STFT helper. Installing `pyhealth` would add a large,
irrelevant dependency surface.

Replaced that import with a local implementation of the paper's declared STFT
parameters: a 200-sample Hann window at 200 Hz, 100-sample hop, magnitude-only
one-sided output, and `center=False`. The official tokenizer architecture and
checkpoint are still used unchanged. Added a Colab-aware regression test for
the expected frequency/time dimensions. The notebook setup cell now also pulls
fast-forward GitHub updates when `/content/EEGTokenizer` already exists, so a
runtime can receive fixes without manually deleting the checkout.

The next Colab smoke test exposed a PyTorch 2.x interface constraint:
`torch.stft` accepted one- or two-dimensional inputs, not the three-dimensional
`batch x channels x time` tensor. Updated the helper to flatten batch and channel
into one batch axis before the STFT and restore both axes afterward. This leaves
the per-channel transform unchanged and supports current Colab PyTorch.
