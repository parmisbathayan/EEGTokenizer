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

## 2026-07-28 — Explicitly reject truncated ZuCo recordings

The initial Drive inspection displayed apparent 81- and 39-channel recordings
for ZDN and ZJM. These were actually very short `105 x 81` and `105 x 39`
recordings that the generic smaller-axis heuristic had transposed. At 500 Hz,
they contain only 0.162 and 0.078 seconds and cannot form a TFM window.

Updated orientation to identify the known 105-channel ZuCo axis first. Added
explicit preprocessing requirements for 105 input channels and at least 500
source samples before channel removal or filtering. The inspection table now
reports `usable_for_tfm`, `too_short`, and `unexpected_channels` separately, and
the notebook prints label-matched, raw-array, and usable totals without conflating
them. Added tests for both orientations and explicit short-record rejection.

## 2026-07-28 — Make full extraction independent and observable

The full extraction cell previously reused the tokenizer and preprocessing
objects created by the optional one-recording smoke test. It now initializes
both objects itself, so after setup and checkpoint preparation the smoke test
can be skipped. Extraction also prints written, reused, and failed counts every
100 recordings. Existing behavior remains resumable: each successful cache is
written immediately and is reused on the next run.

## 2026-07-28 — Persist all diagnostics and the viability decision

The Colab notebook now saves the sentence metadata, recording-level token
diagnostics, descriptive diagnostic table, aggregate diagnostic JSON, and token
quality figure during Cell 7. Cell 8 saves the paired bootstrap result as
`alignment_delta.json`, matching the command-line workflow. Cell 9 saves the
thresholds, observed scores, bootstrap interval, pass/fail value, and decision
text in `viability_gate.json`. These additions do not rerun token extraction or
change the experiment; they complete its auditable result record.

## 2026-07-28 — Prepare bounded V2 frozen token-map probe

V1 completed with mean aligned macro-F1 `0.3116` versus shuffled `0.3401`;
the aligned-minus-shuffled delta was negative for all three seeds. The primary
metrics, predictions, configuration, and token cache remain saved, so the lost
Colab session does not justify rerunning V1.

Prepared the first of three predeclared follow-ups. V2 reuses the 4,532 cached
subject/sentence token arrays and the exact official frozen codebook. It retains
the channel-by-time structure and trains a compact shared temporal CNN, learned
channel positions, channel attention, and a three-class head. Reader recordings
remain separate during training, with equal total loss per sentence, and their
probabilities are averaged only for held-out sentence evaluation.

The evaluation preserves V1's five sentence-stratified folds and seeds
42/52/62, adds a separately trained split-local shuffled control, and applies
the locked multi-criterion gate with a three-version-corrected 98.33% paired
bootstrap interval. Partial predictions and histories are written before each
setup/fold completion marker so Colab restarts safely resume. V2 has its own
Drive results folder and does not overwrite V1.

Simplified the runnable V2 notebook to a single ordered path with five code
cells. Removed the optional duplicate cache/codebook check and its conditional
reuse logic. V2 now reads the codebook tensor directly from the verified
official checkpoint, so it no longer imports the upstream tokenizer model or
installs V1's inference-only packages. The remaining upstream shallow clone is
used only to discover the authors' selected checkpoint and LFS size. All other
cells are necessary for paths, persistent checkpoint caching, evaluation, or
saved-result review.

The first V2 Colab attempt disconnected before `run_signature.json` was written,
so no fold had begun. The cache loader had unnecessarily promoted every stored
`uint16` token map to `int64` while retaining all 4,532 arrays. It now preserves
the compact `uint16` representation in host memory and converts only the active
minibatch to `torch.long`, reducing retained token-array RAM by 75%. Cell 4 now
prints progress every 500 recordings and atomically writes `runtime_status.json`
after codebook loading, token-cache loading, evaluation start, and completion.

## Maintained version comparison

Keep this table updated whenever a version is prepared or completed. It is the
canonical quick comparison; detailed implementation and result notes remain in
the dated entries above and below it.

| Component | V1: frozen histogram | V2: frozen token map |
| --- | --- | --- |
| Status | Complete — gate failed | Prepared; Colab evaluation pending |
| TFM tokenizer | Frozen | Frozen |
| TFM codebook | Used only to produce token IDs | Frozen 8,192 × 64 embedding table |
| Classifier input | One 8,192-bin token histogram per sentence | Full 104-channel × variable-time token map per reader |
| Temporal order | Discarded | Preserved through shared temporal convolutions |
| Electrode identity | Discarded | Preserved with learned channel-position embeddings |
| Reader handling during training | Normalized reader histograms averaged before classification | Reader recordings remain separate with equal total loss per sentence |
| Reader handling during testing | One already-averaged sentence feature | Reader probabilities averaged into one sentence prediction |
| Trainable model | TF-IDF plus balanced logistic regression | Approximately 24k-parameter temporal CNN, channel attention, and linear head |
| TFM parameters updated | None | None |
| Evaluation unit | Sentence | Sentence |
| Generalization claim | Unseen sentences for the known reader pool | Unseen sentences for the known reader pool |
| Main aligned macro-F1 | 0.3116 | Pending |
| Shuffled macro-F1 | 0.3401 | Pending |
| Aligned minus shuffled | −0.0285; negative for all seeds | Pending |
| Decision | Do not advance on V1 | Apply locked V2 gate after all folds |
