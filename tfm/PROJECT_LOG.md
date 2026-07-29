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

| Component | V1: frozen histogram | V2: frozen token map | V3: frozen official encoder | V4: adapted official encoder |
| --- | --- | --- | --- | --- |
| Status | Complete — gate failed | Complete — gate failed | Complete — gate failed | Complete — gate failed; branch closed |
| TFM tokenizer | Frozen | Frozen | Frozen; reuses V1 token IDs | Frozen; reuses V1 token IDs |
| TFM codebook/embedding | Used only to produce token IDs | Frozen 8,192 × 64 embedding table | Part of the frozen official MTP encoder | Official encoder token embedding is initialized from MTP and trainable |
| Classifier input | One 8,192-bin token histogram per sentence | Full 104-channel × variable-time token map per reader | One 64-value pretrained-encoder feature per sentence | Full 104-channel × variable-time token map for a sampled reader |
| Temporal order | Discarded | Preserved through shared temporal convolutions | Contextualized inside each channel group by the official encoder | Contextualized and adapted inside each channel group by the official encoder |
| Electrode identity | Discarded | Preserved with learned channel-position embeddings | Fixed consecutive 16-channel groups; channel-count-weighted group mean | Same fixed groups plus a trainable group-aware mixer initialized to V3 weighting |
| Reader handling during training | Normalized reader histograms averaged before classification | Reader recordings remain separate with equal total loss per sentence | Frozen reader features averaged before the linear probe | One reader sampled per sentence per epoch; reader resampled each epoch |
| Reader handling during testing | One already-averaged sentence feature | Reader probabilities averaged into one sentence prediction | One already-averaged sentence feature | All reader probabilities averaged into one sentence prediction |
| Trainable model | TF-IDF plus balanced logistic regression | Approximately 24k-parameter temporal CNN, channel attention, and linear head | Standardized, class-balanced L2 logistic regression | Complete official MTP encoder plus a new three-class head |
| TFM parameters updated | None | None | None | All differentiable encoder weights; non-floating index tensors and tokenizer remain fixed |
| Evaluation unit | Sentence | Sentence | Sentence | Sentence |
| Generalization claim | Unseen sentences for the known reader pool | Unseen sentences for the known reader pool | Unseen sentences for the known reader pool | Unseen sentences for the known reader pool |
| Main aligned macro-F1 | 0.3116 | 0.2233 ± 0.0647 across folds | 0.3132 ± 0.0481 across folds | 0.2870 ± 0.0514 across folds |
| Shuffled macro-F1 | 0.3401 | 0.2549 ± 0.0463 across folds | 0.3391 ± 0.0667 across folds | 0.3004 ± 0.0712 across folds |
| Aligned minus shuffled | −0.0285; negative for all seeds | −0.0315; negative for all seeds | −0.0259; negative for all seeds | −0.0134; positive for 1 of 3 seeds |
| Decision | Do not advance on V1 | No evidence of useful aligned signal; proceed only to predefined V3 | Frozen encoder did not transfer | Adaptation did not transfer; stop with no V5 |

## 2026-07-28 — Replace thousands of Drive reads with resumable subject packs

The compact-dtype fix removed the RAM excess, but the next free-Colab attempt
remained at `codebook_loaded` because reading 4,532 individual Drive files was
I/O-bound. Added an exact packed-cache layer under
`CachedArtifacts/eeg_tokenizer/tfm/token_records_v2_packed`.

The first run reads up to eight source files concurrently and atomically writes
one compressed pack per subject. A disconnect loses at most the subject being
packed; completed subject files are reused. Subsequent runtimes load about 12
packed files. The packs use flat `uint16` token storage plus offsets and lengths,
so variable time axes are reconstructed without padding or changed values. The
dataset fingerprint is recomputed from the reconstructed arrays and remains
part of the evaluation run signature. This is an execution/storage optimization
only; the V2 model, folds, seeds, weights, controls, and gate are unchanged.

## 2026-07-28 — Preserve completed V2 fits through final-summary dtype fix

The first complete V2 evaluation wrote all 45 fold/setup metric rows: 30 neural
fits and 15 majority baselines. It then stopped during the final paired
bootstrap because prediction columns accumulated from an initially empty pandas
frame retained `object` dtype, which the installed scikit-learn target checker
does not accept even when every value is an integer.

Metric and bootstrap inputs are now explicitly normalized to one-dimensional
`int64` arrays. Cell 4 also reloads the project module after Cell 1 pulls GitHub,
so a resumed runtime cannot silently keep an older imported implementation. The
run signature is unchanged. Rerunning Cell 4 reuses all 45 saved rows and performs
only final aggregation, bootstrap, gate evaluation, and canonical result writes.

## 2026-07-29 — V2 completed and failed the locked gate

The resumed run reused every saved fold and completed the final aggregation.
All canonical results were written under
`Results/eeg_tokenizer/tfm/token_map_v2`. The evaluation covers 4,532 usable
recordings, 400 sentences, and 12 subjects; each sentence has 8–12 readers.

| Result | Aligned token map | Shuffled control | Majority |
| --- | ---: | ---: | ---: |
| Accuracy, fold mean | 0.3433 | 0.3442 | 0.3500 |
| Balanced accuracy, fold mean | 0.3348 | 0.3385 | 0.3333 |
| Macro-F1, fold mean ± SD | 0.2233 ± 0.0647 | 0.2549 ± 0.0463 | 0.1728 ± 0.0000 |

| Locked criterion | Required | Observed | Pass |
| --- | ---: | ---: | :---: |
| Aligned balanced accuracy above chance | > 0.3333 | 0.3348 | Yes |
| Aligned − shuffled macro-F1 | ≥ +0.0150 | −0.0315 | No |
| Seeds with positive aligned − shuffled delta | ≥ 2 of 3 | 0 of 3 | No |
| Corrected 98.33% bootstrap lower bound | > 0 | −0.0789; interval [−0.0789, 0.0032] | No |
| Aligned macro-F1 above majority | > 0.1728 | 0.2233 | Yes |

The per-seed aligned-minus-shuffled macro-F1 differences were −0.0221,
−0.0547, and −0.0178 for seeds 42, 52, and 62. Aligned V2 won 5 of 15
fold-level comparisons, tied 1, and lost 9. The classifier selected very early
epochs (median best epoch 2), and its mean class F1s were 0.0843, 0.3678, and
0.2179 for labels −1, 0, and +1, showing particularly weak recovery of the
negative class.

The tiny 0.0015 excess over the balanced-accuracy chance line is not persuasive:
accuracy remained below the majority baseline, shuffled tokens performed better
on average, every seed delta was negative, and the corrected interval did not
exclude zero. V2 therefore provides no evidence that preserving frozen TFM token
channel/time structure makes the tokens useful for this sentiment task. Per the
predeclared stopping plan, no V2 tuning is authorized; only the predefined V3
remains.

## 2026-07-29 — Prepare final V3 frozen official-encoder probe

Prepared the last of the three predefined TFM transfer versions. V3 reuses the
exact V1 token maps and the compact packed-cache path added for V2. It loads the
authors' MTP-pretrained 64x4 TFM encoder, excludes its obsolete pretraining task
head, freezes every upstream parameter, and captures the 64-dimensional feature
immediately before that head.

A complete 104-channel ZuCo token map would exceed the official encoder's 2,048
sequence-length limit. The fixed montage adapter therefore uses six consecutive
16-channel groups and one eight-channel tail. All seven groups use the same
official encoder; their outputs are averaged in proportion to channel count so
all 104 electrodes contribute once. Reader features are then averaged within a
sentence.

Only a standardized, class-balanced L2 logistic regression is trained. Its
regularization is selected by a three-fold inner search within each of the same
five outer unseen-sentence folds and seeds 42/52/62. The aligned model is compared
with a split-local shuffled feature control and a majority baseline under V2's
unchanged five-part gate and three-version-corrected 98.33% bootstrap interval.

The expensive frozen inference is resumable by subject: each completed shard is
written atomically under `encoder_features_v3` and is validated against the token
dataset fingerprint, encoder SHA-256, and extraction configuration. The entire
4,532 by 64 float32 feature matrix is only about 1.1 MiB, so the limiting cost is
encoder runtime rather than retained feature memory. V3 has no automatic tuning
follow-up; pass or fail, it concludes the bounded TFM sequence.

## 2026-07-29 — V3 completed and failed the locked gate

The Colab run reached `evaluation_complete` and wrote the complete result set
under `Results/eeg_tokenizer/tfm/encoder_probe_v3`. The official encoder loaded
57 checkpoint keys with no unexpected keys; the only missing weights were the
intentionally excluded classification head, and all encoder parameters remained
frozen. Frozen features were produced for 4,532 recordings, 400 sentences, and
12 subjects. All 64 feature dimensions had nonzero variance.

| Result | Aligned encoder probe | Shuffled control | Majority |
| --- | ---: | ---: | ---: |
| Accuracy, fold mean | 0.3150 | 0.3408 | 0.3500 |
| Balanced accuracy, fold mean | 0.3162 | 0.3400 | 0.3333 |
| Macro-F1, fold mean ± SD | 0.3132 ± 0.0481 | 0.3391 ± 0.0667 | 0.1728 ± 0.0000 |

| Locked criterion | Required | Observed | Pass |
| --- | ---: | ---: | :---: |
| Aligned balanced accuracy above chance | > 0.3333 | 0.3162 | No |
| Aligned − shuffled macro-F1 | ≥ +0.0150 | −0.0259 | No |
| Seeds with positive aligned − shuffled delta | ≥ 2 of 3 | 0 of 3 | No |
| Corrected 98.33% bootstrap lower bound | > 0 | −0.0702; interval [−0.0702, 0.0226] | No |
| Aligned macro-F1 above majority | > 0.1728 | 0.3132 | Yes |

The per-seed aligned-minus-shuffled macro-F1 differences were −0.0415,
−0.0194, and −0.0169 for seeds 42, 52, and 62. Aligned V3 won 6 of 15
fold-level comparisons and lost 9. Its macro-F1 was effectively the same as V1
(0.3132 versus 0.3116), while its aligned accuracy was below both shuffled and
majority performance. The frozen official representation therefore did not
recover stable sentence-aligned sentiment information from these ZuCo tokens.

All three predefined versions produced negative aligned-minus-shuffled effects.
Per the bounded plan, the TFM transfer sequence stops here without a V4 or
post-hoc model search. This is evidence about this fixed cross-domain transfer
setup, not a claim that TFM fails on its original clinical tasks or that no EEG
representation can carry sentiment information.

## 2026-07-29 — Clarify the bounded count and prepare final V4 adaptation

After V3, the user clarified that the earlier request for “only three different
versions to try” meant three follow-ups after the V1 baseline: V2, V3, and V4.
V4 is therefore the explicitly final extension. This clarification is recorded
after seeing V1–V3, rather than being retroactively described as a preregistered
choice. Its protocol is locked here before its result; no V5 or V4 variation will
follow a failure.

V4 tests the main unresolved mechanistic alternative: perhaps the MTP-pretrained
clinical encoder is a useful initialization but its frozen representation cannot
express ZuCo reading sentiment. The tokenizer and discrete V1 tokens stay fixed,
while every differentiable weight in the official 64x4 encoder and a new
three-class head are optimized within each training fold. Any integer-valued
index parameter is explicitly reported and remains non-trainable because PyTorch
does not define gradients for integer tensors. The encoder uses a conservative
`1e-5` learning rate; the randomly initialized head uses `3e-4`.

To keep full adaptation feasible on free Colab without letting subjects with more
recordings dominate, every epoch samples exactly one reader for each training
sentence and resamples the reader next epoch. Validation and test inference use
every available reader and average softmax probabilities to one sentence result.
The six 16-channel groups and eight-channel tail remain unchanged. A small
group-aware mixer is initialized to reproduce the V3 channel-count-weighted logits
exactly, then becomes trainable so the seven global montage regions do not remain
exchangeable during adaptation.

The shuffled control is trained from the same MTP checkpoint and random head
initialization. It uses a no-fixed-point, one-to-one permutation of complete
sentence recording sets separately inside fit, validation, and test partitions.
Consequently no EEG crosses a split and no shuffled target remains paired with
its own EEG. Outer test sentences are excluded from both gradient updates and
early stopping.

V4 retains five sentence-stratified folds and seeds 42/52/62. It writes metrics,
predictions, and training histories atomically after each of the 30 aligned or
shuffled setup/fold fits, so a disconnect loses only the active fit. A preliminary
forward/backward smoke test requires finite gradients in both encoder and head.
The run signature binds the dataset fingerprint, checkpoint SHA-256, official
source revision, runtime package versions, and complete configuration.

The five gate criteria and `+0.015` minimum effect remain unchanged. V4 changes
the multiplicity count from three to four and therefore uses a conservative
98.75% paired bootstrap interval (`0.05 / 4`). Pass or fail, this run closes the
TFM transfer branch.

## 2026-07-29 — V4 completed, failed the final gate, and closed the branch

The Colab run reached `evaluation_complete` and wrote the complete canonical
result set under `Results/eeg_tokenizer/tfm/encoder_finetune_v4`: 45 fold/setup
metric rows, 30 neural fits, all out-of-fold predictions, 196 epoch-history rows,
the corrected bootstrap, and the final gate decision. It used all 4,532 cached
recordings representing 400 sentences and 12 subjects.

| Result | Aligned encoder fine-tune | Shuffled control | Majority |
| --- | ---: | ---: | ---: |
| Accuracy, fold mean | 0.3375 | 0.3458 | 0.3500 |
| Balanced accuracy, fold mean | 0.3368 | 0.3457 | 0.3333 |
| Macro-F1, fold mean ± SD | 0.2870 ± 0.0514 | 0.3004 ± 0.0712 | 0.1728 ± 0.0000 |

| Locked criterion | Required | Observed | Pass |
| --- | ---: | ---: | :---: |
| Aligned balanced accuracy above chance | > 0.3333 | 0.3368 | Yes |
| Aligned − shuffled macro-F1 | ≥ +0.0150 | −0.0134 | No |
| Seeds with positive aligned − shuffled delta | ≥ 2 of 3 | 1 of 3 | No |
| Corrected 98.75% bootstrap lower bound | > 0 | −0.0510; interval [−0.0510, 0.0417] | No |
| Aligned macro-F1 above majority | > 0.1728 | 0.2870 | Yes |

The per-seed aligned-minus-shuffled macro-F1 differences were +0.0307,
−0.0152, and −0.0558 for seeds 42, 52, and 62. Aligned V4 won 8 of 15
fold-level comparisons and lost 7, but its losing margins were larger, leaving
both the overall effect and two of three seed effects negative. The paired
bootstrap distribution had mean −0.0042 and its four-version-corrected interval
comfortably crossed zero.

The implementation audits rule out the earlier detached/non-floating-parameter
failure. The official checkpoint loaded 57 encoder keys; only the intentionally
replaced classification-head weights were missing, with no unexpected keys. All
724,608 floating encoder parameters were trainable, while the single reported
16-element integer index remained fixed as required by PyTorch. Every neural fit
recorded a nonzero encoder update (L2 range 0.0372–0.1353, mean 0.0840). Fits ran
4–12 epochs with a mean of 6.53, so the result is not a frozen-encoder artifact,
a detached computation graph, or an incomplete evaluation.

The small `0.0035` balanced-accuracy excess over nominal chance is not evidence
of useful transfer: aligned accuracy remained below the majority baseline,
shuffled training performed better on every aggregate metric, the required
effect had the wrong sign, and the corrected interval did not exclude zero.
Full supervised adaptation therefore did not unlock stable sentence-aligned
sentiment information in this fixed TFM-to-ZuCo setup. In accordance with the
bounded plan, there will be no V5 or post-hoc V4 variation. This conclusion is
limited to the tested transfer protocol and does not claim that TFM fails on its
source clinical tasks or that EEG cannot encode sentiment under other methods.
