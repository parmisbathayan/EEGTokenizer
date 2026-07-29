# NeuroLM transfer project log

This log is chronological. Append implementation changes, problems, results,
and decisions at the bottom. Keep the version-comparison table current.

## 2026-07-28 — Define a bounded EEG-only NeuroLM test

### Starting point

Earlier thesis experiments found no alignment-specific value in 2,496 pooled
classical ZuCo EEG features. The text+EEG model scored similarly with aligned,
shuffled, noise, and zero EEG, so another fusion head over the same summaries is
not justified.

NeuroLM is relevant as a pretrained EEG representation, but its full published
system targets unified instruction-following across clinical and BCI datasets.
Its text alignment is distribution-level and does not pair a reading EEG sample
with the sentence that elicited it. GPT-2 generation therefore does not directly
answer whether reading EEG contains sentiment information.

### Scope decision

- Test ZuCo Task 1 only: about 400 labelled English sentiment sentences and 12
  readers.
- Use no stimulus text and no language-model output.
- Extract only the neural encoder from the official pretrained NeuroLM-B
  checkpoint; do not instantiate GPT-2.
- Freeze every NeuroLM parameter.
- Convert raw sentence EEG to the paper's 200 Hz, one-second patch format.
- Pool encoder outputs using a fixed, non-learned mean/std/temporal-slope rule.
- Average readers only after reader-level encoding.
- Evaluate unique sentences with aligned, split-local shuffled, and majority
  setups across five folds and seeds 42/52/62.

This isolates the cheapest scientifically useful question: whether the learned
NeuroLM EEG representation transfers before any large downstream architecture is
trained.

### Montage decision

The official NeuroLM loader uses a named 10–20-family channel vocabulary. ZuCo's
first 104 usable channels are retained EGI HydroCel sensors; its 105th exported
channel is the flat Cz reference.

The implementation removes Cz and assigns the 104 EGI sensors one-to-one to the
nearest 104 NeuroLM-supported positions using spherical coordinates bundled with
MNE and the Hungarian algorithm. The mapping and angular errors are saved for
audit. Reusing a target position is prohibited. This avoids silently treating
arbitrary EGI indices as NeuroLM channel identities, but it remains an explicit
cross-montage approximation.

### Implementation

Created the self-contained `neurolm/` project:

- `src/zuco_io.py` loads classic and HDF5 MATLAB exports and matches the fixed
  sentiment labels.
- `src/preprocess.py` repairs small gaps, removes Cz, filters, resamples, keeps
  complete one-second windows, and applies the official `/100` input scaling.
- `src/channels.py` constructs and audits the spatial assignment.
- `src/official_neurolm.py` extracts only `tokenizer.*` weights from
  `NeuroLM-B.pt`, instantiates the official encoder, and freezes it.
- `src/extraction.py` atomically caches one compact float16 feature vector per
  reader/sentence so a Colab run can resume.
- `src/features.py` averages readers equally and reports representation
  diagnostics.
- `src/evaluation.py` runs nested sentence-level CV and the paired bootstrap.
- `notebooks/neurolm_zuco_colab.ipynb` is the ordered Colab interface.

The official repository is pinned at commit
`0cda9876d8ce6ee07ed0c43eee5e9a6f5c24b177`. The official NeuroLM-B checkpoint
is approximately 2.38 GB and is downloaded only to Google Drive from Colab.

### Locked viability gate

V1 advances only if aligned minus shuffled fold-mean macro-F1 is at least
`+0.015`, at least two of three seed-level OOF deltas are positive, and the lower
bound of a 98.33% paired bootstrap interval is above zero. The corrected interval
predeclares room for at most three documented NeuroLM versions.

### Status

Code prepared; Colab extraction and evaluation are pending. No scientific result
has been recorded.

## Maintained version comparison

| Component | V1: frozen pooled NeuroLM | V2: raw EEGNet | V3: structured frozen NeuroLM | V4: NeuroLM plus GPT-2 |
| --- | --- | --- | --- | --- |
| Status | Complete — yellow/inconclusive | Prepared — pending Colab run | Prepared — pending Colab run | Planned only |
| Input | NeuroLM channel/time embeddings | Native raw EGI EEG | Factorized NeuroLM channel and per-second sequence | Official full NeuroLM EEG sequence plus fixed instruction |
| NeuroLM source | Frozen NeuroLM-B neural encoder | Not used | Same frozen encoder | Full NeuroLM-B checkpoint |
| Text | None | None | None | Identical instruction only; no stimulus sentence |
| Temporal structure | Global mean, standard deviation and slope | Learned local temporal convolutions | Learned channel and temporal attention | GPT-2 causal attention |
| Reader handling | Feature mean before classification | Reader rows; mean held-out probabilities | Reader rows; mean held-out probabilities | Reader rows; mean label probabilities |
| Trainable model | Standardized logistic regression | One locked compact EEGNet | Small attention probe | Small adapters and label verbalizers |
| Evaluation unit | Unique sentence | Unique sentence | Unique sentence | Unique sentence |
| Aligned macro-F1 | 0.3493 | Pending | Not run | Not run |
| Shuffled macro-F1 | 0.3179 | Pending | Not run | Not run |
| Aligned minus shuffled | +0.0314 across folds | Pending | Not run | Not run |
| Decision | Effect/seed criteria pass; corrected interval crosses zero | Apply new-screen stoplight gate | Apply new-screen stoplight gate | Separate future notebook; final screen |

## 2026-07-29 — Exclude spatial outliers instead of aborting

The first Colab spatial audit produced 104 unique assignments with mean angular
distance `15.9467°` and maximum distance `32.6626°`. The original hard check
aborted because one or more assignments exceeded the conservative `30°` limit.
This was a mapping-policy failure, not a NeuroLM or EEG-data failure.

The code now preserves the locked `30°` credibility limit instead of relaxing
it. All 104 assignments remain in `spatial_mapping.csv`, with a
`use_for_encoder` column. Assignments beyond the limit are excluded from the EEG
encoder, while every retained assignment remains one-to-one. The run aborts only
if fewer than 80 credible channels remain. The exact retained count, excluded
channels, and used-distance statistics are saved in
`spatial_mapping_diagnostics.json`.

The encoder and cache signature now record both NeuroLM channel IDs and the
selected original ZuCo indices. This prevents features generated under a
different channel subset from being silently reused. No checkpoint had been
loaded and no feature extraction had begun before this change.

## 2026-07-29 — Repair Colab Hugging Face compatibility

Cell 4 successfully cached the complete official checkpoint at `2.377 GB`, then
failed before encoder construction. The notebook had pinned
`huggingface_hub==0.34.4`, while the current Colab `transformers` package imports
the newer top-level `is_offline_mode` API. This was a Python-package
compatibility error; the checkpoint and spatial mapping were unaffected.

Updated the Colab-only requirement to `huggingface_hub==1.16.4` and added an
isolated import check immediately after installation. Cell 1 now prints both Hub
and Transformers versions and explicitly requests a runtime restart if an older
Hub module is already resident in the kernel. The 2.377 GB Drive checkpoint is
reused after restarting, so this fix does not repeat the large download.

## 2026-07-29 — V1 completed with a positive but uncertain alignment effect

The full V1 extraction and evaluation completed on 4,532 reader/sentence
recordings covering 400 unique sentences. All 2,304 pooled feature dimensions
were nonconstant. The retained montage contained 102 one-to-one mapped channels;
two assignments above the locked 30-degree limit were excluded.

| Result | Aligned NeuroLM | Shuffled NeuroLM | Majority |
| --- | ---: | ---: | ---: |
| Accuracy, fold mean | 0.3500 | 0.3208 | 0.3500 |
| Balanced accuracy, fold mean | 0.3494 | 0.3204 | 0.3333 |
| Macro-F1, fold mean | 0.3493 | 0.3179 | 0.1728 |

The fold-mean aligned-minus-shuffled macro-F1 was `+0.0314`, exceeding the
`+0.015` effect threshold. The OOF differences were positive for all three
seeds: `+0.0453`, `+0.0263`, and `+0.0163`. However, the corrected 98.33% paired
bootstrap interval was `[-0.0187, +0.0758]`, so its lower bound did not exceed
zero. V1 therefore failed the complete locked gate and is classified as yellow:
suggestive but not eligible for tuning.

The median cosine similarity between reader-level pooled features was `0.9897`.
That does not prove collapse because every dimension varied, but it motivates a
future structured sequence test rather than more tuning of the same global
pooling rule.

## 2026-07-29 — Authorize a bounded three-version exploratory screen

After reviewing V1, the user explicitly authorized exactly three additional
versions that span different representations and classifiers. This supersedes
V1's automatic stop only by adding the predeclared broad screen; it does not
authorize tuning V1 or an open-ended search.

- V2 tests native raw EEG with a compact EEGNet.
- V3 will test unpooled frozen NeuroLM channel/time embeddings with a small
  attention probe.
- V4 will test the official full NeuroLM-B EEG-to-GPT-2 route using only a fixed
  instruction and three sentiment label verbalizers.

The three new versions form an exploratory screening family with a 98.33%
per-version paired-bootstrap interval. Only a green version may later be tuned,
and only after all three screens are complete. If none is green, the branch
stops. Any selected winner still requires an independently locked confirmation.

## 2026-07-29 — Prepare V2 raw-EEG EEGNet

V2 is implemented as a new code path and a separate notebook,
`notebooks/neurolm_raw_eegnet_v2_colab.ipynb`. The V1 notebook and V1 Drive
directories remain unchanged.

V2 reuses only the raw-data reader, fixed labels, and preprocessing. It bypasses
NeuroLM and uses all 104 retained native EGI channels. Preprocessed float16 EEG
is saved atomically as one packed file per subject under `raw_eeg_packs_v2`,
avoiding thousands of repeated Google Drive reads while allowing subject-level
resumption.

The locked model is a compact EEGNet-style network with temporal convolution,
depthwise spatial convolution across all channels, a separable temporal block,
and a three-class head. One reader recording is a training row; its one-second
window logits are averaged before loss calculation. Each sentence receives
equal total training weight, and held-out reader probabilities are averaged into
one sentence prediction. Sentence grouping occurs before readers or windows are
expanded.

The primary negative control trains an independent model after permuting whole
reader bundles inside each train, validation, and test split. An inference-only
50 ms temporal-block shuffle is saved as a secondary diagnostic and does not
enter the gate. Evaluation uses the fixed three seeds, five outer folds, one
inner validation split for early stopping, and no architecture or optimizer
search. Partial metrics, predictions, histories, and completion markers are
written after every setup/fold under `raw_eegnet_v2`.

No package, model, dataset, cache, or environment was downloaded or installed on
the Mac. Colab supplies every V2 runtime dependency; V2 downloads no external
model checkpoint.

## 2026-07-29 — Prepare V3 while V2 evaluates

V3 was locked and implemented before observing V2's result. Waiting for V2 would
not provide a scientific advantage because the versions test different
hypotheses; using its score to redesign V3 would instead turn the bounded screen
into adaptive model search.

The dedicated notebook is
`notebooks/neurolm_structured_probe_v3_colab.ipynb`. It can begin after V2 has
finished creating `raw_eeg_packs_v2`; V2's neural evaluation can continue in a
separate Colab runtime. V3 reuses those preprocessed subject packs and the
existing 2.38 GB NeuroLM-B checkpoint without reading V2 metrics.

The official frozen encoder now exposes the exact `seconds × mapped channels ×
embedding` tensor that was already used internally by V1 pooling. V1's public
feature result is unchanged and a regression test verifies that pooling the new
structured output reproduces the original V1 vector.

To avoid a many-gigabyte full Cartesian cache, V3 saves two float16 views per
reader: 102 channel tokens formed by averaging over seconds, and a variable
number of second tokens formed by averaging over channels. This preserves
spatial identity and temporal order separately while explicitly discarding
their exact interaction. One atomic pack is written per subject under
`structured_features_v3`; the expected total is roughly 0.7-1.0 GB depending on
recording lengths and compression.

The locked approximately 133k-parameter probe shares a `768 → 96` projection,
learns separate channel and time attention, and fuses both summaries with their
absolute difference and elementwise product. NeuroLM remains fully frozen.
Reader rows receive equal total weight per sentence, and reader probabilities
are averaged only for held-out sentence evaluation.

The primary control independently permutes whole reader bundles inside every
train, validation, and test split. A secondary inference-only structure control
permutes both channel identity and time order while preserving token values.
The primary stoplight gate, seeds, folds, early-stopping rule, corrected
interval, and no-tuning policy match V2. Results are isolated under
`structured_probe_v3` and resume at setup/fold granularity.

No Mac dependency or artifact was downloaded or installed. All NeuroLM source,
checkpoint reuse, structured extraction, and PyTorch training remain Colab-only.
