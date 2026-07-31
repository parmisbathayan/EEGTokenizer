# EEGText experiment plan

## 1. Research question

The project tests whether sentence-specific pairing supervision adds useful
language information to a pretrained EEG representation:

> Does contrastive training on natural-reading EEG and the text that elicited
> it improve retrieval of unseen sentences, and does the resulting EEG
> representation improve EEG-only sentiment classification?

This is not a claim of open-vocabulary thought decoding. It is a controlled
transfer study using observed reading EEG, known stimulus text, frozen
pretrained encoders, and a small trainable bridge.

## 2. Motivation

NeuroLM aligns its neural tokenizer with the distribution of language-model
embeddings through domain-adversarial training. The published method calls this
space-wise rather than embedding-wise alignment because it does not require
paired EEG and text. EEGText tests the missing pairwise question directly:
whether the EEG recorded for sentence X can be made closer to the representation
of sentence X than to representations of other sentences.

The existing sentiment experiments remain evidence about EEG-only transfer.
They are not rewritten or treated as preliminary tuning runs for this project.

## 3. Data scope

### 3.1 Initial sources

- ZuCo 1.0 Task 1 / SR: 400 sentiment-labelled movie-review sentences. Existing
  files are reused.
- ZuCo 1.0 Task 2 / NR: normal reading of Wikipedia relation sentences.
- ZuCo 1.0 Task 3 / TSR: task-specific reading of Wikipedia relation sentences.
- TECO: included only after the exact release, license, montage, sample rate,
  storage format, and overlap with ZuCo are identified.

Task 3 represents a different cognitive instruction from normal reading. Its
records will remain identifiable by task and will not silently be pooled with
normal-reading data. Duplicate text across tasks is assigned one global text
group.

### 3.2 Independent unit

The independent evaluation unit is a unique normalized sentence. Multiple
readers and repeated task presentations are dependent views of the same text.
Every view of a sentence stays in the same train, validation, or test partition.

### 3.3 Manifest before modeling

Each reader/sentence recording receives a deterministic row containing source,
task, subject, ordinal, text hash, sample shape, duration, label availability,
usability, and exclusion reason. The manifest is the source of truth for every
later cache and split fingerprint.

## 4. Leakage policy

For an outer sentiment fold:

- external unlabelled EEG-text pairs may be used only when their normalized text
  does not match a held-out sentiment sentence;
- EEG-text pairs from sentiment training sentences may be used for contrastive
  training;
- neither text nor paired representations from sentiment test sentences may be
  used for training or early stopping;
- duplicate sentences across task, release, or dataset are grouped before any
  split is created;
- normalization and cache statistics are fitted using training data only when
  they depend on the sample distribution.

Pretraining on the text of a held-out sentiment sentence would be transductive
and could expose its sentiment through the language model. The implementation
will reject such a split rather than merely warn.

## 5. Version sequence

### D0: corpus inventory and audit

No model is loaded. Inventory official remote metadata, audit local/Drive files,
identify duplicates, quantify coverage, and save a corpus fingerprint. Acceptance
requires explainable counts, valid shapes, stable text hashes, and no unresolved
cross-split duplicate policy.

### V0: frozen zero-shot retrieval

Use frozen NeuroLM EEG representations and frozen GPT-2 text representations
without a learned bridge. Evaluate whether the pretrained space alone retrieves
paired text above controlled baselines.

### V1: sentence-level contrastive adapter

Freeze NeuroLM and GPT-2. Train an identity-initialized residual adapter and a
temperature parameter using symmetric EEG-to-text and text-to-EEG contrastive
loss. Sample one reader per sentence per epoch; use all readers for evaluation.

The primary control trains the same adapter after permuting pairings inside
dataset, task, and length strata. A duration/text-length retrieval baseline and
noise-EEG diagnostic test shortcut learning.

### V2: EEG-only sentiment transfer

Within each existing sentence-held-out sentiment fold, compare representations
from aligned contrastive pretraining, shuffled-pair pretraining, and no
contrastive pretraining. Train the same small sentiment head for each setup.
No stimulus text is available at inference.

### Possible V3: word/fixation-level alignment

Word-level work is not automatic. It requires a separate locked protocol after
the data audit confirms fixation fields and after V1 shows alignment-specific
sentence retrieval under length-matched candidates.

## 6. Representation plan

The first extraction will preserve the official NeuroLM representation and
avoid adding a projection solely to match dimensions. The exact fixed pooling
position will be audited against the official model before V0 is locked.

The trainable V1 bridge is a residual domain adapter, not a replacement EEG
tokenizer:

```text
z_eeg_aligned = normalize(z_eeg + adapter(z_eeg))
```

The provisional adapter is `768 -> 32 -> 768`, with its output projection
zero-initialized. Encoder and language-model weights remain frozen for the
primary experiment.

## 7. Retrieval evaluation

Report pair verification, Recall@1, Recall@5, mean reciprocal rank, and median
rank. Controlled candidate sets will be formed within dataset/task and matched
approximately on word count and recording duration. Evaluation starts with
2-way, 5-way, and 10-way retrieval before any larger candidate pool.

Reader probabilities or similarities are averaged into one sentence-level
prediction. Confidence intervals resample sentences, not reader recordings.

## 8. Sentiment evaluation

Reuse the current five sentence-stratified folds and seeds 42, 52, and 62 where
compatible. Report macro-F1, balanced accuracy, class-wise F1, out-of-fold
predictions, and paired sentence-bootstrap intervals.

The scientifically important comparisons are:

1. aligned contrastive pretraining versus shuffled contrastive pretraining;
2. aligned contrastive pretraining versus no contrastive pretraining;
3. each EEG setup versus the majority reference.

Thresholds and multiplicity correction will be fixed after D0 establishes the
eligible corpus size, but before any V1 result is observed.

## 9. Engineering requirements

- Colab-first execution; no heavyweight local training or data download.
- Minimal notebooks that configure paths and invoke `.py` entry points.
- Atomic subject-level caches and resumable fold-level results.
- Cache signatures bind corpus fingerprint, preprocessing, checkpoint hash,
  source revision, model configuration, and package versions.
- Unit tests cover classic/HDF5 MATLAB routing, orientation, normalization,
  duplicate grouping, split leakage, contrastive labels, resumption, and cache
  rejection.
- Raw data, checkpoints, caches, and results remain ignored by Git.

## 10. Milestones

1. Scaffold and download-free data audit.
2. Run OSF inventory in Colab and lock exact downloadable files and sizes.
3. Download selected files to Drive and complete the corpus audit.
4. Implement and validate frozen representation caches.
5. Run V0 zero-shot retrieval.
6. Lock and run V1 aligned/shuffled contrastive training.
7. Lock and run V2 leakage-safe sentiment transfer.
8. Record a stop/continue decision before considering word-level work.
