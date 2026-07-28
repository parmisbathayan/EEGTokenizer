# Frozen TFM tokenizer transfer to ZuCo

This folder tests one narrow question across a bounded sequence of versions:

> Do tokens from the pretrained TFM tokenizer contain sentence-level sentiment
> information in ZuCo natural-reading EEG?

It uses the authors' official tokenizer and checkpoint as a **frozen** feature
extractor. The first classifier is intentionally only TF-IDF-weighted token
histograms plus logistic regression. A compact model makes a positive result
easier to interpret and a negative result cheaper to obtain.

The upstream paper and code are:

- J. Pradeepkumar et al., *Tokenizing Single-Channel EEG with Time-Frequency
  Motif Learning*, ICLR 2026.
- [Official TFM-Tokenizer repository](https://github.com/Jathurshan0330/TFM-Tokenizer)

## What the experiment does

```text
ZuCo raw sentence EEG (105 channels, 500 Hz)
  -> remove the flat Cz reference channel
  -> repair small NaN gaps
  -> resample to 200 Hz
  -> 0.1-75 Hz bandpass + 50 Hz notch
  -> official frozen TFM tokenizer, channel by channel
  -> one discrete token sequence per channel
  -> average normalized token histograms across readers
  -> nested-CV logistic regression for 3-way sentiment
```

Every reader's EEG for a sentence is pooled before evaluation, so the unit of
cross-validation is the unique sentence. No reader-recording from a held-out
sentence can enter the training fold.

The reported comparisons are:

| setup | purpose |
| --- | --- |
| `tfm_histogram` | tests aligned pretrained token counts |
| `tfm_histogram_shuffled` | shuffles EEG/sentence pairing separately inside train and test splits |
| `majority` | minimum reference |

The extraction also reports codebook coverage, token perplexity, and the largest
token's share. These checks catch a tokenizer that collapses on ZuCo before a
classifier score is interpreted.

V1 is the completed histogram baseline. V2 keeps the tokenizer and official
codebook frozen but replaces histogramming with a small structured token-map
classifier. V3 is the final predefined version: it passes the cached maps through
the official frozen MTP encoder and fits a linear probe to the resulting sentence
features. The fixed protocols are documented in
[`V2_TOKEN_MAP.md`](V2_TOKEN_MAP.md) and
[`V3_ENCODER_PROBE.md`](V3_ENCODER_PROBE.md).

## Run in Colab

Open [`notebooks/tfm_zuco_colab.ipynb`](notebooks/tfm_zuco_colab.ipynb), select a
GPU runtime, and run from top to bottom. The only paths to edit are the three
Google Drive locations in the configuration cell.

After V1 token extraction, open
[`notebooks/tfm_v2_token_map_colab.ipynb`](notebooks/tfm_v2_token_map_colab.ipynb)
for the resumable frozen token-map experiment. V2 reuses `tokens_v1`; it does not
preprocess or tokenize the raw EEG again.

For the final frozen-encoder experiment, open
[`notebooks/tfm_v3_encoder_probe_colab.ipynb`](notebooks/tfm_v3_encoder_probe_colab.ipynb).
It also reuses `tokens_v1`, plus the packed V2 cache when available. Run its five
code cells in order. Frozen features are saved per subject, so rerunning all cells
after a disconnect reuses every completed subject.

The notebook:

1. clones this repository;
2. installs three tokenizer-only packages in the disposable Colab runtime;
3. shallow-clones the official TFM code without downloading all Git LFS files;
4. displays the exact selected checkpoint size from its LFS pointer and downloads
   only that checkpoint to Colab;
5. performs a one-recording smoke test;
6. resumes or creates the token cache on Drive;
7. runs diagnostics and the controlled evaluation.

Nothing from ZuCo is copied into Git. Raw data, the upstream checkout, model
weights, token caches, and results are ignored by `.gitignore`.

## Expected Drive layout

The defaults follow the same `Data` / `CachedArtifacts` / `Results` separation
used by the other thesis projects:

```text
MyDrive/Thesis/
├── Data/
│   ├── zuco_og_raw/results*_SR.mat
│   └── zuco_sentiment_labels_task1_fixed.csv
├── CachedArtifacts/
│   └── eeg_tokenizer/
│       └── tfm/
│           ├── tokens_v1/
│           ├── token_records_v2_packed/
│           ├── encoder_features_v3/
│           └── upstream_checkpoints/
└── Results/
    └── eeg_tokenizer/
        └── tfm/
            ├── tfm_histogram_v1/
            ├── token_map_v2/
            └── encoder_probe_v3/
```

The notebook creates the `eeg_tokenizer/tfm` cache and results directories when
needed. Token extraction is resumable: existing subject/sentence caches are
reused unless `overwrite=True`.

## Local command-line interface

The same implementation is exposed through `run.py`:

```bash
python run.py inspect --raw-dir RAW_DIR --labels-csv LABELS.csv
python run.py extract --raw-dir RAW_DIR --labels-csv LABELS.csv \
  --cache-dir CACHE --tfm-repo /path/to/TFM-Tokenizer
python run.py evaluate --cache-dir CACHE --output-dir RESULTS
```

Local use is optional. The intended run environment is Colab, and this repository
does not create a local virtual environment.

## Outputs

Extraction writes `extraction_manifest.json` and compressed token files under one
folder per subject. The evaluation workflow writes:

- `fold_metrics.csv`
- `oof_predictions.csv`
- `summary.csv`
- `evaluation_config.json`
- `alignment_delta.json`

The Colab notebook additionally preserves its diagnostics and decision record:

- `sentence_metadata.csv`
- `token_diagnostics.json`
- `token_diagnostics_records.csv`
- `token_diagnostics_summary.csv`
- `token_quality_diagnostics.png`
- `viability_gate.json`

The experiment should only motivate a larger frozen-encoder or text-fusion test
if the aligned macro-F1 is stably above shuffled EEG, with an approximate target
of at least `+0.015` and a positive paired bootstrap interval. This is a decision
rule, not a claim that has already been met.

## Known boundary

This is a transfer test, not a reproduction of the paper's clinical benchmark
tables. The checkpoint was trained on different devices, montages, subjects, and
tasks. A failed transfer would be evidence about domain mismatch, not proof that
TFM tokenization is ineffective in its original setting.
