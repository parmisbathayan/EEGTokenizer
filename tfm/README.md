# Frozen TFM tokenizer transfer to ZuCo

This folder tests one narrow question:

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

## Run in Colab

Open [`notebooks/tfm_zuco_colab.ipynb`](notebooks/tfm_zuco_colab.ipynb), select a
GPU runtime, and run from top to bottom. The only paths to edit are the three
Google Drive locations in the configuration cell.

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

The defaults in the notebook assume:

```text
MyDrive/Thesis/
├── Data/
│   ├── zuco_og_raw/results*_SR.mat
│   └── zuco_sentiment_labels_task1_fixed.csv
└── EEGTokenizer/
    └── tfm/
        ├── token_cache/
        └── results/
```

Change the paths in the notebook if your Drive differs. Token extraction is
resumable: existing subject/sentence caches are reused unless `overwrite=True`.

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
folder per subject. Evaluation writes:

- `fold_metrics.csv`
- `oof_predictions.csv`
- `summary.csv`
- `evaluation_config.json`
- `alignment_delta.json`

The experiment should only motivate a larger frozen-encoder or text-fusion test
if the aligned macro-F1 is stably above shuffled EEG, with an approximate target
of at least `+0.015` and a positive paired bootstrap interval. This is a decision
rule, not a claim that has already been met.

## Known boundary

This is a transfer test, not a reproduction of the paper's clinical benchmark
tables. The checkpoint was trained on different devices, montages, subjects, and
tasks. A failed transfer would be evidence about domain mismatch, not proof that
TFM tokenization is ineffective in its original setting.

