# EEGText: paired natural-reading EEG and text

EEGText is a Colab-first project for testing whether paired EEG and text can
improve the language sensitivity of a pretrained EEG representation. The
immediate goal is deliberately narrower than EEG-to-text generation:

> Does pairwise contrastive training on natural-reading EEG and its eliciting
> text improve held-out sentence retrieval, and does that improvement transfer
> to EEG-only sentiment classification?

The project starts with a data audit. No training begins until every source
file, recording, normalized sentence, duplicate-text group, and exclusion is
recorded in a manifest.

## Current milestone

Milestone 1 provides:

- a general ZuCo sentence reader for classic and HDF5 MATLAB files;
- deterministic text normalization and duplicate grouping;
- CSV manifests and JSON summaries for each corpus task;
- an OSF metadata inventory command that lists remote files without downloading
  them;
- a minimal Colab notebook that only calls the Python command-line interface;
- unit tests using synthetic data and mocked remote metadata.

No EEG data, model checkpoint, or dependency is downloaded by this code on the
local machine. The first notebook inventories the official cloud release before
any large Colab download is selected.

## Layout

```text
eegtext/
├── src/                 Python implementation
├── tests/               download-free unit tests
├── notebooks/           minimal Colab entry points
├── EXPERIMENT_PLAN.md   scientific and execution plan
├── PROJECT_LOG.md       chronological decisions and work log
├── requirements-colab.txt
└── run.py               command-line interface
```

## First Colab notebook

Open `notebooks/eegtext_data_audit_colab.ipynb` in Colab. It performs four
steps:

1. clone or fast-forward this repository and run the tests;
2. mount Google Drive and define the data locations;
3. inventory the official OSF release into small JSON/CSV metadata files;
4. audit every ZuCo task directory that is already present in Drive.

The inventory cell does not download the EEG files. Its saved output will be
used to choose exact official artifacts, display their total size, and add a
resumable cloud-only download step in the next milestone.

## Command-line interface

From the `eegtext` directory:

```bash
python run.py inventory-osf \
  --node q3zws \
  --output-dir /path/to/inventory

python run.py audit-zuco \
  --raw-dir /path/to/results/files \
  --dataset zuco \
  --release 1.0 \
  --task SR \
  --pattern 'results*_SR.mat' \
  --output-dir /path/to/audit \
  --labels-csv /optional/sentiment_labels.csv

python run.py combine-manifests \
  --manifest /path/to/sr/recordings.csv \
  --manifest /path/to/nr/recordings.csv \
  --output-dir /path/to/combined
```

The audit command writes:

- `recordings.csv`: one row per reader/sentence recording;
- `summary.json`: counts, exclusions, durations, and duplicate groups;
- `audit_config.json`: the exact source and validation settings.

Writes are atomic. Rerunning a completed audit replaces the three small reports
without touching source data.

## Data and result boundary

Raw data and generated artifacts stay outside Git. The expected Drive layout is:

```text
MyDrive/Thesis/
├── Data/
│   ├── zuco_og_raw/              existing sentiment task
│   ├── zuco_1_task2_nr/          future cloud download
│   ├── zuco_1_task3_tsr/         future cloud download
│   └── teco/                     pending exact release identification
├── CachedArtifacts/eeg_tokenizer/eegtext/
│   ├── corpus_manifests/
│   ├── neurolm_eeg_features/
│   └── gpt2_text_features/
└── Results/eeg_tokenizer/eegtext/
```

These paths match the existing `tfm` and `neurolm` project layout in Drive. The
first notebook defines them centrally as `DATA_ROOT`, `CACHE_ROOT`, and
`RESULTS_ROOT`; no path editing is required for the current Drive structure.
Task 2 and Task 3 do not exist in Drive yet, so their entries are reserved
download destinations rather than expected inputs for the first run.

See `EXPERIMENT_PLAN.md` for the leakage controls and staged model plan.
