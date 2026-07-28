# Frozen NeuroLM-B transfer to ZuCo

This folder tests one narrow EEG-only question:

> Do representations learned by the official pretrained NeuroLM-B neural encoder
> contain sentence-level sentiment information in ZuCo natural-reading EEG?

This is a **transfer test**, not a reproduction of NeuroLM's six clinical and BCI
benchmarks. It deliberately does not use text, GPT-2 generation, or LaBSE. The
large language-model component is not a sensible first test for 400 labelled
sentences; the frozen neural encoder is the part that can fairly be assessed as
an EEG-only representation.

The upstream sources are:

- W.-B. Jiang et al., *NeuroLM: A Universal Multi-task Foundation Model for
  Bridging the Gap Between Language and EEG Signals*, ICLR 2025.
- [Official paper](https://openreview.net/forum?id=Io9yFt7XH7)
- [Official implementation](https://github.com/935963004/NeuroLM)
- [Official checkpoints](https://huggingface.co/Weibang/NeuroLM)

## Experiment

```text
ZuCo sentence EEG (105 channels, 500 Hz)
  -> remove flat Cz reference channel
  -> 0.1-75 Hz, 50 Hz notch, resample to 200 Hz
  -> divide into non-overlapping one-second patches
  -> map EGI sensors one-to-one to nearest NeuroLM-supported positions
  -> exclude assignments farther than 30 degrees (require at least 80 channels)
  -> official frozen NeuroLM-B neural encoder
  -> fixed mean + standard deviation + temporal-slope pooling per reader
  -> equal reader averaging per sentence
  -> nested-CV linear probe for 3-way sentiment
```

The official `NeuroLM-B.pt` checkpoint is about 2.38 GB. The notebook downloads
it only in Colab and persists it in Google Drive. No weights, raw EEG, features,
or results enter Git.

### Controls and decision rule

| setup | purpose |
| --- | --- |
| `neurolm_frozen_probe` | aligned frozen NeuroLM-B sentence features |
| `neurolm_frozen_probe_shuffled` | EEG/sentence pairing shuffled separately inside every train and test split |
| `majority` | minimum reference |

All readers of a sentence remain together. The evaluation therefore measures
generalization to unseen sentences for the known reader pool, not unseen-reader
generalization.

The predeclared gate requires all three:

- aligned minus shuffled mean fold macro-F1 of at least `+0.015`;
- a positive aligned-minus-shuffled OOF delta for at least two of three seeds;
- a positive lower bound of the 98.33% paired sentence-bootstrap interval.

The conservative interval reserves family-wise error for up to three documented
NeuroLM variants. A failed gate stops automatic escalation to a larger model.

## Run in Colab

Open [`notebooks/neurolm_zuco_colab.ipynb`](notebooks/neurolm_zuco_colab.ipynb),
select a GPU runtime, and run every cell in order. Only the Drive paths in Cell 2
may need editing.

The notebook performs these cloud-only downloads:

| artifact | source | approximate size |
| --- | --- | ---: |
| Python additions | PyPI | 20-40 MB, depending on Colab cache |
| Official NeuroLM source at pinned commit | GitHub | under 1 MB |
| `NeuroLM-B.pt` | authors' Hugging Face repository | 2.38 GB |

Extraction is resumable. Each reader/sentence feature is atomically stored as a
small compressed file before the next recording begins.

## Expected Drive layout

```text
MyDrive/Thesis/
├── Data/
│   ├── zuco_og_raw/results*_SR.mat
│   └── zuco_sentiment_labels_task1_fixed.csv
├── CachedArtifacts/eeg_tokenizer/neurolm/
│   ├── upstream_checkpoints/checkpoints/NeuroLM-B.pt
│   └── frozen_features_v1/<subject>/sentence_*.npz
└── Results/eeg_tokenizer/neurolm/frozen_probe_v1/
```

## Important boundary

ZuCo uses an EGI high-density montage that NeuroLM did not name directly. The
code uses MNE's bundled sensor coordinates and a one-to-one minimum-cost spatial
assignment to NeuroLM-supported positions. Assignments beyond 30 degrees remain
in `spatial_mapping.csv` for audit but are excluded from encoder input; at least
80 channels must remain. This is a principled approximation, not exact montage
compatibility, so a failure can still reflect domain or montage mismatch.
