# NeuroLM and raw-EEG transfer screens on ZuCo

This folder contains a bounded EEG-only version series. V1 tested one narrow
transfer question:

> Do representations learned by the official pretrained NeuroLM-B neural encoder
> contain sentence-level sentiment information in ZuCo natural-reading EEG?

This is not a reproduction of NeuroLM's six clinical and BCI benchmarks. V1 used
no text, GPT-2 generation, or LaBSE. After V1 produced a positive but uncertain
alignment effect, the user authorized exactly three broad follow-ups. V2 is a
raw-EEG EEGNet diagnostic; V3 and V4 remain separate future experiments.

The upstream sources are:

- W.-B. Jiang et al., *NeuroLM: A Universal Multi-task Foundation Model for
  Bridging the Gap Between Language and EEG Signals*, ICLR 2025.
- [Official paper](https://openreview.net/forum?id=Io9yFt7XH7)
- [Official implementation](https://github.com/935963004/NeuroLM)
- [Official checkpoints](https://huggingface.co/Weibang/NeuroLM)

## Version series

| Version | Representation | Classifier | Notebook | Status |
| --- | --- | --- | --- | --- |
| V1 | Frozen NeuroLM-B global moments | Balanced logistic regression | [`neurolm_zuco_colab.ipynb`](notebooks/neurolm_zuco_colab.ipynb) | Complete; yellow/inconclusive |
| V2 | Raw 104-channel one-second EEG windows | Compact EEGNet | [`neurolm_raw_eegnet_v2_colab.ipynb`](notebooks/neurolm_raw_eegnet_v2_colab.ipynb) | Prepared; pending Colab run |
| V3 | Factorized frozen NeuroLM channel/time sequence | Small attention probe | [`neurolm_structured_probe_v3_colab.ipynb`](notebooks/neurolm_structured_probe_v3_colab.ipynb) | Prepared; pending Colab run |
| V4 | Full NeuroLM-B EEG-to-GPT-2 path | Three label verbalizers | Separate future notebook | Planned only |

Each version has its own notebook, Drive cache, result directory, and locked
configuration. Earlier notebooks are not rewritten when a new version is added.

## V1 experiment

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

For V1, open
[`notebooks/neurolm_zuco_colab.ipynb`](notebooks/neurolm_zuco_colab.ipynb).
For V2, open
[`notebooks/neurolm_raw_eegnet_v2_colab.ipynb`](notebooks/neurolm_raw_eegnet_v2_colab.ipynb).
For V3, open
[`notebooks/neurolm_structured_probe_v3_colab.ipynb`](notebooks/neurolm_structured_probe_v3_colab.ipynb).
Select a GPU runtime and run every cell in order. Only the Drive paths in Cell 2
may need editing. The full locked V2 protocol is documented in
[`V2_RAW_EEGNET.md`](V2_RAW_EEGNET.md), and V3 is documented in
[`V3_STRUCTURED_PROBE.md`](V3_STRUCTURED_PROBE.md).

The V1 notebook performs these cloud-only downloads:

| artifact | source | approximate size |
| --- | --- | ---: |
| Python additions | PyPI | 20-40 MB, depending on Colab cache |
| Official NeuroLM source at pinned commit | GitHub | under 1 MB |
| `NeuroLM-B.pt` | authors' Hugging Face repository | 2.38 GB |

Extraction is resumable. Each reader/sentence feature is atomically stored as a
small compressed file before the next recording begins.

The V2 notebook installs and downloads no additional model or Python package; it
uses Colab's existing PyTorch and scientific stack. It creates resumable raw-EEG
subject packs only in Google Drive.

## Expected Drive layout

```text
MyDrive/Thesis/
├── Data/
│   ├── zuco_og_raw/results*_SR.mat
│   └── zuco_sentiment_labels_task1_fixed.csv
├── CachedArtifacts/eeg_tokenizer/neurolm/
│   ├── upstream_checkpoints/checkpoints/NeuroLM-B.pt
│   ├── frozen_features_v1/<subject>/sentence_*.npz
│   ├── raw_eeg_packs_v2/<subject>.npz
│   └── structured_features_v3/<subject>.npz
└── Results/eeg_tokenizer/neurolm/
    ├── frozen_probe_v1/
    ├── raw_eegnet_v2/
    └── structured_probe_v3/
```

## Important boundary

ZuCo uses an EGI high-density montage that NeuroLM did not name directly. The
code uses MNE's bundled sensor coordinates and a one-to-one minimum-cost spatial
assignment to NeuroLM-supported positions. Assignments beyond 30 degrees remain
in `spatial_mapping.csv` for audit but are excluded from encoder input; at least
80 channels must remain. This is a principled approximation, not exact montage
compatibility, so a failure can still reflect domain or montage mismatch.

The Colab requirements pin a current `huggingface_hub` compatible with Colab's
`transformers`. If an older Hub module was already imported before setup, restart
the runtime after Cell 1; the checkpoint stored in Drive will not be downloaded
again.
