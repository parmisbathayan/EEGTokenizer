# EEG tokenizer experiments

Colab-first experiments for testing pretrained EEG tokenizers on the thesis
datasets. Each paper lives in its own folder so implementations, notebooks,
results, and decisions do not get mixed together.

## Experiments

| folder | paper | status |
| --- | --- | --- |
| [`tfm/`](tfm/) | *Tokenizing Single-Channel EEG with Time-Frequency Motif Learning* | V1 complete; V2 token-map probe ready |
| [`neurolm/`](neurolm/) | *NeuroLM: A Universal Multi-task Foundation Model for Bridging the Gap Between Language and EEG Signals* | V1 complete; separate V2 raw-EEG EEGNet screen ready |

Raw EEG, model checkpoints, token caches, and experiment outputs are deliberately
kept outside Git. The notebooks store those artifacts in Google Drive.
