# Frozen GPT-2 text-only reference

This experiment measures how predictable the three ZuCo sentiment labels are
from the stimulus sentences alone. It is **Text V1** inside the NeuroLM project:
a sibling reference for interpreting the EEG versions, not an EEG version and
not part of their stoplight family.

## Locked protocol

```text
400 unique labelled stimulus sentences
  -> pinned standard GPT-2 (124M), fully frozen
  -> final non-padding token's 768-dimensional last hidden state
  -> balanced logistic regression
  -> nested sentence-level cross-validation
  -> text-aligned, split-local shuffled-pairing, and majority setups
```

| Component | Choice |
| --- | --- |
| Model | `openai-community/gpt2` |
| Revision | `607a30d783dfa663caf39e06633721c8d4cfcd7e` |
| Input | Exact `sentence` field from the fixed ZuCo sentiment CSV |
| EEG | None |
| Readers | None; each sentence appears exactly once |
| Frozen feature | Final non-padding token, final GPT-2 hidden layer |
| Classifier | Standardized class-balanced logistic regression |
| Regularization | Inner 3-fold choice of `C` from `0.001, 0.01, 0.1, 1, 10` |
| Evaluation | Five stratified folds for seeds 42, 52, and 62 |
| Controls | Split-local shuffled text/sentence pairing and majority |
| Primary metric | Sentence-level macro-F1 |

The final-token state is predeclared because GPT-2 is causal: that position can
attend to the complete preceding sentence. GPT-2 and its tokenizer are never
fine-tuned. This keeps the small dataset from driving 124 million parameters and
makes the result comparable in spirit to the frozen NeuroLM probes.

The result is a **text reference**, not an EEG ceiling. A high score shows that
the labels are recoverable from the sentences and quantifies how much task
signal is available when the true stimulus text is supplied. It does not imply
that EEG should approach the same score.

## Run in Colab

Open
[`notebooks/neurolm_text_gpt2_reference_colab.ipynb`](notebooks/neurolm_text_gpt2_reference_colab.ipynb),
select a GPU runtime, and run every cell. The notebook reads only the fixed label
CSV; the raw EEG files and NeuroLM checkpoint are not needed.

The first run downloads the pinned Hugging Face GPT-2 safetensors weights
(548 MB) plus roughly 3 MB of tokenizer/config files into the dedicated
`text_only_v1` Google Drive cache. The model cache is roughly 552 MB, and the
compressed 400-row feature cache is about 1-2 MB. Later runs reuse both. Nothing
is downloaded or installed on the Mac.

Expected Drive additions:

```text
MyDrive/Thesis/
├── Data/zuco_sentiment_labels_task1_fixed.csv
├── CachedArtifacts/eeg_tokenizer/neurolm/
│   └── text_only_v1/
│       ├── hf_model/
│       └── sentence_features.npz
└── Results/eeg_tokenizer/neurolm/text_only_v1/
    ├── fold_metrics.csv
    ├── oof_predictions.csv
    ├── summary.csv
    ├── extraction_report.json
    ├── evaluation_config.json
    └── text_vs_shuffled_bootstrap.json
```

All Text V1 artifacts are therefore isolated from V1-V5 EEG caches and results.
If `extraction_report.json` reports any truncated sentences, do not interpret
the result until the locked maximum length is reconsidered and the cache is
rebuilt. ZuCo Task 1 sentences are expected to fit comfortably within 128 GPT-2
tokens.
