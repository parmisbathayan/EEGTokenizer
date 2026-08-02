# V5 partially unfrozen NeuroLM/GPT-2

V5 asks whether adapting the language-model side of NeuroLM to ZuCo reveals
EEG-specific sentiment information that the frozen V4 route could not learn.
It remains EEG-only: the stimulus sentence is never supplied.

## Flow

```mermaid
flowchart LR
    A["V2 preprocessed raw EEG"] --> B["Audited 102-channel mapping"]
    B --> C["Start / middle / end seconds"]
    C --> D["Frozen NeuroLM neural tokenizer"]
    D --> E["Frozen lower GPT-2 blocks"]
    E --> F["Trainable final two GPT-2 blocks"]
    F --> G["Trainable final normalization"]
    G --> H["32-unit residual verbalizer adapter"]
    H --> I["Negative / neutral / positive"]
```

## Locked choices

| Component | V5 choice |
| --- | --- |
| Input | EEG only; identical fixed instruction for every recording |
| Reused data | V2 raw subject packs and the existing NeuroLM-B checkpoint |
| Frozen | Neural tokenizer, embeddings, lower GPT-2 blocks and label vectors |
| Trainable | Final two GPT-2 blocks, final normalization and V4 adapter |
| Learning rates | `1e-5` transformer; `5e-4` adapter |
| Training sampling | One randomly selected reader per training sentence and epoch |
| Validation sampling | One fixed reader per validation sentence for early stopping |
| Final test | All held-out reader recordings; probabilities averaged per sentence |
| Splits | Same unseen-sentence 5-fold scheme and seeds 42/52/62 as V4 |
| Control | Independent model trained with whole EEG reader bundles shuffled within each split |
| Resumption | Every completed setup/fold is saved to Drive |

Reader resampling makes end-to-end fine-tuning feasible in Colab. It means V5
is a deliberately resource-bounded adaptation test, not an exact optimizer-only
ablation of V4. The final evaluation still uses every reader available for each
held-out sentence.

## What V5 does not establish

The outer folds hold out sentences, not subjects. V5 therefore measures unseen
sentences within the known reader pool. A positive result would justify a later
locked confirmation that holds out both readers and sentences; a negative
result does not justify adding more fine-tuning variants.

V5 also does not use stimulus text. Text-only and text-plus-EEG controls belong
to a later multimodal experiment so that introducing text is not confounded
with unfreezing NeuroLM.

## Decision

The primary result is aligned minus shuffled macro-F1. V5 reports a 95% paired
sentence bootstrap interval and the established `+0.015` minimum effect. Green
requires all five checks: above-chance balanced accuracy, above-majority
macro-F1, minimum aligned advantage, positive differences in at least two of
three seeds, and a positive interval lower bound.

Run
[`notebooks/neurolm_partial_finetune_v5_colab.ipynb`](notebooks/neurolm_partial_finetune_v5_colab.ipynb).
