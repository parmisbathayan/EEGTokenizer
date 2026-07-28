# V4 full-encoder adaptation

V4 is the final TFM transfer experiment. It tests the remaining hypothesis that
the clinical MTP encoder contains a useful initialization but must adapt to ZuCo
reading EEG before sentiment is linearly visible. Unlike V3, the complete official
TFM encoder is trainable. The tokenizer and cached token IDs remain fixed.

This is the third follow-up after the V1 baseline. It completes the user's
original bounded plan of V2, V3, and V4; no V5 or post-result variation is
authorized.

## End-to-end flow

```mermaid
flowchart LR
    A["V1 cached token maps<br/>4,532 reader recordings"] --> B["One reader sampled<br/>per training sentence/epoch"]
    B --> C["Six 16-channel groups<br/>plus one 8-channel tail"]
    C --> D["Official MTP TFM 64x4<br/>encoder fully trainable"]
    D --> E["New 3-class head"]
    E --> F["Trainable group-aware mixer<br/>initialized as channel-count mean"]
    F --> G["Mean probabilities across<br/>all validation/test readers"]
    G --> H["One prediction per<br/>held-out sentence"]
    H --> I["Aligned vs separately trained<br/>shuffled control"]
    I --> J["Final locked gate"]
```

## What changes from V3

| Component | V3 | V4 |
| --- | --- | --- |
| Official encoder | Frozen | Fully trainable |
| Sentiment head | Logistic regression on cached features | New three-class head trained with the encoder |
| Training row | Reader features averaged before fitting | One randomly selected reader per sentence each epoch |
| Validation/test | Reader features averaged, then classified | Probabilities from every available reader averaged |
| Encoder learning rate | Not applicable | 0.00001 |
| Head learning rate | Logistic-regression regularization search | 0.0003 |
| Maximum epochs | Not applicable | 12, with sentence-level early stopping |
| Compute | Small CPU probes after feature extraction | GPU forward/backward through the official encoder |

## Locked training protocol

| Choice | Fixed value | Reason |
| --- | --- | --- |
| Tokenizer | Frozen V1 tokenizer and token IDs | Isolates encoder adaptation and avoids retraining an 8,192-entry VQ model on 400 labels |
| Initialization | Official MTP-pretrained 64x4 encoder | Tests transfer through supervised adaptation rather than training from scratch |
| Trainable parameters | Entire encoder plus a new three-class head | Directly tests the unresolved frozen-versus-unfrozen question |
| Montage adapter | Six consecutive 16-channel groups plus an 8-channel tail | Uses all 104 channels while respecting the official 2,048-token maximum |
| Group combination | Trainable seven-group mixer initialized to the channel-count-weighted mean | Begins exactly at equal per-channel weighting but can learn that electrode regions contribute differently |
| Training sampling | One reader per training sentence per epoch | Gives every sentence equal influence and keeps free-Colab runtime bounded |
| Reader coverage | Reader resampled each epoch | Exposes the model to multiple readers without multiplying every training epoch by roughly 11 |
| Validation/test pooling | Mean softmax probability across every available reader | Makes the prediction unit the sentence and reduces single-reader noise |
| Loss | Class-balanced cross-entropy | Addresses the three-class imbalance without duplicating sentences |
| Optimization | AdamW; encoder LR 1e-5, head LR 3e-4; weight decay 1e-2 | Uses a conservative adaptation step for pretrained weights and a faster step for the random head |
| Early stopping | 20% sentence-stratified validation split; patience 3; minimum 4, maximum 12 epochs | Selects an epoch without using the outer test fold |
| Outer evaluation | Five stratified sentence folds; seeds 42/52/62 | Preserves the earlier unseen-sentence protocol |

## Alignment control

The shuffled control is not an inference-only trick. For every fold and setup, a
new model starts from the same pretrained checkpoint and the same random head
initialization. Within fit, validation, and test partitions separately, target
sentences are paired through a guaranteed derangement with another sentence's
complete reader-recording set. No target remains paired with its own EEG, and no
EEG crosses a split boundary.

This keeps token values, recording lengths, subjects, class counts, optimization,
and model capacity available to both setups. The intended difference is only
whether EEG and sentiment sentence are correctly aligned.

## Leakage and correctness protections

- Outer test sentences never enter gradient updates or early stopping.
- The validation split is drawn only from the outer training sentences.
- Shuffling is a one-to-one mapping inside each partition and has no fixed points.
- Reader sampling gives one loss contribution per training sentence per epoch.
- Validation and test metrics are calculated only after reader probabilities are
  pooled to one row per sentence.
- Every fold/setup starts again from the verified MTP checkpoint; a prior fold's
  fine-tuned weights cannot leak into the next fold.
- The checkpoint loader rejects missing non-head weights and all unexpected keys.
- The group mixer preserves the identity of the seven montage regions; its exact
  initialization reproduces the V3 channel-count weighting before adaptation.
- The run signature binds results to the token fingerprint, checkpoint SHA-256,
  official source revision, runtime package versions, and full configuration.
- Partial metrics, predictions, and histories are written atomically after each
  completed fold/setup. A disconnect loses at most the active fit.

## Final gate

The effect-size and stability requirements remain unchanged. Because V4 makes
the complete sequence four versions, its paired bootstrap uses a conservative
four-version correction: `0.05 / 4`, producing a 98.75% interval.

| Criterion | Required value |
| --- | ---: |
| Mean aligned balanced accuracy | Greater than 1/3 |
| Mean aligned minus shuffled macro-F1 | At least +0.015 |
| Seeds with positive aligned-minus-shuffled delta | At least 2 of 3 |
| Four-version-corrected paired bootstrap lower bound | Greater than 0 |
| Mean aligned macro-F1 | Greater than majority macro-F1 |

V4 ends the TFM branch whether it passes or fails. A failure supports the narrow
conclusion that neither frozen nor supervised-adapted clinical TFM transfer
provided reliable sentence-aligned ZuCo sentiment signal under these four locked
tests. It does not invalidate the original clinical benchmarks or every possible
reading-EEG tokenizer trained on substantially more unlabeled cognitive EEG.
