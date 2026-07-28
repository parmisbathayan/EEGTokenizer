# NeuroLM transfer project log

This log is chronological. Append implementation changes, problems, results,
and decisions at the bottom. Keep the version-comparison table current.

## 2026-07-28 — Define a bounded EEG-only NeuroLM test

### Starting point

Earlier thesis experiments found no alignment-specific value in 2,496 pooled
classical ZuCo EEG features. The text+EEG model scored similarly with aligned,
shuffled, noise, and zero EEG, so another fusion head over the same summaries is
not justified.

NeuroLM is relevant as a pretrained EEG representation, but its full published
system targets unified instruction-following across clinical and BCI datasets.
Its text alignment is distribution-level and does not pair a reading EEG sample
with the sentence that elicited it. GPT-2 generation therefore does not directly
answer whether reading EEG contains sentiment information.

### Scope decision

- Test ZuCo Task 1 only: about 400 labelled English sentiment sentences and 12
  readers.
- Use no stimulus text and no language-model output.
- Extract only the neural encoder from the official pretrained NeuroLM-B
  checkpoint; do not instantiate GPT-2.
- Freeze every NeuroLM parameter.
- Convert raw sentence EEG to the paper's 200 Hz, one-second patch format.
- Pool encoder outputs using a fixed, non-learned mean/std/temporal-slope rule.
- Average readers only after reader-level encoding.
- Evaluate unique sentences with aligned, split-local shuffled, and majority
  setups across five folds and seeds 42/52/62.

This isolates the cheapest scientifically useful question: whether the learned
NeuroLM EEG representation transfers before any large downstream architecture is
trained.

### Montage decision

The official NeuroLM loader uses a named 10–20-family channel vocabulary. ZuCo's
first 104 usable channels are retained EGI HydroCel sensors; its 105th exported
channel is the flat Cz reference.

The implementation removes Cz and assigns the 104 EGI sensors one-to-one to the
nearest 104 NeuroLM-supported positions using spherical coordinates bundled with
MNE and the Hungarian algorithm. The mapping and angular errors are saved for
audit. Reusing a target position is prohibited. This avoids silently treating
arbitrary EGI indices as NeuroLM channel identities, but it remains an explicit
cross-montage approximation.

### Implementation

Created the self-contained `neurolm/` project:

- `src/zuco_io.py` loads classic and HDF5 MATLAB exports and matches the fixed
  sentiment labels.
- `src/preprocess.py` repairs small gaps, removes Cz, filters, resamples, keeps
  complete one-second windows, and applies the official `/100` input scaling.
- `src/channels.py` constructs and audits the spatial assignment.
- `src/official_neurolm.py` extracts only `tokenizer.*` weights from
  `NeuroLM-B.pt`, instantiates the official encoder, and freezes it.
- `src/extraction.py` atomically caches one compact float16 feature vector per
  reader/sentence so a Colab run can resume.
- `src/features.py` averages readers equally and reports representation
  diagnostics.
- `src/evaluation.py` runs nested sentence-level CV and the paired bootstrap.
- `notebooks/neurolm_zuco_colab.ipynb` is the ordered Colab interface.

The official repository is pinned at commit
`0cda9876d8ce6ee07ed0c43eee5e9a6f5c24b177`. The official NeuroLM-B checkpoint
is approximately 2.38 GB and is downloaded only to Google Drive from Colab.

### Locked viability gate

V1 advances only if aligned minus shuffled fold-mean macro-F1 is at least
`+0.015`, at least two of three seed-level OOF deltas are positive, and the lower
bound of a 98.33% paired bootstrap interval is above zero. The corrected interval
predeclares room for at most three documented NeuroLM versions.

### Status

Code prepared; Colab extraction and evaluation are pending. No scientific result
has been recorded.

## Maintained version comparison

| Component | V1: frozen pooled encoder | V2: structured frozen sequence probe |
| --- | --- | --- |
| Status | Prepared; Colab evaluation pending | Not implemented; permitted only if V1 passes |
| NeuroLM source | Official NeuroLM-B neural encoder | Same frozen encoder |
| Text or GPT-2 | None | None |
| Encoder parameters updated | None | None |
| Reader input | Full 104-channel sentence EEG | Cached channel/time encoder sequence |
| Temporal structure | Fixed embedding slope plus global moments | Learned compact temporal/channel aggregation |
| Reader handling | Encode separately, then equal average | Separate training rows; average held-out probabilities |
| Trainable model | Standardized logistic regression | Small probe, size to be locked before implementation |
| Evaluation unit | Unique sentence | Unique sentence |
| Aligned macro-F1 | Pending | Not run |
| Shuffled macro-F1 | Pending | Not run |
| Aligned minus shuffled | Pending | Not run |
| Decision | Apply locked gate | Do not prepare unless V1 passes |
