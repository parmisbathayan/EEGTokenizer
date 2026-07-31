# EEGText project log

This log is chronological. It records the reasoning, implementation decisions,
data findings, failures, fixes, and observed results. New entries are appended;
earlier entries are not rewritten to make later outcomes look predeclared.

## 2026-07-31 — Define the paired EEG-text project

### Motivation

The existing TFM and NeuroLM branches test whether pretrained EEG
representations contain sentence-level sentiment information. They do not train
on the actual text that elicited each reading EEG recording. NeuroLM's published
tokenizer is text-space aligned at the distribution level, but its training does
not make a ZuCo recording explicitly close to the corresponding sentence.

The new project tests pairwise EEG-text contrastive supervision as a separate
scientific question. Its endpoint remains EEG-only sentiment prediction; text is
privileged training information and is unavailable at sentiment inference.

### Naming and repository boundary

The folder is named `eegtext`. The word `alignment` is intentionally not used as
the project name because another thesis project already uses that terminology.
The new folder is self-contained and does not modify the completed `tfm` or
`neurolm` implementations.

### Initial scope

- Begin with sentence-level recordings, not fixation-level epochs.
- Reuse the existing ZuCo Task 1 files.
- Inventory the official full ZuCo release before choosing any additional large
  artifacts.
- Add Task 2 and Task 3 through a general loader while retaining task identity.
- Treat every normalized duplicate sentence as one split group.
- Add TECO only after its exact release and technical format are identified.
- Use frozen NeuroLM and GPT-2 representations before training a small residual
  contrastive adapter.
- Compare aligned contrastive training with an independently trained shuffled
  pairing control.
- Evaluate downstream sentiment with no text at inference.

The full scientific and leakage plan is recorded in `EXPERIMENT_PLAN.md` before
any model result is observed.

## 2026-07-31 — Implement milestone 1: scaffold and data audit

Created a Colab-first Python package with a minimal command-line interface. The
milestone includes deterministic text normalization, classic and HDF5 ZuCo
sentence readers, recording manifests, audit summaries, and an OSF metadata
inventory. Remote inventory and data access are invoked only from Colab.

The first notebook contains no model logic. It clones or fast-forwards the
repository, runs download-free unit tests, mounts Drive, inventories official
OSF metadata, and audits whichever task directories already exist. The notebook
does not install packages or download EEG artifacts.

The audit reports can be merged after any subset of tasks is available. The
combined manifest recomputes duplicate-text groups across tasks and rejects
duplicate recording identifiers. This establishes the grouping primitive that
later split code will enforce.

### Verification

- Seventeen download-free unit tests pass.
- The command-line parser exposes `inventory-osf`, `audit-zuco`, and
  `combine-manifests`.
- The Colab notebook is valid JSON and every code cell compiles.
- Tests cover text normalization, conflicting labels, channel orientation,
  truncated trials, non-finite thresholds, atomic reports, cross-task duplicate
  groups, split leakage rejection, and recursive/paginated OSF metadata.

Actual MATLAB loading is intentionally deferred to the Colab smoke test because
the source data and Colab's bundled SciPy/HDF5 runtime are not materialized on
the Mac. The audit records file-level failures instead of silently dropping a
subject.

All generated data, manifests, and results use the established
`Data`/`CachedArtifacts`/`Results` Drive separation. The source files and audit
outputs remain outside Git.

No package, dataset, checkpoint, runtime, or environment was downloaded or
installed on the Mac for this milestone.
