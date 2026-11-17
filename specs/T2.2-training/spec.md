# T2.2 — Full training run

**Wave 2. Depends on:** T2.1. **Unblocks:** X3.1 (with E3.1), O6.1.

## Goal
One clean "Save and Run All" of the committed notebook in a fresh Kaggle session trains the recipe for one full epoch on the pinned base. It then selects a checkpoint by validation loss and the §5.2 dev bars, and freezes it (adapter hash, `TrainRun`) before PQA-L ever sees it. This run is the US3 proof (NFR2).

## Design
- **Same notebook, full run.** The T2.1 notebook is rendered at T2.2's commit with no step limit, and it adds one cell: `ggh dev-eval`. The notebook stays thin (D-10). The pilot's checks all run again: the data manifest, the replayed target hashes, the pinned checkpoint loaded, and every loss finite.
- **Checkpoints.** Evaluation and saving happen every 250 steps and at the end (527 optimizer steps), and the best 2 checkpoints by validation loss are kept, as in T2.1. Gradient norms are now kept in the report (a T2.1 known issue).
- **Dev generation (D-4, §5.2).** Greedy generation on the 500 `dev_eval` prompts with each kept checkpoint, in batches with left padding. Each checkpoint produces two replies per prompt:
  - **Constrained:** decoded under EvalKit's answer schema (`answer_schema`), compiled by `xgrammar` with compact separators. This matches how the training targets are written and how llama-server constrains at benchmark time (D-6).
  - **Unconstrained:** the same prompt without the grammar.
  `dev_metrics` (T2.1) turns the replies into `DevMetrics`. Generation stops at the end-of-turn token or after 384 new tokens.
- **Selection (§5.2).** Candidates are tried in order of validation loss. The first that reaches `schema_valid` ≥ 0.98 constrained, `has_citation` ≥ 0.98 and `citation_valid` ≥ 0.98 is selected. If none passes, the run reports which bar failed and selects nothing; a fix then needs a decision (§5.2 step 4).
- **Freeze (TRN-04).** The selected adapter's files are hashed (sha256 over the sorted file list and contents). `TrainRun` records the config hash, the data manifest hash, the repo commit, the Kaggle kernel version, the GPU, the wall clock, the peak VRAM, the steps, the selected step, the validation loss curve and the dev metrics. It is written to `docs/measurements/train_run.json`. The adapter itself is downloaded to `artifacts/adapter/` and checked against the hash. It is published at R4.2.
- **Read the sample (TRN-05, D-18).** 20 dev generations from the selected checkpoint are read and noted in RUN-STATE: answer, citations, label, against the abstract.
- **Ledger (TRN-06).** The session's GPU hours go into the GPU ledger, measured from the log span.

## Outputs
`src/ggh/train/{generate,select}.py`, `ggh dev-eval`, the re-rendered notebook, `docs/measurements/{train_run,train_full}.json`, `docs/measurements/dev_generations_sample.jsonl` (20 rows), a GPU ledger entry, and `artifacts/adapter/` (gitignored, hash-checked).
