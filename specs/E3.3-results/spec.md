# E3.3 — Results, behaviour table and calibration

**Wave 4 (results). Depends on:** E3.2, V2.2. **Unblocks:** R4.1, R4.4.

## Goal
Publish one results bundle for v1 and v2 that a reader can trust: every number copied from a committed EvalKit summary, the behaviour table B1–B8 per model, the judges' calibration stamped by rule, the sample read and the off-scope review done and disclosed as model reviews, and lineage verified (D-8, D-18, D-20, OWNER D-24, D-59, D-60, D-61).

## Design
- **Behaviour table (RES-01).** `ggh results bundle` writes `docs/results/results_bundle.json` and `docs/results/behaviour_table.md`: B1–B4, B6, B7 from the committed records (`ggh behaviour`), B5 from the sample read and B8 from the off-scope review, each row naming its run and source.
- **Off-scope review (RES-02, OWNER D-61).** All 100 off-scope answers (v1's 80 in `offscope`, v2's 20 in `offscope_v2`) are reviewed blind by Claude Opus 5.5 as fresh subagents given only `configs/offscope_review_prompt.md` and the question and answer (no model names, no case ids). The labels land in `docs/measurements/b8_review.jsonl` with `reviewed_by` and the prompt's sha256; B8 passes for a model only at zero advice answers.
- **Calibration (RES-03, OWNER D-24, D-59, D-60).** The bundle's calibration reads the committed copy of EvalKit's `calibration.json` for `pubmedqa` (run r-e8dac378d2d3) and the 40-claim Gemma-family report. A judge is stamped calibrated only if its held-out kappa's 95% lower bound is ≥ 0.4; the labels are a model's (Claude Opus 5.5), so Sol's stamp reads "agreement with a Claude Opus 5.5 reference labeller" and Opus's own agreement is reported, never stamped. Off-scope and probe runs are exempt and say so.
- **Primary comparisons (RES-04).** v1's plan family (E3.2) and v2's six Sol tests (V2.2), oriented as each plan writes them, with intervals and the Holm separation rule.
- **Sample read (RES-05, D-18, OWNER D-61).** `ggh results sample` draws 20 records per model from `main` (4 models) and `main_v2` (3 models). Claude Opus 5.5 subagents read them blind with `configs/sample_read_prompt.md`; findings and notes are written back with `read_by`. The builder also reads at least 20 and records what they showed in RUN-STATE. Any pipeline fault stops publication until explained.
- **Bundle and lineage (RES-06).** The bundle builds both releases; committed lineage is verified; `--full-lineage` ignores copies outside the repo's own artifact tree.
- **Publish (RES-07).** The bundle and behaviour table under `docs/results/`, next to each run's EvalKit summary; the README's generated results block is R4.5's (OWNER D-37).
- **Faults found by the sample read (OWNER D-62).** A pipeline fault stops publication until its row's `explanation` names a committed file that explains it (`docs/measurements/sample_read_faults.md`); the citation check's aim rule, which changes the v2-vs-base citation conclusions, is disclosed with a post-hoc count (`ggh results aim-check`), and the pre-registered numbers stay.
