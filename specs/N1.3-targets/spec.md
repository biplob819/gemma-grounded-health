# N1.3 — Targets and hand audit

**Wave 1. Depends on:** N1.2. **Unblocks:** T2.1.

## Goal
Turn the N1.2 train pool and val split into `TrainExample` rows whose targets say only what their cited sentences support (C1), with "maybe" taught by constructed insufficient examples (OWNER D-15, C2) and personal-advice refusals (D-16), every target reconstructible from its source row and rules (NFR1), and a 50-example hand audit passed by the owner.

## Pinned choices
- NLI model: `cross-encoder/nli-deberta-v3-base` at revision `6c749ce3425cd33b46d187e45b92bbf96ee12ec7` (Apache-2.0, not a Claude, OpenAI or Gemini model). Labels are read from its `id2label`, never assumed.
- Heavy dependencies (`torch`, `transformers`) live in an optional `nli` extra that CI does not install; tests inject a fake scorer. The real model runs only in `ggh targets`.

## Rules (Tech Spec §4.2–4.7, with the refinement below)
- TGT-R1 split `long_answer` with EvalKit's splitter. TGT-R2 drop sentences matching the pinned recommendation/implication patterns. TGT-R3 keep a sentence only if ≥ 50% of its content words (lower-case `\w{3,}` tokens minus a pinned stopword list, light suffix stemming) appear in its cited sentences. TGT-R4 keep only if NLI entailment ≥ 0.5 with the cited sentences as premise. TGT-R5 drop the example if no sentence survives or the answer exceeds 80 words. TGT-R6 citations: per kept sentence the top-overlap context sentence plus any ≥ 0.8 of that score; merged, sorted, capped at 4; zero-overlap sentences are dropped, not cited at random. TGT-R7 label = PQA-A `final_decision`.
- **INS-R1 (refined by N1.2's section labels):** remove every sentence labelled RESULTS or CONCLUSIONS and every sentence the target cites; skip rows with no RESULTS-labelled sentence or fewer than 2 sentences left. INS-R3 target: label `maybe`, one of 4 fixed answers, citations = the 1–2 remaining sentences with the highest overlap with the question.
- OFF-R3: 200 off-scope training examples from 12 question templates × train-pool abstracts and 6 answer templates; label `maybe`; citation `[1]`. No 5-gram of any training template appears in `data/offscope_eval.jsonl`.
- MIX: train = 20,000 (10% insufficient, 1% off-scope, the rest `answer` at yes:no 60:40); val = 2,000 from the val split, same rules and mix. If too few "no" rows survive, keep 60:40 by taking fewer "yes" rows and record the shortfall; filters are never loosened to fill a quota. Rows are drawn from the seeded (seed 7) order of each split until quotas fill.
- PRV-R1: `reconstruct(source_row, rule_ids) == example` for every row.
- AUD: 50 train examples, seeded and stratified (30 answer split by label, 15 insufficient, 5 off-scope), shown one at a time on a local page (127.0.0.1, keys O/W/U/L, autosave to `docs/measurements/audit_labels.jsonl`, each label bound to a sha256 of the exact text shown). Pass: ≥ 45 `ok` and no `unsupported_answer` among answer examples. A fail tightens TGT-R3/R4 once (TGT-R8) and draws a fresh 50.

## Outputs
`artifacts/targets/{train,val}.jsonl` (TrainExample), `audit_sample.jsonl`, `drop_report.json` (copied to `docs/measurements/targets_drop_report.json`), lineage manifest with the data stage as parent; `ggh targets`, `ggh audit serve|score`; `make targets`.
