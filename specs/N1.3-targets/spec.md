# N1.3 — Targets and hand audit

**Wave 1. Depends on:** N1.2. **Unblocks:** T2.1.

## Goal
Turn the N1.2 train pool and val split into `TrainExample` rows whose targets say only what their cited sentences support (C1), with "maybe" taught by constructed insufficient examples (OWNER D-15, C2) and personal-advice refusals (D-16), every target reconstructible from its source row and rules (NFR1), and a 50-example hand audit passed by the owner.

## Pinned choices
- NLI model: `cross-encoder/nli-deberta-v3-base` at revision `6c749ce3425cd33b46d187e45b92bbf96ee12ec7` (Apache-2.0, not a Claude, OpenAI or Gemini model). Labels are read from its `id2label`, never assumed.
- Heavy dependencies (`torch`, `transformers`) live in an optional `nli` extra that CI does not install; tests inject a fake scorer. The real model runs only in `ggh targets`.

## Rules (Tech Spec §4.2–4.7, with the refinement below)
- TGT-R1 split `long_answer` with EvalKit's splitter. TGT-R2 drop sentences matching the pinned recommendation/implication patterns. TGT-R3 keep a sentence only if ≥ 50% of its content words (lower-case `\w{3,}` tokens minus a pinned stopword list, light suffix stemming) appear in its cited sentences. TGT-R4 keep only if NLI entailment ≥ 0.5 with the cited sentences as premise. TGT-R5 drop the example if no sentence survives or the answer exceeds 80 words. TGT-R6 citations (OWNER D-34, D-35): per kept sentence, the smallest of the top 1–4 non-zero-overlap context sentences (ties by position) that passes TGT-R3 and TGT-R4; merged, sorted, capped at 4 by overlap; a sentence with no overlapping context sentence is dropped, not cited at random. TGT-R9 (OWNER D-34): a leading hedge ("These findings suggest that") is stripped from a conclusion sentence before scoring. TGT-R7 label = PQA-A `final_decision`.
- **INS-R1 (refined by N1.2's section labels):** remove every sentence labelled RESULTS or CONCLUSIONS and every sentence the target cites; skip rows with no RESULTS-labelled sentence or fewer than 2 sentences left. INS-R3 target: label `maybe`, one of 4 fixed answers, citations = the 1–2 remaining sentences with the highest overlap with the question.
- OFF-R3: off-scope examples (1% of each split under OWNER D-34, not 200) from 12 question templates × train-pool abstracts and 6 answer templates; label `maybe`; citation `[1]`. No 5-gram of any training template appears in `data/offscope_eval.jsonl`.
- MIX (OWNER D-34): 20,000 train is a ceiling. Train = as many examples as the surviving "no" rows allow at answer yes:no 60:40, plus 10% insufficient and 1% off-scope of the total (measured: ~4,000). Val = one tenth of train, same mix, drawn from N1.2 `val` rows first and then a seeded `train_pool` slice disjoint from train by PMID. The shortfall against 20,000 is recorded in `drop_report.json`; filters are never loosened to fill a quota. Each PMID is used at most once per split.
- PRV-R1: `reconstruct(source_row, rule_ids) == example` for every row.
- AUD: 50 train examples, seeded and stratified (30 answer split by label, 15 insufficient, 5 off-scope), shown one at a time on a local page (127.0.0.1, keys O/W/U/L, autosave to `docs/measurements/audit_labels.jsonl`, each label bound to a sha256 of the exact text shown). Pass: ≥ 45 `ok` and no `unsupported_answer` among answer examples. A fail tightens TGT-R3/R4 once (TGT-R8) and draws a fresh 50.

## Outputs
`artifacts/targets/{train,val}.jsonl` (TrainExample), `audit_sample.jsonl`, `drop_report.json` (copied to `docs/measurements/targets_drop_report.json`), lineage manifest with the data stage as parent; `ggh targets`, `ggh audit serve|score`; `make targets`.
