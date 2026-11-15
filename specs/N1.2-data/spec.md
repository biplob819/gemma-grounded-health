# N1.2 — Data build and leakage check

**Wave 1. Depends on:** N1.1. **Unblocks:** N1.3, E3.1.

## Goal
`make data` turns the pinned PubMedQA files into a train pool (rows with numbered, section-labelled sentences; N1.3 renders the examples it draws), rendered dev files and a frozen off-scope evaluation file, proves the expert set (PQA-L) cannot leak into training, and writes a manifest that reproduces byte for byte.

## Pinned inputs (measured 2026-11-15)
- Dataset `qiaojin/PubMedQA` at revision `9001f2853fb87cab8d220904e0de81ac6973b318`.
- `pqa_artificial/train-00000-of-00001.parquet`, sha256 `cdee2d3c01369f47d2ea7b89cd32ee64b142cc0debcb863bf39b3a81a3cece05`, 211,269 rows: yes 196,144, no 15,125, no maybe.
- `pqa_labeled/train-00000-of-00001.parquet`, sha256 `3d56bd1abc11884579ecc3aab9dd3cdfce8b7cf54715daeca93b8022b7be231c`, 1,000 rows: yes 552, no 338, maybe 110 (same counts EvalKit pins).
- Measured: zero PMID overlap and zero exact question overlap between the two; `long_answer` is 34 words at the median, 98 at p99, 5,644 rows over 80 words; context is 200 words at the median, 348 at p99.
- Context sentences carry section labels (`BACKGROUND`, `OBJECTIVE`, `METHODS`, `RESULTS`, …), kept per sentence for N1.3's insufficient rule.

## Outputs (`src/ggh/data/`)
- `fetch.py`: download to `~/.cache/ggh/pubmedqa-<rev7>/`, verify sha256 before use, refuse a mismatch (never re-download silently over a bad file).
- `build.py` (`ggh data`): load with pyarrow; leakage check (DAT-R3: PMID, normalised question, context 8-gram Jaccard ≥ 0.5 against any PQA-L context, computed with MinHash-free exact shingles over a PQA-L index); seeded split (seed 7): `dev` 2,500 rows (`val` 2,000 + `dev_eval` 500) and `train_pool` (the rest); EvalKit rendering through `evalkit.text.number_sentences` and `evalkit.tasks.render_prompt` with the built-in `pubmedqa` task; outputs `artifacts/data/{train_pool,val,dev_eval}.jsonl` with per-sentence section labels (`val` and `dev_eval` also carry the rendered `prompt`), `overlap_report.json` (explicit zero counts, rows checked, highest context Jaccard), `length_report.json`, both also copied to `docs/measurements/` as committed evidence; `docs/measurements/data_manifest.json` (committed).
- `data/offscope_eval.jsonl` (committed, 20 rows, owner-approved): personal-advice questions paired with `dev_eval` abstracts, loaded and checked by `build.load_offscope` (no separate module needed); its sha256 is pinned in a test and in the manifest, and `ggh data` refuses to write anything if it changed unless `--refreeze-offscope` is passed.
- `DATA_CARD.md`.
- `Makefile` target `data`.

## Rules
- Split is by PMID; `dev` and `train_pool` share none (DAT-R4).
- Rendering is byte-equal to EvalKit's own `render_prompt` for the same `Case` (DAT-R5): the test builds a `Case` and calls EvalKit directly.
- Rendered token lengths are measured with a whitespace-and-punctuation proxy now (Gemma's tokenizer arrives at T2.1, which re-measures); `length_report.json` holds the distribution.
- The manifest records revision, file hashes, EvalKit commit, template hash (sha256 of the `pubmedqa` prompt template string), splitter version (EvalKit commit), seed, counts and drops.
- The data build never reads PQA-L except to build the leakage index; a test asserts no PQA-L PMID, question or context appears in any output except `overlap_report.json` counts.
