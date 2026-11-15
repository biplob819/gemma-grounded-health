# N1.2 acceptance criteria

**DAT-01** — `fetch` downloads both files at the pinned revision into the cache, verifies the two pinned sha256 values, refuses a mismatching file with an error naming it, and records the revision in the manifest.
**DAT-02** — The leakage check drops PQA-A rows sharing a PQA-L PMID, normalised question or a context with 8-gram Jaccard ≥ 0.5; planted duplicates of each kind in a fixture are all dropped and counted by reason; on the real data the report shows 0 PMID and 0 question overlaps (measured) and the 8-gram count, and no PQA-L text appears in any output file.
**DAT-03** — The seeded split puts 2,500 rows in dev (2,000 `val`, 500 `dev_eval`) and the rest in `train_pool`, with no shared PMID; the same seed gives identical files, a different seed different ones.
**DAT-04** — Every rendered prompt equals EvalKit's `render_prompt(builtin_task("pubmedqa"), Case(...))` byte for byte (checked on 20 real rows and a fixture), and sentence numbering comes from `evalkit.text.number_sentences`.
**DAT-05** — `length_report.json` holds the PQA-A label split (196,144 yes / 15,125 no), `long_answer` word quantiles and rendered-length quantiles; values match a recomputation in the test.
**DAT-06** — `data/offscope_eval.jsonl` has 20 rows, each a `dev_eval` abstract plus a personal or dosing advice question; its sha256 is in the manifest; a test fails if the file changes; owner approval is recorded in RUN-STATE.md.
**DAT-07** — `data_manifest.json` lists every output's sha256, counts and drops; running `ggh data` twice gives an identical manifest.
**DAT-08** — `DATA_CARD.md` states sources, licences (PubMedQA MIT), the pinned revision, splits, the leakage result, and the label mix plan, in present tense only for what is true now (EK-L-16).
