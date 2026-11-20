# E3.3 criteria

**RES-01** — `docs/results/behaviour_table.md` and the bundle's `behaviour` rows cover B1–B8 for every model of each release's behaviour run, each with its run and source; no row is typed by hand.
**RES-02** — Every off-scope answer (100) has a B8 label in `docs/measurements/b8_review.jsonl` from a blind Claude Opus 5.5 review under the committed prompt (OWNER D-61), recorded with `reviewed_by` and the prompt hash; the bundle's B8 rows are complete and every place that reports B8 says the review is a model's.
**RES-03** — The calibration stamp is computed from the committed EvalKit `calibration.json` and the Gemma-family report by D-59's rule (kappa lower bound ≥ 0.4); the reference labels are disclosed as Claude Opus 5.5's (D-60); exempt runs say so; nothing calls an API.
**RES-04** — The bundle's primary comparisons are each plan's, oriented as written, with intervals and Holm verdicts equal to the summaries.
**RES-05** — 20 records per model of `main` and `main_v2` have a finding and `read_by`, and every finding other than ok has a note; no pipeline fault is unexplained; the builder's own reading is recorded in RUN-STATE.
**RES-06** — `docs/results/results_bundle.json` is committed, deterministic, round-trips through the schema and reports committed lineage OK.
**RES-07** — The bundle and behaviour table under `docs/results/` are generated from committed results and equal what the generator writes; each run folder keeps its EvalKit summary, and the EvalKit report of the V2.2 main run is shown in the README. The README results block generated from the bundle is spliced in by R4.5 (OWNER D-37).
