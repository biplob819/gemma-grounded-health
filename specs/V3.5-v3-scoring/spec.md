# V3.5 — v3 scoring: the pre-registered main_v3 run and its primary test

**Wave 7 (OWNER D-66). Depends on:** V3.4. **Unblocks:** the v3b nodes' comparison and the release decision.

## Why
configs/eval_plan_v3.yaml registers one scoring run (main_v3) and one primary test: v3 minus v2 on the headline (correct and fully grounded share) over the shared 1,000 PQA-L cases. V3.4 froze the selected checkpoint (checkpoint-550, adapter 61670d6c…, Q4_K_M 2dec4ede…) and committed both hashes before this run. The owner set the cap at $15 (OWNER D-69) and re-confirmed it, with v3 scored at Q4_K_M despite the parity miss (OWNER D-71).

## Design
- **Run (RLS-01).** `uv run evalkit run configs/evalkit_v3.yaml --task configs/tasks/pubmedqa_cit.yaml --n 1000` with EvalKit 0.4.0 (the plan's pin), GPT-6 Luna extracting and GPT-6.1 Sol judging, seed 7, `max_cost_usd: 15`. Only v3 is served (`ggh serve --model ft3`, port 8085); base and v2 servers stay stopped, so their answers must come from EvalKit's store (a miss fails), and their claims and verdicts from the V2.2 run's cache. Estimates before the run: `ggh results estimate` $2.16–$4.31 (cache-aware); EvalKit's uncached dry run $33.30 (`docs/measurements/v3/main_v3_dry_run.json`, every model priced fresh at 4.5 claims per answer; the cap stops the run if reached). Results to `docs/results/main_v3/`, spend to the ledger.
- **Primary test (RLS-02).** `ggh.results.family.family_tests` with the plan (one pair, one metric, one judge; 10,000 resamples, seed 7, EvalKit's stats; Holm over one test). v3 "improves on v2" only if its headline is above v2's and the interval excludes zero; otherwise a tie, whichever way the point falls. Guardrails on v3's own values: citation precision at least 0.70, aim-citation rate at most 0.10. Stretch: base's 0.404. Reported beside the headline: v3's, v2's and base's label distribution and answer length.
