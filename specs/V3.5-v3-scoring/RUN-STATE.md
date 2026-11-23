# V3.5 run state

**Now:** main_v3 starting
**Next:** results, primary test, verifier

## Log
- 2026-11-23 — `configs/evalkit_v3.yaml` (base, v2, v3; endpoints 8080, 8083, 8085; Sol judge, Luna extractor, seed 7, `max_cost_usd: 15`). Estimates, no API calls: `ggh results estimate --plan configs/eval_plan_v3.yaml` $2.16 (1.174 claims per answer) to $4.31 (2.342); EvalKit's uncached dry run $33.30 (4.5 claims per answer for every model, base and v2 priced fresh), committed as `docs/measurements/v3/main_v3_dry_run.json` (V2.2's known issue was that this figure went uncommitted). The run relies on the cache and on EvalKit's stop-at-cap, as main_v2 did ($4.00 against the same $33.30 dry-run figure).
