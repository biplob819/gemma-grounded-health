# V3.7 run state

**Now:** config committed; main_v3b starting
**Next:** results, primary test, verifier

## Log
- 2026-11-24 — Owner rulings (ask widget): "Hold for a Q8_0 check" and "$5 cap"; then, with the check's numbers, "Q4_K_M + Q8_0 extra" (OWNER D-73).
- 2026-11-24 — Q8_0 parity check (`scripts/v3b_q8_parity.py`, the registered bf16 replies against ft3b-Q8_0 on the same 500 dev prompts, llama-server on 8086): label agreement 0.990 (5 differ: 4 yes→no, 1 no→yes), exact citation agreement 0.908, schema validity 1.000; Q8_0 label accuracy 0.904 against bf16's 0.906 (Q4_K_M: 0.924, 0.808, 0.848). Passes X3.1's bars. Read the sample (every 25th row): labels agree on 20/20, answer text identical on 13/20; the differing answers restate the same result with a different emphasis (1471704 cites the elevated-levels sentence rather than the correlation) or restate it more generally (11231951: "PKC-dependent vasoconstrictors" for the three named agonists). The Q4_K_M yes→no drift is a quantization effect.
- 2026-11-24 — `configs/evalkit_v3b.yaml` (base 8080, v2 8083, v3b-Q4_K_M 8086, v3b-Q8_0 8087; Sol judge, Luna extractor, seed 7, `max_cost_usd: 5`). Estimates, no API calls: `ggh results estimate --plan configs/eval_plan_v3b.yaml` $2.16 to $4.31 for the plan's one new model; EvalKit's uncached dry run $44.40 (every model priced fresh), committed as `docs/measurements/v3b/main_v3b_dry_run.json`.
