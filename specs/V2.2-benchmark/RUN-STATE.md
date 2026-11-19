# V2.2 run state

**Now:** plan finalised under OWNER D-57 and hashed; main run next
**Next:** commit and push the plan hash, record its preregistration row, record both estimates, start main_v2 with 8080/8081 stopped

## Log
- 2026-11-19 — Anthropic credit still exhausted (400 "credit balance is too low", checked hourly since 14:35Z on 11-18). OWNER D-57 (owner: "Use Open AI API"; scope chosen in the ask widget, "Re-score the 3 Gemmas"): GPT-6 Luna extracts, GPT-6.1 Sol judges alone; base and v1 answers from EvalKit's store, re-extracted and re-judged with v2; Holm over 6 primary tests; Haiku 5.5 and GPT-6 Luna shown from their published runs, not compared with v2.
- 2026-11-19 — VEV-01 and VEV-02 carried from V2.1 as evidence: `docs/measurements/v2/gguf_ft2_manifest.json` (Q4_K_M 6da2c13e…, rechecked on disk against the file served on 8083), `merge_record.json`; dev parity 0.920 against the 0.95 bar (`docs/measurements/v2/parity.json`; Q8_0 0.996), sent on at Q4_K_M by OWNER D-55.
- 2026-11-19 — plan `configs/eval_plan_v2.yaml` final: EvalKit 0.4.0 (f9cbc8e, D-56), checkpoint adapter e5f97861…, Q4_K_M 6da2c13e…; sha256 b51167ba… in `docs/measurements/eval_plan_v2.sha256`. Caps: main_v2 $16, offscope_v2 and the two probe runs $1.50 each, parity_v2 $4.50 (sum $25, OWNER D-53).
- 2026-11-19 — estimates before any run (no API calls): cache-aware (`ggh results estimate`, Sol-only unit $0.00184 per claim from the two v2 dev-check runs, an upper bound) main_v2 $8.63–$10.78 (1,000 new v2 answers plus 3,516 re-scored base and v1 claims), all V2.2 $9.57–$12.62; EvalKit's uncached dry run of a five-model OpenAI variant $55.85 (4.5 claims per answer, every model priced fresh), so the three-model main run is far below it.
