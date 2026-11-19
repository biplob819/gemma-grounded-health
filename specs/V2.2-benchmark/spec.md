# V2.2 — v2 export, parity and benchmark

**Wave 3 (v2 iteration, OWNER D-43). Depends on:** V2.1. **Unblocks:** E3.3.

## Goal
Measure v2 on the same 1,000 PQA-L questions as v1 under a plan hashed and committed before any v2 test run, compare it with v1 and the base model like-for-like, and log every dollar (D-22, OWNER D-43, D-53, D-57).

## Design
- **Export and dev parity (VEV-01, VEV-02).** The v2 merged bf16 and the Q4_K_M, Q8_0 GGUFs were built in V2.1 by the X3.1 code (`docs/measurements/v2/gguf_ft2_manifest.json`, `merge_record.json`); dev parity was measured there: label agreement 0.920 against the 0.95 bar (Q8_0 0.996). OWNER D-55 sends v2 at Q4_K_M to V2.2 with the miss disclosed; this node records both as evidence, not new work.
- **Plan (VEV-03).** `configs/eval_plan_v2.yaml` is finalised (checkpoint hashes, EvalKit 0.4.0 pin per D-56), hashed into `docs/measurements/eval_plan_v2.sha256`, committed, and appended to `docs/measurements/preregistrations.jsonl` before any v2 PQA-L run.
- **OpenAI only (OWNER D-57).** The Anthropic credit is exhausted. GPT-6 Luna extracts claims and GPT-6.1 Sol is the only judge. Base, v1 and v2 answers (base and v1 from EvalKit's answer cache, never regenerated) are all re-extracted and re-judged, so both primary comparisons (v2 − v1, v2 − base) share one extractor and one judge. Three outcomes (headline, citation precision, aim-citation rate), Holm over the six tests. Haiku 5.5 and GPT-6 Luna are shown from their published runs, not compared.
- **Main run (VEV-04).** `configs/evalkit_v2.yaml` on the 1,000 cases, seed 7, with `citation_support`. The base and v1 llama servers stay stopped, so a cache miss on their answers fails the run instead of regenerating.
- **Side runs (VEV-05).** Off-scope (20), the paraphrase probe (100 + 100) and judged bf16-vs-Q4 parity (200), v2 only, same extractor and judge.
- **Spend (VEV-06).** Each run's dry-run and cache-aware estimate recorded before it starts; caps sum to D-53's $25; each run's cost goes into the spend ledger with the note "v2, built after v1's test results were seen (OWNER D-43)".
