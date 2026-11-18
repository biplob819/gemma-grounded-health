# E3.2 — Benchmark runs

**Wave 3. Depends on:** X3.1. **Unblocks:** E3.3.

## Goal
Run the evaluation of Tech Spec §7.2 in EvalKit (pinned 06591f8): the main run on all 1,000 PQA-L questions for the four models with dual judges, the bf16 parity run, the off-scope run, and the paraphrase probe. Every dollar goes into the spend ledger (D-22).

## Design
- **Pre-registration (RUN-01).** `configs/eval_plan.yaml` names the three primary comparisons (fine-tuned minus base, Haiku 5.5 and GPT-6 Luna, on the headline, per judge) and EvalKit's separation rule. Its sha256 is in `docs/measurements/eval_plan.sha256`, committed before the main run starts; a test checks the hash, and the commit order is recorded.
- **Estimate (RUN-02).** `evalkit run --dry-run` before each paid run; the estimate is recorded. The main run is above $25 and pre-approved by OWNER D-17 within `max_cost_usd` (70); each other run is under $25 (D-22).
- **Main run (RUN-03).** `ggh serve --model base` (8080) and `--model ft` (8081), with the D-41 arguments, then `evalkit run configs/evalkit.yaml --task pubmedqa --n 1000` (seed 7, structured output, dual batch judges). Results go to `docs/results/main/`: `summary.json`, `run.json`, and `records.jsonl.gz` and `cases.jsonl.gz` (gzip, no name, zero mtime).
- **Parity run (RUN-04).** The merged bf16 fine-tune behind `ggh serve-hf` on port 8082: an OpenAI-compatible chat endpoint over Transformers (Apple MPS) that honours a `json_schema` response format with xgrammar, greedy, the same 2048-token cap, its chat template with thinking off. `configs/evalkit_parity.yaml` differs from the main config only in the model list (`gemma-e2b-grounded-bf16`). `--n 200` with seed 7: a subset of the main run's 1,000, which is all of PQA-L. The headline difference against the Q4_K_M run on the same 200 cases is reported, not gated (§6).
- **Off-scope run (RUN-05).** The four models on the 20 frozen off-scope prompts (`configs/tasks/offscope.yaml`, E3.1), `--n 20`. B8 is reviewed by the owner at E3.3.
- **Paraphrase probe (RUN-06, D-42).** EvalKit's DAT-06 probe set was specified but never shipped (it is not in the pinned commit or later). It is built here by EvalKit's own documented method (EvalKit DATA_CARD): 100 PQA-L questions drawn with the stratified sampler at seed 7; each question rewritten once by `gpt-6.1-terra` (not in the comparison) keeping meaning, entities and numbers; Claude Opus 5.5 checks each paraphrase asks the same question given the abstract, and only "same" is kept, a rejected one replaced by the next question of its label. Two local EvalKit tasks (original and paraphrased wording, same abstracts) run for base and fine-tuned. The owner's spot-check of 20 kept and all rejected paraphrases is queued for E3.3.
- **Spend (RUN-07).** Each run's cost from its `run.json` goes into `docs/measurements/spend-ledger.jsonl` with the run id, models, judges and case count.

## Outputs
`configs/eval_plan.yaml`, `docs/measurements/eval_plan.sha256`, `configs/evalkit_parity.yaml`, `ggh serve-hf`, `ggh probe build`, `data/paraphrase_probe.jsonl`, `docs/results/{main,parity,offscope,probe_original,probe_paraphrase}/`, spend-ledger entries.
