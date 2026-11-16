# T2.2 run state

**Now:** building — Kaggle full run (`ggh-t22-train` v1, commit 16ba41f)
**Next:** download outputs, freeze, read the sample, converge, verifier

## History
- 2026-11-17 — spec written. Constrained decoding on Kaggle uses xgrammar 0.2.9; locally it compiles EvalKit's answer schema for the Gemma 4 tokenizer and forces a tiny random model into the answer shape (`{"label": ..., "answer": ...`), with compact separators matching the training targets.
- 2026-11-17 — node_start GO (baseline 43f4fc1). Built `ggh.train.generate` (batched left-padded greedy decoding; constrained by xgrammar under EvalKit's answer schema with separators `, ` and `: `, exactly the training targets' format), `ggh.train.select` (candidates by validation loss, §5.2 bars inclusive at 0.98, adapter hash over sorted names and bytes), `ggh.train.devrun` (`dev_eval` on Kaggle, `freeze` locally), `ggh dev-eval`, `ggh freeze`, and a dev-eval cell in the full-run notebook. Gradient norms are now kept in the report. On the tiny Gemma 4 model on CPU, constrained generation parses in EvalKit's own `parse_answer`.
- 2026-11-17 — dev-eval smoke on Kaggle (`ggh-t22-devsmoke` v1, input: pilot 2's checkpoints, 32 prompts): Unsloth inference with the adapter and xgrammar constraints work; checkpoint 300 passed every bar (schema validity 1.0 constrained and 1.0 unconstrained; the base managed 1/20 unconstrained in E3.1), citation validity 1.0, label accuracy yes 0.96 and no 0.83. Dev-eval took 181 s for 32 prompts in both modes including model load, so about 30 min per candidate for 500. GPU ledger 0.22 h.
- 2026-11-17 — full run pushed: `ggh-t22-train` v1, the committed notebook rendered at 16ba41f (one epoch, then dev-eval).
