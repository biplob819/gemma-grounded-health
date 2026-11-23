# V3.4 — v3: main run frozen, the selected checkpoint, export and parity

**Wave 7 (OWNER D-66). Depends on:** V3.3. **Unblocks:** V3.5 (v3's scoring run, within OWNER D-69's $15 cap).

## Why
v3's main run (Kaggle `ggh-v3-grpo` v1, 678 GRPO steps, 9.19 GPU hours) and its score kernel (`ggh-v3-grpo-score` v1, v2, 14 checkpoints and final on the first 500 dev prompts, 1.82 GPU hours) are complete. The v3 plan's scoring run needs the selected adapter's hash and its Q4_K_M GGUF hash committed first (`configs/eval_plan_v3.yaml`, `checkpoint_selection`).

## Design
- **Freeze (RLX-01).** `docs/measurements/v3/main/` keeps rl_dev.json, every dev summary and all 16 × 500 replies, report.json, both session logs and the selected checkpoint's trainer state. `ggh rl-freeze` applies the plan's selection again (`select_checkpoint`, 500 rows, seven guards, ties to the earlier step), refuses any other checkpoint, a run that did not start from v2's frozen adapter or used another recipe, and copies only `adapter_config.json` and `adapter_model.safetensors` into `artifacts/adapter_v3`, so the hash is the same kind as v1's and v2's. It writes `train_selection.json` (read by `ggh merge`) and `rl_run.json` (`RLRun`: the GRPO counterpart of `TrainRun`, with the dev reward instead of a validation-loss curve).
- **Export (RLX-02).** `ft3` in `ggh.versions` (own adapter, merged, GGUF and measurement paths; `ggh serve` port 8085). `ggh merge --model ft3` and `ggh export --model ft3` reuse X3.1's merge (205 LoRA layers, the base's dropped K/V tensors restored) and converter; the manifest names the frozen adapter hash as its source revision.
- **Parity (RLX-03).** `ggh parity --model ft3`: merged bf16 in Transformers with the grammar against Q4_K_M behind llama-server, on the 500 dev prompts, under X3.1's bars (label agreement 0.95, schema gap). v2 missed the agreement bar at Q4_K_M (0.920) and was sent on by OWNER D-55; v3's result is reported the same way, and a miss goes to the owner before scoring.
- **Exploratory, not in the plan.** `scripts/v3_proxy_checker.py` re-reads the committed dev replies with v3b's checker; it changes nothing in the selection.

## Result
- Selection: checkpoint-550, mean reward 1.372 against v2's 1.318, every guard kept (labels unchanged: accuracy 0.934 vs 0.93, gold-"no" 0.8605 both, "maybe" 0.01 both; aim citations 0.010 vs 0.044; lexical grounding paid 0.886 vs 0.82; 15.9 words vs 13.7).
- Export: adapter 61670d6c…, 205 LoRA layers merged, Q4_K_M 2dec4ede… (sizes equal v2's).
- Parity: label agreement 0.936, a miss on the 0.95 bar (v2 0.920), one-way yes→no drift; OWNER D-71 scores v3 at Q4_K_M as registered and discloses the miss.
- Exploratory: v3b's checker on the 300 held-out rows (the other 200 were its tuning set): right label and checker-supported 0.450 vs v2's 0.453 (all 500 rows: 0.444 vs 0.426); near-copies (similarity ≥ 0.7) 0.22 vs 0.19. The lexical gain barely shows under the stronger proxy.
