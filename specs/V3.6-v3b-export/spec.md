# V3.6 — v3b: main run frozen, the selected checkpoint, export and parity

**Wave 7 (OWNER D-66, D-70; numbering D-72). Depends on:** V3.4. **Unblocks:** V3.7 (v3b's scoring run, under a cap the owner sets before it).

## Why
v3b's main run (Kaggle `ggh-v3b-grpo` v1, notebook at a67891b, 300 GRPO steps under the checker reward, 5.87 GPU hours) and its score kernel (`ggh-v3b-grpo-score` v1: v2, checkpoint-25 … 300 and final on the 300 held-out dev prompts, 1.16 GPU hours) are complete. The v3b plan's scoring run needs the selected adapter's hash and its Q4_K_M GGUF hash committed first (`configs/eval_plan_v3b.yaml`, `checkpoint_selection`).

## Design
- **Freeze (RBX-01).** `docs/measurements/v3b/main/` keeps rl_dev.json, every dev summary and all 14 × 300 replies, report.json, both session logs and the selected checkpoint's trainer state. Every summary is recomputed locally from the replies with the real, canary-checked checker. `ggh rl-freeze --model ft3b` applies the v3b plan's rule (`select_checkpoint` with 300 rows, gold-"no" slack 0.10 and the copy guard, copy share at most v2's plus 0.10; `SELECTION_RULES` in `ggh.cli_rl`), with the same refusals as v3's freeze, and writes `train_selection.json` and an `RLRun` record (reward "v3b").
- **Export (RBX-02).** `ft3b` in `ggh.versions` (own adapter, merged, GGUF and measurement paths; `ggh serve` port 8086). `ggh merge --model ft3b` and `ggh export --model ft3b` reuse X3.1's merge and converter; the manifest names the frozen adapter hash as its source revision.
- **Parity (RBX-03).** `ggh parity --model ft3b`: merged bf16 in Transformers with the grammar against Q4_K_M behind llama-server, on the first 500 dev prompts, under X3.1's bars, as v2 and v3. A miss goes to the owner with the scoring-cap question.

## Result
- Selection: checkpoint-250, v3b reward 0.885 against v2's 0.670 on the 300 held-out rows (250, 275, 300 and final tie at 0.885; ties go to the earlier step). Every candidate keeps every guard except checkpoint-150 (label accuracy 0.930 < 0.9333). checkpoint-250 against v2: right label and checker-supported 0.653 vs 0.457; label accuracy 0.940 vs 0.953; gold-"no" 0.857 both; aim citations 0.007 vs 0.050; copy share (≥ 0.9) 0.153 vs 0.083 (bound 0.183); near-copies (≥ 0.7) 0.29 vs 0.19; mean checker support 0.738; 13.9 words vs 14.1.
- Export: adapter 0537933d…, 205 LoRA layers merged, Q4_K_M 2befee83…, Q8_0 5faa9fb1…, F16 02380e00… (sizes equal v3's).
- Parity: label agreement 0.924, a miss on the 0.95 bar (v2 0.920, v3 0.936), one-way yes→no drift (34 of 38 differences); Q4_K_M label accuracy 0.848 against bf16's 0.906. Goes to the owner with the scoring-cap question.
