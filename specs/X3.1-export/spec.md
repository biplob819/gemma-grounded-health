# X3.1 — Export and parity

**Wave 3. Depends on:** T2.2, E3.1. **Unblocks:** X3.2, E3.2.

## Goal
Turn the frozen T2.2 adapter into the files people and EvalKit run: a merged bf16 checkpoint and fine-tuned Q4_K_M and Q8_0 GGUFs, built with the same pinned llama.cpp and the same steps as the base. Show that the Q4_K_M file behaves like the bf16 model on dev before any benchmark uses it (Tech Spec §6, H2).

## Design
- **Merge (EXP-01).** `ggh merge` checks the adapter in `artifacts/adapter/` against the hash frozen in `docs/measurements/train_selection.json`, loads the pinned base snapshot in bf16 on CPU, applies the adapter with PEFT and merges it (`merge_and_unload`), then saves safetensors to `artifacts/merged/ft-bf16/`. A base that is quantized (a `quantization_config`, or any weight that is not bf16 or fp32) is refused: merging into 4-bit loses precision silently. The base snapshot's tokenizer, chat template, processor and generation config are carried into the merged folder, so the converter and the Space read what training read.
- **Convert (EXP-02).** `ggh export --model ft` runs the E3.1 export function on the merged folder: `convert_hf_to_gguf.py --outtype f16`, then `llama-quantize` to Q4_K_M and Q8_0, with the verified b12421 tools. The base and the fine-tune share the function, so the commit and steps are the same by construction; a test compares the two manifests.
- **Template parity (EXP-03).** `ggh serve-check --model ft` (E3.1) on the fine-tuned Q4_K_M: the GGUF's embedded template renders the 20 dev-smoke prompts token for token as the training tokenizer's `apply_chat_template` does, constrained replies parse in EvalKit, and reruns are byte-identical.
- **Dev parity (EXP-04).** `ggh parity` generates greedy constrained replies on the 500 `dev_eval` prompts twice: with the merged bf16 model in Transformers (on Apple MPS, xgrammar under EvalKit's answer schema, the T2.2 generator), and with `ft-Q4_K_M.gguf` behind `llama-server` (EvalKit's JSON-schema request, as at benchmark time). `ParityReport` records label agreement, exact citation agreement and constrained schema validity for each. The bar: label agreement ≥ 0.95 and schema validity within 1 point. The judged headline difference on 200 PQA-L cases is E3.2's to report, not a gate here.
- **Manifests (EXP-05).** Each output gets an `ExportArtifact` record (kind, model, quant, path, sha256, llama.cpp tag, parent key) in `docs/measurements/export_artifacts.json`. Parent keys are lineage keys (§2.1): a `train` stage manifest is written locally from the frozen record (inputs: the targets' train and val files and the recipe; output: the adapter; parent: the targets stage), the `export` stage's parent is `train`, and `parity`'s parent is `export`. `ggh lineage verify` walks the chain back to the dataset.
- **Read the sample (EXP-06, D-18).** 20 dev prompts with the Q4_K_M and bf16 replies side by side, read and noted in RUN-STATE.

## Outputs
`src/ggh/export/{merge,parity,artifacts}.py`, `ggh merge`, `ggh export --model ft`, `ggh parity`, `artifacts/merged/ft-bf16/` and `artifacts/gguf/ft-*.gguf` (gitignored), `docs/measurements/{gguf_ft_manifest,serving_checks_ft,parity_report,export_artifacts}.json`, `docs/measurements/parity_sample.jsonl` (20 rows), and the stage manifests.
