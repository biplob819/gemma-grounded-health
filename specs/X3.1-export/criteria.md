# X3.1 acceptance criteria

**EXP-01** — The adapter is checked against its frozen hash, merged into the pinned base loaded in bf16 on CPU, and saved as safetensors with the base's tokenizer and chat template. A quantized base is refused, and a changed adapter is refused.
**EXP-02** — `ft-Q4_K_M.gguf` and `ft-Q8_0.gguf` (and the F16 they come from) are built by the same export function, llama.cpp release and steps as the base; the two manifests agree on everything but the name, source and file hashes.
**EXP-03** — Template parity holds for the fine-tuned Q4_K_M: 20/20 prompts token for token, constrained replies parse in EvalKit, reruns byte-identical.
**EXP-04** — On the 500 `dev_eval` prompts, Q4_K_M versus merged bf16: label agreement ≥ 0.95 and constrained schema validity within 1 point, recorded in a valid `ParityReport`.
**EXP-05** — Every export output has an `ExportArtifact` record with its sha256 and a parent key, and the lineage chain from the parity report back to the dataset verifies.
**EXP-06** — Read the sample: 20 Q4_K_M answers side by side with bf16, read and noted in RUN-STATE.
