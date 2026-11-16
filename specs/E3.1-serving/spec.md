# E3.1 — Local serving and EvalKit integration

**Wave 3. Depends on:** N1.2. **Unblocks:** X3.1 (with T2.2).

## Goal
Serve Gemma 4 E2B (base now, fine-tuned after X3.1) locally through a pinned llama.cpp, reach it from EvalKit through the supported `endpoints` route (D-7), and prove the plumbing: template parity, constrained JSON, determinism, behaviour checks, and a measured dev smoke run with the base model.

## Pinned inputs (checked 2026-11-16)
- llama.cpp release `b12421`: `llama-b12421-bin-macos-arm64.tar.gz`, sha256 `bdeff25d862f1c9b9526ce2c373492e67656659b2ac386308c28b4808f4d074d` (GitHub's published digest, verified on download). The source tarball of the same tag supplies `convert_hf_to_gguf.py` and `gguf-py`; its sha256 is pinned when first fetched.
- Base model `google/gemma-4-E2B-it` at `4adf15748de9e1a0097b6274cee9c15836c41c37` (one 10.25 GB safetensors file, Apache-2.0, not gated).
- Running the release binaries needs the owner's permission (the session's permission check blocks running downloaded executables); see RUN-STATE.

## Outputs
- `src/ggh/serve/llama.py`: fetch and verify the pinned release and source into the cache; never run an unverified file.
- `src/ggh/export/gguf.py` (shared with X3.1): HF safetensors → F16 GGUF → Q4_K_M and Q8_0 with the pinned tools; manifest with hashes.
- `ggh serve --model base|ft --quant Q4_K_M` (ports 8080/8081): all layers on Metal, `--parallel 1`, fixed seed and threads, waits for `/health`.
- Template parity test: 20 dev prompts through `/apply-template` versus `transformers` `apply_chat_template` of the pinned tokenizer, token for token.
- `configs/evalkit.yaml` (§7.1) and a cache-reuse check.
- Local JSONL tasks for the dev smoke run (20 `dev_eval` rows) and the off-scope set.
- `ggh behaviour` B1, B3, B4, B8 over EvalKit records.
- Determinism test: 20 cases twice, identical outputs.
- Dev smoke run with base: claims per answer and seconds per answer written into the cost model (§11).
