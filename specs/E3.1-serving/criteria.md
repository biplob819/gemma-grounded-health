# E3.1 acceptance criteria

**EVK-01** — The pinned llama.cpp release and source are fetched into the cache and refused unless their sha256 matches; the pinned build loads Gemma 4 E2B (E6, checked first); the base Q4_K_M GGUF is built by the shared export script with its hashes recorded.
**EVK-02** — `ggh serve` starts `llama-server` for base (8080) or fine-tuned (8081) with all layers on Metal, one slot, fixed seed and threads, and returns only after `/health` is ok; a second start on a busy port fails clearly.
**EVK-03** — For 20 dev prompts, `/apply-template` output tokenises identically to the pinned tokenizer's `apply_chat_template`.
**EVK-04** — EvalKit's structured-output path against llama-server returns schema-valid JSON for 20 dev prompts; unconstrained validity is measured and reported separately.
**EVK-05** — `configs/evalkit.yaml` matches §7.1 (endpoints, seed 7, pinned EvalKit commit); a test checks the cache-reuse conditions before any reuse is claimed.
**EVK-06** — `ggh behaviour` computes B1, B3, B4 and the B8 pre-sort flag from EvalKit records, each tested on fixture records.
**EVK-07** — The dev smoke and off-scope tasks load as EvalKit local JSONL tasks and render the same prompts as the built-in `pubmedqa` task.
**EVK-08** — 20 cases run twice through `ggh serve` give byte-identical outputs.
**EVK-09** — A dev smoke run with the base model records claims per answer and seconds per answer, and the cost model in the Tech Spec §11 is updated from them.
