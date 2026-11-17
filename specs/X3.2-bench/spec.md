# X3.2 — Laptop benchmark

**Wave 3. Depends on:** X3.1. **Unblocks:** R4.1.

## Goal
Measure what the GGUFs cost to run on the owner's Apple Silicon laptop, under the fixed §8 protocol (D-19), with every number written by a script and none copied by hand (US2, H7).

## Design
- **`ggh bench` and `make bench` (BEN-01).** For each variant it starts `llama-server` on its own port with exactly `ggh serve`'s arguments (`llama.server_args`: all layers on Metal, one slot, seed 7, greedy, thinking off with D-41's budget), waits for `/health`, and sends the workload through the chat endpoint with EvalKit's answer schema, as EvalKit does: 50 `dev_eval` prompts drawn with seed 7, at most 256 new tokens, the prompt cache off so every prompt is processed fresh. 3 warm-up prompts are discarded, then the 50 run 3 times. Each request's server `timings` give prompt-processing and generation tokens per second and time to first token (prompt time plus one generated token); its wall time is the per-answer latency. Each metric is the median across the 3 repetitions of that repetition's p50 (or p95).
- **Memory and hardware (BEN-02).** A thread samples the server process's resident memory every 50 ms (`ps -o rss=`) and keeps the peak. The record holds the chip, RAM, macOS version, power source (`pmset`), llama.cpp tag and model file size.
- **`llama-bench` (BEN-03).** `llama-bench -m <gguf> -p 512 -n 128 -ngl 999 -o json` on the same file; pp512 and tg128 tokens per second go into `llama_bench`.
- **Variants (BEN-04).** Q4_K_M (shipped) and Q8_0, base and fine-tuned: four `LaptopBench` records in `docs/measurements/laptop_bench.json`, keyed `<model>-<quant>` (the Tech Spec §2.2 name; the schema has no model field, so the key carries it). Per-token speed should match between base and fine-tuned, and is compared on llama-bench tg128 (a fixed-length generation); per-request generation speed and per-answer latency move with answer length and are reported for both, not compared.
- **Guards (BEN-05).** The bench refuses to start while anything listens on the `ggh serve` ports (8080, 8081: an E3.2 generation run) and refuses on battery power. It runs one server at a time.

## Outputs
`src/ggh/serve/bench.py`, `ggh bench`, `make bench`, `docs/measurements/laptop_bench.json`.
