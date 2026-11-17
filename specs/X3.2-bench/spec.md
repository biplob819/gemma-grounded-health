# X3.2 — Laptop benchmark

**Wave 3. Depends on:** X3.1. **Unblocks:** R4.1.

## Goal
Measure what the GGUFs cost to run on the owner's Apple Silicon laptop, under the fixed §8 protocol (D-19), with every number written by a script and none copied by hand (US2, H7).

## Design
- **`ggh bench` and `make bench` (BEN-01).** For each variant it starts `llama-server` on its own port with exactly `ggh serve`'s arguments (`llama.server_args`: all layers on Metal, one slot, seed 7, greedy, thinking off with D-41's budget), waits for `/health`, and sends the workload through the chat endpoint with EvalKit's answer schema, as EvalKit does: 50 `dev_eval` prompts drawn with seed 7, at most 256 new tokens, the prompt cache off so every prompt is processed fresh. 3 warm-up prompts are discarded, then the 50 run 3 times. Each request's server `timings` give prompt-processing and generation tokens per second and time to first token (prompt time plus one generated token); its wall time is the per-answer latency. Each metric is the median across the 3 repetitions of that repetition's p50 (or p95).
- **Memory and hardware (BEN-02).** A thread samples the server process's resident memory every 50 ms (`ps -o rss=`) and keeps the peak. The record holds the chip, RAM, macOS version, power source (`pmset`), llama.cpp tag and model file size.
- **`llama-bench` (BEN-03).** Before the workloads, `llama-bench -m <gguf> -p 512 -n 128 -ngl 999 -r 5 -o json` runs on every variant in turn, 3 rounds, so slow drift in the machine's state falls on all variants alike; each variant's pp512 and tg128 are its median across rounds, and every round is kept in the file's `details`.
- **Variants (BEN-04).** Q4_K_M (shipped) and Q8_0, base and fine-tuned: four `LaptopBench` records in `docs/measurements/laptop_bench.json`, keyed `<model>-<quant>` (the Tech Spec §2.2 name; the schema has no model field, so the key carries it). Per-token speed should match between base and fine-tuned, and is compared on llama-bench tg128 (a fixed-length generation); per-request generation speed and per-answer latency move with answer length and are reported for both, not compared.
- **Guards (BEN-05).** The bench refuses to start while anything listens on the `ggh serve` ports (8080, 8081: an E3.2 generation run) and refuses on battery power. It runs one server at a time.

- **Recorded deviations from §8.** Context 8192 (`ggh serve`'s, where §8 also says "default context"); `cache_prompt` false so every prompt is processed fresh; 256 tokens and temperature 0 per §8 where EvalKit sends 2048 and leaves temperature to the (already greedy) server; the file is `laptop_bench.json` with a `measured_at` stamp rather than a dated name. These, the metric definitions and the memory caveat (on Apple Silicon `ps` RSS may leave out Metal private buffers such as the KV cache) are written into the file's `protocol` block.

## Outputs
`src/ggh/serve/bench.py`, `ggh bench`, `make bench`, `docs/measurements/laptop_bench.json`.
