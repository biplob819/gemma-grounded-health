# X3.2 acceptance criteria

**BEN-01** — `make bench` runs §8: 50 seeded `dev_eval` prompts, constrained JSON through the chat endpoint, at most 256 new tokens, greedy, `ggh serve`'s server arguments, prompt cache off, 3 warm-ups discarded, 3 repetitions, and each metric the median of the per-repetition medians (p95 likewise); the aggregation is tested on fixed numbers.
**BEN-02** — Peak resident memory of the server process is sampled every 50 ms, and the chip, RAM, macOS version, power source, llama.cpp tag and model file size are recorded.
**BEN-03** — `llama-bench` pp512 and tg128 are captured for each variant from its JSON output.
**BEN-04** — Four valid `LaptopBench` records (base and fine-tuned, Q4_K_M and Q8_0) are written, and base and fine-tuned generation speeds agree within 10 % on both measures: llama-bench tg128 (a fixed 128-token generation, the median of 3 rounds that alternate the variants) and the workload's per-request p50. Per-answer latency is reported for both, not compared: it moves with answer length.
**BEN-05** — The bench refuses to start while a `ggh serve` port is in use or on battery power.
