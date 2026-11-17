# X3.2 run state

**Now:** spec written
**Next:** node_start, tests, build

## History
- 2026-11-17 — spec written. The machine is an Apple M3 Max with 64 GB on mains power.
- 2026-11-17 — node_start GO (baseline 7f851b3). Built `ggh.serve.bench` (seeded prompts, EvalKit's constrained chat request with 256 tokens and the prompt cache off, server `timings`, the median of per-repetition percentiles, a 50 ms resident-memory sampler, hardware and power record, `llama-bench` parsing, the port and battery guards), `ggh bench` and `make bench`. The server process type now names `pid`.
- 2026-11-17 — first bench run: the base/fine-tuned gap in per-request generation speed was 14 % (Q4_K_M) and 17 % (Q8_0), in opposite directions, and `llama-bench` tg128 for ft Q4_K_M read 87 against the base's 115. Three alternating `llama-bench` reruns on the two Q4_K_M files gave 111.6/108.7, 116.4/118.4 and 117.6/114.9 tokens/s: the 87 was a one-off. The per-request rate also moves with answer length (fine-tuned answers are about a quarter as long), so BEN-04 now compares per-token speed on tg128 (a fixed 128-token generation) and reports the per-request numbers without comparing them.
- 2026-11-17 — bench (Apple M3 Max, 64 GB, macOS 26.5.2, mains power, b12421; 50 prompts × 3 after 3 warm-ups, 9.5 min for all four):

  | Variant | File MB | tg128 | pp512 | gen p50 | TTFT p50 | latency p50 / p95 | peak RSS MB |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | base Q4_K_M | 3,269 | 116.0 | 6,963 | 109.7 | 101 ms | 809 / 1,272 ms | 4,520 |
  | ft Q4_K_M | 3,269 | 108.2 | 6,938 | 107.8 | 131 ms | 583 / 725 ms | 4,509 |
  | base Q8_0 | 4,737 | 81.7 | 7,186 | 82.8 | 114 ms | 1,310 / 1,914 ms | 5,975 |
  | ft Q8_0 | 4,737 | 75.4 | 7,036 | 85.0 | 142 ms | 707 / 886 ms | 5,969 |

  Per-token gap on tg128: 6.7 % (Q4_K_M), 7.7 % (Q8_0), inside the 10 % bar but larger than the reruns' spread suggests; this machine's run-to-run noise is several percent. The fine-tuned model answers faster (shorter answers) but starts 30–40 ms later, and its per-request prompt rate is lower (3,974 against 5,206 tokens/s on Q4_K_M) although pp512 matches; a likely cause is D-41's forced end-of-thinking tokens being processed after the prompt, not verified.
