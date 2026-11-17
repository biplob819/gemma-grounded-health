# X3.2 run state

**Now:** building — verifier round 1 PASS, fixes applied and the bench rerun
**Next:** converge, verifier round 2

## History
- 2026-11-17 — spec written. The machine is an Apple M3 Max with 64 GB on mains power.
- 2026-11-17 — node_start GO (baseline 7f851b3). Built `ggh.serve.bench` (seeded prompts, EvalKit's constrained chat request with 256 tokens and the prompt cache off, server `timings`, the median of per-repetition percentiles, a 50 ms resident-memory sampler, hardware and power record, `llama-bench` parsing, the port and battery guards), `ggh bench` and `make bench`. The server process type now names `pid`.
- 2026-11-17 — first bench run: the base/fine-tuned gap in per-request generation speed was 14 % (Q4_K_M) and 17 % (Q8_0), in opposite directions, and `llama-bench` tg128 for ft Q4_K_M read 87 against the base's 115. Three alternating `llama-bench` reruns on the two Q4_K_M files gave 111.6/108.7, 116.4/118.4 and 117.6/114.9 tokens/s: the 87 was a one-off. The per-request rate also moves with answer length (fine-tuned answers are about half to two-thirds as long; see the token counts below), so BEN-04 first moved to tg128 (a fixed 128-token generation) and reports the per-request numbers without comparing them.
- 2026-11-17 — bench (Apple M3 Max, 64 GB, macOS 26.5.2, mains power, b12421; 50 prompts × 3 after 3 warm-ups, 9.5 min for all four):

  | Variant | File MB | tg128 | pp512 | gen p50 | TTFT p50 | latency p50 / p95 | peak RSS MB |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | base Q4_K_M | 3,269 | 116.0 | 6,963 | 109.7 | 101 ms | 809 / 1,272 ms | 4,520 |
  | ft Q4_K_M | 3,269 | 108.2 | 6,938 | 107.8 | 131 ms | 583 / 725 ms | 4,509 |
  | base Q8_0 | 4,737 | 81.7 | 7,186 | 82.8 | 114 ms | 1,310 / 1,914 ms | 5,975 |
  | ft Q8_0 | 4,737 | 75.4 | 7,036 | 85.0 | 142 ms | 707 / 886 ms | 5,969 |

  Per-token gap on tg128: 6.7 % (Q4_K_M), 7.7 % (Q8_0), inside the 10 % bar but larger than the reruns' spread suggests; this machine's run-to-run noise is several percent. The fine-tuned model answers faster (shorter answers) but starts 30–40 ms later, and its per-request prompt rate is lower (3,974 against 5,206 tokens/s on Q4_K_M) although pp512 matches; a likely cause is D-41's forced end-of-thinking tokens being processed after the prompt, not verified.
- 2026-11-17 — verifier round 1: PASS, no blocker. It found the "quarter as long" claim wrong (the committed numbers imply about half to two-thirds), the reruns and llama-bench spreads unrecorded, the per-request token counts dropped, the §8 deviations and the memory caveat unwritten, and the BEN-04 swap weakly justified (on that run the per-request gap would have passed and tg128 was the noisier measure). Fixed: per-request prompt and answer token counts are kept; the §8 deviations, metric definitions and the RSS caveat are in the file's `protocol` block; the sampler test now checks it stops; BEN-04 now needs both measures within 10 %. A run with llama-bench at 10 repetitions per variant, each right after its workload, still showed tg128 gaps of 15 % (Q4_K_M, base faster) and 13 % (Q8_0, ft faster): opposite directions again, so the machine's state at each call, not the weights. llama-bench now runs before the workloads, alternating the variants over 3 rounds, median per variant.
- 2026-11-17 — bench rerun (same machine and protocol; `measured_at` in the file):

  | Variant | tg128 (rounds) | gen p50 | answer tokens p50 | TTFT p50 | latency p50 / p95 | peak RSS MB |
  | --- | --- | --- | --- | --- | --- | --- |
  | base Q4_K_M | 125.6 (124.9, 125.6, 125.8) | 113.1 | 79 | 97 ms | 804 / 1,217 ms | 4,470 |
  | ft Q4_K_M | 124.9 (126.2, 124.9, 108.8) | 105.1 | 49.5 | 137 ms | 604 / 759 ms | 4,462 |
  | base Q8_0 | 96.0 (96.0, 97.7, 94.1) | 79.3 | 98.5 | 113 ms | 1,357 / 2,099 ms | 5,926 |
  | ft Q8_0 | 95.6 (96.8, 95.6, 90.0) | 84.3 | 48.5 | 142 ms | 711 / 888 ms | 5,905 |

  Gaps, base against fine-tuned: tg128 0.5 % (Q4_K_M) and 0.3 % (Q8_0); per-request 7.1 % (Q4_K_M, ft slower) and 5.9 % (Q8_0, ft faster). Both measures pass. Interleaving removed the tg128 gap; single dips still happen (ft Q4_K_M round 3, 108.8) and the median absorbs them. Every request processed 486 prompt tokens at the median for both models, so the fine-tuned model's later first token (137 against 97 ms on Q4_K_M) and lower per-request prompt rate are not from a longer prompt; D-41's forced end-of-thinking tokens remain the likely cause, not verified.
