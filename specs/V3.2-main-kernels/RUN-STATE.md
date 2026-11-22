# V3.2 run state

**Now:** building: the trial passed; main-run kernels and the selection rule written
**Next:** verifier, converge x2, merge, push both kernels

## Log
- 2026-11-23 — Trial attempt 3 (kernel version 3, repo at aa1b5f5) completed in 0.87 GPU hours. It ran 40 steps in 1,858 s. On 200 dev prompts, v2 scored 1.2975 and checkpoint-40 and the final adapter 1.2875 (identical replies). Every summary recomputed from the replies matches `rl_dev.json`. Stop rules: parse failures 0, reward drop 0.010, maybe share 0 vs 0, yes share 0.85 vs at most 0.90: all passed, so the plan goes on to the main run. 26 of 200 replies changed, mostly paraphrases: one flipped to a wrong "no", one to a right "yes", and two newly cite an aim sentence. Training reward stayed flat (1.38 over steps 1–10, 1.23 over steps 31–40), as expected for 40 steps at 5e-6 with little headroom.
- 2026-11-23 — Main-run time from the trial log: about 679 steps × 47 s ≈ 8.9 h of training, plus 15 adapters × about 5.5 min ≈ 1.4 h of scoring, plus about 10 min of setup. That is too close to Kaggle's 12-hour limit, so the main run is split into a train kernel and a score kernel. The training code is unchanged, so the plan's pins hold.
