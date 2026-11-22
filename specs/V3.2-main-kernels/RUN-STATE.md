# V3.2 run state

**Now:** green after 1 verifier round; the main-run kernels rendered at 305bcac
**Next:** V3.3: push the train kernel, then the score kernel on its output; select the checkpoint; freeze v3's TrainRun

## Log
- 2026-11-23 — Trial attempt 3 (kernel version 3, repo at aa1b5f5) completed in 0.87 GPU hours. It ran 40 steps in 1,858 s. On 200 dev prompts, v2 scored 1.2975 and checkpoint-40 and the final adapter 1.2875 (identical replies). Every summary recomputed from the replies matches `rl_dev.json`. Stop rules: parse failures 0, reward drop 0.010, maybe share 0 vs 0, yes share 0.85 vs at most 0.90: all passed, so the plan goes on to the main run. 26 of 200 replies changed, mostly paraphrases: one flipped to a wrong "no", one to a right "yes", and two newly cite an aim sentence. Training reward was noisy with no upward trend (1.38 over steps 1–10, 1.23 over steps 31–40, per-step standard deviation 0.17), as expected for 40 steps at 5e-6 with little headroom.
- 2026-11-23 — Main-run time from the trial log: about 679 steps × 47 s ≈ 8.9 h of training, plus 16 adapters × about 5.5 min ≈ 1.5 h of scoring, plus about 10 min of setup. That is too close to Kaggle's 12-hour limit, so the main run is split into a train kernel and a score kernel. The training code is unchanged, so the plan's pins hold.
- 2026-11-23 — Converge 1 failed on three mechanical gates: a Literal type in a test, the writeup's node count, and the changelog entry. All fixed. Verifier round 1 PASS with no blockers. It confirmed the selection and trial rules against the plan text, ran the score notebook's glob against a fake read-only tree (exactly one run folder found), checked that the trial's PEFT checkpoints load, and checked the pins and that the trial notebook renders byte-identically. Notes applied: the scoring is 16 adapters, and the reward wording. Notes carried to V3.3:
  - checkpoint-679 and final are the same step, and the tie goes to checkpoint-679 by order, so V3.3 names the folder it hashes;
  - check that the candidate set is complete against save_steps before selecting;
  - a crash late in training leaves checkpoints but no report.json or final/, and with save_only_model there is no resume;
  - smoke-test the score kernel's kernel-source mount on the trial's output while the train kernel runs;
  - the plan's spend clause says the cap is asked with the ask widget, so re-confirm OWNER D-69's $15 in the widget before the V3.4 scoring run.
