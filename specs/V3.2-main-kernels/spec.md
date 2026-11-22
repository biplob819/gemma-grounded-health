# V3.2 — v3 trial verdict, main-run kernels and the selection rule

**Wave 7 (v3, OWNER D-66). Depends on:** V3.1. **Unblocks:** V3.3 (the main run's outputs frozen, the checkpoint selected, v3's `TrainRun`, export and parity), then V3.4 (the one scoring run and the release).

## Goal
Record the trial's go decision from its own outputs, so anyone can recompute it, and get the main GRPO run onto Kaggle in a form that fits the session limit, with the plan's checkpoint-selection rule ready as tested code before any main-run output exists.

## Design
- **Trial verdict (RLM-01).** The third trial attempt (kernel `biplob819/ggh-v3-grpo-trial` version 3, repo at aa1b5f5, after D-67 and D-68) is frozen in `docs/measurements/v3/trial/`: `report.json`, `rl_dev.json`, and each adapter's summary and 200 replies. `ggh.rl.select.trial_verdict` applies the plan's four stop rules to the final adapter against v2 (parse failures ≤ 0.02, mean reward within 0.05 of v2's, "maybe" share within ±0.05, "yes" share ≤ v2's + 0.05), and reports, without applying, the gold-"no" accuracy. A test recomputes every summary from the committed replies with `dev_summary` (where `artifacts/` is built) and checks that each rule can stop the plan. Result: every rule passed. v2 scored 1.2975 and the final adapter 1.2875; 26 of 200 replies changed, mostly paraphrases.
- **Main-run kernels (RLM-02).** At 47 s per GRPO step (trial log), one epoch of about 679 steps is about 8.9 h, and scoring 16 adapters (v2, checkpoint-50 to 650, checkpoint-679 and final) on 500 dev prompts adds about 1.5 h. Together that is too close to Kaggle's 12-hour limit. `ggh.rl.notebook.render(commit, None, part=...)` therefore renders two kernels at one commit:
  - `notebooks/v3/main/train/`, kernel `biplob819/ggh-v3-grpo`: data, targets replay, then `ggh grpo` only.
  - `notebooks/v3/main/score/`, kernel `biplob819/ggh-v3-grpo-score`: attaches the train kernel's output as a kernel source, finds the single run folder holding `report.json` and `final/` under `/kaggle/input`, then runs `ggh rl-dev --n 500`.

  The trial notebook (`part="all"`) renders byte-identically. Training and scoring code are unchanged, so the plan's pins hold.
- **Selection rule (RLM-03).** `ggh.rl.select.select_checkpoint` applies the plan's `checkpoint_selection` text to a 500-row `rl_dev.json`:
  - It applies seven guards against v2: parse failures ≤ 0.01; "maybe" share within ±0.05; "yes" share ≤ v2's + 0.05; gold-"no" accuracy ≥ v2's − 0.05; mean words ≥ 0.8 × v2's; label accuracy ≥ v2's − 0.02; aim citations ≤ v2's.
  - Among candidates that keep every guard, it takes the highest mean reward. Ties go to the earlier step, and the final adapter counts as the last step.
  - It returns no selection when no candidate keeps every guard. It reports whether the pick beats v2, because the plan asks the owner before scoring a v3 that does not.
- **Owner cap (OWNER D-69).** The $15 cap for the v3 scoring run, approved in chat on 2026-11-21, is recorded now. It is enforced in V3.4 by `max_cost_usd: 15`.
- **GPU hours (D-38).** The three trial attempts are logged in `docs/measurements/gpu-ledger.jsonl`.
