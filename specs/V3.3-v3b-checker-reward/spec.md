# V3.3 — v3b: a checker reward aligned with the headline, plan and kernels

**Wave 7 (OWNER D-70). Depends on:** V3.2. **Unblocks:** V3.4 (run outputs, selection, export) and V3.5 (scoring and release).

## Why
A free dev check showed the problem before v3's main run finished. On the 200 dev grounding cases, using the committed answers and GPT-6.1 Sol's committed verdicts (`scripts/v3_reward_alignment.py`), v3's reward and the headline rank the models in opposite orders:

| Model | v3 reward | Sol headline |
|---|---|---|
| v2 | 1.113 | 0.350 |
| base | 0.875 | 0.520 |
| v1 | 0.773 | 0.355 |

Two causes: the reward pays a full point for the label alone, and its word-overlap grounding check passes answers that over-claim. In 41–46 % of trial steps, every sampled reply in a group scored the same. Optimising that reward is unlikely to raise the headline. The owner ruled (OWNER D-70) that v3's run should finish under its own plan while v3b is built around a local entailment check.

## Design
- **Checker (RLB-01).** Five local checkers were compared against Sol on 600 committed dev answers (`scripts/v3b_checker_scores.py`, `scripts/v3b_checker_eval.py`): deberta-v3-base (cached), deberta-v3-large, MiniCheck-RoBERTa-L, MiniCheck-Flan-T5-L, and the mean of two of them, each once with the cited sentences and once with the whole abstract as evidence.
  - **Best:** the mean of MiniCheck-RoBERTa-L and nli-deberta-v3-large on the cited sentences.
    - AUC 0.868 among right-label answers, against 0.60 for v3's lexical check.
    - 5-fold cross-validated agreement with Sol 0.82, against 0.69.
    - Its proxy headline ranks base first, as Sol does: base 0.445 > v2 0.355 > v1 0.22. Sol has base 0.52 and v1 and v2 tied at 0.355 and 0.35.
  - **Rejected:** whole-abstract evidence is much worse (AUC 0.50–0.73).
  - **Threshold:** 0.65, chosen in 4 of 5 folds. Agreement is flat from 0.50 to 0.80, so the exact value matters little.
  - **Caveat:** the pair itself was picked from nine variants on the same answers without nested cross-validation, so its lead over the next best (0.81, 0.80) is within noise.
  - **Blind spot (verifier):** the checker never sees the question. Copied or generic result sentences pass (0.98–0.99), while an answer that draws the question's conclusion fails (0.02) (`docs/measurements/v3/checker_probes.json`). Sol's headline shares this. OWNER D-70 adds a copy guard and a non-copy headline.
  - **Startup check:** before training and before dev scoring, the checker must reproduce three committed canary pairs within 0.01.
  - **Implementation:** `src/ggh/rl/checker.py`, revisions pinned, and it reproduces the experiment's scores exactly.
  - **Held out:** the 200 tuning cases (`docs/measurements/v3/checker_tuning_ids.json`) are excluded from v3b's selection.
- **Reward (RLB-02).** `src/ggh/rl/reward_b.py`:
  - unparsed −1;
  - +0.25 for the gold label;
  - +1.0 more when the label is right, the structure passes (v3's checks without word overlap) and the checker's support is ≥ 0.65;
  - +0.5 instead of +1.0 for a right "maybe" with sound structure, which is not sent to the checker;
  - −0.5 for an aim citation.

  On the 600 committed dev answers, with the checker replayed from the committed scores, this reward averages base 0.546 > v2 0.511 > v1 0.143 (`scripts/v3b_checker_eval.py`); label credit 0.5 would nearly tie base and v2. "Maybe" answers skip the checker because none of v2's 299 maybe targets pass it: they say what is missing. Their smaller credit removes a premium: a right maybe earns 0.75, below a right yes/no's expected 0.81. Of v2's 2,423 yes/no gold answers, 56 % pass, which leaves headroom that v3's reward lacked. Copying a cited result sentence is paid, since it is supported. The answer's similarity to its nearest cited sentence is recorded, and the plan guards the copy share. The checker runs once per batch, and only for right-label yes/no replies with sound structure.
- **Trainer and dev check (RLB-03).** `src/ggh/rl/grpo_b.py` reuses v3's pinned trainer pieces with v3b's reward, so v3's pinned files are untouched.
  - **Where the checker runs:** on the CPU during training, adding about 4 s per step locally and an estimated 12–20 s on Kaggle; on the GPU during the dev check.
  - **Dev check:** `ggh rl-dev-b` scores v2 and each checkpoint, each in its own process, on the 300 dev rows outside the tuning set.
- **Recipe and kernels (RLB-04).** `configs/recipe_v3b_grpo.yaml` is v3's recipe with checkpoints every 25 steps. `ggh.rl.notebook.render(..., variant="v3b")` renders three kernels at one commit; v3's notebooks render byte-identically:
  - a 40-step trial scored on 200 held-out dev rows;
  - a 300-step train kernel;
  - a score kernel on 300 held-out dev rows.
- **Plan (RLB-05).** `configs/eval_plan_v3b.yaml` is hashed before any v3b training step and pins the reward, checker, trainer, the reused v3 parts, rules, schema, recipe, packages and tuning IDs. It has v3's primary test, guardrails, stretch and scoring setup.
  - **Trial stop rules:** v3's, plus the copy guard, plus a mean step time (report.json's wall_clock_s / steps) over 115 s.
  - **Selection:** on the held-out 300 rows with v3's guards, except gold-"no" accuracy at v2's −0.10, since 21 rows make 0.05 a single row. A copy guard is added: copy share at most v2's + 0.10 (OWNER D-70). v3's lexical reward and the share at similarity 0.7 are reported beside it.
  - **Reporting:** the headline among non-copied answers is reported for v3b, v2 and base beside the headline.
  - **Honesty:** both v3 and v3b are reported if scored, and neither is picked after the fact.
  - **Spend:** the scoring cap is asked before that run.
