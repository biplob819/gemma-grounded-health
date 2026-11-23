# V3.3 run state

**Now:** verifier round 2 PASS. The plan is recorded and the three v3b kernels are rendered.
**Next:** converge x2 and merge. The v3b trial goes to the portfolio session only with the owner's go-ahead.

## Log
- 2026-11-23 — The owner asked whether RL is improving and how to know early. The free checks ran with the owner's go-ahead in the widget. Reward alignment on the 200 dev grounding cases: v3's reward ranks v2 > base > v1, while Sol's headline ranks base > v1 ≈ v2. More than half of v2's maximum-reward answers are claims Sol judges unsupported (over-generalisation). Headroom: 41–46 % of trial steps had zero-spread groups. Owner ruling (OWNER D-70): finish v3's main run and build v3b around a local entailment check.
- 2026-11-23 — Checker search, dev only, with owner-approved downloads:
  - The cached deberta-v3-base reached 0.80 agreement but ranked base only 0.03 above v2. Checking each sentence separately added nothing; checking each citation separately was worse.
  - The three stronger checkers were then downloaded at pinned revisions, with licences verified (Apache-2.0, MIT, MIT). The MiniCheck .bin files were loaded with torch's weights-only loader, and the unpinned minicheck package was not installed.
  - Best: the mean of MiniCheck-RoBERTa-L and nli-deberta-v3-large, at 0.82 agreement and a base−v2 gap of +0.09 (Sol +0.17).
  - Gold targets: 56 % of yes/no pass, and 0 % of maybe.
  - The 200 tuning cases all lie within dev_eval's 500, so v3b selects on the other 300 (21 gold "no").
  - `HFChecker` reproduces the experiment's scores exactly.
- 2026-11-23 — Plan hashed at 54849c9. Verifier round 1 FAIL with six blockers:
  - (B1) building the checker imported transformers before Unsloth. Now only the weights prefetch before Unsloth, and the checker is built after it.
  - (B2) the batch cache ignored context. Now the abstract is in the key, with a test.
  - (B3) the checker passes copied and generic result sentences: real-checker probes show copies at 0.98–0.99 and an honest conclusion at 0.02. The owner chose a guard plus reporting in the ask widget (OWNER D-70): copy share at most v2's + 0.10 as a trial stop rule and a selection guard, plus the headline among non-copied answers.
  - (B4) the step-time rule was undefined. Now it is report.json's wall_clock_s / steps, implemented in `select.trial_verdict_b`.
  - (B5) the plan promised a support trace the code cannot produce. Dropped; the trace is the reward, its grounding credit and the copy probe.
  - (B6) the docstring numbers came from the proxy. They now come from the actual reward with the checker replayed (base 0.546, v2 0.511, v1 0.143), and the text no longer calls base > v2 > v1 "the headline's order".
  Notes applied:
  - the "maybe" credit is cut to 0.5 so "maybe" carries no premium, with planted maybe-filler and maybe-aim tests;
  - the golden hack test now computes its rewards;
  - real-checker probes are committed;
  - a startup canary check runs on Kaggle;
  - v3's lexical reward and the share at similarity 0.7 are reported beside the dev summary;
  - the caveat about the cross-validated model choice is recorded;
  - the provenance scripts for gold_checker.json and the deberta-base column are committed (checker_scores.jsonl merges their outputs by case and model);
  - node numbering: v3's scoring now falls in V3.5.
  The plan was re-hashed before any v3b training.
- 2026-11-23 — Verifier round 2 PASS with no blockers. It confirmed all six fixes and the Unsloth import order end to end, and the canary reproduces within about 4e-5 on CPU and MPS. Notes applied before recording the plan:
  - a fourth canary pair whose deberta score sits mid-range (0.49), checked per model and as the mean, so drift in deberta alone shows (the real checker passes it);
  - the "maybe" credit cut to 0.4, since a right maybe at 0.75 still beat v2's own dev pass rate (69 of 159, an expected 0.68); recorded in OWNER D-70;
  - `grpo-b` reports a canary failure as a FAIL line;
  - the spec's generic-probe figure corrected (0.81).
  Notes left as they are: the copy guard covers near-verbatim copies only (the owner chose guard + report; `copied_close` at 0.7 and the non-copy headline make paraphrases visible), and the merge into checker_scores.jsonl is described, not scripted. Plan re-hashed to ec30faf7… before any v3b training.
- 2026-11-23 — Converge failed on pyright in reward_b.py: a citation index typed as Any picked Sequence's slice overload. Fixed with a renamed variable and an explicit int, behaviour unchanged (all reward tests pass). The plan was re-pinned and re-hashed to 03ece981…, the v3b row replaced and the kernels re-rendered, still before any v3b training.
- 2026-11-23 — v3b trial (kernel ggh-v3b-grpo-trial v1, repo at 6f01154) complete: 40 steps, 2683.7 s of training (67.1 s per step), 1.17 GPU hours. The startup canary passed (the run raises on failure). Every stop rule passed: parse 0, mean reward 0.7025 against v2's 0.7037, maybe 0 against 0, yes 0.90 against 0.905, copy share 0.075 against 0.085, 67.1 s per step against 115. The four dev summaries were recomputed locally from the committed replies with the real checker and matched exactly. Its 200 dev rows are the held-out ones (tuning ids excluded), so v2's label accuracy (0.955, 16 gold "no") differs from v3's trial (0.915, 21) on a different 200. Zero-spread groups averaged 14 % over the 40 steps, against 41-46 % under v3's reward. Bundle in docs/measurements/v3/trial_v3b/.
