# V3.3 run state

**Now:** building v3b. Verifier round 1's blockers are fixed and the plan is re-hashed.
**Next:** verifier round 2, then record the plan, render the kernels, converge x2 and merge.

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
