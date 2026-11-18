# V2.1 run state

**Now:** started; support-based citation rebuild
**Next:** blind audit, dev bars, Kaggle retrain

## History
- 2026-11-18 — work done in parallel lanes before the node opened (merged from branch v2.1-prep): TGT-R10 (v2, then tightened after OWNER D-46), the per-sentence audit page and scorer (AUD-R4), the owner audit (not passed, 15 of 30 answers) and the blind re-audit (not passed, 11 of 30), and the v1 dev baseline (`docs/results/dev_ground_v1/`, EvalKit ee59699, $5.50): v1 citation precision 0.356 (Opus) against base 0.770, aim-citation rate 0.574 against 0.046.
- 2026-11-18 — EvalKit reports both issues fixed on its main at 317c73f (its D-78; independent verification pending, ships in v0.4.0): missing batch verdicts now keep a run pending and are resent; a final judge failure is recorded per claim and counted in `missing_verdicts`; the cap no longer stops early when two batches reserve at once. Plan: move the pin to the verified commit before V2.2's runs, and re-collect the two missing verdicts and the 9 citation-support gaps from cache then.
- 2026-11-18 — EvalKit's verifier confirmed its D-78 fix at 317c73f (19 attack tests: no route leaves a claim unjudged while a run reads complete). Note for reruns: while a run is pending under `--wait`, read the pending status, not `missing_verdicts`. Pin to v0.4.0 when tagged.
- 2026-11-18 — TGT-R11 (citations by per-sentence support, OWNER D-48): every cited sentence entails its claim alone (100 %, against 20–22 % in v1, v2 and v2.1); 2,722 train / 272 val; 1.10 citations per answer; the floor 0.3 sits in the gap between the audits' weak (≤ 0.22) and supporting (≥ 0.32) sentences, though 55 of 73 owner- and reviewer-judged 'supports' sentences score ≤ 0.08 alone, so NLI alone under-credits support that a reader sees. The helper stopped at the ~3,000 floor as instructed; owner adopted v2.2 (OWNER D-49).
- 2026-11-18 — owner audit ruling OWNER D-50 (train v2.2, disclose the round-3 miss); v2 recipe `configs/recipe_v2.yaml` (D-51: 2 epochs, checkpoint every 50 steps) and the v2 dev grounding bars `configs/v2_dev_bars.yaml` (D-52) committed before the run and before any v2 measurement.
- 2026-11-18 — v2 full run pushed 09:20Z: Kaggle `ggh-v21-train` v1, notebook rendered at 8ff61b9 (recipe_v2, unsloth, two epochs, then dev-eval).
