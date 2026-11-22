# V3.3 run state

**Now:** building v3b. The checker was chosen, and the reward, trainer, notebooks and plan are written.
**Next:** hash the plan, render the kernels, verifier, converge x2, merge.

## Log
- 2026-11-23 — The owner asked whether RL is improving and how to know early. The free checks ran with the owner's go-ahead in the widget. Reward alignment on the 200 dev grounding cases: v3's reward ranks v2 > base > v1, while Sol's headline ranks base > v1 ≈ v2. More than half of v2's maximum-reward answers are claims Sol judges unsupported (over-generalisation). Headroom: 41–46 % of trial steps had zero-spread groups. Owner ruling (OWNER D-70): finish v3's main run and build v3b around a local entailment check.
- 2026-11-23 — Checker search, dev only, with owner-approved downloads:
  - The cached deberta-v3-base reached 0.80 agreement but ranked base only 0.03 above v2. Checking each sentence separately added nothing; checking each citation separately was worse.
  - The three stronger checkers were then downloaded at pinned revisions, with licences verified (Apache-2.0, MIT, MIT). The MiniCheck .bin files were loaded with torch's weights-only loader, and the unpinned minicheck package was not installed.
  - Best: the mean of MiniCheck-RoBERTa-L and nli-deberta-v3-large, at 0.82 agreement and a base−v2 gap of +0.09 (Sol +0.17).
  - Gold targets: 56 % of yes/no pass, and 0 % of maybe.
  - The 200 tuning cases all lie within dev_eval's 500, so v3b selects on the other 300 (21 gold "no").
  - `HFChecker` reproduces the experiment's scores exactly.
