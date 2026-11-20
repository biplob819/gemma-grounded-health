# R4.4 — EvalKit upstream

**Wave 5 (release). Depends on:** E3.3, R4.2. **Unblocks:** R4.5.

## Goal
Give back to EvalKit what GGH learned running it on a local model, built under EvalKit's own harness, and publish GGH's v2 report on EvalKit's site (FR10, US6, OWNER D-64).

## Design
- **The PR (UPS-01).** EvalKit node B5.17 on branch `ggh-gemma-example`: a "Gemma through llama-server" worked example in `docs/plugins.md` (exact llama-server flags, the real config, the gotchas: thinking off, one slot, no effort or sampling parameters, cache keys by model name) and `examples/llama-server/evalkit.yaml`, tested offline. https://github.com/biplob819/evalkit/pull/1
- **The page (UPS-02).** `docs/gemma/`: EvalKit's own report for GGH's v2 main run, rendered by `render_report` from GGH's committed summary (copied byte for byte at GGH 22579b3), with an "About this run" card (judge set-up, both calibration stamps, the aim-rule caveat, Hub links) and no per-case data. Deployed by EvalKit's Pages workflow when the PR merges.
- **EvalKit's gates (UPS-03).** B5.17 converges 12/12 under EvalKit's own gates; a fresh verifier passed GEM-01..04; EvalKit's session reviews and merges with the owner's approval there.
- **The rubric issue (OWNER D-64).** https://github.com/biplob819/evalkit/issues/2: the aim rule applied to aim-restating claims, with GGH's evidence; no rubric change.
