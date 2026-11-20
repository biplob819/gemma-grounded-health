# R4.4 run state

**Now:** PR open; waiting for EvalKit's review and the owner's merge approval in EvalKit's session
**Next:** merge, /gemma/ live, links, converge, verifier

## Log
- 2026-11-21 — OWNER D-64 (ask widget): PR plus the /gemma/ page; the aim rule as an issue only. EvalKit's session confirmed node number B5.17, the Pages workflow change, and that merges need the owner's approval in its own session.
- 2026-11-21 — EvalKit node B5.17 built in a worktree on `ggh-gemma-example` (from main d49a585): worked example, example config (GGH's `evalkit_v2.yaml` cut to v2), `/gemma/` page with `scripts/build_gemma_page.py` (`--check`, `--ggh`, `--self-test`), Pages workflow copy step. EvalKit gates: ruff, format, pyright clean; pytest 3270 passed; `node_converge.py B5.17` 12/12 three times.
- 2026-11-21 — EvalKit verifier round 1 (fresh, read-only): GEM-01..04 PASS. Notes applied in 47af04f (GGH decision ids labelled, both calibration stamps explained, aim link pinned to 22579b3, placeholder-key wording). Raised for EvalKit's session: EvalKit's README, changelog, decisions and case study call the r-e8dac378d2d3 calibration labels "the owner's hand labels", while GGH OWNER D-60 records they were Claude Opus 5.5's; the kappas are identical, and the /gemma/ page quotes GGH's wording. One record must be corrected before /gemma/ deploys.
- 2026-11-21 — pushed `ggh-gemma-example`; PR https://github.com/biplob819/evalkit/pull/1; issue https://github.com/biplob819/evalkit/issues/2 (rubric be34ebe577bc96ae; numbers from GGH's `aim_rule_check.json` at 22579b3; no case ids). EvalKit's session asked to review and merge with the owner's approval there.
