# R4.5 run state

**Now:** cards re-uploaded; records; converge
**Next:** converge twice, verifier, green, merge, then tag v1.0.0 and the GitHub release (OWNER D-65)

## Log
- 2026-11-21 — OWNER D-65 (ask widget): publish v1.0.0 when R4.5 is green (tag, GitHub release with generated notes, both corrected Hub cards re-uploaded and checked); every future approval through the ask widget.
- 2026-11-21 — prepared on `r4.5-prep` (7c3122c, five commits): reproduce guide with a docs test, README rebuilt around the generated blocks (draft removed), CITATION 1.0.0, `ggh release-notes`, fresh-clone check (passed offline at 07ecd6c: clone, sync, doctor, 173 fast tests), walkthrough regenerated (ten screens, v2), portfolio.yml images and priorities, Hub card fixes; package version 1.0.0. Merged into `r4.5` from main (ca9e821): the README draft removed, the EvalKit report link no longer says pending (PR #1 merged).
- 2026-11-21 — Hub cards re-uploaded under OWNER D-65: main README.md (sha256 de6ae083…) and GGUF README.md (d068032d…), each downloaded back from tag v2.0.0 (moved to the new commits) with its planned hash; the Hub now tags the main repo `base_model:finetune:google/gemma-4-E2B-it` (it said adapter). `release_record_v2.json` records both; release notes regenerated (every file "published"); the R4.2 known issue closed.
