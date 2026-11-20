# P5.1 run state

**Now:** write-up drafted and generated (`ggh writeup`); tests green locally; awaiting converge and verifier
**Next:** node_converge P5.1, verifier round 1, then merge

## Log
- 2026-11-21 — v1.0.0 released (R4.5); P5.1 started.
- 2026-11-21 — Built on branch `p5.1-prep`: `ggh writeup` renders `docs/writeup/how-i-built-it.md` (PUB-01), `project-page.md` (PUB-02), `decisions-summary.md` (PUB-03) and `video-script.md` (PUB-04) from number-free templates (`src/ggh/release/templates/writeup/`) and lists in `src/ggh/release/writeup.py`; every number comes from the results bundle or a committed evidence file (aim-rule count, GPU ledger, v1 targets report, Q8_0 parity, release records, build graph, converge gates, changelog). `prd-one-page.md` is hand-written; a test checks each of its numbers is in the PRD. `tests/release/test_writeup.py`: committed equals generator, no digits or number words in the templates, every decimal traced, note length, caveat coverage, prose verdicts equal the bundle's, decision ids, dates and quotes, video timing (90 s) with the offline segment, forbidden claims, links.
- 2026-11-21 — Write-up link added to `ggh.release.links` and the README links block. The v1.0.0 release notes stay byte-identical to the published body: `links.AFTER_RELEASE` makes `ggh release-notes` render the write-up as it was at release (pending), so `--check` still passes and the published file is never rewritten.
- 2026-11-21 — Read it (D-18): read all four generated files and the PRD summary end to end against the bundle, the decisions and the known issues. Corrections made while reading: v2's first target audit was the owner's, not blind (wording fixed); the post-hoc aim count compares base with v2's corrected figures, not its published ones; the demo GIF is not described as recorded with Wi-Fi off (only the Ollama capture still to record is).
