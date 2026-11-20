# R4.5 — Open-source release v1.0.0

**Wave 5 (release). Depends on:** R4.3, R4.4. **Unblocks:** P5.1.

## Goal
Release v1.0.0: anyone can reproduce the work from a fresh clone, read the results from a README generated from the bundle, cite it, and download the published artifacts by hash (US3, US4, US7, OWNER D-27, D-37, D-65).

## Design
- **Reproduce (OSS-01).** `docs/reproduce.md`: data → targets → training (Kaggle) → export → evaluation → release, each step a real command; a test runs the cheap steps and dry-runs the rest (each dry-run says why).
- **README (OSS-02, OSS-06).** The generated results and links blocks (`ggh readme-table --write README.md`), the v2 story with the OWNER D-62 wording, model and GGUF links, the local demo (OWNER D-63), a run-it-offline snippet (`ollama run --think=false …` or llama-server), the research-only notice, real screenshots and GIFs, a step-by-step guide and every important link; a test checks links offline and that every image file is real.
- **Citation (OSS-03).** `CITATION.cff` version 1.0.0, validated against CFF 1.2.0's structure.
- **Release notes (OSS-04).** `ggh release-notes` writes `docs/release_notes_v1.0.0.md` from the bundle and the release record (results, comparisons with the D-62 caveat, every artifact with sha256, known limits, links); the GitHub release `v1.0.0` uses it as its body (OWNER D-65).
- **Fresh clone (OSS-05).** `scripts/fresh_clone_check.py` clones the tagged commit, syncs, runs `ggh doctor` and a fast test set; CI on GitHub runs the full suite; every README and release-notes link resolves.
- **Hub cards (OWNER D-65).** The two corrected cards re-uploaded and checked by download-back.
