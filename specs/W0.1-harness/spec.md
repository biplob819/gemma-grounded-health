# W0.1 — Repo and harness

**Wave 0. Depends on:** nothing. **Unblocks:** N1.1, and through it every other node.

## Goal
A public repo that installs as `gemma-grounded-health`, runs `ggh --version`, and enforces the build process exactly as EvalKit does: no node starts before its dependencies are green, and no node turns green without proof that passes every gate.

## Inputs
- Tech Spec §13 (build harness), §1 (decisions), §12 (testing), Appendix B (carried lessons).
- EvalKit's harness: `~/EvalKit/scripts/{harness_lib,verify_graph_eligibility,node_start,node_converge,status,secret_scan}.py` and `tests/harness/`. Ported, not rewritten.

## Outputs
- `pyproject.toml` (hatchling, src layout, Python ≥3.11, Apache-2.0 with NOTICE), `src/ggh/` with `ggh --version`, `ggh --help`, `ggh doctor`.
- `src/ggh/ledger.py`: spend and GPU ledger writer and totals; `scripts/status.py` prints both totals.
- `.github/workflows/ci.yml`: lint, format, types, tests, secret scan, graph check, gate self-test, build; Linux and macOS × Python 3.11 and 3.12.
- `docs/graph/graph.md` (17 nodes) and `docs/graph/catalog.tsv` (107 rows), generated from Tech Spec §13.2 and §13.4.
- The six harness scripts, with node ids `[WNTEXRPO]n.n`, 13 gates (EvalKit's 12 with `judge` replaced by `data` and `recipe`), smoke commands `ggh --version`, `ggh --help`, `ggh doctor`.
- Secret scan extended with Hugging Face and Kaggle tokens; model and data binaries skipped.
- Open-source community files (OWNER D-27): README, CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, PR and issue templates (bug, model behaviour), CITATION.cff.
- `docs/decisions.md` (D-1 to D-31), `docs/lessons.md` (16 carried EvalKit lessons and this project's own), `changelog.md`.

## Out of scope
- Any data, training or evaluation code (from N1.1).
- The GitHub release workflow (no PyPI release, D-21).

## Constraints
- Harness scripts use only the standard library.
- The repo is public from its first push (OWNER D-25); the secret scan must be clean before any push.
