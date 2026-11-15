# N1.1 run state

**Now:** green
**Next:** N1.2 (data build)

## History
- 2026-11-15 — spec and criteria written while the W0.1 verifier ran.
- 2026-11-15 — node_start GO after W0.1 green (baseline 28724d3). Built schemas, recipe loader, lineage manifests and `ggh lineage verify`, fixtures and tests. Base model confirmed on the Hub: google/gemma-4-E2B-it, revision 4adf157, Apache-2.0, not gated; an official QAT q4_0 GGUF also exists (noted for E3.1).
- 2026-11-15 — verifier round 1 confirmed SCH-02, 03, 05; disproved SCH-04 (verify failed intact chains over unrelated manifests and off-chain duplicates) and SCH-01 (a trailing newline passed the id rule). Fixed, plus should-fix: child inputs checked against parent outputs (a re-hashed edit is caught), outputs must be inside the root and not symlinks, input paths enter the key (D-32), strict floats and recipe bools, duplicate YAML keys rejected, fixtures compared byte for byte, installed EvalKit commit checked, per-thread temp names, unreadable outputs reported.
- 2026-11-15 — verifier round 2 confirmed SCH-01, 02, 03, 05; disproved SCH-04 (another spelling of a path skipped the input check; a directory symlink inside the root passed). Fixed at the source: manifest paths must be canonical (URIs for outside sources); every path component checked for symlinks; source inputs no parent produced are checked; dot-folders other than .git/.venv*/node_modules/__pycache__ count for duplicates; unhashable YAML keys are a ConfigError; a comment on decimals in the recipe.
- 2026-11-15 — verifier round 3 (the cap, OWNER D-33) confirmed SCH-01, 02, 03, 05; disproved SCH-04 narrowly (a repo file named by a `file://`-style URI dodged the input check). Fixed before green under D-33's wrong-number exception: only `hf://` and `cache://` URIs are accepted. Also fixed: unreadable source inputs reported, recipe decode/nesting errors become ConfigError, seed bounded, tests for symlinked source paths and NUL. Two contrived findings logged in docs/known-issues.md. Converge passed twice; rows flipped green.
