# R4.5 criteria

**OSS-01** — `docs/reproduce.md` covers data, training, export, evaluation and release from a fresh clone; each step is a command the docs test runs or dry-runs, and each dry-run names its reason.
**OSS-02** — The README opens with the generated results block, model, GGUF and demo links, a three-line run-it-offline snippet and the research-only notice; its numbers equal the bundle.
**OSS-03** — `CITATION.cff` validates and its version (1.0.0) and links match the release.
**OSS-04** — A tagged v1.0.0 GitHub release exists whose notes equal `ggh release-notes` output (results, artifacts with hashes, known limits).
**OSS-05** — A clean checkout at the tag passes CI and `ggh doctor`, and every link in the README and release notes resolves.
**OSS-06** — The README has real screenshots (the demo, the EvalKit report), a step-by-step guide and every important link; a test checks links and screenshot files.
