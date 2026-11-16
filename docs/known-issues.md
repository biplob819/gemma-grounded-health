# Known issues

Findings a verifier raised that were not fixed before a node went green (OWNER D-33), and
should-fix items ruled out of scope. Each names its node, realism and severity. A realistic
finding that would leak a secret or publish a wrong number is never parked here.

| Node | Finding | Realism | Severity | Status |
| --- | --- | --- | --- | --- |
| W0.1 | A Kaggle key read in one notebook cell and displayed by another cell is not paired, so the scan misses it (GitHub push protection does not cover bare hex Kaggle keys either) | Plausible | Medium | Open; avoid the pattern in the T2.1 notebook |
| W0.1 | Per-secret hit counts are best-effort (D-31): distinct same-kind secrets in one cell can share one report | Contrived | Low | Accepted |
| N1.1 | A case or NFD spelling of a parent output on macOS is reported as "source input changed" rather than as a mismatch with the parent (still caught) | Contrived | Low | Accepted |
| N1.1 | A duplicate parent manifest behind a symlinked folder under the root is not counted, because `rglob` does not follow symlinks | Contrived | Low | Accepted |
| N1.2 | Real-row checks (50 parquet rows through the build code, 20 `dev_eval` prompts, the UNLABELLED count) skip where the cache or local artifacts are absent, so CI never runs them; the full rebuild test is opt-in (`GGH_DATA_REBUILD=1`) | Plausible | Low | Accepted; run locally before any data change |
| N1.2 | `length_report(kept)` in place of `all_rows` would go unnoticed, because 0 rows are dropped on the real data | Contrived | Low | Accepted |
| N1.2 | Output section labels are not compared with real rows in the default suite (the verifier checked all 2,500 `val`/`dev_eval` rows by hand) | Contrived | Low | Accepted; N1.3's INS-R1 tests read the labels |
| N1.2 | The lineage manifest `artifacts/data/manifest.json` is not listed in `data_manifest.json` (it holds timestamps and that manifest's own hash) | Contrived | Low | Accepted by design |
| N1.3 | TGT-R2's `need to\b` misses "needs to", and "may be considered" is not a pattern: a few hedged-recommendation sentences reach real answers (verifier round 1 found 3 of the latter). The owner's audit passed 50/50, so the patterns stay as built; widening them needs a decision and a rebuild | Plausible | Low | Open; revisit with any later target rebuild |
| N1.3 | INS-R3 cites sentence 1 by position when no remaining sentence shares a content word with the question (17 of 928 insufficient examples) | Plausible | Low | Accepted; the target text says the abstract does not answer the question, so the citation carries no claim |
| N1.3 | `AuditLabel` has no timestamp or reviewer, so the audit's provenance rests on RUN-STATE and the commit | Contrived | Low | Accepted |
| N1.3 | TGT-R9's hedge pattern misses past tenses and "the current study" ("Our results indicated that…" opens 46 of 8,259 real answer sentences, which keep the hedge) | Plausible | Low | Open; a wider pattern needs a decision and a rebuild |
| N1.3 | CI does not run the audit page's JavaScript (keys, one-at-a-time navigation, autosave); the API it calls is tested, and the owner used the page for the real audit | Plausible | Low | Accepted |
| N1.3 | Untested in CI after round 3 (mutations survive): `HFScorer` truncation, softmax axis and batch sort (the model output order is tested); ranking on the raw rather than hedge-stripped sentence when the hedge words are absent from the context; the stale-note field in `/api/items`; `build` passing `seed` to `select` (the audit-sample seed is tested) | Plausible | Low | Accepted; the local real-data test replays all 9,279 examples |
| N1.3 | Tech Spec TGT-R6 says "cap at 4 (highest-overlap first)"; the code drops a whole sentence, in sentence order, when its citations would push the union past 4 (233 sentences on the real data), so each kept sentence keeps the citations that support it | Plausible | Low | Accepted; documented here and in `assemble` |
