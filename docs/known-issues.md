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
