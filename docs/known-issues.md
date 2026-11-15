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
