# N1.1 acceptance criteria

**SCH-01 — data schemas**
- `TrainExample`, `DropRecord`, `DataManifest`, `AuditLabel` validate their fixtures and round-trip through JSON byte-identically.
- `TrainExample` rejects: an id that is not exactly `pqa-a-<ASCII digits>[:kind]` (no trailing newline, no non-ASCII digits), citations outside 1..n, unsorted or duplicate citations, no citations, an id that does not match its kind, `insufficient`/`offscope` with a label other than `maybe`, `answer` whose label differs from `source_label`, `support` or `nli_entail` outside [0, 1], NaN anywhere. Float fields are strict: an int is accepted, a bool or a numeric string is not.
- `DataManifest` requires 64-hex sha256 values for every file and for `offscope_eval_hash` and `template_hash`; `AuditLabel.verdict` is one of the four verdicts.

**SCH-02 — training schemas**
- `RecipeConfig` loads `configs/recipe.yaml` (the §5.1 values) through `load_recipe`; rejects out-of-range values, unknown keys, duplicate YAML keys, and bools or strings where a number belongs (and numbers where a bool belongs), with an error naming the field.
- `DevMetrics` rates are in [0, 1]; `TrainRun.selected_step` must appear in `val_loss_curve`.

**SCH-03 — release schemas**
- `ExportArtifact`, `ParityReport`, `LaptopBench`, `BehaviourRow`, `ResultsBundle` validate fixtures and round-trip.
- `ExportArtifact` requires `quant` and `llama_cpp_commit` exactly when `kind="gguf"`.
- `ResultsBundle` refuses `calibrated` without `gemma_family_check` (OWNER D-24).

**SCH-04 — stage contract and lineage**
- `stage_key` is stable under dict key order (property test) and changes when any of stage, code_version, an input's path or hash, params or seed changes, so swapping two inputs' contents changes it (D-32).
- Manifest paths have one spelling: every output, and every input that is not an `hf://` or `cache://` URI (the only schemes, for sources outside the repo), is a canonical relative POSIX path (no `./`, `//`, `.`/`..` segments, backslashes, absolute paths or trailing slash); anything else is refused when the manifest is built.
- `write_manifest` is atomic (no partial file after a simulated crash) and `verify` walks parents, judging only manifests on the chain: it fails on a missing parent, a parent key with several manifests (in any folder except `.git`, `.venv*`, `node_modules`, `__pycache__`), a recorded output that no longer matches its file, an output reached through any symlinked component, a child input that does not match its parent's recorded output (a parent edited and re-hashed later), a source input in the repo that no parent produced and that is missing or changed, or a cycle; it passes on an intact chain even when unrelated `manifest.json` files sit under the root.

**SCH-05 — EvalKit import contract**
- Every EvalKit symbol this repo uses (`evalkit.schemas.Answer`, `Case`, `Interval`, `RunSummary`, `evalkit.text.number_sentences`, `evalkit.tasks.builtin_task`, `render_prompt`) imports from the pinned EvalKit, checked against a list written in the test, not derived from the code (EK-L-10).
- The pinned EvalKit commit in `pyproject.toml` equals the one recorded in D-28, and the installed `evalkit-harness` came from that exact commit.
