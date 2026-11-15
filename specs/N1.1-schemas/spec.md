# N1.1 — Schemas and config

**Wave 1. Depends on:** W0.1. **Unblocks:** N1.2, and through it every pipeline node.

## Goal
Every artifact the pipeline writes has one typed, frozen schema (Tech Spec §3), every stage writes a lineage manifest (§2.1), and the EvalKit symbols this repo relies on are pinned and checked.

## Outputs
- Dependencies: `pydantic>=2.7`, `pyyaml>=6`, and `evalkit-harness` pinned to EvalKit commit `06591f8de9b178078c971940a83019a6530fd119` by git URL (EvalKit is not on PyPI yet; D-28).
- `src/ggh/schemas/_base.py`: `Model` (frozen, `extra="forbid"`, `allow_inf_nan=False`), `Interval` re-exported from EvalKit, strict field types (`Prob` in [0, 1], `NonNegFloat`, `NonNegInt`, `Sha256` = 64 lowercase hex). Sequence fields are tuples; mappings are frozen (EK-L-8, EK-L-9).
- `src/ggh/schemas/data.py`: `TrainExample`, `DropRecord`, `DataManifest`, `AuditLabel`.
- `src/ggh/schemas/train.py`: `RecipeConfig`, `DevMetrics`, `TrainRun`.
- `src/ggh/schemas/release.py`: `ExportArtifact`, `ParityReport`, `LaptopBench`, `BehaviourRow`, `ResultsBundle`.
- `src/ggh/config.py`: `load_recipe(path)` reading YAML into `RecipeConfig` with errors that name the field; `configs/recipe.yaml` holding the §5.1 starting values.
- `src/ggh/lineage.py`: `stage_key(...)`, `write_manifest(...)` (atomic temp-and-rename), `read_manifest`, `verify(path)` walking parents.
- `tests/fixtures/schemas/*.json`: one valid fixture per schema.

## Rules
- `TrainExample.target` is EvalKit's `Answer`; citations must be within 1..len(context), sorted, unique and non-empty (B1, B7); `kind="insufficient"` and `kind="offscope"` must have label `maybe`; `kind="answer"` must have label equal to `source_label`.
- `TrainExample.id` is `pqa-a-<pmid>` for `answer`, with `:insufficient` or `:offscope` suffixes for the other kinds.
- `RecipeConfig`: `compute_dtype="bfloat16"` is refused when the target GPU is T4 or P100 (D-9) only at run time (T2.1), not in the schema; the schema checks ranges (lr in (0, 1), lora_r ≥ 1, dropout in [0, 1), `max_seq_len` a multiple of 128, epochs in (0, 3]).
- `ResultsBundle` refuses `calibration_status="calibrated"` unless `gemma_family_check` is present (OWNER D-24).
- `ExportArtifact`: `quant` required for `gguf`, forbidden otherwise; `llama_cpp_commit` required for `gguf`.
- Lineage key: sha256 of canonical JSON of (stage, code_version, sorted input hashes, params, seed); stable under dict key order; any change to any part changes the key.

## Out of scope
- Building any of the artifacts (N1.2 onwards).
