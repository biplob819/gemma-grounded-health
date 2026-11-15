# Changelog

Format: `[TYPE] [NODE] — summary`. Types: FEAT, FIX, DOCS, HARNESS, RELEASE.

## 2026-11-15
- [FEAT] [N1.1] — green after 3 verifier rounds (D-32 path-and-hash keys, canonical manifest paths, chain-only verify with input and source checks; OWNER D-33 caps rounds at 3). Schemas and config: `ggh.schemas` (data, training and release schemas, frozen and strict), `configs/recipe.yaml` pinned to google/gemma-4-E2B-it@4adf157, stage lineage manifests with `ggh lineage verify`, EvalKit pinned by commit with an import contract test
- [HARNESS] [W0.1] — green after 17 verifier rounds (lessons L-1 to L-18; decisions D-29 to D-31 on self-test standards, enumerated contracts and detection scope). Repo and harness: uv project and `ggh` CLI, EvalKit's harness scripts ported (18-node graph, 113-row catalog, 13 gates with self-tests), Hugging Face and Kaggle token patterns in the secret scan, spend and GPU ledgers, decisions D-1 to D-31, a neutral public PRD (OWNER D-26), open-source community files and CITATION.cff (OWNER D-27), lessons carried from EvalKit, CI on Linux and macOS
