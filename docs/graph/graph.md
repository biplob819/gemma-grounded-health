# Gemma Grounded Health build graph

Source of truth for build-node dependencies. Parsed by `scripts/harness_lib.py`.
A node may start only when every node in its `Depends on` cell is green (all its rows in `catalog.tsv` are green).
Edge rules: no cycles; a node inherits the constraints of what it depends on; `Verified by` must pass before a node flips green.
Generated from the Tech Spec §13.2 on 2026-11-15; edit here from now on and keep the Tech Spec in sync.


## Wave 0 — harness

| Node | Title | Depends on | Verified by | Deliverable | Stories |
| --- | --- | --- | --- | --- | --- |
| **W0.1** | Repo and harness | — | node_converge + verifier | Public repo with open-source community files, uv project, CI, ported harness scripts with `--self-test`, records seeded (D-1 to D-31, lessons from Appendix B) | — |

## Wave 1 — data

| Node | Title | Depends on | Verified by | Deliverable | Stories |
| --- | --- | --- | --- | --- | --- |
| **N1.1** | Schemas and config | W0.1 | node_converge + verifier | `ggh.schemas` (§3), config loaders, fixtures, EvalKit import contract | US1, US3 |
| **N1.2** | Data build and leakage check | N1.1 | node_converge + verifier | `make data`: pinned load, overlap check, splits, EvalKit rendering, off-scope evaluation set frozen, `DATA_CARD.md` | US1, US3 |
| **N1.3** | Targets and hand audit | N1.2 | node_converge + verifier | Answer and citation targets, filters, label mix, insufficient and off-scope examples (as ruled), provenance check, 50-example audit passed | US1, US4 |

## Wave 2 — training

| Node | Title | Depends on | Verified by | Deliverable | Stories |
| --- | --- | --- | --- | --- | --- |
| **T2.1** | Training recipe and pilot | N1.3 | node_converge + verifier | `ggh.train`, recipe config, thin notebook, `pytest -m recipe`, Kaggle pilot measurements, D-9 confirmed | US3 |
| **T2.2** | Full training run | T2.1 | node_converge + verifier | Clean Run-All on Kaggle, selected checkpoint, `TrainRun`, dev metrics, US3 proof | US1, US3 |

## Wave 3 — serving and evaluation

| Node | Title | Depends on | Verified by | Deliverable | Stories |
| --- | --- | --- | --- | --- | --- |
| **E3.1** | Local serving and EvalKit integration | N1.2 | node_converge + verifier | Pinned llama.cpp, base GGUF, `ggh serve`, template parity, EvalKit endpoints config, behaviour checks, dev smoke run | US1, US6 |
| **E3.2** | Benchmark runs | X3.1 | node_converge + verifier | Main, parity, off-scope and probe runs in EvalKit; spend logged | US1 |
| **E3.3** | Results, behaviour table and calibration | E3.2, V2.2 | node_converge + verifier | Behaviour table, owner reviews, Gemma-family check (as ruled), sample read, `results_bundle.json` | US1, US4 |

## Wave 3 — v2 iteration (OWNER D-43)

| Node | Title | Depends on | Verified by | Deliverable | Stories |
| --- | --- | --- | --- | --- | --- |
| **V2.1** | Citation-fixed targets and retrain (v2) | E3.2 | node_converge + verifier | `ggh targets` v2 that never cites an aim or objective sentence for an answer, dev grounding judged by EvalKit on a dev task (v1 baseline and v2), one Kaggle run, selection with a dev grounding bar, frozen v2 `TrainRun` | US1, US3 |
| **V2.2** | v2 export, parity and benchmark | V2.1 | node_converge + verifier | v2 merged bf16 and GGUFs, dev parity, a v2 plan hashed before its runs, the main run with v2 added (the other models from the cache), off-scope and probe runs for v2, spend logged | US1, US2 |

## Wave 3 — export and laptop

| Node | Title | Depends on | Verified by | Deliverable | Stories |
| --- | --- | --- | --- | --- | --- |
| **X3.1** | Export and parity | T2.2, E3.1 | node_converge + verifier | Merged bf16, fine-tuned Q4_K_M and Q8_0 GGUFs, parity report on dev | US2, US7 |
| **X3.2** | Laptop benchmark | X3.1 | node_converge + verifier | `make bench` under the §8 protocol, `LaptopBench` file | US2 |

## Wave 4 — release

| Node | Title | Depends on | Verified by | Deliverable | Stories |
| --- | --- | --- | --- | --- | --- |
| **R4.1** | Model card | E3.3, X3.2 | node_converge + verifier | Generated `MODEL_CARD.md`, Hub metadata, licence and NOTICE | US4 |
| **R4.2** | Hugging Face release | R4.1 | node_converge + verifier | Two Hub repos at a tagged revision, download-back and Ollama checks | US4, US7 |
| **R4.3** | Gradio Space | R4.2 | node_converge + verifier | ZeroGPU Space (or recorded local fallback) with side-by-side answers | US5 |
| **R4.4** | EvalKit upstream | E3.3, R4.2 | node_converge + verifier | EvalKit PR (docs, example config) and the Gemma report page, built under EvalKit's harness | US6 |
| **R4.5** | Open-source release v1.0.0 | R4.3, R4.4 | node_converge + verifier | Reproduce-from-scratch docs, README with results, CITATION.cff, tagged v1.0.0 GitHub release with notes linking model, GGUF, Space and results | US3, US4, US7 |

## Wave 5 — write-up

| Node | Title | Depends on | Verified by | Deliverable | Stories |
| --- | --- | --- | --- | --- | --- |
| **P5.1** | Write-up | R4.5 | node_converge + verifier | "How I built it" note, decision log summary, project page text with generated numbers, video script | — |

## Wave 6 — cut

O6.1 (optional DPO: rule-based preference pairs, one DPO run, an exploratory EvalKit result in its own table) was cut by OWNER D-58; its capabilities DPO-01..04 are removed from the catalog.

## Wave 7 — v3 (OWNER D-66)

| Node | Title | Depends on | Verified by | Deliverable | Stories |
| --- | --- | --- | --- | --- | --- |
| **V3.1** | v3 GRPO: reward, recipe, pre-registration and Kaggle runner | R4.5 | node_converge + verifier | Programmatic GRPO reward proven against planted hacks, a recipe continuing v2's adapter, `ggh grpo` and `ggh rl-dev`, the v3 plan hashed before training, a pinned Kaggle notebook | US1, US3 |
| **V3.2** | v3 trial verdict, main-run kernels and the selection rule | V3.1 | node_converge + verifier | The trial's outputs frozen with every stop rule recomputed (passed), the main run split into a train kernel and a score kernel rendered at one commit, the plan's dev checkpoint-selection rule as tested code, the owner's $15 scoring cap recorded | US1, US3 |
| **V3.3** | v3b: a checker reward aligned with the headline, plan and kernels | V3.2 | node_converge + verifier | A reward that tracks the headline: two local grounding checkers chosen against Sol on committed dev answers, a headline-shaped v3b reward proven against planted hacks, its trainer and held-out dev check, v3b's Kaggle kernels, and the v3b plan hashed before training | US1, US3 |

V3.4 (v3's and v3b's run outputs frozen, checkpoints selected under each plan, `TrainRun`s, export and parity) and V3.5 (the scoring runs the owner approves, the release after the owner's go-ahead) are added when their inputs exist.
