# Reproduce from scratch

Every step from a fresh clone to the release, as the commands that built v1.0.0 (the shipped model is v2, `gemma-e2b-grounded-v2-q4`). Each step names what it reads, what it writes and what it checks. The build graph, decisions and every committed number live in this repository, so a step you skip can be checked against the committed file it would have written.

Each command block is marked for `tests/test_reproduce_doc.py`:

- **run**: the test runs every command in the block on the committed files, with no network, no keys and no spend, and checks that nothing tracked by git changes (the committed outputs are what the commands write).
- **dry-run**: the block needs something a test cannot have (a GPU, a Kaggle account, the network, paid API calls, or the large files in the gitignored `artifacts/`). The test checks that each `ggh` command parses with the real CLI, each `make` target exists, and each repo path it names is committed; the reason is given on the block.

You need Python 3.11+, [uv](https://docs.astral.sh/uv/), git and about 50 GB of free disk for the full path (merged weights and GGUFs). Research only, not medical advice.

## 0. Clone and check the machine

<!-- reproduce: dry-run: network (clones the repo and installs packages) -->
```bash
git clone https://github.com/biplob819/gemma-grounded-health
cd gemma-grounded-health
uv sync
```

<!-- reproduce: run -->
```bash
uv run ggh --version
uv run ggh doctor
```

`ggh doctor` reports which credentials are set by name only (never a value); none is needed for the tests or for any **run** block here. `make check` runs what CI runs (ruff, pyright, pytest, the secret scan, the graph check and the package build).

## 1. Data

<!-- reproduce: dry-run: network (downloads PubMedQA from the Hugging Face Hub) -->
```bash
uv run ggh data
```

Builds the training pool (PQA-A) and the dev splits into `artifacts/data/`, removes every question that overlaps the 1,000-question benchmark, and checks each split's row count and sha256 against `docs/measurements/data_manifest.json`. A second run with `--offline` uses the cache only.

## 2. Training targets and the hand audit

<!-- reproduce: dry-run: needs step 1's artifacts/data and an NLI model download (the nli dependency group) -->
```bash
uv run --group nli ggh targets
uv run ggh targets --replay
```

`ggh targets` writes the answer targets (label, short answer, cited sentences) into `artifacts/targets/` with its lineage manifest; `--replay` uses the committed NLI scores (`data/nli_scores.jsonl.gz`) instead of the model and checks every hash. The drop counts are in `docs/measurements/targets_drop_report.json`.

<!-- reproduce: run -->
```bash
uv run ggh audit score
```

Applies the pass bar to the committed hand-audit labels (`docs/measurements/audit_labels.jsonl`). The v2 per-sentence audit (`uv run ggh audit score --v2`) exits non-zero by design: its round-3 sample missed the bar and the owner waived it (OWNER D-50, disclosed in `DATA_CARD.md`).

## 3. Training on Kaggle

<!-- reproduce: dry-run: needs a Kaggle account with a T4 GPU and the Kaggle CLI; edit the kernel id in notebooks/v2/kernel-metadata.json to your own -->
```bash
kaggle kernels push -p notebooks/v2
kaggle kernels output biplob819/ggh-v21-train -p artifacts/kaggle/ggh-v21-train
uv run ggh freeze --model ft2 --report artifacts/kaggle/ggh-v21-train/run/report.json --selection artifacts/kaggle/ggh-v21-train/dev/selection.json --adapter artifacts/kaggle/ggh-v21-train/dev/adapter --kernel-version 1 --repo-commit 8ff61b977580700c592ee2c8443d46570e0126f1 --generations artifacts/kaggle/ggh-v21-train/dev/generations.jsonl --log artifacts/kaggle/ggh-v21-train/ggh-v21-train.log
```

The notebook (`notebooks/v2/train_kaggle.ipynb`, rendered from `ggh.train.notebook` and tested) clones this repo at a pinned commit, rebuilds the data, replays the targets with every hash checked, trains `configs/recipe_v2.yaml` with LoRA and picks the checkpoint on dev data only. `ggh freeze` checks the downloaded adapter against the selection and writes `docs/measurements/v2/train_run.json`. GPU hours are logged in `docs/measurements/gpu-ledger.jsonl`.

## 4. Export and parity

<!-- reproduce: dry-run: needs step 3's adapter, the base model download (network) and about 30 GB of disk; parity needs `ggh serve --model ft2` running in a second terminal -->
```bash
uv run ggh merge --model ft2
uv run ggh llama fetch
uv run ggh export --model ft2
uv run ggh serve --model ft2
uv run ggh parity --model ft2
uv run ggh export-records --model ft2
uv run ggh lineage verify artifacts/parity_v2/manifest.json
```

Merges the adapter into the bf16 base on CPU, converts it to GGUF (Q4_K_M and Q8_0) with the pinned llama.cpp release, and compares the Q4_K_M file with the merged weights on 500 dev prompts (`docs/measurements/v2/parity.json`). `ggh lineage verify` walks the chain back from parity to the data.

## 5. Evaluation with EvalKit

<!-- reproduce: run -->
```bash
uv run ggh results plan-check
uv run ggh results estimate --plan configs/eval_plan_v2.yaml
```

`plan-check` confirms each evaluation plan's hash was committed before its run (pre-registration); `estimate` prices the plan from EvalKit's cache with no API call.

<!-- reproduce: dry-run: paid API calls (OpenAI judge and extractor) and the three GGUF servers (base 8080, v1 8081, v2 8083), each in its own terminal -->
```bash
uv run ggh serve --model base
uv run ggh serve --model ft
uv run ggh serve --model ft2
uv run evalkit run configs/evalkit_v2.yaml --task configs/tasks/pubmedqa_cit.yaml --n 1000
uv run evalkit run configs/evalkit_v2_parity.yaml --task configs/tasks/pubmedqa_cit.yaml --n 200
uv run evalkit run configs/evalkit_v2_solo.yaml --task configs/tasks/offscope.yaml --n 20
uv run evalkit run configs/evalkit_v2_solo.yaml --task configs/tasks/probe_original.yaml --n 100
uv run evalkit run configs/evalkit_v2_solo.yaml --task configs/tasks/probe_paraphrase.yaml --n 100
```

Each EvalKit run is copied into `docs/results/<run>/` (summary, records and cases); spend is logged in `docs/measurements/spend-ledger.jsonl`. Everything after this point reads the committed runs and costs nothing:

<!-- reproduce: run -->
```bash
uv run ggh probe report --release v2
uv run ggh parity-headline --model ft2
uv run ggh results aim-check
uv run ggh results bundle
```

The paraphrase probe, the judged parity headline, the post-hoc aim-rule count (OWNER D-62) and the results bundle (`docs/results/results_bundle.json`), the single source every published number is generated from (D-20).

## 6. Laptop benchmark

<!-- reproduce: dry-run: needs the GGUF files and a quiet laptop on mains power (Tech Spec §8) -->
```bash
uv run ggh bench --model ft2
```

Writes `docs/measurements/v2/laptop_bench.json` (tokens per second, first-token time, latency and memory).

## 7. Cards, README and release

<!-- reproduce: run -->
```bash
uv run ggh card
uv run ggh readme-table --write README.md
uv run ggh release-notes
```

`MODEL_CARD.md`, the README's results and links blocks and `docs/release_notes_v1.0.0.md`, all generated from the bundle.

<!-- reproduce: dry-run: needs the merged weights and GGUFs in artifacts/; pushing also needs HF_TOKEN and the owner's go-ahead -->
```bash
make release
uv run ggh release --owner biplob123 --plan-json docs/measurements/release_plan_v2.json
```

The release dry run lists every file for the two Hub repos with its size and sha256, verifies the lineage, and pushes nothing (`docs/measurements/release_plan_v2.json`). With `--push` it uploads, downloads every file back to check its sha256 and runs the GGUF through Ollama on one sample (`docs/measurements/release_record_v2.json`).

## A fresh-clone check

<!-- reproduce: dry-run: clones and installs into a temporary folder (several minutes); tests/test_fresh_clone.py runs it only with GGH_FRESH_CLONE=1 -->
```bash
uv run python scripts/fresh_clone_check.py
```

Clones this checkout at `HEAD` into a temporary folder, runs `uv sync`, `uv run ggh doctor` and a fast test subset there, and fails on any error. The full CI (`make check`) runs on GitHub on every push.
