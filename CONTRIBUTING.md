# Contributing to Gemma Grounded Health

Thanks for helping. Bug reports, recipe improvements, evaluation fixes and documentation are all welcome.

## Set up

```bash
git clone https://github.com/biplob819/gemma-grounded-health
cd gemma-grounded-health
uv sync
uv run pytest -q
```

The test suite needs no GPU, makes no paid calls and downloads no models or datasets. (The type checker's PyPI wrapper may fetch Node.js once on first use.) As the pipeline lands, training will be tested on a tiny random-weight model on CPU (build node T2.1) and evaluation on recorded responses (E3.3). You do not need API keys, a Kaggle account or a Hugging Face token to contribute.

## How the project is built

Work is organised as build nodes in a dependency graph (`docs/graph/graph.md`), each with a spec, acceptance criteria and proofs (`specs/<node>-<slug>/`, `docs/graph/catalog.tsv`). Check the build status with `uv run python scripts/status.py` and the full gate set for a node with `uv run python scripts/node_converge.py <node>`. The Tech Spec §13 explains the process.

## Before you open a pull request

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest -q
uv run python scripts/secret_scan.py
uv run python scripts/verify_graph_eligibility.py
uv run python scripts/node_converge.py --self-test
uv build
```

Or run `make check` (every step above except the self-test) and `make selftest`. CI runs the same checks on Python 3.11 and 3.12 on Linux and macOS.

- **Tests first.** A bug fix comes with a test that fails without it; a feature comes with tests for its behaviour and its error cases.
- **Never make a paid call or download a model from a test.**
- **No Claude or OpenAI output may enter training data or rewards.** They are used only to evaluate (PRD NFR1). Every training example must trace back to a PubMedQA row and a rule (Tech Spec §4.7).
- **Never tune on the expert set.** Recipe and data changes are judged on the dev split only (Tech Spec D-4).
- **Tokens never appear in code, notebooks, tests, logs or output.** `scripts/secret_scan.py` runs in CI and catches Anthropic, OpenAI, Hugging Face and Kaggle tokens.
- **Text files are UTF-8**, and every `open`/`read_text`/`write_text` names its encoding (a test enforces this).
- Add a line to `changelog.md`.

## Proposing a larger change

Open an issue first describing the problem and the change you have in mind. Design decisions are recorded in `docs/decisions.md`; grep it before proposing a change it already rules on.

By contributing you agree that your work is released under the project's Apache-2.0 licence, and to follow the [code of conduct](CODE_OF_CONDUCT.md).
