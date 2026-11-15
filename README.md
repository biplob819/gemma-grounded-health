# Gemma Grounded Health

This project is building a fine-tuned small open Gemma model (Gemma 4 E2B) that answers health research questions **only from a given abstract** and cites the sentences it used. The training recipe and a Kaggle pilot come at build node T2.1, the full training run at T2.2. The plan: score it with [EvalKit](https://github.com/biplob819/evalkit) against its own base model, Claude Haiku 5.5 and GPT-6 Luna (build node E3.2), and release it on Hugging Face with the training recipe, a GGUF for laptops and a demo (nodes R4.2 and R4.3).

> **Status: under construction.** Results, model and demo links appear here once the build reaches them. Research only, not medical advice.

- Product: [Gemma Grounded Health PRD](Gemma%20Grounded%20Health%20PRD.md)
- Design: [Gemma Grounded Health Tech Spec](Gemma%20Grounded%20Health%20Tech%20Spec.md)
- Build status: `uv run python scripts/status.py`

The project is built node by node with the same harness as EvalKit: a dependency-gated build graph (`docs/graph/`), a capability catalog with proof cells, converge gates (`scripts/node_converge.py`) and a skeptical verifier. See Tech Spec §13.

Contributions welcome: [CONTRIBUTING.md](CONTRIBUTING.md) · [Code of conduct](CODE_OF_CONDUCT.md) · [Security](SECURITY.md) · [Report a bug or a model behaviour](.github/ISSUE_TEMPLATE) · [Pull request checklist](.github/PULL_REQUEST_TEMPLATE.md) · Cite with [CITATION.cff](CITATION.cff).

Data: [PubMedQA](https://pubmedqa.github.io/) (Jin et al., 2019, MIT licence). Base model: Gemma 4 E2B (Apache-2.0). Code: Apache-2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE); the model will be released under Apache-2.0 too (node R4.2).
