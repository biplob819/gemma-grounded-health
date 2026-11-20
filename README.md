# Gemma Grounded Health

This project fine-tunes a small open Gemma model (Gemma 4 E2B, 3.3 GB at 4-bit, runs offline on a laptop) to answer health research questions **only from a given abstract**, with a yes / no / maybe label, a short answer and the numbered sentences it relied on. Every model is scored with [EvalKit](https://github.com/biplob819/evalkit) on all 1,000 expert-labelled PubMedQA questions.

> **Status.** v1 and v2 are trained and benchmarked. Next: the results bundle (build node E3.3), then the Hugging Face release with the recipe, a laptop GGUF and a demo (nodes R4.2 and R4.3). Research only, not medical advice.

## v2 result: the citations are fixed, the headline is not

![Base, v1 and v2 on 1,000 PubMedQA questions: correct-and-grounded 0.404, 0.331, 0.329; cited sentences that support a claim 0.747, 0.362, 0.736; answers citing an unsupporting aim sentence 0.138, 0.560, 0.084, each with a 95% interval](docs/assets/headline_v2.svg)

v1 cited the study's aim instead of its results in more than half its answers, so it lost to its own base model on "correct and fully grounded". v2 was retrained on targets whose citations must each support the answer (OWNER D-43, D-48). On the same 1,000 questions, under a plan hashed before the run (`configs/eval_plan_v2.yaml`, OWNER D-53, D-57):

| V2.2 main run (`r-715fc213a8a5`) | Correct and fully grounded | Cited sentences that support a claim | Answers citing an unsupporting aim sentence | Label accuracy |
| --- | --- | --- | --- | --- |
| Gemma base | **0.404** | **0.747** | 0.138 | 0.501 |
| Gemma v1 | 0.331 | 0.362 | 0.560 | **0.728** |
| Gemma v2 | 0.329 | 0.736 | **0.084** | 0.678 |

- **v2 vs v1:** citation precision +0.374 and aim-citation rate −0.476, both separated; headline −0.002, a tie.
- **v2 vs base:** aim-citation rate −0.054, separated (v2 better); citation precision a tie; headline −0.075, separated (base better).
- Six pre-registered tests, paired bootstrap, Holm-corrected. About half of v2's claims still go beyond the sentence it cites, which is why the headline did not move.
- **How it was scored:** GPT-6 Luna extracted claims and GPT-6.1 Sol judged them (OWNER D-57: the Anthropic API credit ran out, so V2.2 ran on OpenAI only). Base and v1 answers come from EvalKit's cache, byte-identical to the v1 run, and were re-scored the same way. Sol's agreement with a Claude Opus 5.5 reference labeller is kappa 0.60 [0.47, 0.71] on held-out claims (OWNER D-59, D-60; the labels are a model's, not a human expert's).

![v1 and v2 give the same answer to "Is there a connection between sublingual varices and hypertension?", but v1 cites sentence 2, the study's aim, which GPT-6.1 Sol marks as not supporting the claim, while v2 cites sentence 11, the result (OR 2.25, p<0.002), which it marks as supporting](docs/assets/screenshots/v1_vs_v2_citation.png)

<details><summary>v1's published numbers (E3.2, Claude Haiku 5.5 extracting, Claude Opus 5.5 and GPT-6.1 Sol judging; not compared with v2)</summary>

| v1 main run (`r-f0f3b432a413`, 1,000 questions) | Label accuracy | Correct and fully grounded (Opus / Sol) |
| --- | --- | --- |
| Gemma base | 0.501 | 0.418 / 0.394 |
| **Gemma fine-tuned (v1)** | **0.728** | 0.388 / 0.330 |
| Claude Haiku 5.5 | 0.685 | 0.345 / 0.231 |
| GPT-6 Luna | 0.579 | 0.431 / 0.290 |

</details>

![EvalKit's report for the V2.2 main run: headline with 95% intervals and the per-model table of headline, groundedness, citation precision, aim-citation rate and accuracy](docs/assets/screenshots/evalkit_report_main_v2.png)

The full story, including how the v1 failure was found: [case study](docs/case-study.md). Every number: `docs/results/`. Known limits: [known issues](docs/known-issues.md).

[![Walkthrough: results, the citation failure, laptop benchmark, CLI and audit page](https://raw.githubusercontent.com/biplob819/gemma-grounded-health/main/docs/assets/walkthrough.gif)](https://github.com/biplob819/gemma-grounded-health/blob/main/docs/walkthrough.md)

- Product: [Gemma Grounded Health PRD](Gemma%20Grounded%20Health%20PRD.md)
- Design: [Gemma Grounded Health Tech Spec](Gemma%20Grounded%20Health%20Tech%20Spec.md)
- Build status: `uv run python scripts/status.py`

The project is built node by node with the same harness as EvalKit: a dependency-gated build graph (`docs/graph/`), a capability catalog with proof cells, converge gates (`scripts/node_converge.py`) and a skeptical verifier. See Tech Spec §13.

Contributions welcome: [CONTRIBUTING.md](CONTRIBUTING.md) · [Code of conduct](CODE_OF_CONDUCT.md) · [Security](SECURITY.md) · [Report a bug or a model behaviour](.github/ISSUE_TEMPLATE) · [Pull request checklist](.github/PULL_REQUEST_TEMPLATE.md) · Cite with [CITATION.cff](CITATION.cff).

Data: [PubMedQA](https://pubmedqa.github.io/) (Jin et al., 2019, MIT licence). Base model: Gemma 4 E2B (Apache-2.0). Code: Apache-2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE); the model will be released under Apache-2.0 too (node R4.2).
