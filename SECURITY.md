# Security

## Reporting a vulnerability

Please report security problems privately through GitHub's [private vulnerability reporting](https://github.com/biplob819/gemma-grounded-health/security/advisories/new) for this repository, not in a public issue. Include what you found, how to reproduce it, and the version or commit you tested. You will get an acknowledgement within a week; once a fix is released, the advisory is published with credit to you unless you prefer otherwise.

Supported versions: the latest release. There is no release yet; v1.0.0 is tagged at build node R4.5.

## Keys and tokens

- Today nothing in the repo reads a key: `ggh doctor` only checks which environment variables are set. From build node E3.1 (evaluation, `ANTHROPIC_API_KEY` and `OPENAI_API_KEY`, read by EvalKit) and T2.1 (Hugging Face and Kaggle), keys come from environment variables, a local `.env`, or Kaggle Secrets. `.env`, `.env.*` and `kaggle*.json` are gitignored.
- Nothing in this repo prints, logs or writes a token value. `ggh doctor` reports which credentials are set by name only.
- The training notebook arrives at build node T2.1. It will be committed with no stored outputs, and a safety test added at that node will fail if it contains any token.
- `scripts/secret_scan.py` runs in CI and pre-commit.

## The demo

The Hugging Face Space arrives at build node R4.3. Its design (Tech Spec §9) logs nothing a visitor types, only timings and errors, and that node's tests will enforce it.

## Safety of the model

Once released (build node R4.2), the model is a research artifact. It is not a medical device and must not be used for diagnosis, treatment or dosing. Report harmful behaviour (for example personal medical advice) as an issue with the input. The off-scope evaluation set will be frozen at node N1.2, before training, so reports feed later evaluation sets.
