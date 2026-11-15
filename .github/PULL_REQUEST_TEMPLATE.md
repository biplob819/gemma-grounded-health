## What this changes and why

## How it was tested

- [ ] `uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest -q && uv run python scripts/secret_scan.py`
- [ ] New behaviour and its error cases have tests; no test makes a paid call or downloads a model
- [ ] No Claude or OpenAI output enters training data; nothing was tuned on the expert set
- [ ] `changelog.md` has a line
- [ ] No tokens, `.env` contents or private data anywhere in the diff
