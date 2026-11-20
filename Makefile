# Pipeline targets are added by the nodes that build them (Tech Spec §2.2): `data` (N1.2), `targets` (N1.3), `bench` (X3.2).
# The other targets run the build harness.
.PHONY: check status converge selftest data targets bench card release

# check runs every CI step except the self-tests, which selftest runs
check:
	uv run ruff check .
	uv run ruff format --check .
	uv run pyright
	uv run pytest -q
	uv run python scripts/secret_scan.py
	uv run python scripts/verify_graph_eligibility.py
	uv build

data:
	uv run ggh data

targets:
	uv run --group nli ggh targets

# laptop benchmark, Tech Spec §8 (X3.2): mains power, no evaluation run in progress
bench:
	uv run ggh bench

# model card from the results bundle (R4.1, D-20)
card:
	uv run ggh card

# Hub release plan (R4.2, Tech Spec §10.2): a dry run that lists every file and checks lineage
# and sha256s. Pushing needs `uv run ggh release --push --owner <owner>`, HF_TOKEN and the
# owner's go-ahead. It follows the shipped release (v2: parity_v2 -> export_ft2 -> train_v2 ->
# targets_v2 -> data); `--release v1` plans v1
release:
	uv run ggh lineage verify artifacts/parity_v2/manifest.json
	uv run ggh release --dry-run

status:
	uv run python scripts/status.py

converge:
	uv run python scripts/node_converge.py $(NODE)

selftest:
	uv run python scripts/verify_graph_eligibility.py --self-test
	uv run python scripts/node_start.py --self-test
	uv run python scripts/node_converge.py --self-test
	uv run python scripts/secret_scan.py --self-test
