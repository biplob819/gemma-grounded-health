"""Builds tests/fixtures/release/results_bundle_v1.json from the committed v1 results, plus the
golden card and README block generated from it.

Run: uv run python -m tests.release.build_v1_bundle

The bundle is `ggh.results.bundle.build` over the committed files (E3.3, RES-06), limited to the
v1 release: what `ggh results bundle` writes to docs/results/results_bundle.json, less v2. The
fixture is a frozen copy so the generators are tested on real v1 numbers (the v1 card, its
recipe and targets lineage). V2.2's runs are committed, so the full bundle now holds v1 and v2;
tests/results/test_bundle.py checks it on the real data, and tests/release/test_card_v2.py runs
the same generators on a v1 + v2 bundle. Only `spend_usd` (the whole ledger) moves with v2.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests" / "fixtures" / "release"
BUNDLE = OUT / "results_bundle_v1.json"
CARD = OUT / "MODEL_CARD_v1.md"
README_BLOCK = OUT / "readme_block_v1.md"
SUMMARIES = (
    "docs/results/main/summary.json",
    "docs/results/main_citation/summary.json",
    "docs/results/offscope/summary.json",
    "docs/results/parity/summary.json",
    "docs/results/probe_original/summary.json",
    "docs/results/probe_paraphrase/summary.json",
)


def bundle_json() -> str:
    from ggh.results.bundle import build

    return build(ROOT, versions=("v1",)).bundle.model_dump_json(indent=2) + "\n"


def bundle() -> dict[str, Any]:
    return json.loads(bundle_json())


def main() -> None:
    from ggh.release import card, readme, sources

    OUT.mkdir(parents=True, exist_ok=True)
    BUNDLE.write_text(bundle_json(), encoding="utf-8", newline="\n")
    src = sources.load(ROOT, sources.Paths(bundle=BUNDLE.relative_to(ROOT)))
    CARD.write_text(card.render(src), encoding="utf-8")
    README_BLOCK.write_text(readme.results_block(src), encoding="utf-8")
    print(f"wrote {BUNDLE.relative_to(ROOT)}, {CARD.name}, {README_BLOCK.name}")


if __name__ == "__main__":
    main()
