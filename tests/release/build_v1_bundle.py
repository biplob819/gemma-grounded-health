"""Builds tests/fixtures/release/results_bundle_v1.json from the committed v1 results, plus the
golden card and README block generated from it.

Run: uv run python -m tests.release.build_v1_bundle

The bundle is `ggh.results.bundle.build` over the committed files (E3.3, RES-06), exactly what
`ggh results bundle` writes to docs/results/results_bundle.json; the fixture is a frozen copy so
the generators are tested on real v1 numbers. Only v1 is in it until V2.2's main run lands, and
the same generators then run unchanged on the v1 + v2 bundle.
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

    return build(ROOT).bundle.model_dump_json(indent=2) + "\n"


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
