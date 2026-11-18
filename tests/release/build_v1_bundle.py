"""Builds tests/fixtures/release/results_bundle_v1.json from the committed v1 results, plus the
golden card and README block generated from it.

Run: uv run python -m tests.release.build_v1_bundle

This is a test fixture, not the release bundle: E3.3 writes docs/results/results_bundle.json
with the real behaviour table and calibration. The fixture only uses what is committed today
(the E3.2 summaries, X3.1 parity, X3.2 bench, T2.2 run, N1.3 audit), so the generators can be
tested on real v1 numbers now and rerun unchanged on the v2 bundle later.
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
M = Path("docs") / "measurements"
SUMMARIES = (
    "docs/results/main/summary.json",
    "docs/results/offscope/summary.json",
    "docs/results/probe_original/summary.json",
    "docs/results/probe_paraphrase/summary.json",
)
BASE, FT = "gemma-e2b-base-q4", "gemma-e2b-grounded-q4"


def _json(rel: str | Path) -> Any:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def _neg(d: dict[str, float]) -> dict[str, float]:
    return {"point": -d["point"], "lo": -d["hi"], "hi": -d["lo"]}


def bundle() -> dict[str, Any]:
    main = _json(SUMMARIES[0])
    primary: dict[str, Any] = {}
    for c in main["comparisons"]:  # diff is model_a minus model_b
        if {c["model_a"], c["model_b"]} != {BASE, FT}:
            continue
        diff = c["diff"] if c["model_a"] == FT else _neg(c["diff"])
        key = f"{c['metric']}: {FT} minus {BASE}" + (f" ({c['judge']})" if c["judge"] else "")
        primary[key] = diff
    behaviour = [
        {
            "rule": "B7",
            "model": m["model"],
            "metric": "citation_validity",
            "value": m["citation_validity"],
            "n": m["n"],
            "source": "evalkit:citation_validity",
        }
        for m in main["models"]
    ]
    parity = _json(M / "parity_report.json")
    parity["judged_headline_delta"] = _json(M / "parity_headline.json")["bf16_minus_q4_headline"]
    labels = [
        json.loads(line)
        for line in (ROOT / M / "audit_labels.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {
        "schema_version": "1.0",
        "evalkit_run_ids": [_json(p)["run_id"] for p in SUMMARIES],
        "evalkit_version": _json(M / "data_manifest.json")["evalkit_version"],
        "judge_mode": main["judge_mode"],
        "calibration_status": main["calibration_status"],
        "gemma_family_check": None,
        "summary_paths": list(SUMMARIES),
        "primary": primary,
        "behaviour": behaviour,
        "parity": parity,
        "bench": _json(M / "laptop_bench.json")["variants"]["ft-Q4_K_M"],
        "train": _json(M / "train_run.json"),
        "audit_pass_rate": sum(lab["verdict"] == "ok" for lab in labels) / len(labels),
        "artifacts": _json(M / "export_artifacts.json"),
        "lineage_ok": True,
    }


def main() -> None:
    from ggh.release import card, readme, sources

    OUT.mkdir(parents=True, exist_ok=True)
    BUNDLE.write_text(json.dumps(bundle(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    src = sources.load(ROOT, sources.Paths(bundle=BUNDLE.relative_to(ROOT)))
    CARD.write_text(card.render(src), encoding="utf-8")
    README_BLOCK.write_text(readme.results_block(src), encoding="utf-8")
    print(f"wrote {BUNDLE.relative_to(ROOT)}, {CARD.name}, {README_BLOCK.name}")


if __name__ == "__main__":
    main()
