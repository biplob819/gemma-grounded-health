#!/usr/bin/env python3
"""Does the v3 GRPO reward rank models the way the judged headline does? (owner request, 2026-11-23)

Scores the committed dev answers of base, v1 and v2 (the 200-case dev grounding runs
`docs/results/dev_ground_v1` and `dev_ground_v2_sol`, Q4_K_M through llama-server) with
`ggh.rl.reward` and sets them beside GPT-6.1 Sol's correct-and-grounded verdict on the same
answers. Dev data only; no model call, no API call.
Writes docs/measurements/v3/reward_alignment.json.
"""

from __future__ import annotations

import gzip
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

from ggh.rl import reward as R

ROOT = Path(__file__).resolve().parents[1]
RUNS = ("dev_ground_v1", "dev_ground_v2_sol")
NAMES = {"gemma-e2b-base-q4": "base", "gemma-e2b-grounded-q4": "v1",
         "gemma-e2b-grounded-v2-q4": "v2"}  # fmt: skip
JUDGE = "gpt-6.1-sol"
OUT = ROOT / "docs/measurements/v3/reward_alignment.json"


def _load(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> dict:
    cases = {c["id"]: c for d in RUNS for c in _load(ROOT / "docs/results" / d / "cases.jsonl.gz")}
    by: dict[str, dict[str, dict]] = defaultdict(dict)
    for d in RUNS:
        for r in _load(ROOT / "docs/results" / d / "records.jsonl.gz"):
            by[r["completion"]["model"]][r["case_id"]] = r
    ids = sorted(cases)
    models, levels = {}, defaultdict(list)
    for model, name in NAMES.items():
        rows = []
        for i in ids:
            r, c = by[model][i], cases[i]
            s = R.score(r["completion"]["raw_text"] or "", c["reference_label"], c["context"])
            head = bool(r["scores"]["correct_and_grounded"].get(JUDGE))
            rows.append((s, head))
            levels[s.reward].append(head)
        top = [h for s, h in rows if s.reward == R.R_LABEL + R.R_GROUNDED]
        models[name] = {
            "mean_reward": round(st.fmean(s.reward for s, _ in rows), 4),
            "sol_headline": round(st.fmean(h for _, h in rows), 4),
            "label_accuracy": round(st.fmean(s.label_correct for s, _ in rows), 4),
            "reward_grounding_paid": round(
                st.fmean(s.label_correct and s.grounded for s, _ in rows), 4
            ),
            "aim_cited": round(st.fmean(s.aim_cited for s, _ in rows), 4),
            "at_max_reward": len(top),
            "sol_headline_among_max_reward": round(st.fmean(top), 4) if top else None,
        }
    result = {
        "cases": len(ids),
        "gold_mix": dict(Counter(cases[i]["reference_label"] for i in ids)),
        "judge": JUDGE,
        "models": models,
        "sol_headline_by_reward_value": {
            str(k): {"answers": len(v), "sol_headline": round(st.fmean(v), 4)}
            for k, v in sorted(levels.items())
        },
        "rank_by_reward": sorted(models, key=lambda m: -models[m]["mean_reward"]),
        "rank_by_sol_headline": sorted(models, key=lambda m: -models[m]["sol_headline"]),
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
