#!/usr/bin/env python3
"""v3b reward design (owner request, OWNER D-70): how well does each local grounding checker agree
with GPT-6.1 Sol's correct-and-grounded verdict on the committed dev answers of base, v1 and v2?

Reads docs/measurements/v3/checker_scores.jsonl (written by scripts/v3b_checker_scores.py and the
deberta-v3-base prototype: one row per answer, 200 dev cases x 3 models, each checker's support
probability with the cited sentences or the whole abstract as evidence). Pure analysis, no model:
the AUC among right-label answers, 5-fold cross-validated agreement of "right label, checker at or
above its threshold, no aim citation" with Sol, and that proxy's headline per model.
Writes docs/measurements/v3/checker_eval.json.
"""

from __future__ import annotations

import json
import random
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCORES = ROOT / "docs/measurements/v3/checker_scores.jsonl"
OUT = ROOT / "docs/measurements/v3/checker_eval.json"
MODELS = ("base", "v1", "v2")
GRID = [i / 100 for i in range(5, 100, 5)]


def _auc(pos: list[float], neg: list[float]) -> float:
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def _proxy(r: dict, k: str, t: float) -> bool:
    return bool(r["label_correct"] and r[k] >= t and not r["aim_cited"])


def main() -> dict:
    rows = [json.loads(x) for x in SCORES.read_text(encoding="utf-8").splitlines() if x.strip()]
    for r in rows:
        r["ens_minicheck_roberta_deberta_large_cited"] = (
            r["minicheck_roberta_cited"] + r["deberta_large_cited"]
        ) / 2
    cases = sorted({r["case"] for r in rows})
    random.Random(7).shuffle(cases)
    folds = [set(cases[i::5]) for i in range(5)]
    right = [r for r in rows if r["label_correct"]]
    keys = [k for k in rows[0] if k.endswith(("_cited", "_abstract")) and k != "aim_cited"]
    out: dict = {
        "answers": len(rows),
        "sol_headline": {
            m: round(st.fmean(r["sol_headline"] for r in rows if r["model"] == m), 4)
            for m in MODELS
        },
        "checkers": {},
    }
    for k in keys:
        held, ts = [], []
        for f in folds:
            train = [r for r in rows if r["case"] not in f]
            t = max(
                GRID, key=lambda t: st.fmean(_proxy(r, k, t) == r["sol_headline"] for r in train)
            )
            ts.append(t)
            held += [_proxy(r, k, t) == r["sol_headline"] for r in rows if r["case"] in f]
        t = st.median(ts)
        out["checkers"][k] = {
            "auc_among_right_labels": round(
                _auc(
                    [r[k] for r in right if r["sol_headline"]],
                    [r[k] for r in right if not r["sol_headline"]],
                ),
                4,
            ),
            "cv_agreement_with_sol": round(st.fmean(held), 4),
            "fold_thresholds": ts,
            "threshold": t,
            "proxy_headline": {
                m: round(st.fmean(_proxy(r, k, t) for r in rows if r["model"] == m), 4)
                for m in MODELS
            },
        }
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
