#!/usr/bin/env python3
"""v3b reward design (owner request, OWNER D-70): how well does each local grounding checker agree
with GPT-6.1 Sol's correct-and-grounded verdict on the committed dev answers of base, v1 and v2?

Reads docs/measurements/v3/checker_scores.jsonl (written by scripts/v3b_checker_scores.py and the
deberta-v3-base prototype: one row per answer, 200 dev cases x 3 models, each checker's support
probability with the cited sentences or the whole abstract as evidence). Pure analysis, no model:
the AUC among right-label answers, 5-fold cross-validated agreement of "right label, checker at or
above its threshold, no aim citation" with Sol, and that proxy's headline per model.
It also scores the committed answers with the actual v3b reward (`ggh.rl.reward_b`), replaying
the committed checker scores instead of the models.
Writes docs/measurements/v3/checker_eval.json.
"""

from __future__ import annotations

import gzip
import json
import random
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCORES = ROOT / "docs/measurements/v3/checker_scores.jsonl"
OUT = ROOT / "docs/measurements/v3/checker_eval.json"
MODELS = ("base", "v1", "v2")
RUNS = ("dev_ground_v1", "dev_ground_v2_sol")
NAMES = {"gemma-e2b-base-q4": "base", "gemma-e2b-grounded-q4": "v1",
         "gemma-e2b-grounded-v2-q4": "v2"}  # fmt: skip
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
    out["v3b_reward_by_model"] = v3b_reward(rows)
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def _gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def v3b_reward(rows: list[dict]) -> dict[str, float]:
    """The v3b reward's mean per model on the committed answers, the checker replayed from the
    committed per-answer scores (the mean of MiniCheck-RoBERTa-L and deberta-v3-large)."""
    from ggh.rl import reward as R
    from ggh.rl import reward_b as RB

    support = {
        (r["case"], r["model"]): r["ens_minicheck_roberta_deberta_large_cited"] for r in rows
    }
    cases = {c["id"]: c for d in RUNS for c in _gz(ROOT / "docs/results" / d / "cases.jsonl.gz")}
    out: dict[str, list[float]] = {m: [] for m in MODELS}
    for d in RUNS:
        for rec in _gz(ROOT / "docs/results" / d / "records.jsonl.gz"):
            key = (rec["case_id"], NAMES[rec["completion"]["model"]])
            c, text = cases[rec["case_id"]], rec["completion"]["raw_text"] or ""
            a = R.parse(text)
            table = {}
            if a is not None:
                table[(RB.evidence(a, c["context"]) if all(1 <= i <= len(c["context"])
                       for i in a.citations) else "", a.answer)] = support[key]  # fmt: skip

            class Replay:
                def support(self, pairs, table=table):
                    return [table[p] for p in pairs]

            s = RB.score_batch([text], [c["reference_label"]], [c["context"]], Replay())[0]
            out[key[1]].append(s.reward)
    return {m: round(st.fmean(v), 4) for m, v in out.items()}


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
