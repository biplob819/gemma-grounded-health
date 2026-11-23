#!/usr/bin/env python3
"""V3.3 provenance: the deberta-v3-base prototype (the cached NLI model of N1.3) on the committed
dev answers of base, v1 and v2, joined-citation and per-sentence variants. Its joined-citation
score is the deberta_base_cited column of docs/measurements/v3/checker_scores.jsonl, merged with
scripts/v3b_checker_scores.py's output by (case, model).
Usage: uv run --group nli python scripts/v3b_deberta_base_scores.py OUT.json
"""

import gzip
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "src")
from ggh.rl import reward as R
from ggh.targets.nli import HFScorer


def load(p):
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


RUNS = ("dev_ground_v1", "dev_ground_v2_sol")
cases = {c["id"]: c for d in RUNS for c in load(f"docs/results/{d}/cases.jsonl.gz")}
names = {
    "gemma-e2b-base-q4": "base",
    "gemma-e2b-grounded-q4": "v1",
    "gemma-e2b-grounded-v2-q4": "v2",
}


def sents(t):
    return [s for s in re.split(r"(?<=[.!?])\s+", t.strip()) if len(s.split()) >= 3] or [t]


rows = []
need = set()
for d in RUNS:
    for r in load(f"docs/results/{d}/records.jsonl.gz"):
        c = cases[r["case_id"]]
        text = r["completion"]["raw_text"] or ""
        a = R.parse(text)
        s = R.score(text, c["reference_label"], c["context"])
        cites = []
        if a is not None and a.citations and all(1 <= i <= len(c["context"]) for i in a.citations):
            cites = sorted(set(a.citations))
        row = dict(
            case=r["case_id"],
            model=names[r["completion"]["model"]],
            reward=s.reward,
            lc=s.label_correct,
            aim=s.aim_cited,
            head=bool(r["scores"]["correct_and_grounded"].get("gpt-6.1-sol")),
            gold=c["reference_label"],
        )
        if cites and a is not None:
            joined = " ".join(c["context"][i - 1] for i in cites)
            hs = sents(a.answer)
            row["pairs_joined_whole"] = (joined, a.answer)
            row["pairs_joined_sent"] = [(joined, h) for h in hs]
            row["pairs_each_whole"] = [(c["context"][i - 1], a.answer) for i in cites]
            row["pairs_each_sent"] = [(c["context"][i - 1], h) for i in cites for h in hs]
            need.add(row["pairs_joined_whole"])
            need.update(row["pairs_joined_sent"])
            need.update(row["pairs_each_whole"])
            need.update(row["pairs_each_sent"])
        rows.append(row)
need = sorted(need)
sc = HFScorer()
P = dict(zip(need, sc.entail(need), strict=True))
for r in rows:
    if "pairs_joined_whole" in r:
        r["joined_whole"] = P[r["pairs_joined_whole"]]
        r["joined_sent_min"] = min(P[p] for p in r["pairs_joined_sent"])
        r["each_whole_max"] = max(P[p] for p in r["pairs_each_whole"])
        hs = sorted({h for _, h in r["pairs_each_sent"]})
        r["each_sent_minmax"] = min(
            max(
                P[(pr, h)]
                for pr, _ in r["pairs_each_whole"]
                for (pr2, h2) in [(pr, h)]
                if (pr, h) in P
            )
            for h in hs
        )
    else:
        for k in ("joined_whole", "joined_sent_min", "each_whole_max", "each_sent_minmax"):
            r[k] = 0.0
    for k in list(r):
        if k.startswith("pairs_"):
            del r[k]
Path(sys.argv[1]).write_text(json.dumps(rows), encoding="utf-8")
print("pairs scored", len(need))
