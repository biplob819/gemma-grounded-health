"""Exploratory, not in v3's plan (V3.4): v3's main-run dev replies read by v3b's checker.

v3's plan selects on its own lexical reward, which ranks models against Sol's headline
(docs/measurements/v3/reward_alignment.json). This re-reads v2's and three v3 candidates'
committed dev replies with v3b's checker (MiniCheck-RoBERTa-L and nli-deberta-v3-large, the
better proxy for Sol), on all 500 rows and on the 300 outside the checker's tuning set. It
changes nothing in the selection; the v3 result is the scoring run.

    uv run python scripts/v3_proxy_checker.py \
        docs/measurements/v3/main docs/measurements/v3/main/proxy_v3b_checker.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ggh.rl import grpo_b as G

NAMES = ("v2", "checkpoint-250", "checkpoint-550", "final")


def main(folder: Path, out: Path) -> None:
    root = Path(".")
    lines = (root / "artifacts/data/dev_eval.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(x) for x in lines][:500]
    held = G.tuning_ids(root)
    keep = [i for i, r in enumerate(rows) if r["pmid"] not in held]
    chk = G.checker(root)  # canary-checked
    result = {}
    for name in NAMES:
        text = (folder / f"dev_{name}.jsonl").read_text(encoding="utf-8").splitlines()
        rep = [json.loads(x)["reply"] for x in text]
        result[name] = {
            "all500": G.dev_summary(rep, rows, chk),
            "held": G.dev_summary([rep[i] for i in keep], [rows[i] for i in keep], chk),
        }
    out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    for name, parts in result.items():
        for part, s in parts.items():
            print(name, part, s["n"], s["mean_reward"], s["grounded_paid"], s["copied_close"])


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
