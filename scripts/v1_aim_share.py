"""Recompute the share of v1's training answers that cite an aim sentence (TGT-R10's classifier).

    uv run python scripts/v1_aim_share.py            # check the committed measurement
    uv run python scripts/v1_aim_share.py --write    # rewrite it

Needs v1's training targets locally (artifacts/targets_v1_local/train.jsonl, git-ignored); the
file's sha256 must equal v1's committed lineage (docs/measurements/lineage/targets.json).
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from ggh.targets import rules_r10

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "artifacts/targets_v1_local/train.jsonl"
OUT = ROOT / "docs/measurements/v1_targets_aim_share.json"
LINEAGE = ROOT / "docs/measurements/lineage/targets.json"


def measure() -> dict:
    raw = TRAIN.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    lineage = json.loads(LINEAGE.read_text(encoding="utf-8"))
    want = lineage["outputs"]["artifacts/targets/train.jsonl"]
    if sha != want:
        raise SystemExit(f"v1_aim_share: {TRAIN} is not v1's training set ({sha} != {want})")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    answers = [r for r in rows if r["kind"] == "answer"]

    def cites_aim(r: dict) -> bool:
        ctx, cites = r["context"], r["target"]["citations"]
        return any(rules_r10.is_aim_sentence(ctx[c - 1]) for c in cites if 0 < c <= len(ctx))

    n_aim = sum(cites_aim(r) for r in answers)
    rules_sha = hashlib.sha256(Path(rules_r10.__file__).read_bytes()).hexdigest()
    return {
        "input": "artifacts/targets/train.jsonl (v1, local copy artifacts/targets_v1_local/)",
        "input_sha256": sha,
        "classifier": "ggh.targets.rules_r10.is_aim_sentence (TGT-R10)",
        "classifier_sha256": rules_sha,
        "rows": len(rows),
        "kinds": dict(sorted(Counter(r["kind"] for r in rows).items())),
        "answer_labels": dict(sorted(Counter(r["target"]["label"] for r in answers).items())),
        "answers": len(answers),
        "answers_citing_an_aim_sentence": n_aim,
        "share": round(n_aim / len(answers), 4),
        "note": (
            "The 43 % phrase-based share is the figure recorded in OWNER D-43 at E3.2; its phrase "
            "list was not committed, so it is cited, not recomputed. An earlier figure of 3,785 "
            "did not reproduce under either version of rules_r10 (both give this count)."
        ),
    }


if __name__ == "__main__":
    got = measure()
    if "--write" in sys.argv:
        OUT.write_text(json.dumps(got, indent=2) + "\n", encoding="utf-8")
        print(f"v1_aim_share: wrote {OUT.relative_to(ROOT)}")
    else:
        old = json.loads(OUT.read_text(encoding="utf-8"))
        same = {k: old[k] for k in got if k != "classifier_sha256"} == {
            k: got[k] for k in got if k != "classifier_sha256"
        }
        print(f"v1_aim_share: {'current' if same else 'STALE'}")
        sys.exit(0 if same else 1)
