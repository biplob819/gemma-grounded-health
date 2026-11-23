#!/usr/bin/env python3
"""V3.3 verifier follow-up: the real v3b checker on hand-written probe replies to one dev abstract
(dev-1471704, gold "no"), to record its blind spot: it never sees the question, so copied or
generic result sentences pass while an answer drawing the question's conclusion fails.
Needs `uv sync --group nli` and the two models. Writes docs/measurements/v3/checker_probes.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.rl.test_reward import CONTEXT, GOOD  # noqa: E402

from ggh.rl.checker import HFChecker  # noqa: E402

PROBES = [
    ("verbatim copy of the cited sentence", CONTEXT[6], [7]),
    ("copy of an unrelated result sentence", CONTEXT[9], [10]),
    ("generic study description", "The study measured immunoreactive beta-endorphin levels in "
     "patients with polycystic ovary syndrome and controls.", [6]),
    ("honest answer drawing the question's conclusion", "Elevated immunoreactive beta-endorphin in "
     "polycystic ovary syndrome is not caused by increased pituitary secretion.", [6]),
    ("the reward tests' good answer", GOOD, [6]),
]  # fmt: skip


def main() -> list[dict]:
    pairs = [(" ".join(CONTEXT[c - 1] for c in cites), claim) for _, claim, cites in PROBES]
    got = HFChecker(device="cpu").support(pairs)
    out = [{"probe": n, "claim": c, "citations": k, "support": round(s, 4)}
           for (n, c, k), s in zip(PROBES, got, strict=True)]  # fmt: skip
    (ROOT / "docs/measurements/v3/checker_probes.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8")  # fmt: skip
    return out


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
