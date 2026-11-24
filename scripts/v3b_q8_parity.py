"""Exploratory, asked for by the owner (V3.7, OWNER D-73): v3b's dev parity at Q8_0.

v3b's registered parity (`ggh parity --model ft3b`, docs/measurements/v3b/parity.json) compares the
merged bf16 model with Q4_K_M and missed X3.1's 0.95 label-agreement bar (0.924). Before deciding
how v3b is scored, the owner asked for the same check at Q8_0. This reuses the committed bf16
replies (same 500 dev prompts, same grammar-constrained requests) and asks ft3b-Q8_0 behind
llama-server for its replies. It writes its own files and changes nothing registered.

    uv run python scripts/v3b_q8_parity.py
"""

from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path

from ggh.data.build import read_jsonl
from ggh.export import parity
from ggh.serve import checks, llama
from ggh.train.devrun import DEV_EVAL

ROOT = Path(".")
V3B = Path("docs/measurements/v3b")
OUT = V3B / "parity_q8_0.json"
GENERATIONS = V3B / "parity_q8_0_generations.jsonl.gz"


def labels(replies: list[str]) -> list[str | None]:
    out = []
    for r in replies:
        a = parity.parsed(r)
        out.append(a.label if a is not None else None)
    return out


def summary(rows: list[dict], bf16: list[str], other: list[str]) -> dict:
    rep = parity.build_report(bf16, other)
    lb, lo = labels(bf16), labels(other)
    gold = [r["label"] for r in rows]
    moves = Counter(f"{a}->{b}" for a, b in zip(lb, lo, strict=True) if a != b)
    return {
        "report": json.loads(rep.model_dump_json()),
        "failures": parity.failures(rep),
        "moves": dict(sorted(moves.items())),
        "label_accuracy": round(sum(a == g for a, g in zip(lo, gold, strict=True)) / len(gold), 4),
    }


def main() -> None:
    raw = gzip.decompress((ROOT / V3B / "parity_generations.jsonl.gz").read_bytes())
    committed = [json.loads(x) for x in raw.decode("utf-8").splitlines() if x]
    rows = read_jsonl(ROOT / DEV_EVAL)[: len(committed)]
    assert [r["pmid"] for r in rows] == [c["pmid"] for c in committed]
    bf16 = [c["bf16"] for c in committed]
    q4 = [c["q4_k_m"] for c in committed]
    port = llama.PORTS["ft3b"]
    proc = llama.start(ROOT, "ft3b", "Q8_0", port, log=ROOT / llama.GGUF_DIR / "serve-ft3b-q8.log")
    try:
        complete = checks.evalkit_complete(f"http://{llama.HOST}:{port}/v1")
        q8 = parity.server_replies([r["prompt"] for r in rows], complete, "gemma-e2b-ft3b-q8")
    finally:
        proc.terminate()
        proc.wait(timeout=30)
    gold = [r["label"] for r in rows]
    result = {
        "note": "exploratory (OWNER D-73); the registered parity is parity.json (Q4_K_M)",
        "n": len(rows),
        "bf16_label_accuracy": round(
            sum(a == g for a, g in zip(labels(bf16), gold, strict=True)) / len(gold), 4
        ),
        "q8_0_vs_bf16": summary(rows, bf16, q8),
        "q4_k_m_vs_bf16": summary(rows, bf16, q4),
    }
    lines = [
        json.dumps(
            {"pmid": r["pmid"], "label": r["label"], "bf16": a, "q8_0": b}, ensure_ascii=True
        )
        for r, a, b in zip(rows, bf16, q8, strict=True)
    ]
    (ROOT / GENERATIONS).write_bytes(parity.gzip_bytes(("\n".join(lines) + "\n").encode("utf-8")))
    (ROOT / OUT).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
