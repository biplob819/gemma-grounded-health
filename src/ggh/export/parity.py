"""EXP-04, EXP-06: dev parity of the fine-tuned Q4_K_M GGUF against the merged bf16 model
(Tech Spec §6, H2).

Both sides answer the 500 `dev_eval` prompts greedily under EvalKit's answer schema: bf16 in
Transformers with the T2.2 xgrammar generator, Q4_K_M behind llama-server through EvalKit's
own adapter (the JSON-schema request a benchmark run sends). The bar: label agreement at
least 0.95 and constrained schema validity within 1 point. Disagreements count against
agreement, and so does a reply that does not parse on either side.
"""

from __future__ import annotations

import gzip
import io
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from evalkit.pipeline.answer import SchemaError, parse_answer

from ggh.schemas import ParityReport
from ggh.serve import checks

LABELS = ("yes", "no", "maybe")
AGREEMENT_BAR = 0.95
SCHEMA_GAP = 0.01
MEASURE = Path("docs") / "measurements"
REPORT = MEASURE / "parity_report.json"
GENERATIONS = MEASURE / "parity_generations.jsonl.gz"
SAMPLE = MEASURE / "parity_sample.jsonl"
Q4_NAME = "gemma-e2b-ft-q4"


def parsed(text: str) -> Any | None:
    try:
        return parse_answer(text, LABELS)
    except SchemaError:
        return None


def build_report(bf16: Sequence[str], q4: Sequence[str]) -> ParityReport:
    if len(bf16) != len(q4) or not bf16:
        raise ValueError("bf16 and Q4_K_M replies must be non-empty and aligned")
    a = [parsed(t) for t in bf16]
    b = [parsed(t) for t in q4]
    n = len(a)
    both = [(x, y) for x, y in zip(a, b, strict=True) if x is not None and y is not None]
    return ParityReport(
        n=n,
        label_agreement=sum(x.label == y.label for x, y in both) / n,
        citation_exact_match=sum(list(x.citations) == list(y.citations) for x, y in both) / n,
        schema_valid={
            "bf16": sum(x is not None for x in a) / n,
            "q4_k_m": sum(y is not None for y in b) / n,
        },
    )


def passes(rep: ParityReport) -> bool:
    gap = abs(rep.schema_valid["bf16"] - rep.schema_valid["q4_k_m"])
    return rep.label_agreement >= AGREEMENT_BAR and gap <= SCHEMA_GAP + 1e-12


def failures(rep: ParityReport) -> list[str]:
    out = []
    if rep.label_agreement < AGREEMENT_BAR:
        out.append(f"label agreement {rep.label_agreement:.3f} < {AGREEMENT_BAR}")
    gap = abs(rep.schema_valid["bf16"] - rep.schema_valid["q4_k_m"])
    if gap > SCHEMA_GAP + 1e-12:
        out.append(f"schema validity differs by {gap:.3f} > {SCHEMA_GAP}")
    return out


def server_replies(prompts: Sequence[str], complete: checks.Complete) -> list[str]:
    """Constrained replies from the running ft server, one EvalKit request per prompt."""
    return [complete(checks.request(p, Q4_NAME, constrained=True)).raw_text for p in prompts]


def load_bf16(merged: Path) -> tuple[Any, Any]:  # pragma: no cover - 10 GB model
    import torch  # pyright: ignore[reportMissingImports]
    from transformers import (  # pyright: ignore[reportMissingImports]
        AutoModelForImageTextToText,
        AutoTokenizer,
    )

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForImageTextToText.from_pretrained(merged, dtype=torch.bfloat16)
    model.to(device).eval()  # pyright: ignore[reportArgumentType]
    return model, AutoTokenizer.from_pretrained(merged)


def gzip_bytes(raw: bytes) -> bytes:
    """Reproducible gzip: no file name, zero mtime."""
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0) as g:
        g.write(raw)
    return buf.getvalue()


def write(root: Path, rows: Sequence[dict[str, Any]], bf16: Sequence[str], q4: Sequence[str],
          rep: ParityReport) -> None:  # fmt: skip
    lines = [
        json.dumps(
            {"pmid": r["pmid"], "label": r["label"], "bf16": a, "q4_k_m": b}, ensure_ascii=True
        )
        for r, a, b in zip(rows, bf16, q4, strict=True)
    ]
    (root / MEASURE).mkdir(parents=True, exist_ok=True)
    (root / GENERATIONS).write_bytes(gzip_bytes(("\n".join(lines) + "\n").encode("utf-8")))
    (root / SAMPLE).write_text("\n".join(lines[::25]) + "\n", encoding="utf-8")
    (root / REPORT).write_text(rep.model_dump_json(indent=2) + "\n", encoding="utf-8")


def run(
    root: Path,
    rows: Sequence[dict[str, Any]],
    *,
    bf16_gen: Callable[[Sequence[str]], list[str]],
    q4_gen: Callable[[Sequence[str]], list[str]],
) -> ParityReport:
    prompts = [r["prompt"] for r in rows]
    bf16 = bf16_gen(prompts)
    q4 = q4_gen(prompts)
    rep = build_report(bf16, q4)
    write(root, rows, bf16, q4, rep)
    return rep
