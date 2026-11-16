"""RCP-05: deterministic dev metrics (D-4, Tech Spec §5.2). Never a model judge.

`dev_metrics` reads generated texts: the constrained replies are the scored path (D-6) and
give every metric except `schema_valid_unconstrained`, which comes from the unconstrained
replies of the same prompts. Parsing is EvalKit's own `parse_answer`, so "valid" here means
what it means in the benchmark.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from evalkit.pipeline.answer import SchemaError, parse_answer

from ggh.schemas import DevMetrics

LABELS = ("yes", "no", "maybe")
MAX_WORDS = 80


def _parse(text: str) -> Any:
    try:
        return parse_answer(text, LABELS)
    except SchemaError:
        return None


def dev_metrics(
    rows: Sequence[Mapping[str, Any]], constrained: Sequence[str], unconstrained: Sequence[str]
) -> DevMetrics:
    """`rows` are dev_eval rows (`sentences`, `label`), aligned with both text lists."""
    if not (len(rows) == len(constrained) == len(unconstrained)) or not rows:
        raise ValueError("rows, constrained and unconstrained must be non-empty and aligned")
    parsed = [_parse(t) for t in constrained]
    valid = [a for a in parsed if a is not None]
    n = len(rows)
    cites_ok = has_cite = over = 0
    words = 0
    hits: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    for row, a in zip(rows, parsed, strict=True):
        ref = row["label"]
        totals[ref] += 1
        if a is None:
            continue  # a schema failure counts against every metric
        k = len(row["sentences"])
        has_cite += bool(a.citations)
        cites_ok += bool(a.citations) and all(1 <= c <= k for c in a.citations)
        hits[ref] += a.label == ref
        w = len(a.answer.split())
        words += w
        over += w > MAX_WORDS
    return DevMetrics(
        n=n,
        schema_valid=len(valid) / n,
        schema_valid_unconstrained=sum(_parse(t) is not None for t in unconstrained) / n,
        citation_valid=cites_ok / n,
        has_citation=has_cite / n,
        label_accuracy={lab: hits[lab] / totals[lab] for lab in sorted(totals)},
        mean_words=words / len(valid) if valid else 0.0,
        over_80_words=over / len(valid) if valid else 0.0,
    )


def passes_selection(m: DevMetrics, bar: float = 0.98) -> bool:
    """Tech Spec §5.2 step 2."""
    return m.schema_valid >= bar and m.has_citation >= bar and m.citation_valid >= bar
