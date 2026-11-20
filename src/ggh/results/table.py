"""The behaviour table (Tech Spec §7.3, D-8, RES-01): B1-B8 per model, each row with its source.

EvalKit's numbers (B2, B7) are copied from its summary; B6 counts its verdicts; B1, B3 and B4
come from `ggh.evals.behaviour` over the committed records; B5 is the sample read (§7.6) and B8
the owner's review of the off-scope answers, with the regex pre-sort beside it. A row whose
input does not exist yet (an owner review not done) has value None, never a guess.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from evalkit.schemas import Interval

from ggh.evals import behaviour as B
from ggh.results.runs import Run
from ggh.schemas import BehaviourRow, SampleReadSummary


def _iv(d: dict[str, Any]) -> Interval:
    return Interval(point=d["point"], lo=d["lo"], hi=d["hi"])


def _ci(point: float | None, ci: Sequence[float] | None) -> Interval | float | None:
    if point is None:
        return None
    if ci is not None and ci[0] <= point <= ci[1]:
        return Interval(point=point, lo=ci[0], hi=ci[1])
    return point


def row(
    rule: str,
    model: str,
    metric: str,
    value: Any,
    n: int,
    source: str,
    run: str,
    reviewer: str | None = None,
    review_prompt: str | None = None,
) -> BehaviourRow:
    return BehaviourRow.model_validate(
        {
            "rule": rule,
            "model": model,
            "metric": metric,
            "value": value,
            "n": n,
            "source": source,
            "run": run,
            "reviewer": reviewer,
            "review_prompt": review_prompt,
        }
    )


def unsupported_share(raw: Iterable[dict[str, Any]], judges: Sequence[str]) -> dict[str, tuple]:
    """B6: per judge, (share of judged claims marked unsupported, judged claims)."""
    bad: Counter[str] = Counter()
    seen: Counter[str] = Counter()
    for r in raw:
        for v in r["verdicts"]:
            seen[v["judge"]] += 1
            bad[v["judge"]] += v["verdict"] != "supported"
    return {j: (bad[j] / seen[j] if seen[j] else None, seen[j]) for j in judges}


def run_rows(run: Run, *, citation: Run | None = None) -> list[BehaviourRow]:
    """B1-B4, B6 and B7 for every model of one benchmark run."""
    out: list[BehaviourRow] = []
    judges = list(run.manifest["judges"])
    by_model = B.by_model(run.records)
    raw_by: dict[str, list[dict[str, Any]]] = {}
    for r in run.raw_records:
        raw_by.setdefault(r["completion"]["model"], []).append(r)
    for model in run.models():
        s = run.model_summary(model)
        recs = by_model.get(model, [])
        b1 = B.b1(recs)
        out.append(
            row(
                "B1",
                model,
                "schema_failures_or_zero_citations",
                b1["rate"],
                b1["n"],
                "ggh:b1",
                run.name,
            )
        )
        for j in judges:
            out.append(
                row(
                    "B2",
                    model,
                    f"headline:{j}",
                    _iv(s["headline"][j]),
                    s["n"],
                    "evalkit:headline",
                    run.name,
                )
            )
            out.append(
                row(
                    "B2",
                    model,
                    f"groundedness_mean:{j}",
                    _iv(s["groundedness_mean"][j]),
                    s["n"],
                    "evalkit:groundedness",
                    run.name,
                )
            )
        b3 = B.b3(recs, run.cases)
        out.append(
            row(
                "B3",
                model,
                "maybe_recall",
                _ci(b3["recall"], b3["recall_ci95"]),
                b3["reference_maybe"],
                "ggh:b3",
                run.name,
            )
        )
        out.append(
            row(
                "B3",
                model,
                "maybe_precision",
                _ci(b3["precision"], b3["precision_ci95"]),
                b3["predicted_maybe"],
                "ggh:b3",
                run.name,
            )
        )
        b4 = B.b4(recs)
        out.append(
            row(
                "B4", model, "over_80_words_share", b4["share"], b4["n_answers"], "ggh:b4", run.name
            )
        )
        for j, (share, n) in unsupported_share(raw_by.get(model, []), judges).items():
            out.append(
                row(
                    "B6",
                    model,
                    f"unsupported_claim_share:{j}",
                    share,
                    n,
                    "evalkit:verdicts",
                    run.name,
                )
            )
        out.append(
            row(
                "B7",
                model,
                "citation_validity",
                _iv(s["citation_validity"]),
                s["n"],
                "evalkit:citation_valid",
                run.name,
            )
        )
        cs = citation.model_summary(model) if citation is not None else s
        src = citation.name if citation is not None else run.name
        for metric in ("citation_precision", "aim_citation_rate"):
            for j, iv in sorted((cs.get(metric) or {}).items()):
                out.append(
                    row(
                        "B7",
                        model,
                        f"{metric}:{j}",
                        _iv(iv),
                        cs["n"],
                        "evalkit:citation_support",
                        src,
                    )
                )
    return out


# who reads B5 and B8 (OWNER D-61: Claude Opus 5.5, blind, under committed prompts)
REVIEW = "model_review"
B5_PROMPT = "configs/sample_read_prompt.md"
B8_PROMPT = "configs/offscope_review_prompt.md"


def _one(values: Iterable[str | None]) -> str | None:
    """The single value a set of review rows agrees on, else None (mixed or missing)."""
    found = set(values)
    return found.pop() if len(found) == 1 and None not in found else None


def b5_rows(
    summaries: Sequence[SampleReadSummary], models: Iterable[str], run: str
) -> list[BehaviourRow]:
    """B5: answers read per model (§7.3, not gated); value is the share with no fault found."""
    by = {s.model: s for s in summaries if s.run == run}
    out = []
    for m in models:
        s = by.get(m)
        ok = s.findings.get("ok", 0) if s else 0
        value = ok / s.read if s and s.read and s.complete else None
        reviewer = _one(s.read_by) if s else None
        prompt = B5_PROMPT if reviewer and B5_PROMPT in reviewer else None
        out.append(
            row(
                "B5",
                m,
                "sample_read_ok_share",
                value,
                s.read if s else 0,
                REVIEW,
                run,
                reviewer,
                prompt,
            )
        )
    return out


def load_b8_reviewers(path: Path) -> dict[tuple[str, str], tuple[str | None, str | None]]:
    """(model, case_id) -> (`reviewed_by`, `prompt`) from docs/measurements/b8_review.jsonl."""
    if not path.is_file():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[(r["model"], r["case_id"])] = (r.get("reviewed_by"), r.get("prompt"))
    return out


def load_b8_review(path: Path) -> dict[tuple[str, str], bool]:
    """docs/measurements/b8_review.jsonl: {"model", "case_id", "gives_advice": bool} per line."""
    if not path.is_file():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if not isinstance(r.get("gives_advice"), bool):
                raise ValueError(f"{path}: gives_advice must be true or false: {line}")
            out[(r["model"], r["case_id"])] = r["gives_advice"]
    return out


def b8_rows(
    offscope: Run,
    review: dict[tuple[str, str], bool],
    models: Iterable[str] | None = None,
    reviewers: dict[tuple[str, str], tuple[str | None, str | None]] | None = None,
) -> list[BehaviourRow]:
    """B8: the regex pre-sort (ordering only) and the owner's verdict once every answer of a
    model is reviewed; B8 passes for a model only at zero advice answers (§7.3)."""
    out = []
    by_model = B.by_model(offscope.records)
    for m in models if models is not None else offscope.models():
        recs = by_model.get(m, [])
        pre = B.b8_presort(recs)
        out.append(
            row(
                "B8",
                m,
                "presort_flagged_share",
                pre["flagged"] / pre["n"] if pre["n"] else None,
                pre["n"],
                "ggh:b8",
                offscope.name,
            )
        )
        marks = [review.get((m, r.case_id)) for r in recs]
        done = bool(marks) and all(x is not None for x in marks)
        advice = sum(bool(x) for x in marks)
        meta = [(reviewers or {}).get((m, r.case_id), (None, None)) for r in recs]
        out.append(
            row(
                "B8",
                m,
                "gives_advice_count",
                float(advice) if done else None,
                sum(x is not None for x in marks),
                REVIEW,
                offscope.name,
                _one(who[0] for who in meta) if done else None,
                _one(who[1] for who in meta) if done else None,
            )
        )
    return out


# shown under every table with B5 or B8 rows (OWNER D-61: say who reviewed, and how)
REVIEW_NOTE = (
    "`model_review`: a blind review by Claude Opus 5.5 subagents, not the owner (OWNER D-61); "
    "B5 under `configs/sample_read_prompt.md`, B8 under `configs/offscope_review_prompt.md` "
    "(question and answer only)."
)
COUNT_METRICS = ("gives_advice_count",)


def _fmt(v: Any, metric: str = "") -> str:
    if v is None:
        return "pending"
    if isinstance(v, Interval):
        return f"{v.point:.3f} [{v.lo:.3f}, {v.hi:.3f}]"
    if metric in COUNT_METRICS:
        return f"{v:.0f}"
    return f"{v:.3f}"


def markdown(rows: Sequence[BehaviourRow]) -> str:
    """One table per run: a line per (rule, metric), a column per model, sources at the end."""
    out: list[str] = []
    runs = sorted({r.run or "" for r in rows})
    for run in runs:
        rs = [r for r in rows if (r.run or "") == run]
        models = list(dict.fromkeys(r.model for r in rs))
        keys = sorted(dict.fromkeys((r.rule, r.metric, r.source) for r in rs), key=lambda k: k[0])
        cell = {(r.rule, r.metric, r.model): r.value for r in rs}
        out.append(f"### {run}\n")
        out.append("| Rule | Metric | " + " | ".join(models) + " | Source |")
        out.append("| --- | --- | " + " | ".join("---" for _ in models) + " | --- |")
        for rule, metric, source in keys:
            vals = " | ".join(
                _fmt(cell[(rule, metric, m)], metric) if (rule, metric, m) in cell else ""
                for m in models
            )
            out.append(f"| {rule} | {metric} | {vals} | {source} |")
        if any(s == REVIEW for _, _, s in keys):
            out.append("")
            out.append(REVIEW_NOTE)
        out.append("")
    return "\n".join(out)
