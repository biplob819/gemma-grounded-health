"""Primary comparisons over a plan's own Holm family (OWNER D-53, V2.2).

EvalKit's summary Holm-corrects across every model pair in a run and compares the headline and
accuracy only. The V2.2 plan instead names a family: two pairs (v2 minus v1, v2 minus base),
three outcomes (headline, citation precision, aim-citation rate) and both judges, twelve tests,
Holm-corrected together. This module computes those tests from EvalKit's per-record values
with EvalKit's own bootstrap functions (`evalkit.stats`: seeded resamples, paired percentile
interval, two-sided p with the +1 correction, Holm step-down), so the method is EvalKit's and
only the family differs.

Per-case values, each reproducing EvalKit's summary number for the model when pooled:
- headline: `scores.correct_and_grounded[judge]` (0 or 1).
- citation_precision: per answer, (distinct cited sentences supporting some claim, distinct
  cited sentences), from `citation_support[judge]`; pooled as sum over sum.
- aim_citation_rate: per answer with citation_support entries, 1 if a cited sentence's role
  (the judge's most common role for it, EvalKit's tie order) is `aim` and it supports no
  claim; pooled over those answers. Both follow EvalKit's `citation_counts` (B5.12).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from typing import Any

from evalkit.schemas import Interval
from evalkit.stats import Resamples, holm

from ggh.results.runs import ResultsError, Run

RATIO_METRICS = ("citation_precision", "aim_citation_rate")


# EvalKit's tie order for a sentence's role (B5.12): an aim reading is never hidden.
ROLE_TIE_ORDER = ("aim", "background", "method", "conclusion", "result")


def role(entries: Sequence[dict[str, Any]]) -> str:
    """The most common role a judge gave one cited sentence, ties broken by ROLE_TIE_ORDER."""
    counts = Counter(e["role"] for e in entries)
    return max(ROLE_TIE_ORDER, key=lambda r: (counts[r], -ROLE_TIE_ORDER.index(r)))


def case_values(rec: dict[str, Any], metric: str, judge: str) -> tuple[float, float]:
    """(numerator, denominator) for one record; the pooled value is sum(num) / sum(den)."""
    if metric == "headline":
        return float(bool(rec["scores"]["correct_and_grounded"].get(judge))), 1.0
    entries = (rec.get("citation_support") or {}).get(judge, [])
    by_sentence: dict[int, list[dict[str, Any]]] = {}
    for e in entries:
        by_sentence.setdefault(e["n"], []).append(e)
    supports = {n: any(e["supports"] for e in es) for n, es in by_sentence.items()}
    if metric == "citation_precision":
        return float(sum(supports.values())), float(len(supports))
    if metric == "aim_citation_rate":
        if not by_sentence:
            return 0.0, 0.0
        aim = any(not supports[n] and role(es) == "aim" for n, es in by_sentence.items())
        return float(aim), 1.0
    raise ResultsError(f"no per-case definition for {metric!r}")


def pooled(nums: Sequence[float], dens: Sequence[float]) -> float:
    den = math.fsum(dens)
    if den == 0:
        raise ResultsError("a pooled metric has no denominator")
    return math.fsum(nums) / den


def _boot(resamples: Resamples, nums: list[float], dens: list[float]) -> list[float]:
    gn, gd = nums.__getitem__, dens.__getitem__
    out = []
    for row in resamples.rows:
        d = math.fsum(map(gd, row))
        out.append(math.fsum(map(gn, row)) / d if d else math.nan)
    return out


def _quantile(xs: Sequence[float], q: float) -> float:
    pos = q * (len(xs) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def paired_ratio(
    a: tuple[list[float], list[float]],
    b: tuple[list[float], list[float]],
    resamples: Resamples,
    alpha: float,
) -> tuple[Interval, float]:
    """As `evalkit.stats.paired`, for pooled ratios: the difference of the two pooled values
    on each resample of the shared cases; percentile interval and two-sided p (+1 corrected)."""
    point = pooled(*a) - pooled(*b)
    ba, bb = _boot(resamples, *a), _boot(resamples, *b)
    diffs = sorted(
        x - y for x, y in zip(ba, bb, strict=True) if not (math.isnan(x) or math.isnan(y))
    )
    big = len(diffs) + 1
    le = (sum(d <= 0 for d in diffs) + 1) / big
    ge = (sum(d >= 0 for d in diffs) + 1) / big
    lo, hi = _quantile(diffs, alpha / 2), _quantile(diffs, 1 - alpha / 2)
    iv = Interval(point=round(point, 6), lo=round(min(lo, point), 6), hi=round(max(hi, point), 6))
    return iv, min(1.0, 2 * min(le, ge))


def family_tests(run: Run, plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Every primary test of the plan, with its raw and Holm-adjusted decision."""
    boot = plan.get("bootstrap", {})
    alpha = float(boot.get("alpha", 0.05))
    by: dict[tuple[str, str], dict[str, Any]] = {
        (r["completion"]["model"], r["case_id"]): r for r in run.raw_records
    }
    models = {m for pair in plan["primary"] for m in pair}
    shared = sorted(set.intersection(*({c for (m, c) in by if m == x} for x in models)))
    if not shared:
        raise ResultsError(f"{run.name}: the primary models share no cases")
    resamples = Resamples(len(shared), int(boot.get("resamples", 10_000)), int(boot.get("seed", 7)))
    tests: list[dict[str, Any]] = []
    for a, b in plan["primary"]:
        for spec in plan["primary_metrics"]:
            metric, better = spec["metric"], spec["better"]
            for j in plan["judges"]:
                va = [case_values(by[(a, c)], metric, j) for c in shared]
                vb = [case_values(by[(b, c)], metric, j) for c in shared]
                cols_a = ([x for x, _ in va], [y for _, y in va])
                cols_b = ([x for x, _ in vb], [y for _, y in vb])
                diff, p = paired_ratio(cols_a, cols_b, resamples, alpha)
                tests.append(
                    {
                        "model_a": a,
                        "model_b": b,
                        "metric": metric,
                        "judge": j,
                        "better": better,
                        "diff": diff,
                        "p_value": p,
                    }
                )
    keys = list(range(len(tests)))
    rejected = holm({k: tests[k]["p_value"] for k in keys}, alpha)
    for k in keys:
        tests[k]["separated"] = rejected[k]
        tests[k]["family_size"] = len(tests)
    return tests
