"""RLS-02 (V3.5): v3's registered result from the committed main_v3 run (configs/eval_plan_v3.yaml).

The primary test is `family_tests` with the plan as written (v3 minus v2 on the headline, EvalKit's
paired bootstrap, Holm over a family of one). The verdict follows the plan's rule: "improves" only
if v3's headline is above v2's and the interval excludes zero, otherwise a tie whichever way the
point falls. Guardrails are bars on v3's own values; the stretch goal is reported, not tested. The
label and length report sits beside the headline. v3 minus base is exploratory, not Holm-corrected.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from ggh.results import family as F
from ggh.results.runs import Run

PLAN = Path("configs/eval_plan_v3.yaml")
RUN = "main_v3"
RESULT = Path("docs/results/main_v3/result.json")
JUDGE = "gpt-6.1-sol"
OUTCOMES = (
    {"metric": "headline", "better": "higher"},
    {"metric": "citation_precision", "better": "higher"},
    {"metric": "aim_citation_rate", "better": "lower"},
)


def _point(m: dict[str, Any], metric: str) -> float:
    v = m[metric]
    v = v.get(JUDGE, v) if isinstance(v, dict) else v
    return float(v["point"] if isinstance(v, dict) else v)


def _diff(t: dict[str, Any]) -> dict[str, Any]:
    d = t["diff"]
    return {"point": d.point, "lo": d.lo, "hi": d.hi, "p_value": round(float(t["p_value"]), 6),
            "separated": bool(t["separated"])}  # fmt: skip


def answers_report(run: Run) -> dict[str, Any]:
    """Each model's label distribution and mean and maximum answer length in words."""
    out: dict[str, Any] = {}
    for model in run.models():
        answers = [r["completion"]["answer"] or {} for r in run.raw_records
                   if r["completion"]["model"] == model]  # fmt: skip
        words = [len((a.get("answer") or "").split()) for a in answers]
        labels = Counter(a.get("label") for a in answers)
        out[model] = {
            "labels": {k: labels.get(k, 0) for k in ("yes", "no", "maybe")},
            "unparsed": len(answers) - sum(labels.get(k, 0) for k in ("yes", "no", "maybe")),
            "mean_words": round(statistics.fmean(words), 2),
            "max_words": max(words),
        }
    return out


def verdict(v3_headline: float, v2_headline: float, separated: bool) -> str:
    """The plan's rule: above v2 with an interval that excludes zero, else a tie."""
    return "improves on v2" if v3_headline > v2_headline and separated else "tie"


def result(root: Path) -> dict[str, Any]:
    plan = yaml.safe_load((root / PLAN).read_text(encoding="utf-8"))
    run = Run(root, RUN)
    (a, b), = plan["primary"]  # fmt: skip
    (primary,) = F.family_tests(run, plan)
    point = {m["model"]: {k: _point(m, k) for k in ("headline", "citation_precision",
             "aim_citation_rate")} for m in run.summary["models"]}  # fmt: skip
    guards = {}
    for g in plan["guardrails"]:
        v = point[a][g["metric"]]
        ok = v >= g["min"] if "min" in g else v <= g["max"]
        guards[g["metric"]] = {**{k: g[k] for k in ("min", "max") if k in g}, "value": v, "ok": ok}
    (stretch,) = plan["stretch"]
    base = next(m for m in run.models() if m not in (a, b))
    exploratory = {}
    for other in (b, base):
        p = dict(plan, primary=[[a, other]], primary_metrics=list(OUTCOMES))
        exploratory[f"{a} minus {other}"] = {t["metric"]: _diff(t) for t in F.family_tests(run, p)}
    return {
        "run_id": run.manifest["run_id"],
        "plan": PLAN.as_posix(),
        "primary": {
            "model_a": a,
            "model_b": b,
            "metric": "headline",
            "judge": JUDGE,
            "family_size": primary["family_size"],
            **_diff(primary),
        },
        "verdict": verdict(point[a]["headline"], point[b]["headline"], bool(primary["separated"])),
        "headline": {m: v["headline"] for m, v in point.items()},
        "guardrails": guards,
        "stretch": {
            "reach": stretch["reach"],
            "value": point[a]["headline"],
            "reached": point[a]["headline"] >= stretch["reach"],
        },
        "answers": answers_report(run),
        "exploratory": exploratory,
        "spent_usd": round(float(run.manifest["spent_usd"]), 6),
    }


def write(root: Path) -> Path:
    out = root / RESULT
    out.write_text(json.dumps(result(root), indent=2) + "\n", encoding="utf-8")
    return out
