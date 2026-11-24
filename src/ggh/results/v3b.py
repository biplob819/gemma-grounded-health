"""RBS-02 (V3.7): v3b's registered result from the committed main_v3b run
(configs/eval_plan_v3b.yaml), computed as v3's (`ggh.results.v3`) plus what v3b's plan reports
beside the headline: the copy share and the headline among answers that are not copies (copy
similarity under 0.9 to every cited sentence, `ggh.rl.reward_b.copy_similarity`), so a gain that
comes from copying is visible (OWNER D-70). That measure misses verbatim cut-down extractions
(V3.6), so the non-copy headline is an upper bound on what copying explains away, not a floor.
v3b at Q8_0 is the owner's exploratory extra (OWNER D-73), outside the primary test.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from evalkit.schemas import Answer

from ggh.results import family as F
from ggh.results.runs import Run
from ggh.results.v3 import JUDGE, OUTCOMES, _diff, _point, answers_report, verdict
from ggh.rl.reward_b import COPY_RATIO, copy_similarity

PLAN = Path("configs/eval_plan_v3b.yaml")
RUN = "main_v3b"
RESULT = Path("docs/results/main_v3b/result.json")
Q8 = "gemma-e2b-grounded-v3b-q8"
NEAR_COPY = 0.7


def copy_report(run: Run) -> dict[str, Any]:
    """Per model: the copy share (>= 0.9), the near-copy share (>= 0.7), and the headline among
    the answers that are not copies."""
    out: dict[str, Any] = {}
    for model in run.models():
        sims, good = [], []
        for r in run.raw_records:
            if r["completion"]["model"] != model:
                continue
            a = r["completion"]["answer"]
            ctx = run.cases[r["case_id"]].context
            sims.append(copy_similarity(Answer.model_validate(a), ctx) if a else 0.0)
            good.append(bool(r["scores"]["correct_and_grounded"].get(JUDGE)))
        keep = [g for s, g in zip(sims, good, strict=True) if s < COPY_RATIO]
        out[model] = {
            "copied": round(sum(s >= COPY_RATIO for s in sims) / len(sims), 4),
            "near_copied": round(sum(s >= NEAR_COPY for s in sims) / len(sims), 4),
            "non_copy_n": len(keep),
            "non_copy_headline": round(sum(keep) / len(keep), 4),
        }
    return out


def unjudged(run: Run) -> dict[str, int]:
    """Cited sentences of extracted claims the citation judge returned no result for, per model."""
    out: dict[str, int] = {m: 0 for m in run.models()}
    for r in run.raw_records:
        seen = {(e["claim_id"], e["n"]) for e in (r.get("citation_support") or {}).get(JUDGE, [])}
        for c in r.get("claims") or []:
            out[r["completion"]["model"]] += sum(
                (c["id"], n) not in seen for n in c.get("cited") or []
            )
    return out


def result(root: Path) -> dict[str, Any]:
    plan = yaml.safe_load((root / PLAN).read_text(encoding="utf-8"))
    run = Run(root, RUN)
    ((a, b),) = plan["primary"]
    (primary,) = F.family_tests(run, plan)
    metrics = ("headline", "citation_precision", "aim_citation_rate")
    point = {m["model"]: {k: _point(m, k) for k in metrics} for m in run.summary["models"]}
    guards = {}
    for g in plan["guardrails"]:
        v = point[a][g["metric"]]
        ok = v >= g["min"] if "min" in g else v <= g["max"]
        guards[g["metric"]] = {**{k: g[k] for k in ("min", "max") if k in g}, "value": v, "ok": ok}
    (stretch,) = plan["stretch"]
    base = next(m for m in run.models() if m not in (a, b, Q8))
    exploratory = {}
    for x, y in ((a, base), (Q8, a), (Q8, b)):
        p = dict(plan, primary=[[x, y]], primary_metrics=list(OUTCOMES))
        exploratory[f"{x} minus {y}"] = {t["metric"]: _diff(t) for t in F.family_tests(run, p)}
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
        "copies": copy_report(run),
        "exploratory": exploratory,
        "exploratory_models": [Q8],
        "unjudged_citations": unjudged(run),
        "spent_usd": round(float(run.manifest["spent_usd"]), 6),
    }


def write(root: Path) -> Path:
    out = root / RESULT
    out.write_text(json.dumps(result(root), indent=2) + "\n", encoding="utf-8")
    return out
