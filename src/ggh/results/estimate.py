"""A cache-aware spend estimate for a plan's runs (V2.2, RUN-02, D-22), from committed evidence.

EvalKit's `--dry-run` prices every model as uncached, so for V2.2's main run (four of five
models served from the store) it overstates the spend several-fold. This estimate prices only
the models a run answers fresh (`new_models` in the plan), from unit costs measured on runs
already paid for:

- `per_claim`: extraction plus both claim judges (batch), per extracted claim: the E3.2 main
  run's spend, less its frontier generation, over its claims.
- `per_claim_citation`: both citation judges (OWNER D-44), per claim: the v1 citation rescore's
  spend over its claims (it re-extracted and re-judged nothing else).
- claims per answer for a new model are unknown before it runs, so two scenarios bracket it:
  v1's rate on the same task (v2 is trained like v1 on shorter, single-citation targets) and
  the base model's (the most claims a Gemma model made). `--claims-per-answer` overrides both.

Endpoint models generate for $0 (EvalKit prices them free). A backtest prices the v1 dev
grounding baseline (fresh, citation_support on) with the same units, against what it cost.
No API is called and nothing is read outside the repo.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ggh.results.plan import Plan
from ggh.results.runs import Run

V1 = "gemma-e2b-grounded-q4"
BASE = "gemma-e2b-base-q4"
APPROVAL_LINE_USD = 25.0  # D-22: the owner approves any single run above this
# the E3.2 run whose v1 and base claim rates stand in for a new model on the same task
REFERENCE_RUN = {
    "pubmedqa_cit": "main",
    "pubmedqa": "main",
    "offscope": "offscope",
    "probe_original": "probe_original",
    "probe_paraphrase": "probe_paraphrase",
}


def _claims(run: Run, model: str | None = None) -> int:
    return sum(
        len(r["claims"])
        for r in run.raw_records
        if model is None or r["completion"]["model"] == model
    )


def _answers(run: Run, model: str) -> int:
    return sum(r["completion"]["model"] == model for r in run.raw_records)


def units(root: Path) -> dict[str, Any]:
    main, cit = Run(root, "main"), Run(root, "main_citation")
    gen = sum(float(r["completion"]["cost_usd"] or 0.0) for r in main.raw_records)
    per_claim = (main.manifest["spent_usd"] - gen) / _claims(main)
    per_cite = cit.manifest["spent_usd"] / _claims(cit)
    out: dict[str, Any] = {
        "per_claim": round(per_claim, 6),
        "per_claim_citation": round(per_cite, 6),
        "from": {
            "per_claim": main.manifest["run_id"],
            "per_claim_citation": cit.manifest["run_id"],
        },
    }
    if Run.exists(root, "dev_ground_v1"):
        dev = Run(root, "dev_ground_v1")
        predicted = _claims(dev) * (per_claim + per_cite)
        out["backtest"] = {
            "run_id": dev.manifest["run_id"],
            "predicted_usd": round(predicted, 2),
            "actual_usd": round(dev.manifest["spent_usd"], 2),
        }
    return out


def claims_per_answer(root: Path, task_stem: str) -> dict[str, float]:
    ref = Run(root, REFERENCE_RUN.get(task_stem, "main"))
    return {
        who: round(_claims(ref, m) / _answers(ref, m), 3)
        for who, m in (("like_v1", V1), ("like_base", BASE))
    }


def estimate(root: Path, plan: Plan, cpa: float | None = None) -> dict[str, Any]:
    u = units(root)
    lines = []
    totals = {"like_v1": 0.0, "like_base": 0.0}
    for r in plan.runs:
        spec = r.spec
        new = list(spec.get("new_models", []))
        n = int(spec["n"])
        stem = Path(spec["task"]).stem
        rates = {"like_v1": cpa, "like_base": cpa} if cpa else claims_per_answer(root, stem)
        unit = u["per_claim"] + (u["per_claim_citation"] if spec.get("citation_support") else 0)
        usd = {k: round(len(new) * n * rate * unit, 2) for k, rate in rates.items()}
        for k in totals:
            totals[k] += usd[k]
        lines.append(
            {
                "run": r.name,
                "answers": len(new) * n,
                "cached_models": list(spec.get("cached_models", [])),
                "citation_support": bool(spec.get("citation_support")),
                "claims_per_answer": rates,
                "usd": usd,
                "over_approval_line": max(usd.values()) > APPROVAL_LINE_USD,
                "evalkit_dry_run": f"{spec['command']} --dry-run" if spec.get("command") else None,
            }
        )
    return {
        "plan": plan.path.as_posix(),
        "plan_status": plan.status,
        "units": u,
        "runs": lines,
        "total_usd": {k: round(v, 2) for k, v in totals.items()},
        "note": "cache-aware; EvalKit's --dry-run prices every model uncached (an upper bound)",
    }


def text(est: dict[str, Any]) -> str:
    u = est["units"]
    out = [
        f"cache-aware estimate for {est['plan']} ({est['plan_status']}); no API calls made",
        f"units: ${u['per_claim']:.5f} per claim (extract + 2 judges, {u['from']['per_claim']}),"
        f" ${u['per_claim_citation']:.5f} per claim citation judging "
        f"({u['from']['per_claim_citation']})",
    ]
    if "backtest" in u:
        b = u["backtest"]
        out.append(
            f"backtest {b['run_id']}: predicted ${b['predicted_usd']:.2f}, "
            f"actual ${b['actual_usd']:.2f}"
        )
    for ln in est["runs"]:
        c, d = ln["claims_per_answer"], ln["usd"]
        flag = "  OVER $25: owner approval (D-22)" if ln["over_approval_line"] else ""
        out.append(
            f"{ln['run']:<20} {ln['answers']:>5} new answers  "
            f"${d['like_v1']:.2f} ({c['like_v1']} claims/answer) to "
            f"${d['like_base']:.2f} ({c['like_base']}){flag}"
        )
    t = est["total_usd"]
    out.append(f"total ${t['like_v1']:.2f} to ${t['like_base']:.2f}")
    return "\n".join(out)


def dumps(est: dict[str, Any]) -> str:
    return json.dumps(est, indent=2, sort_keys=True)
