"""A post-hoc check of the aim-citation rule (E3.3, OWNER D-62). Nothing here changes a published
number: the pre-registered V2.2 results stay as EvalKit computed them, and this check is
disclosed as post-hoc, made after the results were read.

Why. EvalKit 0.4.0's citation judge (rubric hash `be34ebe577bc96ae`) is told that "a sentence
that only asks a question or states an aim or a hypothesis does not establish" a claim. GPT-6.1
Sol applies that literally: when a claim only restates what the study set out to do ("The study
assessed whether X affects Y"), Sol marks the aim sentence it cites as not supporting, while the
claim judge (the same model, a different rubric) rules the claim supported with that very
sentence as its evidence. EvalKit's citation metrics count a cited sentence as supporting if it
supports any claim, give it its most common role (ties to `aim`) and count an aim hit for an
answer citing an `aim` sentence that supports no claim, so these restatements read as aim
citations.

The rule (`restates_aim`). A citation-judge entry (claim c, sentence n) is counted as supporting
when all of these hold, and every other entry keeps the citation judge's own `supports`:
1. the citation judge gave the pair the role `aim`;
2. the claim judge (same judge name) ruled claim c `supported`;
3. n is in that verdict's `evidence`;
4. the claim's text restates the study's aim or design: `AIM_OR_DESIGN` matches it
   (a word list ported unchanged from the read-only investigation that found the effect).
Everything downstream is EvalKit's definition unchanged (`ggh.results.family.case_values`, the
GGH copy of EvalKit's `citation_counts`), applied to records whose entries the rule rewrote.

What is written (`docs/measurements/aim_rule_check.json`, `ggh results aim-check`):
- per run, model and judge: published and corrected citation precision and aim-citation rate,
  with the counts behind the change; the published values are asserted equal to EvalKit's
  `summary.json` points (main_v2, and main_citation for context);
- the plan's primary family (configs/eval_plan_v2.yaml: v2 minus v1 and v2 minus base, three
  outcomes, Sol) under published and corrected counts, with the plan's own bootstrap (10,000
  resamples, seed 7, `ggh.results.family`, the code the results bundle publishes from); each
  published gap is asserted equal to the difference of the two summary points;
- extractor counts per model (main, main_citation, main_v2): answers with no claims although the
  answer text is non-empty, and label-only claims (`LABEL_ONLY`: "Yes.", "No.", "Maybe.");
- the effect of v1's label-only claims on the Sol headline of gpt-6-luna and claude-haiku-5-5 in
  `main`: the headline with those claims dropped (an answer left with no claims is not grounded,
  as EvalKit scores it).
"""

from __future__ import annotations

import copy
import json
import math
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ggh.results import family as F
from ggh.results.plan import load_plan
from ggh.results.runs import ResultsError, Run

OUT = Path("docs") / "measurements" / "aim_rule_check.json"
PLAN = "configs/eval_plan_v2.yaml"
RUN = "main_v2"
CONTEXT_RUNS = ("main_citation",)  # v1's citation re-judge, for context (not in the family)
EXTRACTOR_RUNS = ("main", "main_citation", "main_v2")
LABEL_RUN = "main"
LABEL_JUDGE = "gpt-6.1-sol"
LABEL_MODELS = ("gpt-6-luna", "claude-haiku-5-5")
CITATION_RUBRIC_HASH = "be34ebe577bc96ae"  # EvalKit 0.4.0's CITATION_PROMPT_HASH
TOL = 1e-9

# Ported unchanged from the investigation: a claim that names the study's aim (aim, objective,
# purpose, sought to, was to ...), its act of inquiry (investigated, assessed, evaluated ...) or
# its design (randomised, prospective, cohort, trial ...). Case-insensitive, word-start anchored.
AIM_OR_DESIGN = re.compile(
    r"\b(aim|aimed|objective|purpose|goal|sought|investigat|assess|evaluat|examin|explor|"
    r"determin|hypothes|this study|the study|the investigation|was to|were to|design|"
    r"randomi[sz]ed|prospective|retrospective|cohort|trial)",
    re.IGNORECASE,
)
# A claim that is only the yes/no/maybe label, with punctuation at most.
LABEL_ONLY = re.compile(r"^\W*(yes|no|maybe)\W*$", re.IGNORECASE)

Rule = Callable[[dict[str, Any], dict[str, Any] | None, str], bool]


def restates_aim(entry: dict[str, Any], verdict: dict[str, Any] | None, claim_text: str) -> bool:
    """The post-hoc rule: does this (claim, sentence) entry count as supporting after all?"""
    return (
        entry["role"] == "aim"
        and verdict is not None
        and verdict["verdict"] == "supported"
        and entry["n"] in verdict["evidence"]
        and AIM_OR_DESIGN.search(claim_text) is not None
    )


def _flipped(rec: dict[str, Any], judge: str, rule: Rule) -> list[dict[str, Any]]:
    """The judge's citation entries with the rule applied (a copy; `supports` only ever turns
    from false to true)."""
    verdicts = {v["claim_id"]: v for v in rec["verdicts"] if v["judge"] == judge}
    texts = {c["id"]: c["text"] for c in rec["claims"]}
    out = []
    for e in (rec.get("citation_support") or {}).get(judge, []):
        flip = not e["supports"] and rule(
            e, verdicts.get(e["claim_id"]), texts.get(e["claim_id"], "")
        )
        out.append({**e, "supports": True} if flip else e)
    return out


def corrected_record(
    rec: dict[str, Any], judges: Sequence[str], rule: Rule = restates_aim
) -> dict[str, Any]:
    """The record with the rule applied to each judge's citation entries; nothing else changes."""
    cs = rec.get("citation_support") or {}
    if not cs:
        return rec
    return {
        **rec,
        "citation_support": {
            j: _flipped(rec, j, rule) if j in judges else v for j, v in cs.items()
        },
    }


def _r(x: float) -> float:
    return round(x, 6)


def _pooled(recs: Sequence[dict[str, Any]], metric: str, judge: str) -> float:
    vals = [F.case_values(r, metric, judge) for r in recs]
    return F.pooled([n for n, _ in vals], [d for _, d in vals])


def _counts(
    recs: Sequence[dict[str, Any]], fixed: Sequence[dict[str, Any]], judge: str
) -> dict[str, int]:
    """What the rule touched, for one model and judge."""
    c: dict[str, int] = dict.fromkeys(
        (
            "supported_verdicts_with_evidence",
            "aim_evidence_entries",
            "aim_evidence_entries_not_supporting",
            "entries_flipped",
            "sentences_flipped",
            "aim_hits_cleared",
            "cited_sentences",
            "supporting_sentences_published",
            "supporting_sentences_corrected",
            "answers_with_citations",
            "aim_hits_published",
            "aim_hits_corrected",
        ),
        0,
    )
    for rec, fx in zip(recs, fixed, strict=True):
        ents = {
            (e["claim_id"], e["n"]): e for e in (rec.get("citation_support") or {}).get(judge, [])
        }
        for v in rec["verdicts"]:
            if v["judge"] != judge or v["verdict"] != "supported" or not v["evidence"]:
                continue
            c["supported_verdicts_with_evidence"] += 1
            for n in v["evidence"]:
                e = ents.get((v["claim_id"], n))
                if e is not None and e["role"] == "aim":
                    c["aim_evidence_entries"] += 1
                    c["aim_evidence_entries_not_supporting"] += not e["supports"]
        before = (rec.get("citation_support") or {}).get(judge, [])
        after = (fx.get("citation_support") or {}).get(judge, [])
        c["entries_flipped"] += sum(
            a["supports"] != b["supports"] for a, b in zip(after, before, strict=True)
        )
        pn, pd = F.case_values(rec, "citation_precision", judge)
        cn, _ = F.case_values(fx, "citation_precision", judge)
        pa, ad = F.case_values(rec, "aim_citation_rate", judge)
        ca, _ = F.case_values(fx, "aim_citation_rate", judge)
        c["cited_sentences"] += int(pd)
        c["supporting_sentences_published"] += int(pn)
        c["supporting_sentences_corrected"] += int(cn)
        c["sentences_flipped"] += int(cn - pn)
        c["answers_with_citations"] += int(ad)
        c["aim_hits_published"] += int(pa)
        c["aim_hits_corrected"] += int(ca)
        c["aim_hits_cleared"] += int(pa - ca)
    return c


def _by_model(run: Run) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {m: [] for m in run.models()}
    for r in run.raw_records:
        out.setdefault(r["completion"]["model"], []).append(r)
    return out


def citation_table(run: Run) -> dict[str, Any]:
    """Per model and judge: published vs corrected citation metrics, with the counts. Raises if
    the published branch does not reproduce EvalKit's summary point."""
    judges = list(run.manifest["judges"])
    out: dict[str, Any] = {}
    for model, recs in _by_model(run).items():
        summ = run.model_summary(model)
        fixed = [corrected_record(r, judges) for r in recs]
        per: dict[str, Any] = {}
        for j in judges:
            if j not in (summ.get("citation_precision") or {}):
                continue
            pub = {m: _pooled(recs, m, j) for m in F.RATIO_METRICS}
            for m, v in pub.items():
                want = summ[m][j]["point"]
                if abs(v - want) > TOL:
                    raise ResultsError(
                        f"{run.name} {model} {j}: {m} {v} is not the published {want}"
                    )
            cor = {m: _pooled(fixed, m, j) for m in F.RATIO_METRICS}
            per[j] = {
                "published": {m: _r(v) for m, v in pub.items()},
                "corrected": {m: _r(v) for m, v in cor.items()},
                "change": {m: _r(cor[m] - pub[m]) for m in F.RATIO_METRICS},
                "counts": _counts(recs, fixed, j),
            }
        out[model] = per
    return out


def _test_row(t: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_a": t["model_a"],
        "model_b": t["model_b"],
        "metric": t["metric"],
        "judge": t["judge"],
        "better": t["better"],
        "diff": t["diff"].model_dump(),
        "p_value": _r(t["p_value"]),
        "separated": bool(t["separated"]),
        "family_size": t["family_size"],
    }


def family(run: Run, plan: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """The plan's primary family on the published records and on the corrected ones, with the
    same bootstrap (`ggh.results.family.family_tests`). The headline does not use citations, so
    only the two citation outcomes can move."""
    published = F.family_tests(run, plan)
    for t in published:  # the published branch is EvalKit's summary, model minus model
        a, b = (run.model_summary(t["model_a"]), run.model_summary(t["model_b"]))
        want = a[t["metric"]][t["judge"]]["point"] - b[t["metric"]][t["judge"]]["point"]
        if abs(t["diff"].point - want) > 1e-6:
            raise ResultsError(
                f"{run.name}: published {t['metric']} gap {t['diff'].point} is not {want}"
            )
    fixed = copy.copy(run)
    fixed.__dict__["raw_records"] = [
        corrected_record(r, list(plan["judges"])) for r in run.raw_records
    ]
    corrected = F.family_tests(fixed, plan)
    for p, c in zip(published, corrected, strict=True):
        if p["metric"] == "headline" and p["diff"] != c["diff"]:
            raise ResultsError("the aim rule moved the headline")
    return {
        "published": [_test_row(t) for t in published],
        "corrected": [_test_row(t) for t in corrected],
    }


def _answer_text(rec: dict[str, Any]) -> str:
    return str((rec["completion"].get("answer") or {}).get("answer") or "").strip()


def extractor_counts(run: Run) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for model, recs in _by_model(run).items():
        claims = [c for r in recs for c in r["claims"]]
        label = [r for r in recs if any(LABEL_ONLY.match(c["text"]) for c in r["claims"])]
        out[model] = {
            "records": len(recs),
            "empty_claims": sum(not r["claims"] for r in recs),
            "empty_claims_nonempty_answer": sum(
                not r["claims"] and bool(_answer_text(r)) for r in recs
            ),
            "claims": len(claims),
            "label_only_claims": sum(bool(LABEL_ONLY.match(c["text"])) for c in claims),
            "records_with_label_only_claims": len(label),
        }
    return out


def _fully(rec: dict[str, Any], judge: str, drop: Callable[[str], bool]) -> bool:
    """EvalKit's fully grounded (every claim judged and supported, at least one claim), over
    the claims `drop` keeps; an answer that failed upstream stays ungrounded."""
    if rec["completion"].get("error") is not None or rec["completion"].get("answer") is None:
        return False
    verdicts = {v["claim_id"]: v["verdict"] for v in rec["verdicts"] if v["judge"] == judge}
    kept = [c for c in rec["claims"] if not drop(c["text"])]
    return bool(kept) and all(verdicts.get(c["id"]) == "supported" for c in kept)


def label_only_headline(run: Run, judge: str, models: Sequence[str]) -> dict[str, Any]:
    """Each model's headline (correct and fully grounded) as published and with label-only
    claims dropped. The published branch is recomputed from the verdicts and asserted equal to
    every record's score and to the summary point."""
    by = _by_model(run)
    out: dict[str, Any] = {}
    for model in models:
        recs = by[model]
        pub, drop = [], []
        verdicts: dict[str, int] = {}
        for r in recs:
            ok = bool(r["scores"]["label_correct"])
            f = _fully(r, judge, lambda _t: False)
            if f != bool(r["scores"]["fully_grounded"].get(judge)):
                raise ResultsError(
                    f"{run.name} {r['case_id']} {model}: fully grounded does not recompute"
                )
            pub.append(ok and f)
            drop.append(ok and _fully(r, judge, lambda t: LABEL_ONLY.match(t) is not None))
            label_ids = {c["id"] for c in r["claims"] if LABEL_ONLY.match(c["text"])}
            for v in r["verdicts"]:
                if v["judge"] == judge and v["claim_id"] in label_ids:
                    verdicts[v["verdict"]] = verdicts.get(v["verdict"], 0) + 1
        p, d = math.fsum(pub) / len(recs), math.fsum(drop) / len(recs)
        want = run.model_summary(model)["headline"][judge]["point"]
        if abs(p - want) > TOL:
            raise ResultsError(f"{run.name} {model}: headline {p} is not the published {want}")
        out[model] = {
            "published": _r(p),
            "label_only_dropped": _r(d),
            "change": _r(d - p),
            "answers_gained": sum(b and not a for a, b in zip(pub, drop, strict=True)),
            "answers_lost": sum(a and not b for a, b in zip(pub, drop, strict=True)),
            "label_only_verdicts": dict(sorted(verdicts.items())),
        }
    return out


def build(root: Path) -> dict[str, Any]:
    plan = load_plan(root, PLAN)
    run = Run(root, RUN)
    runs = {name: Run(root, name) for name in (RUN, *CONTEXT_RUNS, *EXTRACTOR_RUNS)}
    return {
        "post_hoc": True,
        "decision": "OWNER D-62",
        "note": (
            "Post-hoc robustness check, made after the V2.2 results were read. The pre-registered "
            "numbers stay as published; 'corrected' counts an aim-role cited sentence as "
            "supporting when the claim judge used it as evidence for a supported claim that "
            "restates the study's aim or design."
        ),
        "citation_rubric_hash": CITATION_RUBRIC_HASH,
        "rule": {
            "name": "restates_aim",
            "conditions": [
                "citation-judge role for the (claim, sentence) pair is 'aim'",
                "claim judge's verdict for the claim is 'supported'",
                "the sentence is in that verdict's evidence",
                "the claim text matches aim_or_design_pattern",
            ],
            "aim_or_design_pattern": AIM_OR_DESIGN.pattern,
            "label_only_pattern": LABEL_ONLY.pattern,
        },
        "plan": PLAN,
        "sources": {
            name: {"records_sha256": r.records_sha256, "summary_sha256": r.summary_sha256}
            for name, r in sorted(runs.items())
        },
        "citation": {name: citation_table(runs[name]) for name in (RUN, *CONTEXT_RUNS)},
        "family": {RUN: family(run, plan.data)},
        "extractor": {name: extractor_counts(runs[name]) for name in EXTRACTOR_RUNS},
        "label_only_headline": {
            "run": LABEL_RUN,
            "judge": LABEL_JUDGE,
            "models": label_only_headline(runs[LABEL_RUN], LABEL_JUDGE, LABEL_MODELS),
        },
    }


def dumps(result: dict[str, Any]) -> str:
    return json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def write(root: Path, result: dict[str, Any], out: Path = OUT) -> Path:
    path = root / out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(result), encoding="utf-8")
    return path
