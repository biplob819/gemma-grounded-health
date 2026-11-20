"""The judges' calibration stamp (E3.3, RES-03; OWNER D-24, D-59, D-60).

EvalKit stamps calibration per run folder, and no committed GGH summary is rewritten (D-59), so
every GGH summary still says `uncalibrated`. The bundle's stamp is computed here instead, from
committed files only, with no API call:

- the committed copy of EvalKit's `calibration.json` for its `pubmedqa` calibration run
  (r-e8dac378d2d3), validated with EvalKit's own `CalibrationReport`, which must be for the
  configured task and for EvalKit's current rubric (`JUDGE_PROMPT_HASH`), the rubric every
  verdict in the GGH `pubmedqa` runs carries, and must hold held-out agreement for every judge
  those runs used;
- the 40-claim Gemma-family report (OWNER D-24) and the labeller's 20-answer relabel.

A judge is stamped `calibrated` when its held-out kappa's 95% lower bound is at least the bar
(D-59: 0.4) and it is not the reference labeller's own model: the labels are Claude Opus 5.5's
(D-60), so Opus's agreement is the same model agreeing with itself and is reported, never
stamped. A judge under the bar is reported with its kappa and its numbers stay published; that
is a disclosure, not a publish blocker. A missing or invalid calibration input is a blocker.
Runs of another task (off-scope, probes) are exempt and say so.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evalkit.pipeline.judge import JUDGE_PROMPT_HASH
from evalkit.schemas import CalibrationReport, Interval
from pydantic import ValidationError

from ggh.results.runs import Run
from ggh.schemas import GemmaFamilyAgreement, JudgeCalibration, LabellerConsistency

# display names for the reference labeller's model id, for the stamp's wording (OWNER D-60)
LABELLER_NAMES = {"claude-opus-5-5": "Claude Opus 5.5"}


@dataclass(frozen=True)
class Stamp:
    task: str | None  # the task the calibration covers; None when not configured
    judges: tuple[JudgeCalibration, ...]
    labeller: LabellerConsistency | None
    problems: tuple[str, ...]  # publish blockers: a missing or invalid calibration input

    def by_judge(self) -> dict[str, JudgeCalibration]:
        return {j.judge: j for j in self.judges}

    def release_status(self, judges: Iterable[str]) -> str:
        """`calibrated` when every judge a release relies on is stamped and Gemma-family
        checked, and no calibration input is missing or invalid."""
        need = sorted(set(judges))
        by = self.by_judge()
        ok = (
            bool(need)
            and not self.problems
            and all(
                (c := by.get(j)) is not None and c.status == "calibrated" and c.gemma_family
                for j in need
            )
        )
        return "calibrated" if ok else "uncalibrated"

    def exempt_note(self, task: str) -> str | None:
        if self.task is None or task == self.task:
            return None
        return (
            f"exempt: task {task} is not {self.task}, so the {self.task} calibration cannot "
            "cover it (OWNER D-59); its judges' numbers here carry no calibrated stamp"
        )


def _f3(x: float) -> str:
    return f"{x:.3f}"


def _iv(k: Interval) -> str:
    return f"{_f3(k.point)} [{_f3(k.lo)}, {_f3(k.hi)}]"


def _load(root: Path, rel: str | None, what: str, problems: list[str]) -> Any:
    if not rel:
        problems.append(f"calibration: no {what} is configured (configs/results.yaml)")
        return None
    path = root / rel
    if not path.is_file():
        problems.append(f"calibration: {rel} is missing ({what})")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        problems.append(f"calibration: {rel} is not valid JSON: {exc}")
        return None


def _iso(t: Any) -> str:
    return t.isoformat().replace("+00:00", "Z") if hasattr(t, "isoformat") else str(t)


def _model_id(labeller: str) -> str:
    return labeller.split()[0] if labeller.split() else ""


def verdict_hashes(run: Run) -> set[str]:
    return {str(v.get("judge_prompt_hash")) for r in run.raw_records for v in r.get("verdicts", [])}


def evalkit_report(
    root: Path, rel: str | None, task: str, judges: list[str], problems: list[str]
) -> CalibrationReport | None:
    """EvalKit's current calibration report, refused (None, with the reason in problems) when
    it is missing, invalid, for another task or under another rubric."""
    raw = _load(root, rel, "EvalKit calibration", problems)
    if raw is None:
        return None
    try:
        report = CalibrationReport.model_validate(
            raw.get("current") if isinstance(raw, dict) else raw
        )
    except ValidationError as exc:
        problems.append(f"calibration: {rel} is not an EvalKit calibration report: {exc}")
        return None
    if report.task != task:
        problems.append(f"calibration: {rel} is for task {report.task!r}, not {task!r}")
        return None
    if report.judge_prompt_hash != JUDGE_PROMPT_HASH:
        problems.append(
            f"calibration: {rel} is under rubric {report.judge_prompt_hash}, not EvalKit's "
            f"{JUDGE_PROMPT_HASH}"
        )
        return None
    for a in report.agreements:
        if a.split != "heldout" or a.judge_prompt_hash != JUDGE_PROMPT_HASH:
            problems.append(
                f"calibration: {rel}: {a.judge}'s agreement is on split {a.split!r} under rubric "
                f"{a.judge_prompt_hash}, not held-out under {JUDGE_PROMPT_HASH}"
            )
            return None
    have = {a.judge for a in report.agreements}
    for j in judges:
        if j not in have:
            problems.append(f"calibration: {rel} has no held-out agreement for judge {j}")
    return report


def gemma_family(
    root: Path,
    rel: str | None,
    check_rel: str | None,
    reference: str,
    runs: dict[str, Run],
    problems: list[str],
) -> dict[str, GemmaFamilyAgreement]:
    raw = _load(root, rel, "Gemma-family report", problems)
    if raw is None:
        return {}
    out: dict[str, GemmaFamilyAgreement] = {}
    try:
        if _model_id(str(raw["labeller"])) != reference:
            problems.append(
                f"calibration: {rel} was labelled by {raw['labeller']!r}, not {reference}"
            )
        run = runs.get(raw["run"])
        if run is not None and run.manifest["run_id"] != raw["run_id"]:
            problems.append(f"calibration: {rel} is for run {raw['run_id']}, not {raw['run']}'s")
        for j in raw["judges"]:
            if j["judge_prompt_hash"] != JUDGE_PROMPT_HASH:
                problems.append(
                    f"calibration: {rel}: {j['judge']} is under rubric {j['judge_prompt_hash']}"
                )
                continue
            out[j["judge"]] = GemmaFamilyAgreement(
                run=raw["run"],
                run_id=raw["run_id"],
                judge_prompt_hash=j["judge_prompt_hash"],
                n_claims=j["n_claims"],
                coverage=j["coverage"],
                agreement=j["agreement"],
                kappa=Interval.model_validate(j["kappa"]),
                by_model=j.get("by_model", {}),
                source=rel or "",
            )
    except (KeyError, TypeError, ValidationError) as exc:
        problems.append(f"calibration: {rel} is not a Gemma-family report: {exc}")
        return {}
    check = _load(root, check_rel, "Gemma-family check", problems)
    if check is not None and check != {j: g.agreement for j, g in out.items()}:
        problems.append(f"calibration: {check_rel} disagrees with {rel}")
    return out


def labeller(
    root: Path, rel: str | None, reference: str, prompt: str | None, problems: list[str]
) -> LabellerConsistency | None:
    raw = _load(root, rel, "labeller record", problems)
    if raw is None:
        return None
    try:
        sc = raw["self_consistency"]
        lc = LabellerConsistency(
            labeller=raw["labeller"],
            prompt=raw["prompt"],
            prompt_sha256=raw["prompt_sha256"],
            claims=sc["claims"],
            answers=sc["answers"],
            identical=sc["identical"],
            kappa=Interval.model_validate(sc["kappa"]),
            method=sc["method"],
            source=rel or "",
        )
    except (KeyError, TypeError, ValidationError) as exc:
        problems.append(f"calibration: {rel} has no labeller self-consistency: {exc}")
        return None
    if _model_id(lc.labeller) != reference:
        problems.append(f"calibration: {rel} names labeller {lc.labeller!r}, not {reference}")
    if prompt is not None and lc.prompt != prompt:
        problems.append(f"calibration: {rel} used prompt {lc.prompt}, not {prompt}")
    elif prompt is not None:
        p = root / prompt
        if not p.is_file():
            problems.append(f"calibration: the labeller prompt {prompt} is missing")
        elif hashlib.sha256(p.read_bytes()).hexdigest() != lc.prompt_sha256:
            problems.append(f"calibration: {prompt} no longer has the sha256 {rel} records")
    return lc


def wording(c: JudgeCalibration, name: str) -> str:
    kappa = f"held-out kappa {_iv(c.kappa)} on {c.n_claims} {c.task} claims"
    ref = f"a {name} reference labeller (a model's labels, not a human expert's; OWNER D-60)"
    if c.status == "calibrated":
        text = (
            f"{c.judge}: calibrated, {kappa}, lower bound at least {c.bar} (OWNER D-59): "
            f"agreement with {ref}"
        )
    elif c.status == "below_bar":
        text = (
            f"{c.judge}: not calibrated, {kappa}: the lower bound is under the {c.bar} bar "
            f"(OWNER D-59), against {ref}; its numbers stay published with this miss disclosed"
        )
    else:
        bar = "" if c.meets_bar else f", and its lower bound is under the {c.bar} bar"
        text = (
            f"{c.judge}: never stamped calibrated: the reference labels are its own model's "
            f"({name}), so its {kappa} is the same model agreeing with itself (OWNER D-60)"
            f"{bar}; reported only, its numbers stay published"
        )
    g = c.gemma_family
    if g is not None:
        text += (
            f". Gemma-family check ({g.n_claims} claims from {g.run}): agreement "
            f"{_f3(g.agreement)}, kappa {_iv(g.kappa)}"
        )
    return text


def stamp(
    root: Path, cfg: dict[str, Any] | None, runs: dict[str, Run], check_rel: str | None
) -> Stamp:
    """Every judge of the configured task's runs, stamped by D-59/D-60's rule."""
    problems: list[str] = []
    if not cfg:
        return Stamp(
            None, (), None, ("calibration: not configured (configs/results.yaml, RES-03)",)
        )
    task = str(cfg["task"])
    bar = float(cfg["bar"])
    reference = str(cfg["reference_labeller"])
    name = LABELLER_NAMES.get(reference, reference)
    prompt = cfg.get("labeller_prompt")
    covered = [r for _, r in sorted(runs.items()) if r.manifest["task"] == task]
    judges = sorted({j for r in covered for j in r.manifest["judges"]})
    for r in covered:
        hashes = verdict_hashes(r)
        if hashes != {JUDGE_PROMPT_HASH}:
            problems.append(
                f"calibration: {r.name}'s verdicts use rubric {sorted(hashes)}, not EvalKit's "
                f"{JUDGE_PROMPT_HASH}"
            )
    report = evalkit_report(root, cfg.get("evalkit"), task, judges, problems)
    gf = gemma_family(root, cfg.get("gemma_family_report"), check_rel, reference, runs, problems)
    lc = labeller(root, cfg.get("labeller"), reference, prompt, problems)
    out: list[JudgeCalibration] = []
    agreements = {a.judge: a for a in report.agreements} if report else {}
    for j in judges:
        a = agreements.get(j)
        if a is None or report is None:
            continue
        meets = a.kappa.lo >= bar
        status = (
            "not_stampable_self_reference"
            if j == reference
            else ("calibrated" if meets else "below_bar")
        )
        fields: dict[str, Any] = {
            "judge": j,
            "task": a.task,
            "judge_prompt_hash": a.judge_prompt_hash,
            "split": a.split,
            "n_claims": a.n_claims,
            "coverage": a.coverage,
            "agreement": a.agreement,
            "kappa": a.kappa,
            "bar": bar,
            "meets_bar": meets,
            "reference_labeller": f"{reference} ({prompt}, OWNER D-60)",
            "reference_model": reference,
            "status": status,
            "wording": "-",
            "source": cfg["evalkit"],
            "scored_at": _iso(report.scored_at),
            "gemma_family": gf.get(j),
        }
        draft = JudgeCalibration.model_validate(fields)
        out.append(draft.model_copy(update={"wording": wording(draft, name)}))
    return Stamp(task, tuple(out), lc, tuple(problems))
