"""The Gemma-family judge check (E3.3, RES-03, OWNER D-24, Tech Spec §7.5).

A reference labeller labels 40 claims drawn from Gemma answers, and each judge's verdicts on
those claims are scored against the labels (OWNER D-60: Claude Opus 5.5 with
`configs/labeller_prompt.md`, since the owner is not a domain expert). Nothing here calls an
API: the verdicts are the ones already in the run's records (V2.2, OWNER D-57: GPT-6.1 Sol
on GPT-6 Luna's claims in `main_v2`).

- `draw` picks the claims from one committed run, seeded: half from the base model and the rest
  split evenly over the fine-tunes (§7.5's 20 base / 20 fine-tuned), one claim per answer and
  one answer per question, so two models' claims on one abstract never sit side by side.
- `write_sheet` writes `docs/measurements/gemma_family_sheet.json`. It is the private join (it
  names models and claim ids) and holds no verdicts; the labeller never sees it.
- `serve` opens EvalKit's own labelling page (`evalkit.pipeline.labeller`, 127.0.0.1 only) on
  the sheet's blind rows: the claim and its cited sentences, nothing else. Labels save to
  `docs/measurements/gemma_family_labels.json` in EvalKit's `LabelFile` shape as they are made.
- `score` joins labels to the run's verdicts and writes `gemma_family_check.json` (judge ->
  agreement, the map `ResultsBundle.gemma_family_check` reads) and `gemma_family_report.json`
  (kappa with its interval, confusion, coverage and agreement per model).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from evalkit.calibration import CalibrationPair, cluster_kappa
from evalkit.pipeline.calibrate import LabelFile, SheetRow
from evalkit.pipeline.judge import JUDGE_PROMPT_HASH, redact

from ggh.results.runs import ResultsError, Run, canonical_sha256
from ggh.schemas._base import FrozenMap, Model, NonBlank, NonNegInt, PosInt, Sha256

N = 40
SEED = 24  # OWNER D-24
RUN = "main_v2"
MEASUREMENTS = Path("docs") / "measurements"
SHEET = MEASUREMENTS / "gemma_family_sheet.json"
LABELS = MEASUREMENTS / "gemma_family_labels.json"
CHECK = MEASUREMENTS / "gemma_family_check.json"
REPORT = MEASUREMENTS / "gemma_family_report.json"
MIN_COVERAGE = 0.95  # as EvalKit's calibration: a judge must rule on 95% of the labelled claims


class FamilyItem(Model):
    key: NonBlank  # opaque claim key shown on the page
    answer: NonBlank  # opaque answer key
    model: NonBlank  # private: never rendered
    case_id: NonBlank
    claim_id: NonBlank
    claim_sha256: Sha256  # the claim as drawn (id, text, cited), to catch a changed run
    claim_text: str  # redacted, as the page shows it
    cited: tuple[tuple[int, str], ...]  # (sentence number, redacted text)


class FamilySheet(Model):
    sheet_id: NonBlank
    run: NonBlank
    run_id: NonBlank
    seed: int
    allocation: FrozenMap[NonNegInt]
    items: tuple[FamilyItem, ...]

    def rows(self) -> list[SheetRow]:
        """EvalKit's blind rows, so its labelling page and label files work unchanged."""
        return [
            SheetRow(
                sheet_id=self.sheet_id,
                key=i.key,
                answer=i.answer,
                split="heldout",
                relabel=False,
                claim_id=i.claim_id,
                claim_text=i.claim_text,
                cited=i.cited,
            )
            for i in self.items
        ]


class JudgeCheck(Model):
    judge: NonBlank
    judge_prompt_hash: NonBlank
    n_claims: PosInt
    coverage: float
    agreement: float
    kappa: dict[str, float]  # point, lo, hi
    confusion: tuple[tuple[int, int], tuple[int, int]]  # ((tp, fn), (fp, tn)), reference rows
    by_model: FrozenMap[float]  # agreement per Gemma model


LABELLER = "claude-opus-5-5 with configs/labeller_prompt.md (OWNER D-60)"


class FamilyReport(Model):
    sheet_id: NonBlank
    run: NonBlank
    run_id: NonBlank
    scored_at: NonBlank
    n_labelled: PosInt
    extraction_errors: NonNegInt
    labeller: NonBlank
    reference_supported: NonNegInt
    judges: tuple[JudgeCheck, ...]


def _h(*parts: object) -> str:
    return hashlib.sha256(":".join(str(p) for p in parts).encode()).hexdigest()


def allocation(models: Sequence[str], n: int = N) -> dict[str, int]:
    """Half to the base model (the run's first), the rest evenly over the fine-tunes in run
    order, any remainder to the earlier ones: 20/10/10 for base, v1, v2."""
    if not models:
        raise ResultsError("the run has no models")
    if len(models) == 1:
        return {models[0]: n}
    base, tuned = models[0], list(models[1:])
    rest = n - n // 2
    per, extra = divmod(rest, len(tuned))
    out = {base: n // 2}
    out.update({m: per + (i < extra) for i, m in enumerate(tuned)})
    return out


def parse_allocation(text: str) -> dict[str, int]:
    """`model=count,model=count` from the command line."""
    out: dict[str, int] = {}
    for part in text.split(","):
        name, _, count = part.partition("=")
        if not name.strip() or not count.strip().isdigit():
            raise ResultsError(f"allocation {part!r} is not model=count")
        out[name.strip()] = int(count)
    return out


def _claim_digest(claim: dict[str, Any]) -> str:
    return canonical_sha256({k: claim[k] for k in ("id", "text", "cited")})


def draw(
    run: Run, *, n: int = N, seed: int = SEED, alloc: dict[str, int] | None = None
) -> FamilySheet:
    models = list(run.manifest["models"])
    alloc = dict(alloc) if alloc is not None else allocation(models, n)
    unknown = sorted(set(alloc) - set(models))
    if unknown:
        raise ResultsError(f"{run.name} has no model(s) {', '.join(unknown)}")
    if sum(alloc.values()) != n:
        raise ResultsError(f"allocation adds up to {sum(alloc.values())}, not {n}")
    by_model: dict[str, list[dict[str, Any]]] = {}
    for r in run.raw_records:
        if r["claims"]:
            by_model.setdefault(r["completion"]["model"], []).append(r)
    taken: set[str] = set()
    items: list[FamilyItem] = []
    for model in models:
        want = alloc.get(model, 0)
        recs = sorted(by_model.get(model, []), key=lambda r: _h(seed, run.name, r["case_id"]))
        got = 0
        for rec in recs:
            if got == want:
                break
            if rec["case_id"] in taken:
                continue
            claims = rec["claims"]
            c = claims[int(_h(seed, "claim", rec["case_id"], model), 16) % len(claims)]
            case = run.cases[rec["case_id"]]
            cited = tuple(
                (int(k), redact(case.context[k - 1]))
                for k in c["cited"]
                if 1 <= k <= len(case.context)
            )
            text = redact(c["text"])
            items.append(
                FamilyItem(
                    key="k" + _h("ggh-family", seed, c["id"], text, cited)[:12],
                    answer="a" + _h("ggh-family", seed, rec["case_id"], model)[:10],
                    model=model,
                    case_id=rec["case_id"],
                    claim_id=c["id"],
                    claim_sha256=_claim_digest(c),
                    claim_text=text,
                    cited=cited,
                )
            )
            taken.add(rec["case_id"])
            got += 1
        if got < want:
            raise ResultsError(
                f"{run.name}: only {got} labellable answers for {model}; need {want}"
            )
    items.sort(key=lambda i: _h(seed, "order", i.key))  # blind order: models interleaved
    sheet_id = "gf" + _h(run.manifest["run_id"], *sorted(i.key for i in items))[:12]
    return FamilySheet(
        sheet_id=sheet_id,
        run=run.name,
        run_id=run.manifest["run_id"],
        seed=seed,
        allocation=alloc,
        items=tuple(items),
    )


def load_sheet(root: Path) -> FamilySheet:
    path = root / SHEET
    if not path.is_file():
        raise ResultsError(f"{SHEET} is missing: run `ggh calibrate sample` first")
    return FamilySheet.model_validate_json(path.read_text(encoding="utf-8"))


def load_labels(root: Path) -> LabelFile | None:
    path = root / LABELS
    if not path.is_file():
        return None
    return LabelFile.model_validate_json(path.read_text(encoding="utf-8"))


def write_sheet(root: Path, sheet: FamilySheet, *, force: bool = False) -> Path:
    """Refuses to replace a sheet that already has labels (they only fit that sheet), and any
    other existing sheet without `force`."""
    path = root / SHEET
    labels = load_labels(root)
    if labels is not None and labels.labels:
        raise FileExistsError(f"{LABELS} holds labels for the current sheet; it is not redrawn")
    if path.is_file() and not force:
        raise FileExistsError(f"{SHEET} exists (use --force to redraw; no labels yet)")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(sheet.model_dump_json(indent=1) + "\n", encoding="utf-8", newline="\n")
    return path


def serve(root: Path, *, port: int = 0) -> Any:
    """EvalKit's labelling page on this sheet, bound to 127.0.0.1 (the caller serves it)."""
    from evalkit.pipeline.labeller import serve_labeller

    sheet = load_sheet(root)
    return serve_labeller(sheet.rows(), root / LABELS, round_="first", seed=sheet.seed, port=port)


def _verdicts(run: Run, judge: str) -> dict[str, dict[str, Any]]:
    return {v["claim_id"]: v for r in run.raw_records for v in r["verdicts"] if v["judge"] == judge}


def score(
    root: Path, *, results_root: Path | None = None, now: datetime | None = None
) -> FamilyReport:
    """Every judge of the sheet's run against the reference labels. Refuses an incomplete label
    file, labels for another sheet, a run that changed since the draw, verdicts under another
    rubric, and a judge that ruled on fewer than 95% of the eligible claims."""
    sheet = load_sheet(root)
    labels = load_labels(root)
    if labels is None or labels.sheet_id != sheet.sheet_id or labels.round != "first":
        raise ResultsError(f"{LABELS} has no first-round labels for sheet {sheet.sheet_id}")
    missing = [i.key for i in sheet.items if i.key not in labels.labels]
    if missing:
        raise ResultsError(f"{len(missing)} of {len(sheet.items)} claims are not labelled yet")
    run = Run(results_root or root, sheet.run)
    if run.manifest["run_id"] != sheet.run_id:
        raise ResultsError(f"{sheet.run} is run {run.manifest['run_id']}, not {sheet.run_id}")
    claims = {c["id"]: c for r in run.raw_records for c in r["claims"]}
    for i in sheet.items:
        c = claims.get(i.claim_id)
        if c is None or _claim_digest(c) != i.claim_sha256:
            raise ResultsError(f"claim {i.key} changed in {sheet.run} since it was drawn")
    eligible = [i for i in sheet.items if not labels.labels[i.key].extraction_error]
    if not eligible:
        raise ResultsError("every labelled claim is flagged as an extraction error")
    checks = []
    for judge in run.manifest["judges"]:
        vs = _verdicts(run, judge)
        covered = [i for i in eligible if i.claim_id in vs]
        coverage = len(covered) / len(eligible)
        if coverage < MIN_COVERAGE:
            raise ResultsError(f"{judge} ruled on {len(covered)} of {len(eligible)} claims")
        if any(vs[i.claim_id]["judge_prompt_hash"] != JUDGE_PROMPT_HASH for i in covered):
            raise ResultsError(f"{judge}'s verdicts were made under another rubric")
        pairs = [
            (
                i,
                CalibrationPair(
                    i.answer,
                    labels.labels[i.key].label == "supported",
                    vs[i.claim_id]["verdict"] == "supported",
                ),
            )
            for i in covered
        ]
        tp = sum(p.human and p.judge for _, p in pairs)
        fn = sum(p.human and not p.judge for _, p in pairs)
        fp = sum(not p.human and p.judge for _, p in pairs)
        tn = len(pairs) - tp - fn - fp
        k = cluster_kappa([p for _, p in pairs], seed=sheet.seed)
        by_model: dict[str, float] = {}
        for m in sorted({i.model for i, _ in pairs}):
            mine = [p for i, p in pairs if i.model == m]
            by_model[m] = round(sum(p.human == p.judge for p in mine) / len(mine), 6)
        checks.append(
            JudgeCheck(
                judge=judge,
                judge_prompt_hash=JUDGE_PROMPT_HASH,
                n_claims=len(pairs),
                coverage=round(coverage, 6),
                agreement=round((tp + tn) / len(pairs), 6),
                kappa={"point": k.point, "lo": k.lo, "hi": k.hi},
                confusion=((tp, fn), (fp, tn)),
                by_model=by_model,
            )
        )
    stamp = (now or datetime.now().astimezone()).isoformat(timespec="seconds")
    return FamilyReport(
        sheet_id=sheet.sheet_id,
        run=sheet.run,
        run_id=sheet.run_id,
        scored_at=stamp,
        n_labelled=len(sheet.items),
        extraction_errors=len(sheet.items) - len(eligible),
        labeller=LABELLER,
        reference_supported=sum(labels.labels[i.key].label == "supported" for i in eligible),
        judges=tuple(checks),
    )


Written = Literal["check", "report"]


def write_report(root: Path, report: FamilyReport) -> dict[Written, Path]:
    """`gemma_family_check.json` holds judge -> agreement only (the bundle's map); the report
    holds the rest."""
    check = {j.judge: j.agreement for j in report.judges}
    paths: dict[Written, Path] = {"check": root / CHECK, "report": root / REPORT}
    paths["check"].write_text(json.dumps(check, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["report"].write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return paths
