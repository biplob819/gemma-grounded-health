"""Read the sample (Tech Spec §7.6, D-18, RES-05; EvalKit L-37).

`draw` picks 20 records per model from a committed run, seeded, and `write_sheet` writes them
to `docs/measurements/sample_read_<run>.jsonl` (one `SampleReadRow` per line): the question,
the answer, the cited sentences, every claim with each judge's verdict, rationale and
per-sentence citation support, and empty `finding`, `note` and `read_by` fields. The reader
fills those three in place. `summarise` validates the sheet against the run (a record edited
after it was drawn is caught by its digest) and counts the findings per model; any
`pipeline_fault` stops publication until its `explanation` names a committed file that
explains it.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

from ggh.results.runs import Run, canonical_sha256
from ggh.schemas import SampleReadRow, SampleReadSummary

PER_MODEL = 20
SEED = 7
SHEET = "docs/measurements/sample_read_{run}.jsonl"


def _rank(seed: int, run: str, case_id: str) -> str:
    return hashlib.sha256(f"{seed}:{run}:{case_id}".encode()).hexdigest()


def _claims(rec: dict[str, Any]) -> list[dict[str, Any]]:
    support = rec.get("citation_support") or {}
    out = []
    for c in rec["claims"]:
        verdicts = {}
        for v in rec["verdicts"]:
            if v["claim_id"] != c["id"]:
                continue
            cs = [
                {"n": s["n"], "supports": s["supports"], "role": s["role"]}
                for s in support.get(v["judge"], [])
                if s["claim_id"] == c["id"]
            ]
            verdicts[v["judge"]] = {
                "verdict": v["verdict"],
                "evidence": list(v["evidence"]),
                "rationale": v["rationale"],
                "citation_support": cs,
            }
        out.append(
            {"id": c["id"], "text": c["text"], "cited": list(c["cited"]), "verdicts": verdicts}
        )
    return out


def draw(run: Run, per_model: int = PER_MODEL, seed: int = SEED) -> list[SampleReadRow]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for r in run.raw_records:
        by_model.setdefault(r["completion"]["model"], []).append(r)
    rows: list[SampleReadRow] = []
    for model in run.models():
        recs = sorted(by_model.get(model, []), key=lambda r: _rank(seed, run.name, r["case_id"]))
        for rec in recs[:per_model]:
            case = run.cases[rec["case_id"]]
            ans = rec["completion"].get("answer") or {}
            cites = [int(n) for n in ans.get("citations", [])]
            cited = {str(n): case.context[n - 1] for n in cites if 1 <= n <= len(case.context)}
            rows.append(
                SampleReadRow.model_validate(
                    {
                        "run": run.name,
                        "model": model,
                        "case_id": rec["case_id"],
                        "record_sha256": canonical_sha256(rec),
                        "reference_label": case.reference_label,
                        "label": ans.get("label"),
                        "answer": ans.get("answer", rec["completion"].get("raw_text", "")),
                        "citations": [n for n in cites if n >= 1],
                        "cited_sentences": cited,
                        "claims": _claims(rec),
                    }
                )
            )
    return rows


def sheet_path(root: Path, run: str) -> Path:
    return root / SHEET.format(run=run)


def write_sheet(root: Path, run: Run, *, force: bool = False) -> Path:
    path = sheet_path(root, run.name)
    if path.is_file() and not force:
        existing = load_sheet(path)
        if any(r.finding is not None for r in existing):
            raise FileExistsError(f"{path} already holds findings; it is not redrawn")
    text = "".join(r.model_dump_json() + "\n" for r in draw(run))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def load_sheet(path: Path) -> list[SampleReadRow]:
    return [
        SampleReadRow.model_validate_json(x)
        for x in path.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]


def summarise(root: Path, run: Run) -> tuple[list[SampleReadSummary], list[str]]:
    """Per-model summary of the sheet, and its problems (stale or foreign rows, pipeline faults)."""
    path = sheet_path(root, run.name)
    if not path.is_file():
        return [], []
    rows = load_sheet(path)
    digests = {
        (r["completion"]["model"], r["case_id"]): canonical_sha256(r) for r in run.raw_records
    }
    problems = []
    for r in rows:
        want = digests.get((r.model, r.case_id))
        if r.run != run.name or want is None:
            problems.append(f"{path.name}: {r.model} {r.case_id} is not a record of {run.name}")
        elif want != r.record_sha256:
            problems.append(f"{path.name}: {r.model} {r.case_id} changed since it was drawn")
        if r.finding == "pipeline_fault":
            doc = r.explanation.split("#", 1)[0].split(":", 1)[0]
            if not r.explanation:
                problems.append(f"{path.name}: pipeline fault on {r.model} {r.case_id}: {r.note}")
            elif not (root / doc).is_file():
                problems.append(f"{path.name}: {r.model} {r.case_id} is explained by {doc}, "
                                "which is not committed")  # fmt: skip
    out = []
    for model in run.models():
        mine = [r for r in rows if r.model == model]
        if not mine:
            continue
        found = Counter(r.finding for r in mine if r.finding is not None)
        read = sum(found.values())
        out.append(
            SampleReadSummary(
                run=run.name,
                model=model,
                drawn=len(mine),
                read=read,
                findings=dict(sorted(found.items())),
                complete=read == len(mine),
            )
        )
    return out, problems
