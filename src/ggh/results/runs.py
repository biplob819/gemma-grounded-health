"""Reading committed EvalKit run folders under docs/results/ (E3.2 layout).

A folder holds `run.json`, `summary.json`, and `records.jsonl[.gz]` and `cases.jsonl[.gz]`.
Records are parsed into EvalKit's `Record` and `Case` with fields the pinned EvalKit does not
know (such as `citation_support` from a later pin) kept aside, so a run made at a newer pin
still reads; the summary is read as JSON for the same reason.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

from evalkit.schemas import Case, Record

RESULTS = Path("docs") / "results"


class ResultsError(ValueError):
    """A committed results file that is missing, unreadable or inconsistent."""


def _read_maybe_gz(folder: Path, stem: str) -> bytes:
    for name in (f"{stem}.jsonl.gz", f"{stem}.jsonl"):
        p = folder / name
        if p.is_file():
            raw = p.read_bytes()
            return gzip.decompress(raw) if name.endswith(".gz") else raw
    raise ResultsError(f"{folder}: no {stem}.jsonl or {stem}.jsonl.gz")


def jsonl(raw: bytes) -> list[dict[str, Any]]:
    return [json.loads(x) for x in raw.decode("utf-8").splitlines() if x.strip()]


def canonical_sha256(obj: Any) -> str:
    blob = json.dumps(obj, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("ascii")).hexdigest()


def _known(cls: type, row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if k in cls.model_fields}  # type: ignore[attr-defined]


@dataclass
class Run:
    """One committed run folder."""

    root: Path
    name: str
    folder: Path = field(init=False)

    def __post_init__(self) -> None:
        self.folder = self.root / RESULTS / self.name
        if not (self.folder / "summary.json").is_file():
            raise ResultsError(f"{self.folder}: no summary.json")

    @staticmethod
    def exists(root: Path, name: str) -> bool:
        return (root / RESULTS / name / "summary.json").is_file()

    @cached_property
    def manifest(self) -> dict[str, Any]:
        return json.loads((self.folder / "run.json").read_text(encoding="utf-8"))

    @cached_property
    def summary(self) -> dict[str, Any]:
        return json.loads((self.folder / "summary.json").read_text(encoding="utf-8"))

    @cached_property
    def raw_records(self) -> list[dict[str, Any]]:
        return jsonl(_read_maybe_gz(self.folder, "records"))

    @cached_property
    def records(self) -> list[Record]:
        return [Record.model_validate(_known(Record, r)) for r in self.raw_records]

    @cached_property
    def cases(self) -> dict[str, Case]:
        rows = jsonl(_read_maybe_gz(self.folder, "cases"))
        return {(c := Case.model_validate(_known(Case, r))).id: c for r in rows}

    @cached_property
    def summary_sha256(self) -> str:
        return hashlib.sha256((self.folder / "summary.json").read_bytes()).hexdigest()

    @cached_property
    def records_sha256(self) -> str:
        return hashlib.sha256(_read_maybe_gz(self.folder, "records")).hexdigest()

    def model_summary(self, model: str) -> dict[str, Any]:
        for m in self.summary["models"]:
            if m["model"] == model:
                return m
        raise ResultsError(f"{self.name}: no model {model!r} in summary.json")

    def models(self) -> list[str]:
        return [m["model"] for m in self.summary["models"]]

    @property
    def summary_rel(self) -> str:
        return (RESULTS / self.name / "summary.json").as_posix()
