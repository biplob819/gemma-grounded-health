"""Pre-registered evaluation plans (Tech Spec §7.4, H9; RUN-01, VEV-03).

A plan is `configs/eval_plan*.yaml`. Its sha256 goes in `docs/measurements/<stem>.sha256`, and the
commit that added that hash, with its commit time, is appended to
`docs/measurements/preregistrations.jsonl`. `preregistration_problems` checks, for every run a
plan lists that has committed results, that the plan still has its hash and that the hash was
committed before the run started. It is data-driven: a new plan or run is covered as soon as it
exists, with no test to edit.

The E3.2 plan predates the `status` and `runs` fields: a plan without `status` is final, and a
plan without `runs` covers its main run, `docs/results/main/`.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ggh.results.runs import RESULTS, Run

PREREG = Path("docs") / "measurements" / "preregistrations.jsonl"
MEASUREMENTS = Path("docs") / "measurements"


def parse_time(text: str) -> dt.datetime:
    t = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    if t.tzinfo is None:
        raise ValueError(f"timestamp {text!r} has no timezone")
    return t


@dataclass(frozen=True)
class PlannedRun:
    name: str
    results: Path  # relative to the repo root
    spec: dict[str, Any]


@dataclass(frozen=True)
class Plan:
    path: Path  # relative to the repo root
    data: dict[str, Any]
    sha256: str

    @property
    def status(self) -> str:
        return str(self.data.get("status", "final"))

    @property
    def hash_file(self) -> Path:
        return MEASUREMENTS / f"{self.path.stem}.sha256"

    @property
    def runs(self) -> list[PlannedRun]:
        specs = self.data.get("runs") or [{"name": "main", "results": str(RESULTS / "main")}]
        return [PlannedRun(s["name"], Path(s["results"]), s) for s in specs]

    @property
    def primary(self) -> list[tuple[str, str]]:
        return [(a, b) for a, b in self.data["primary"]]


def load_plan(root: Path, rel: Path | str) -> Plan:
    rel = Path(rel)
    raw = (root / rel).read_bytes()
    return Plan(rel, yaml.safe_load(raw), hashlib.sha256(raw).hexdigest())


def all_plans(root: Path) -> list[Plan]:
    return [
        load_plan(root, p.relative_to(root)) for p in sorted(root.glob("configs/eval_plan*.yaml"))
    ]


def preregistrations(root: Path) -> list[dict[str, Any]]:
    path = root / PREREG
    if not path.is_file():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _git(root: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)
    except OSError:
        return None
    return out.stdout if out.returncode == 0 else None


def plan_problems(root: Path, plan: Plan) -> list[str]:
    """Static checks on a plan, whatever has been run."""
    p: list[str] = []
    has_hash = (root / plan.hash_file).is_file()
    if plan.status == "draft":
        if has_hash:
            p.append(f"{plan.path}: a draft plan must not have a committed hash ({plan.hash_file})")
    elif plan.status == "final":
        if not has_hash:
            p.append(f"{plan.path}: a final plan needs {plan.hash_file}")
        elif (root / plan.hash_file).read_text(encoding="utf-8").strip() != plan.sha256:
            p.append(f"{plan.path}: its bytes no longer match {plan.hash_file}")
    else:
        p.append(f"{plan.path}: status {plan.status!r} is neither draft nor final")
    for key in ("metric", "judges", "primary", "rule"):
        if key not in plan.data:
            p.append(f"{plan.path}: no {key!r}")
    rule = str(plan.data.get("rule", ""))
    if "Holm" not in rule or "tie" not in rule:
        p.append(f"{plan.path}: the rule must name Holm and the tie")
    return p


def run_started(root: Path, run: PlannedRun) -> dt.datetime | None:
    """When a planned run started, from its committed run.json; None if it has no results."""
    manifest = root / run.results / "run.json"
    if not Run.exists(root, run.results.name) or not manifest.is_file():
        return None
    return parse_time(json.loads(manifest.read_text(encoding="utf-8"))["started_at"])


def preregistration_problems(root: Path, plan: Plan) -> list[str]:
    """For every run of the plan that has results: the plan is final, its hash is committed
    and recorded, and that commit is older than the run's start (git checks it when present)."""
    ran = [(r, t) for r in plan.runs if (t := run_started(root, r)) is not None]
    if not ran:
        return []
    p: list[str] = []
    names = ", ".join(r.name for r, _ in ran)
    if plan.status != "final":
        return [f"{plan.path}: is {plan.status} but has results ({names})"]
    hash_path = root / plan.hash_file
    if not hash_path.is_file() or hash_path.read_text(encoding="utf-8").strip() != plan.sha256:
        return [f"{plan.path}: no committed hash matching its bytes, yet {names} ran"]
    rows = [r for r in preregistrations(root) if r.get("plan") == plan.path.as_posix()]
    if len(rows) != 1:
        return [f"{PREREG}: needs exactly one row for {plan.path}, found {len(rows)}"]
    row = rows[0]
    if row.get("sha256") != plan.sha256 or row.get("hash_file") != plan.hash_file.as_posix():
        p.append(f"{PREREG}: the row for {plan.path} names another hash or hash file")
    committed = parse_time(row["committed_at"])
    for run, started in ran:
        if not committed < started:
            p.append(
                f"{run.name} started {started.isoformat()}, before the plan hash was "
                f"committed ({committed.isoformat()})"
            )
    when = _git(root, "log", "-1", "--format=%cI", row["commit"])
    if when and when.strip():  # absent in a shallow CI clone
        if parse_time(when.strip()) != committed:
            p.append(
                f"{PREREG}: {row['commit'][:12]} was committed at {when.strip()}, "
                f"not {row['committed_at']}"
            )
        shown = _git(root, "show", f"{row['commit']}:{plan.hash_file.as_posix()}")
        if shown is None or shown.strip() != plan.sha256:
            p.append(f"{row['commit'][:12]} does not hold {plan.hash_file} with the plan's hash")
    return p
