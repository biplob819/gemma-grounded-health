#!/usr/bin/env python3
"""Run every gate for a build node and report all of them, even after a failure.

Usage:
    python scripts/node_converge.py W0.1          # run all 13 gates
    python scripts/node_converge.py --self-test   # prove every gate can fail

Exit 0 only when no gate FAILs. SKIP means the gate had nothing to check yet
(for example no golden tests exist before node E3.3); SKIP never hides a failure.

Gates:
   1 claims-work     at least one capability row of the node is building or green
   2 eligibility     graph has no cycles and nothing builds ahead of its dependencies
   3 proofs          every row has a proof for all five dimensions, and each proof resolves
   4 ruff            uv run ruff check .
   5 pyright         uv run pyright
   6 pytest          unit, property and contract tests
   7 golden          pytest -m golden
   8 data            pytest -m data (manifests, PQA-L isolation, provenance)
   9 recipe          pytest -m recipe (real training code, tiny model, CPU)
  10 safety          pytest -m safety, plus the secret scan
  11 build           uv build, install the wheel in a clean venv, run the smoke commands
  12 changed-tests   if source changed since node start, tests changed too
  13 records         changelog entry, criteria cover every row, RUN-STATE.md exists
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import secret_scan
from harness_lib import (
    PROOF_DIMENSIONS,
    ROOT,
    SPECS_DIR,
    Cap,
    HarnessError,
    Node,
    cap_mentioned,
    eligibility_problems,
    load_catalog,
    load_graph,
    parse_catalog,
    parse_graph,
    spec_dir,
)

# Commands run inside the clean venv by gate 11.
SMOKE_COMMANDS: list[list[str]] = [["ggh", "--version"], ["ggh", "--help"], ["ggh", "doctor"]]
PYTEST_UNIT_MARKERS = "not golden and not data and not recipe and not safety"
NA_MIN_REASON = 10
NA_FORBIDDEN = {"logical"}  # every capability needs at least a real unit test


@dataclass
class Result:
    status: str  # PASS | FAIL | SKIP
    detail: str


@dataclass
class Ctx:
    node: str
    root: Path
    nodes: dict[str, Node]
    caps: list[Cap]
    specs: Path


def _run(cmd: list[str], cwd: Path, timeout: int = 900) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        return 127, str(exc)
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout}s"
    return p.returncode, (p.stdout + p.stderr).strip()


def _tail(text: str, n: int = 12) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-n:])


# ---------- gates ----------


def gate_claims_work(ctx: Ctx) -> Result:
    rows = [c for c in ctx.caps if c.node == ctx.node]
    active = [c for c in rows if c.status in ("building", "green")]
    if not rows:
        return Result("FAIL", f"{ctx.node} has no catalog rows")
    if not active:
        return Result(
            "FAIL", f"no row of {ctx.node} is building or green (run node_start --mark-building)"
        )
    return Result("PASS", f"{len(active)}/{len(rows)} rows active")


def gate_eligibility(ctx: Ctx) -> Result:
    problems = eligibility_problems(ctx.nodes, ctx.caps)
    return Result("FAIL", "; ".join(problems)) if problems else Result("PASS", "eligible")


PROOF_RE = re.compile(r'^(?P<path>[^›"]+)›"(?P<test>[^"]+)"$')


def _test_functions(path: Path) -> dict[str, bool]:
    """Map each real test function name to whether it is skipped.

    Only `def test_*` at module level or inside a `Test*` class count, so names that
    appear in comments, strings or as prefixes of other names do not.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: dict[str, bool] = {}

    def record(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if fn.name.startswith("test"):
            found[fn.name] = any("skip" in ast.unparse(d) for d in fn.decorator_list)

    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            record(node)
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            for item in node.body:
                if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                    record(item)
    return found


def proof_problems(cap: Cap, root: Path) -> list[str]:
    if not cap.proof:
        return [f"{cap.cap_id}: no proof"]
    entries: dict[str, str] = {}
    for part in cap.proof.split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            return [f"{cap.cap_id}: malformed proof entry {part!r}"]
        key, value = (s.strip() for s in part.split("=", 1))
        if key in entries:
            return [f"{cap.cap_id}: duplicate proof key {key!r}"]
        entries[key] = value
    problems: list[str] = []
    tests_root = (root / "tests").resolve()
    for dim in PROOF_DIMENSIONS:
        value = entries.get(dim)
        if value is None:
            problems.append(f"{cap.cap_id}: missing {dim} proof")
            continue
        if value.startswith("n/a:"):
            if dim in NA_FORBIDDEN:
                problems.append(f"{cap.cap_id}: {dim} proof cannot be n/a")
            elif len(value[4:].strip()) < NA_MIN_REASON:
                problems.append(f"{cap.cap_id}: {dim} n/a needs a reason of {NA_MIN_REASON}+ chars")
            continue
        m = PROOF_RE.match(value)
        if not m:
            problems.append(f'{cap.cap_id}: {dim} proof must be path›"test" or n/a:<reason>')
            continue
        rel, test = m["path"].strip(), m["test"].strip()
        path = (root / rel).resolve()
        if not path.is_relative_to(tests_root) or path.suffix != ".py":
            problems.append(f"{cap.cap_id}: {dim} proof file {rel} must be a .py file under tests/")
            continue
        if not path.is_file():
            problems.append(f"{cap.cap_id}: {dim} proof file {rel} not found")
            continue
        try:
            tests = _test_functions(path)
        except SyntaxError as exc:
            problems.append(f"{cap.cap_id}: {dim} proof file {rel} does not parse: {exc.msg}")
            continue
        if test not in tests:
            problems.append(f"{cap.cap_id}: {dim} test {test!r} is not a test function in {rel}")
        elif tests[test]:
            problems.append(f"{cap.cap_id}: {dim} test {test!r} is skipped")
    extra = set(entries) - set(PROOF_DIMENSIONS)
    if extra:
        problems.append(f"{cap.cap_id}: unknown proof keys {sorted(extra)}")
    return problems


def gate_proofs(ctx: Ctx) -> Result:
    rows = [c for c in ctx.caps if c.node == ctx.node]
    problems = [p for c in rows for p in proof_problems(c, ctx.root)]
    if problems:
        shown = "; ".join(problems[:6]) + (
            f"; … {len(problems) - 6} more" if len(problems) > 6 else ""
        )
        return Result("FAIL", shown)
    return Result("PASS", f"{len(rows)} rows × {len(PROOF_DIMENSIONS)} dimensions resolve")


def _tool_gate(cmd: list[str], root: Path, label: str) -> Result:
    code, out = _run(cmd, root)
    if code == 0:
        return Result("PASS", label)
    return Result("FAIL", _tail(out))


def gate_ruff(ctx: Ctx) -> Result:
    return _tool_gate(
        [sys.executable, "-m", "ruff", "check", "--output-format", "concise", "."],
        ctx.root,
        "ruff clean",
    )


def gate_pyright(ctx: Ctx) -> Result:
    return _tool_gate([sys.executable, "-m", "pyright"], ctx.root, "pyright clean")


def _pytest(root: Path, marker: str, label: str) -> Result:
    code, out = _run([sys.executable, "-m", "pytest", "-q", "-m", marker], root)
    if code == 0:
        last = out.splitlines()[-1] if out else ""
        return Result("PASS", f"{label}: {last}")
    if code == 5:
        return Result("SKIP", f"no {label} tests collected yet")
    return Result("FAIL", _tail(out))


def gate_pytest(ctx: Ctx) -> Result:
    return _pytest(ctx.root, PYTEST_UNIT_MARKERS, "unit")


MIN_PYTHON = "3.11"  # requires-python floor; float and stdlib behaviour can differ (L-27)


def gate_golden(ctx: Ctx) -> Result:
    """Golden runs compare bytes, so run them on this interpreter and on the minimum
    supported Python (a separate .venv-py311), when uv can provide it."""
    here = _pytest(ctx.root, "golden", "golden")
    if here.status != "PASS" or os.environ.get("GGH_SKIP_MIN_PYTHON") or not shutil.which("uv"):
        return here
    env = {
        **os.environ,
        "UV_PROJECT_ENVIRONMENT": str(ctx.root / f".venv-py{MIN_PYTHON.replace('.', '')}"),
    }
    cmd = [
        "uv",
        "run",
        "--python",
        MIN_PYTHON,
        "--all-extras",
        "python",
        "-m",
        "pytest",
        "-q",
        "-m",
        "golden",
    ]
    proc = subprocess.run(cmd, cwd=ctx.root, env=env, capture_output=True, text=True, timeout=900)
    out = (proc.stdout + proc.stderr).strip()
    if proc.returncode != 0:
        return Result("FAIL", f"golden on Python {MIN_PYTHON}: " + _tail(out))
    return Result("PASS", f"{here.detail}; also on Python {MIN_PYTHON}")


def gate_data(ctx: Ctx) -> Result:
    return _pytest(ctx.root, "data", "data")


def gate_recipe(ctx: Ctx) -> Result:
    return _pytest(ctx.root, "recipe", "recipe")


def gate_safety(ctx: Ctx) -> Result:
    hits = secret_scan.scan(secret_scan.candidate_files(ctx.root))
    tests = _pytest(ctx.root, "safety", "safety")
    if hits:
        return Result("FAIL", "secret scan: " + "; ".join(hits[:5]))
    if tests.status == "FAIL":
        return tests
    return Result("PASS", f"secret scan clean; {tests.detail}")


def _venv_bin(venv: Path, exe: str) -> Path:
    sub = "Scripts" if os.name == "nt" else "bin"
    suffix = ".exe" if os.name == "nt" else ""
    return venv / sub / f"{exe}{suffix}"


def gate_build(ctx: Ctx, smoke: list[list[str]] | None = None) -> Result:
    smoke = SMOKE_COMMANDS if smoke is None else smoke
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "dist"
        code, log = _run(["uv", "build", "--wheel", "--out-dir", str(out)], ctx.root)
        if code != 0:
            return Result("FAIL", "uv build: " + _tail(log))
        wheels = sorted(out.glob("*.whl"))
        if not wheels:
            return Result("FAIL", "uv build produced no wheel")
        venv = Path(tmp) / "venv"
        code, log = _run(["uv", "venv", "-q", str(venv)], ctx.root)
        if code != 0:
            return Result("FAIL", "uv venv: " + _tail(log))
        py = _venv_bin(venv, "python")
        code, log = _run(
            ["uv", "pip", "install", "-q", "--python", str(py), str(wheels[0])], ctx.root
        )
        if code != 0:
            return Result("FAIL", "install wheel: " + _tail(log))
        for cmd in smoke:
            exe = _venv_bin(venv, cmd[0])
            code, log = _run([str(exe), *cmd[1:]], Path(tmp))
            if code != 0:
                return Result("FAIL", f"{' '.join(cmd)}: {_tail(log)}")
        return Result(
            "PASS", f"{wheels[0].name} installs; smoke: {', '.join(' '.join(c) for c in smoke)}"
        )


def changed_files(root: Path, base: str) -> list[str] | None:
    """Files changed since base: committed, staged, unstaged and untracked.

    Returns None when git is unusable.
    """
    code, _ = _run(["git", "rev-parse", "--git-dir"], root)
    if code != 0:
        return None
    files: set[str] = set()
    code, _ = _run(["git", "rev-parse", "HEAD"], root)
    if code == 0 and base not in ("", "none"):
        code, out = _run(["git", "diff", "--name-only", base], root)
        if code != 0:
            raise HarnessError(f"git diff against base {base!r} failed: {_tail(out, 2)}")
        files.update(out.splitlines())
    else:
        code, out = _run(["git", "ls-files"], root)
        files.update(out.splitlines())
    code, out = _run(["git", "ls-files", "--others", "--exclude-standard"], root)
    files.update(out.splitlines())
    return sorted(f for f in files if f)


def changed_tests_problem(files: list[str]) -> str | None:
    src = [
        f for f in files if f.startswith("src/") or (f.startswith("scripts/") and f.endswith(".py"))
    ]
    tests = [f for f in files if f.startswith("tests/") and not f.endswith("__init__.py")]
    if src and not tests:
        return f"{len(src)} source files changed but no tests changed (e.g. {src[0]})"
    return None


def gate_changed_tests(ctx: Ctx) -> Result:
    d = spec_dir(ctx.node, ctx.specs)
    base_file = d / ".base" if d else None
    base = (
        base_file.read_text(encoding="utf-8").strip()
        if base_file and base_file.is_file()
        else "none"
    )
    try:
        files = changed_files(ctx.root, base)
    except HarnessError as exc:
        return Result("FAIL", str(exc))
    if files is None:
        return Result("SKIP", "not a git repository")
    problem = changed_tests_problem(files)
    if problem:
        return Result("FAIL", problem)
    return Result(
        "PASS", f"{len(files)} files changed since {base[:10]}; tests changed with source"
    )


def records_problems(node: str, caps: list[Cap], root: Path, specs: Path) -> list[str]:
    problems: list[str] = []
    changelog = root / "changelog.md"
    if not changelog.is_file() or f"[{node}]" not in changelog.read_text(encoding="utf-8"):
        problems.append(f"changelog.md has no [{node}] entry")
    d = spec_dir(node, specs)
    if d is None:
        return [*problems, f"no spec folder for {node}"]
    criteria = d / "criteria.md"
    text = criteria.read_text(encoding="utf-8") if criteria.is_file() else ""
    missing = [c.cap_id for c in caps if c.node == node and not cap_mentioned(c.cap_id, text)]
    if missing:
        problems.append(f"criteria.md misses {', '.join(missing)}")
    if not (d / "RUN-STATE.md").is_file():
        problems.append("RUN-STATE.md missing")
    return problems


def gate_records(ctx: Ctx) -> Result:
    problems = records_problems(ctx.node, ctx.caps, ctx.root, ctx.specs)
    return Result("FAIL", "; ".join(problems)) if problems else Result("PASS", "records complete")


GATES: list[tuple[str, Callable[[Ctx], Result]]] = [
    ("claims-work", gate_claims_work),
    ("eligibility", gate_eligibility),
    ("proofs", gate_proofs),
    ("ruff", gate_ruff),
    ("pyright", gate_pyright),
    ("pytest", gate_pytest),
    ("golden", gate_golden),
    ("data", gate_data),
    ("recipe", gate_recipe),
    ("safety", gate_safety),
    ("build", gate_build),
    ("changed-tests", gate_changed_tests),
    ("records", gate_records),
]


# ---------- self-test: every gate must be shown to fail on bad input ----------


def self_test() -> int:
    """Feed each gate a planted defect and require FAIL that names the defect."""
    failures = 0

    def check(name: str, ok: bool) -> None:
        nonlocal failures
        failures += 0 if ok else 1
        print(f"  [{'ok' if ok else 'WRONG'}] {name}")

    graph = parse_graph("| **W0.1** | Harness | — |\n| **N1.1** | Schemas | W0.1 |\n")
    hdr = "cap_id\tnode\tstatus\tproof\tnote\n"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        specs = root / "specs"
        (specs / "W0.1-harness").mkdir(parents=True)

        def ctx(cat: str, node: str = "W0.1", at: Path = root) -> Ctx:
            return Ctx(node, at, graph, parse_catalog(hdr + cat), specs)

        # 1 claims-work
        r = gate_claims_work(ctx("H-1\tW0.1\tspec\t\t\n"))
        check(
            "1 claims-work fails naming the node when nothing started",
            r.status == "FAIL" and "W0.1" in r.detail and "building or green" in r.detail,
        )
        r = gate_claims_work(ctx("H-1\tW0.1\tbuilding\t\t\n"))
        check("1 claims-work passes when building", r.status == "PASS")
        r = gate_claims_work(ctx("H-1\tW0.1\tnot-started\t\t\n"))
        check("1 claims-work fails when the only row is not-started", r.status == "FAIL")
        r = gate_claims_work(ctx("S-1\tN1.1\tbuilding\t\t\n"))
        check(
            "1 claims-work fails naming a node with no rows",
            r.status == "FAIL" and "W0.1 has no catalog rows" in r.detail,
        )

        # 2 eligibility
        r = gate_eligibility(ctx("H-1\tW0.1\tbuilding\t\t\nS-1\tN1.1\tbuilding\t\t\n", "N1.1"))
        check("2 eligibility fails when building ahead", r.status == "FAIL" and "W0.1" in r.detail)
        r = gate_eligibility(ctx("H-1\tW0.1\tgreen\t\t\nS-1\tN1.1\tbuilding\t\t\n", "N1.1"))
        check("2 eligibility passes on a valid order", r.status == "PASS")
        cyc = parse_graph("| **W0.1** | H | N1.1 |\n| **N1.1** | S | W0.1 |\n")
        r = gate_eligibility(Ctx("W0.1", root, cyc, parse_catalog(hdr), specs))
        check("2 eligibility fails naming a cycle", r.status == "FAIL" and "cycle" in r.detail)
        unk = parse_graph("| **W0.1** | H | — |\n| **N1.1** | S | W0.9 |\n")
        r = gate_eligibility(Ctx("W0.1", root, unk, parse_catalog(hdr), specs))
        check(
            "2 eligibility fails naming an unknown dependency",
            r.status == "FAIL" and "unknown node W0.9" in r.detail,
        )

        # 3 proofs
        tests_dir = root / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_x.py").write_text(
            "import pytest\n\n"
            "# test_in_comment\n"
            "NAME = 'test_in_string'\n\n"
            "def test_real():\n    pass\n\n"
            "def test_realistic():\n    pass\n\n"
            "@pytest.mark.skip\ndef test_skipped():\n    pass\n\n"
            "@pytest.mark.skipif(True, reason='x')\ndef test_skipif():\n    pass\n",
            encoding="utf-8",
        )
        (root / "outside.py").write_text("def test_real():\n    pass\n", encoding="utf-8")
        (tests_dir / "test_broken.py").write_text("def test_real(:\n", encoding="utf-8")
        ok = "; ".join(
            [f'{d}=tests/test_x.py›"test_real"' for d in PROOF_DIMENSIONS[:3]]
            + ["repro=n/a:no state to reproduce here", "safety=n/a:no secrets or spend involved"]
        )
        r = gate_proofs(ctx(f"H-1\tW0.1\tgreen\t{ok}\t\n"))
        check("3 proofs pass on resolvable proof", r.status == "PASS")
        # each planted defect must be named in the detail, not just produce some FAIL
        bad = {
            "empty proof": ("", "no proof"),
            "missing dimension": ('logical=tests/test_x.py›"test_real"', "missing contract proof"),
            "unknown test": (ok.replace('"test_real"', '"test_ghost"', 1), "'test_ghost'"),
            "name only in a comment": (
                ok.replace('"test_real"', '"test_in_comment"', 1),
                "'test_in_comment' is not a test function",
            ),
            "name only in a string": (
                ok.replace('"test_real"', '"test_in_string"', 1),
                "'test_in_string' is not a test function",
            ),
            "prefix of another test": (
                ok.replace('"test_real"', '"test_rea"', 1),
                "'test_rea' is not a test function",
            ),
            "skipped test": (ok.replace('"test_real"', '"test_skipped"', 1), "is skipped"),
            "skipif test": (ok.replace('"test_real"', '"test_skipif"', 1), "is skipped"),
            "path outside tests/": (
                ok.replace("tests/test_x.py", "outside.py", 1),
                "outside.py must be a .py file under tests/",
            ),
            "path escaping repo": (
                ok.replace("tests/test_x.py", "tests/../../x.py", 1),
                "must be a .py file under tests/",
            ),
            "short n/a reason": (
                ok.replace("n/a:no state to reproduce here", "n/a:x"),
                "repro n/a needs a reason",
            ),
            "logical n/a": (
                ok.replace(
                    'logical=tests/test_x.py›"test_real"', "logical=n/a:nothing to unit test here"
                ),
                "logical proof cannot be n/a",
            ),
            "duplicate key": (ok + '; logical=tests/test_x.py›"test_real"', "duplicate proof key"),
            "unknown key": (ok + "; vibes=n/a:not a real proof dimension", "unknown proof keys"),
            "missing file": (ok.replace("tests/test_x.py", "tests/test_gone.py", 1), "not found"),
            "bad format": (
                ok.replace('tests/test_x.py›"test_real"', "tests/test_x.py", 1),
                "must be path",
            ),
            "malformed entry": (ok + "; just words", "malformed proof entry"),
            "file does not parse": (
                ok.replace("tests/test_x.py", "tests/test_broken.py", 1),
                "does not parse",
            ),
        }
        for label, (proof, expect) in bad.items():
            r = gate_proofs(ctx(f"H-1\tW0.1\tgreen\t{proof}\t\n"))
            check(f"3 proofs fail on {label}", r.status == "FAIL" and expect in r.detail)

        # 4 ruff — must name the planted rule, so a missing tool cannot pass for a caught defect
        lint = root / "lint"
        lint.mkdir()
        (lint / "bad.py").write_text("import os\n", encoding="utf-8")
        r = gate_ruff(ctx("H-1\tW0.1\tbuilding\t\t\n", at=lint))
        check("4 ruff fails naming F401", r.status == "FAIL" and "F401" in r.detail)
        # the gate must honour the project's ruff config, not a fixed rule set
        (lint / "ruff.toml").write_text('[lint]\nselect = ["E", "F"]\n', encoding="utf-8")
        (lint / "bad.py").write_text("x = 1\nif x == None:\n    pass\n", encoding="utf-8")
        r = gate_ruff(ctx("H-1\tW0.1\tbuilding\t\t\n", at=lint))
        check(
            "4 ruff fails naming E711 from the project config",
            r.status == "FAIL" and "E711" in r.detail,
        )
        (lint / "bad.py").write_text("x = 1\n", encoding="utf-8")
        check(
            "4 ruff passes on clean code",
            gate_ruff(ctx("H-1\tW0.1\tbuilding\t\t\n", at=lint)).status == "PASS",
        )

        # 5 pyright
        types = root / "types"
        types.mkdir()
        (types / "bad.py").write_text('x: int = "a"\n', encoding="utf-8")
        (types / "pyrightconfig.json").write_text("{}", encoding="utf-8")
        r = gate_pyright(ctx("H-1\tW0.1\tbuilding\t\t\n", at=types))
        check("5 pyright fails naming bad.py", r.status == "FAIL" and "bad.py" in r.detail)
        (types / "bad.py").write_text("x: int = 1\n", encoding="utf-8")
        check(
            "5 pyright passes on clean code",
            gate_pyright(ctx("H-1\tW0.1\tbuilding\t\t\n", at=types)).status == "PASS",
        )

        # 6-10 pytest-backed gates, each on its own failing fixture
        pt = root / "pt"
        pt.mkdir()
        (pt / "test_fail.py").write_text(
            "import pytest\n\n"
            "def test_unit_fails():\n    assert False\n\n"
            "@pytest.mark.golden\ndef test_golden_fails():\n    assert False\n\n"
            "@pytest.mark.data\ndef test_data_fails():\n    assert False\n\n"
            "@pytest.mark.recipe\ndef test_recipe_fails():\n    assert False\n\n"
            "@pytest.mark.safety\ndef test_safety_fails():\n    assert False\n",
            encoding="utf-8",
        )
        at_pt = ctx("H-1\tW0.1\tbuilding\t\t\n", at=pt)
        r = gate_pytest(at_pt)
        check(
            "6 pytest fails naming the failing test",
            r.status == "FAIL" and "test_unit_fails" in r.detail,
        )
        r = gate_golden(at_pt)
        check(
            "7 golden fails naming the failing test",
            r.status == "FAIL" and "test_golden_fails" in r.detail,
        )
        r = gate_data(at_pt)
        check(
            "8 data fails naming the failing test",
            r.status == "FAIL" and "test_data_fails" in r.detail,
        )
        r = gate_recipe(at_pt)
        check(
            "9 recipe fails naming the failing test",
            r.status == "FAIL" and "test_recipe_fails" in r.detail,
        )
        r = gate_safety(ctx("H-1\tW0.1\tbuilding\t\t\n", at=pt))
        check(
            "10 safety fails on a failing safety test",
            r.status == "FAIL" and "test_safety_fails" in r.detail,
        )
        empty = root / "empty"
        empty.mkdir()
        (empty / "test_none.py").write_text("def test_ok():\n    pass\n", encoding="utf-8")
        at_empty = ctx("H-1\tW0.1\tbuilding\t\t\n", at=empty)
        for i, marker, gate in (
            (7, "golden", gate_golden),
            (8, "data", gate_data),
            (9, "recipe", gate_recipe),
        ):
            check(
                f"{i} {marker} gate reports SKIP, not PASS, with no {marker} tests",
                gate(at_empty).status == "SKIP",
            )
        check("6 pytest gate passes on a passing suite", gate_pytest(at_empty).status == "PASS")
        broken_tests = root / "broken_tests"
        broken_tests.mkdir()
        (broken_tests / "test_imp.py").write_text(
            "import no_such_module_ggh\n\ndef test_x():\n    pass\n", encoding="utf-8"
        )
        at_broken = ctx("H-1\tW0.1\tbuilding\t\t\n", at=broken_tests)
        for i, name, gate in (
            (6, "pytest", gate_pytest),
            (7, "golden", gate_golden),
            (8, "data", gate_data),
            (9, "recipe", gate_recipe),
        ):
            r = gate(at_broken)
            check(
                f"{i} {name} gate fails, never SKIPs, on a collection error",
                r.status == "FAIL" and "no_such_module_ggh" in r.detail,
            )
        leak = root / "leak"
        leak.mkdir()
        (leak / "k.py").write_text('K = "sk-ant-' + "Q" * 30 + '"\n', encoding="utf-8")
        r = gate_safety(ctx("H-1\tW0.1\tbuilding\t\t\n", at=leak))
        check("10 safety fails on a leaked key", r.status == "FAIL" and "anthropic key" in r.detail)
        (leak / "k.py").write_text("K = 1\n", encoding="utf-8")
        (leak / "notes.md").write_text("token: hf_" + "Ab3d" * 9 + "\n", encoding="utf-8")
        r = gate_safety(ctx("H-1\tW0.1\tbuilding\t\t\n", at=leak))
        check(
            "10 safety fails on a leak in a non-.py file",
            r.status == "FAIL" and "hugging face token" in r.detail,
        )
        (leak / "notes.md").unlink()
        check(
            "10 safety passes on a clean tree",
            gate_safety(ctx("H-1\tW0.1\tbuilding\t\t\n", at=leak)).status == "PASS",
        )

        # 11 build
        broken = root / "broken"
        broken.mkdir()
        (broken / "pyproject.toml").write_text("[project\nname=", encoding="utf-8")
        r = gate_build(Ctx("W0.1", broken, graph, [], specs), smoke=[])
        check("11 build fails on a broken project", r.status == "FAIL" and "uv build" in r.detail)
        tiny = root / "tiny"
        (tiny / "tinypkg").mkdir(parents=True)
        (tiny / "tinypkg" / "__init__.py").write_text(
            "def main():\n    print('tiny ok')\n", encoding="utf-8"
        )
        (tiny / "pyproject.toml").write_text(
            '[project]\nname = "tinypkg"\nversion = "0.0.1"\n'
            '[project.scripts]\nggh-selftest-tiny = "tinypkg:main"\n'
            '[build-system]\nrequires = ["hatchling"]\nbuild-backend = "hatchling.build"\n',
            encoding="utf-8",
        )
        # The tiny package needs its build backend (hatchling) from the uv cache or PyPI. Check
        # that independently of gate_build, so a broken gate is still caught when the backend is
        # available, and an offline machine with an empty cache reports a skip, not a failure.
        probe_out = root / "probe-dist"
        backend_ok = (
            _run(["uv", "build", "--wheel", "--out-dir", str(probe_out)], tiny, timeout=300)[0] == 0
        )
        if not backend_ok:
            print("  [skip] 11 tiny-package build checks: build backend unavailable (offline?)")
        else:
            # the console script exists only inside the wheel's venv, never on PATH
            r = gate_build(Ctx("W0.1", tiny, graph, [], specs), smoke=[["ggh-selftest-tiny"]])
            check("11 build passes running the wheel's own console script", r.status == "PASS")
            failing_smoke = [
                ["ggh-selftest-tiny"],
                ["python", "-c", "pass"],
                ["python", "-c", "raise SystemExit(3)"],
            ]
            r = gate_build(Ctx("W0.1", tiny, graph, [], specs), smoke=failing_smoke)
            check(
                "11 build fails naming a failing smoke command",
                r.status == "FAIL" and "raise SystemExit(3)" in r.detail,
            )
            global SMOKE_COMMANDS
            saved = SMOKE_COMMANDS
            SMOKE_COMMANDS = failing_smoke  # the default (smoke=None) path must use the module list
            try:
                r = gate_build(Ctx("W0.1", tiny, graph, [], specs))
            finally:
                SMOKE_COMMANDS = saved
            check(
                "11 build runs the module smoke list by default",
                r.status == "FAIL" and "raise SystemExit(3)" in r.detail,
            )
        check(
            "11 smoke commands are ggh --version, --help and doctor",
            SMOKE_COMMANDS == [["ggh", "--version"], ["ggh", "--help"], ["ggh", "doctor"]],
        )

        # 12 changed-tests
        check(
            "12 fails when only source changed",
            changed_tests_problem(["src/ggh/x.py"]) is not None,
        )
        check(
            "12 an __init__.py does not count as a test change",
            changed_tests_problem(["src/ggh/x.py", "tests/__init__.py"]) is not None,
        )
        check(
            "12 passes when tests changed",
            changed_tests_problem(["src/ggh/x.py", "tests/test_x.py"]) is None,
        )
        repo = root / "repo"
        repo.mkdir()
        for c in (
            ["git", "init", "-q"],
            [
                "git",
                "-c",
                "user.name=t",
                "-c",
                "user.email=t@t",
                "commit",
                "-q",
                "--allow-empty",
                "-m",
                "a",
            ],
        ):
            _run(c, repo)
        try:
            changed_files(repo, "deadbeef")
            check("12 an unknown base commit fails", False)
        except HarnessError:
            check("12 an unknown base commit fails", True)
        # the real gate, end to end, on a git repo with a recorded baseline
        head = _run(["git", "rev-parse", "HEAD"], repo)[1].strip()
        rspecs = repo / "specs" / "W0.1-harness"
        rspecs.mkdir(parents=True)
        (rspecs / ".base").write_text(head + "\n", encoding="utf-8")
        (repo / "src" / "ggh").mkdir(parents=True)
        (repo / "src" / "ggh" / "y.py").write_text("Y = 1\n", encoding="utf-8")
        rctx = Ctx("W0.1", repo, graph, [], repo / "specs")
        r = gate_changed_tests(rctx)
        check(
            "12 gate fails naming the source file when no test changed",
            r.status == "FAIL" and "src/ggh/y.py" in r.detail,
        )
        git = ["git", "-c", "user.name=t", "-c", "user.email=t@t"]
        _run(["git", "add", "src"], repo)
        _run([*git, "commit", "-q", "-m", "src only"], repo)
        r = gate_changed_tests(rctx)
        check(
            "12 gate fails naming a committed source change with no test",
            r.status == "FAIL" and "src/ggh/y.py" in r.detail,
        )
        (repo / "tests").mkdir()
        (repo / "tests" / "test_y.py").write_text("def test_y():\n    pass\n", encoding="utf-8")
        check("12 gate passes once a test changed", gate_changed_tests(rctx).status == "PASS")
        repo2 = root / "repo2"
        (repo2 / "tests").mkdir(parents=True)
        (repo2 / "tests" / "test_old.py").write_text("def test_o():\n    pass\n", encoding="utf-8")
        _run(["git", "init", "-q"], repo2)
        _run(["git", "add", "."], repo2)
        _run([*git, "commit", "-q", "-m", "tests before base"], repo2)
        base2 = _run(["git", "rev-parse", "HEAD"], repo2)[1].strip()
        (repo2 / "specs" / "W0.1-harness").mkdir(parents=True)
        (repo2 / "specs" / "W0.1-harness" / ".base").write_text(base2 + "\n", encoding="utf-8")
        (repo2 / "scripts").mkdir()
        (repo2 / "scripts" / "tool.py").write_text("T = 1\n", encoding="utf-8")
        _run(["git", "add", "scripts"], repo2)
        _run([*git, "commit", "-q", "-m", "script after base"], repo2)
        r = gate_changed_tests(Ctx("W0.1", repo2, graph, [], repo2 / "specs"))
        check(
            "12 gate measures from .base: a test committed before it does not count",
            r.status == "FAIL" and "scripts/tool.py" in r.detail,
        )
        (repo2 / "specs" / "W0.1-harness" / ".base").write_text("deadbeef\n", encoding="utf-8")
        r = gate_changed_tests(Ctx("W0.1", repo2, graph, [], repo2 / "specs"))
        check("12 gate fails on an unknown baseline", r.status == "FAIL" and "deadbeef" in r.detail)
        (repo2 / "scripts" / "tool.py").write_text("T = 2\n", encoding="utf-8")  # unstaged
        (repo2 / "specs" / "W0.1-harness" / ".base").write_text(
            _run(["git", "rev-parse", "HEAD"], repo2)[1].strip() + "\n", encoding="utf-8"
        )
        r = gate_changed_tests(Ctx("W0.1", repo2, graph, [], repo2 / "specs"))
        check("12 gate counts unstaged edits", r.status == "FAIL" and "scripts/tool.py" in r.detail)
        repo3 = root / "repo3"  # a repo with no recorded baseline (.base absent or "none")
        (repo3 / "src" / "ggh").mkdir(parents=True)
        (repo3 / "src" / "ggh" / "z.py").write_text("Z = 1\n", encoding="utf-8")
        _run(["git", "init", "-q"], repo3)
        _run(["git", "add", "."], repo3)
        _run([*git, "commit", "-q", "-m", "src"], repo3)
        r = gate_changed_tests(Ctx("W0.1", repo3, graph, [], repo3 / "specs"))
        check(
            "12 gate with no baseline counts tracked files and fails on source without tests",
            r.status == "FAIL" and "src/ggh/z.py" in r.detail,
        )
        nogit = root / "nogit"
        nogit.mkdir()
        r = gate_changed_tests(Ctx("W0.1", nogit, graph, [], nogit / "specs"))
        check("12 gate reports SKIP, not PASS, outside git", r.status == "SKIP")

        # 13 records
        r = gate_records(ctx("S-1\tN1.1\tgreen\t\t\n", node="N1.1"))
        check(
            "13 records fails naming a missing spec folder",
            r.status == "FAIL" and "no spec folder for N1.1" in r.detail,
        )
        rec = ctx("H-1\tW0.1\tgreen\t\t\n")
        wdir = specs / "W0.1-harness"
        (wdir / "criteria.md").write_text("H-1\n", encoding="utf-8")
        (wdir / "RUN-STATE.md").write_text("ok\n", encoding="utf-8")
        r = gate_records(rec)
        check(
            "13 records fails naming a missing changelog entry",
            r.status == "FAIL" and "changelog.md has no [W0.1] entry" in r.detail,
        )
        (root / "changelog.md").write_text(
            "[FEAT] [N1.1] — other node; mentions W0.1\n", encoding="utf-8"
        )
        r = gate_records(rec)
        check(
            "13 records fails when the changelog exists but has no entry for the node",
            r.status == "FAIL" and "changelog.md has no [W0.1] entry" in r.detail,
        )
        (root / "changelog.md").write_text("[FEAT] [W0.1] — harness\n", encoding="utf-8")
        check("13 records passes when complete", gate_records(rec).status == "PASS")
        (wdir / "criteria.md").write_text("nothing here\n", encoding="utf-8")
        r = gate_records(rec)
        check(
            "13 records fails naming a capability missing from criteria",
            r.status == "FAIL" and "criteria.md misses H-1" in r.detail,
        )
        (wdir / "criteria.md").write_text("H-10 only\n", encoding="utf-8")
        r = gate_records(rec)
        check(
            "13 records needs the cap id as a whole token (H-1 is not in H-10)",
            r.status == "FAIL" and "criteria.md misses H-1" in r.detail,
        )
        (wdir / "criteria.md").write_text("H-1\n", encoding="utf-8")
        (wdir / "RUN-STATE.md").unlink()
        r = gate_records(rec)
        check(
            "13 records fails naming a missing RUN-STATE.md",
            r.status == "FAIL" and "RUN-STATE.md missing" in r.detail,
        )
        # the runner reports every gate even after a failure, and counts failures
        try:
            rep, nfail = run_gates(
                ctx("H-1\tW0.1\tbuilding\t\t\n"),
                [
                    ("first", lambda c: Result("FAIL", "planted-first\nsecond-line")),
                    ("crash", lambda c: (_ for _ in ()).throw(KeyError("planted-crash"))),
                    ("exits", lambda c: sys.exit(0)),
                    ("returns none", lambda c: None),  # type: ignore[arg-type,return-value]
                    ("bad status", lambda c: Result("fail", "lower-case status")),
                    ("unprintable", lambda c: (_ for _ in ()).throw(ValueError(10**5000))),
                    ("repr exits", lambda c: (_ for _ in ()).throw(_ReprExits())),
                    ("repr str exits", lambda c: (_ for _ in ()).throw(_ReprIsExitingStr())),
                    ("status equals all", lambda c: Result(_StatusEqualsAll("BOGUS"), "x")),
                    ("status lies", lambda c: Result(_StatusLies("FAIL"), "real failure")),
                    ("status exits", lambda c: Result(_ExitingStr("PASS"), "x")),
                    ("result subclass", lambda c: _ResultSub("PASS", "x")),
                    ("detail subclass", lambda c: Result("PASS", _StatusLies("x"))),
                    ("last", lambda c: Result("PASS", "reached-last")),
                ],
                "h",
            )
        except (Exception, SystemExit) as exc:  # the runner must survive any gate, even an exit
            rep, nfail = f"runner crashed: {exc!r}", -1
        check(
            "runner reports every gate after failures and counts crashes as FAIL",
            nfail == 13
            and "planted-first" in rep
            and "second-line" in rep
            and "gate crashed" in rep
            and "planted-crash" in rep
            and "reached-last" in rep
            and "SystemExit" in rep
            and "RESULT: FAIL (13 failing gates)" in rep
            and "unprintable" in rep,
        )
        logs = root / "logs"
        paths = {write_log(logs, f"r{i}") for i in range(3)}
        check("write_log never reuses a name", len(paths) == 3)
        import threading

        def burst(n: int, where: Path) -> bool:
            """n writers released together by a barrier, so a check-then-write would collide."""
            gate = threading.Barrier(n)
            out: list[Path] = []

            def one(i: int) -> None:
                gate.wait()
                out.append(write_log(where, f"c{i}"))

            threads = [threading.Thread(target=one, args=(i,)) for i in range(n)]
            for th in threads:
                th.start()
            for th in threads:
                th.join()
            texts = sorted(p.read_text(encoding="utf-8") for p in where.glob("*.txt"))
            return len(set(out)) == n and texts == sorted(f"c{i}\n" for i in range(n))

        check(
            "write_log keeps every log when 32 writers start at once (3 bursts)",
            all(burst(32, root / f"clogs{k}") for k in range(3)),
        )
        lroot = root / "harness-logs"
        quiet = [("ok", lambda c: Result("PASS", "fine"))]
        import contextlib
        import io

        with contextlib.redirect_stdout(io.StringIO()):
            for _ in range(3):  # same second, through the real converge() logging path
                converge(ctx("H-1\tW0.1\tbuilding\t\t\n"), quiet, lroot)
        check(
            "converge logs every run, even several in one second",
            len(list((lroot / "W0.1").glob("converge-*.txt"))) == 3,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            converge(ctx("H-1\tW0.1\tbuilding\t\t\n"), quiet, None)
        check("converge with no log root writes nothing", len(list(lroot.rglob("*.txt"))) == 3)
        strict = io.TextIOWrapper(io.BytesIO(), encoding="utf-8", errors="strict")
        surr_root = root / "surr-logs"
        try:
            with contextlib.redirect_stdout(strict):  # a strict UTF-8 console, like macOS
                code_sur = converge(
                    ctx("H-1\tW0.1\tbuilding\t\t\n"),
                    [("odd", lambda c: Result("FAIL", "bad name \udcff here"))],
                    surr_root,
                )
        except Exception:
            code_sur = -1
        surr_logs = list((surr_root / "W0.1").glob("converge-*.txt"))
        strict.flush()
        console = strict.buffer.getvalue().decode("utf-8")  # type: ignore[attr-defined]
        check(
            "converge on a strict console returns 1, logs and prints undecodable text escaped",
            code_sur == 1
            and len(surr_logs) == 1
            and "\\udcff" in surr_logs[0].read_text(encoding="utf-8")
            and "\\udcff" in console,
        )

        class _BrokenStdout(io.StringIO):
            def write(self, s: str) -> int:
                raise BrokenPipeError("console went away")

        broken_root = root / "broken-logs"
        try:
            with contextlib.redirect_stdout(_BrokenStdout()):
                converge(ctx("H-1\tW0.1\tbuilding\t\t\n"), quiet, broken_root)
        except BrokenPipeError:
            pass
        check(
            "converge writes the log before printing (a dead console cannot lose it)",
            len(list((broken_root / "W0.1").glob("converge-*.txt"))) == 1,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            code_fail = converge(
                ctx("H-1\tW0.1\tbuilding\t\t\n"),
                [("bad", lambda c: Result("FAIL", "x")), ("ok", lambda c: Result("PASS", "y"))],
                None,
            )
            code_ok = converge(ctx("H-1\tW0.1\tbuilding\t\t\n"), quiet, None)
        check(
            "converge exits 1 when a gate fails and 0 when all pass", (code_fail, code_ok) == (1, 0)
        )
    print("self-test", "PASSED" if failures == 0 else f"FAILED ({failures})")
    return 0 if failures == 0 else 1


class _ReprExits(Exception):
    """Self-test fixture: an exception whose repr raises SystemExit."""

    def __repr__(self) -> str:
        raise SystemExit(3)


class _ExitingStr(str):
    """Self-test fixture: a str whose length and formatting raise SystemExit."""

    def __len__(self) -> int:
        raise SystemExit(4)

    def __format__(self, spec: str) -> str:
        raise SystemExit(5)


@dataclass
class _ResultSub(Result):
    """Self-test fixture: a Result subclass, which the runner must not trust."""


class _StatusEqualsAll(str):
    def __eq__(self, other: object) -> bool:
        return True

    __hash__ = str.__hash__


class _StatusLies(str):
    def __str__(self) -> str:
        return "PASS"


class _ReprIsExitingStr(Exception):
    def __repr__(self) -> str:
        return _ExitingStr("sneaky")


def _describe(obj: object) -> str:
    """repr() for a report line, falling back to the type name when repr itself fails
    (for example an exception carrying an integer past Python's digit limit)."""
    try:
        raw = repr(obj)
        if not isinstance(raw, str):
            raise TypeError("repr did not return a str")
        text = str.__str__(raw)  # an exact str: a subclass's __len__/__format__ never runs
        return text if len(text) <= 300 else text[:297] + "..."
    except BaseException:  # even a repr that raises SystemExit must not stop the runner
        try:
            return f"<{str.__str__(type(obj).__name__)}, unprintable>"
        except BaseException:
            return "<unprintable object>"


def run_gates(ctx: Ctx, gates: list[tuple[str, Callable[[Ctx], Result]]], header: str):
    """Run every gate and report each one, even after a failure. Returns (report, n_failed)."""
    lines = [header]
    failed = 0
    for i, (name, fn) in enumerate(gates, start=1):
        try:
            r = fn(ctx)
            # exact types only: a str subclass could lie in __eq__, __str__ or __format__
            if (
                type(r) is not Result
                or type(r.status) is not str
                or type(r.detail) is not str
                or r.status not in ("PASS", "FAIL", "SKIP")
            ):
                raise TypeError(f"gate returned {_describe(r)}, not a valid Result")
            status = r.status
            detail_lines = str.splitlines(r.detail) or [""]
        except (Exception, SystemExit) as exc:  # a crashing or exiting gate is a failing gate
            status, detail_lines = "FAIL", [f"gate crashed: {_describe(exc)}"]
        failed += status == "FAIL"
        first, *rest = detail_lines
        lines.append(f"{i:>2} {name:<14} {status:<4}  {first}")
        lines.extend(f"{'':>22}{ln}" for ln in rest)
    lines.append(
        f"RESULT: {'FAIL' if failed else 'PASS'} "
        f"({failed} failing gate{'s' if failed != 1 else ''})"
    )
    return "\n".join(lines), failed


def write_log(log_dir: Path, report: str) -> Path:
    """Write the converge report with an exclusive create, so concurrent or same-second runs
    never overwrite each other: a taken name gets a -2, -3, … suffix."""
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    n = 1
    while True:
        path = log_dir / (f"converge-{stamp}.txt" if n == 1 else f"converge-{stamp}-{n}.txt")
        try:
            with path.open("x", encoding="utf-8", errors="backslashreplace") as fh:
                fh.write(report + "\n")
            return path
        except FileExistsError:
            n += 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("node", nargs="?")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--no-log", action="store_true", help="do not write a log under .harness/")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    if not args.node:
        ap.error("node is required")
    try:
        ctx = Ctx(args.node, ROOT, load_graph(), load_catalog(), SPECS_DIR)
    except (HarnessError, OSError) as exc:
        print(f"node-converge: cannot read inputs: {exc}")
        return 1
    if args.node not in ctx.nodes:
        print(f"node-converge: unknown node {args.node}")
        return 1
    return converge(ctx, GATES, log_root=None if args.no_log else ROOT / ".harness")


def converge(
    ctx: Ctx, gates: list[tuple[str, Callable[[Ctx], Result]]], log_root: Path | None
) -> int:
    """Run the gates, print the report, and log it under log_root/<node>/ (None: no log)."""
    header = f"node-converge {ctx.node} — {dt.datetime.now().isoformat(timespec='seconds')}"
    report, failed = run_gates(ctx, gates, header)
    if log_root is not None:  # log first: printing must never be able to lose the log
        write_log(log_root / ctx.node, report)
    _print_safely(report)
    return 1 if failed else 0


def _print_safely(text: str) -> None:
    """Print without ever raising on text the console cannot encode (lone surrogates, …)."""
    stream = sys.stdout
    encoding = getattr(stream, "encoding", None) or "utf-8"
    try:
        safe = text.encode(encoding, errors="backslashreplace").decode(encoding)
    except LookupError:
        safe = text.encode("ascii", errors="backslashreplace").decode("ascii")
    print(safe, file=stream)


if __name__ == "__main__":
    sys.exit(main())
