"""OSS-01: docs/reproduce.md is a working recipe. Every command block is marked `run` or
`dry-run: <reason>`. A `run` block's `ggh` commands are executed here on the committed files
(no network, no keys, no spend) and must leave every file git tracks as it was, so the committed
outputs are what the commands write. A `dry-run` block needs a GPU, Kaggle, the network, paid
API calls or the gitignored artifacts: its `ggh` commands must parse with the real CLI, its
`make` targets must exist and every repo path it names must be committed."""

import fcntl
import hashlib
import re
import shlex
import subprocess
from pathlib import Path

import pytest

from ggh.cli import build_parser, main

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "reproduce.md"
MARKED = re.compile(r"<!-- reproduce: (run|dry-run: [^>]+?) -->\n```bash\n(.*?)```", re.S)
ANY_BLOCK = re.compile(r"^```bash$", re.M)
# tools a dry-run block may name besides ggh and make; each needs what its block's reason says
OTHER_TOOLS = {"git", "cd", "kaggle", "mv", "ollama", "hf"}
REPO_PATH = re.compile(r"^(configs|docs|notebooks|scripts|data|src)/")
STEPS = ("Data", "targets", "Kaggle", "Export", "Evaluation", "Laptop", "release")


def blocks() -> list[tuple[str, list[list[str]]]]:
    text = DOC.read_text(encoding="utf-8")
    out = []
    for mark, body in MARKED.findall(text):
        cmds = [shlex.split(ln, comments=True) for ln in body.splitlines()]
        out.append((mark, [c for c in cmds if c]))
    return out


def ggh_args(cmd: list[str]) -> list[str] | None:
    """`uv run [--group G] ggh ARGS` -> ARGS; anything else -> None."""
    if cmd[:2] != ["uv", "run"]:
        return None
    rest = cmd[2:]
    while rest and rest[0] == "--group":
        rest = rest[2:]
    return rest[1:] if rest[:1] == ["ggh"] else None


def _snapshot() -> dict[Path, bytes | None]:
    """Bytes of every tracked file under the paths the run steps write, and the untracked set."""
    tracked = subprocess.run(
        ["git", "ls-files", "README.md", "MODEL_CARD.md", "docs"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.split()  # fmt: skip
    snap: dict[Path, bytes | None] = {ROOT / p: (ROOT / p).read_bytes() for p in tracked}
    for p in ROOT.joinpath("docs").rglob("*"):
        if p.is_file() and p not in snap:
            snap[p] = p.read_bytes()
    return snap


def _restore(snap: dict[Path, bytes | None]) -> None:
    for p, data in snap.items():
        if data is not None and (not p.is_file() or p.read_bytes() != data):
            p.write_bytes(data)
    for p in ROOT.joinpath("docs").rglob("*"):
        if p.is_file() and p not in snap:
            p.unlink()  # written by a run step and not there before


def _state() -> dict[str, str]:
    """Every changed or untracked path with its content hash: a write shows up even on a
    checkout that already had local changes."""
    out = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    state = {}
    for line in out.splitlines():
        p = ROOT / line[3:].strip('"')
        state[line] = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else ""
    return state


def test_every_command_block_is_marked_and_the_steps_are_there():
    text = DOC.read_text(encoding="utf-8")
    assert len(ANY_BLOCK.findall(text)) == len(MARKED.findall(text))
    marks = [m for m, _ in blocks()]
    assert "run" in marks and any(m.startswith("dry-run: ") and len(m) > 20 for m in marks)
    for step in STEPS:
        assert step in text, step
    assert "Research only, not medical advice" in text


def test_dry_run_blocks_parse_and_name_committed_paths():
    parser = build_parser()
    targets = set(re.findall(r"^([a-z]+):", (ROOT / "Makefile").read_text(encoding="utf-8"), re.M))
    for mark, cmds in blocks():
        if mark == "run":
            continue
        for cmd in cmds:
            args = ggh_args(cmd)
            if args is not None:
                parser.parse_args(args)  # SystemExit on a command the CLI does not have
            elif cmd[0] == "make":
                assert set(cmd[1:]) <= targets, cmd
            elif cmd[:2] == ["uv", "run"]:
                assert cmd[2] in {"evalkit", "python"}, cmd
            elif cmd[:2] != ["uv", "sync"]:
                assert cmd[0] in OTHER_TOOLS, cmd
            for tok in cmd:
                if REPO_PATH.match(tok):
                    assert (ROOT / tok).exists(), tok


def test_run_blocks_run_offline_and_change_nothing(monkeypatch: pytest.MonkeyPatch, capsys):
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "HF_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.chdir(ROOT)
    before = _state()
    # the run steps regenerate committed outputs in place: snapshot them and put them back, so
    # the test detects a stale output without overwriting a developer's edits or hiding the
    # staleness from the other generator tests
    lock = open(ROOT / ".git" / "ggh-reproduce-test.lock", "w", encoding="utf-8")  # noqa: SIM115
    fcntl.flock(lock, fcntl.LOCK_EX)  # one run at a time per checkout (a second run waits)
    snap = _snapshot()
    ran = []
    try:
        for mark, cmds in blocks():
            if mark != "run":
                continue
            for cmd in cmds:
                args = ggh_args(cmd)
                assert args is not None, f"a run block runs only ggh commands: {cmd}"
                try:
                    code = main(args)
                except SystemExit as exc:  # --version exits through argparse
                    code = exc.code
                assert code in (0, None), (cmd, capsys.readouterr().out[-2000:])
                ran.append(args[0])
        after = _state()
    finally:
        _restore(snap)
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    assert {"doctor", "card", "readme-table", "release-notes", "results"} <= set(ran)
    assert after == before, "a run step changed a committed file: commit its output"
