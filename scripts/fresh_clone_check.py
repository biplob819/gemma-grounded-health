#!/usr/bin/env python3
"""OSS-05: a clean clone at HEAD installs, passes `ggh doctor` and a fast test subset.

Clones this checkout at its current commit into a temporary folder (committed files only: no
`artifacts/`, no `.env`, no local edits), runs `uv sync --frozen`, `uv run ggh doctor` and the
fast subset below with every credential removed from the environment, and exits non-zero on
the first failure. The full CI (`make check`) runs on GitHub on every push; this is the local
check before tagging.

Usage:
    uv run python scripts/fresh_clone_check.py            # clone HEAD, check, delete
    uv run python scripts/fresh_clone_check.py --keep     # keep the clone and print its path
    uv run python scripts/fresh_clone_check.py --offline  # uv sync from the local cache only
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CREDENTIAL_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "KAGGLE", "HF_", "PASSKEY")
# fast, offline tests that need only committed files (no artifacts/, no GPU, no network)
FAST_TESTS = (
    "tests/test_cli.py",
    "tests/test_lineage.py",
    "tests/schemas",
    "tests/release/test_links.py",
    "tests/release/test_citation.py",
    "tests/release/test_readme.py",
    "tests/release/test_release_notes.py",
    "tests/release/test_card_fixes.py",
    "tests/test_reproduce_doc.py",
)


def clean_env(env: Mapping[str, str]) -> dict[str, str]:
    """The environment without credentials or the parent's virtualenv."""
    out = {k: v for k, v in env.items() if not any(h in k.upper() for h in CREDENTIAL_HINTS)}
    out.pop("VIRTUAL_ENV", None)
    return out


def steps(sha: str, clone: Path, offline: bool = False) -> list[tuple[str, list[str], Path]]:
    """(label, command, working folder) in order."""
    sync = ["uv", "sync", "--frozen", *(["--offline"] if offline else [])]
    return [
        ("clone", ["git", "clone", "--quiet", "--no-hardlinks", str(ROOT), str(clone)], ROOT),
        ("checkout", ["git", "checkout", "--quiet", "--detach", sha], clone),
        ("uv sync", sync, clone),
        ("ggh doctor", ["uv", "run", "ggh", "doctor"], clone),
        ("fast tests", ["uv", "run", "pytest", "-q", "-p", "no:cacheprovider", *FAST_TESTS], clone),
    ]


def head_sha(root: Path = ROOT) -> str:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    ap.add_argument("--keep", action="store_true", help="keep the temporary clone")
    ap.add_argument("--offline", action="store_true", help="uv sync from the cache only")
    args = ap.parse_args(argv)
    sha = head_sha()
    tmp = Path(tempfile.mkdtemp(prefix="ggh-fresh-"))
    clone = tmp / "gemma-grounded-health"
    env = clean_env(os.environ)
    print(f"fresh clone of {sha[:12]} into a temporary folder")
    try:
        for label, cmd, cwd in steps(sha, clone, args.offline):
            r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, check=False)
            tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
            print(f"[{'ok' if r.returncode == 0 else 'FAIL'}] {label}")
            for line in tail:
                print(f"    {line}")
            if r.returncode != 0:
                return 1
        print(f"fresh clone check passed at {sha[:12]}")
        return 0
    finally:
        if args.keep:
            print(f"kept: {clone}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
