"""Command-line interface: `ggh`.

stdlib argparse keeps the install light. Later nodes add `data`, `serve`, `bench`, `eval`,
`export` and `release` (Tech Spec §2.2). Exit codes: 0 success, 2 configuration or runtime error.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys

from ggh import __version__

MIN_PYTHON = (3, 11)
# Credentials later nodes use. doctor reports which are set by NAME only, never a value.
CREDENTIALS = (
    ("ANTHROPIC_API_KEY", "evaluation (EvalKit)"),
    ("OPENAI_API_KEY", "evaluation (EvalKit)"),
    ("HF_TOKEN", "Hugging Face download and release"),
    ("KAGGLE_USERNAME", "Kaggle training runs"),
    ("KAGGLE_KEY", "Kaggle training runs"),
)
EXIT_ERROR = 2


def doctor_report() -> tuple[bool, list[str]]:
    """Checks that need nothing installed beyond ggh. Later nodes append their own."""
    lines: list[str] = []
    ok = sys.version_info[:2] >= MIN_PYTHON
    py = ".".join(map(str, sys.version_info[:3]))
    floor = ".".join(map(str, MIN_PYTHON))
    lines.append(f"[{'ok' if ok else 'FAIL'}] python {py} (needs >= {floor})")
    lines.append(f"[ok] platform {platform.system()} {platform.machine()}")
    for name, use in CREDENTIALS:
        state = "set" if os.environ.get(name) else "not set"
        lines.append(f"[info] {name} {state} ({use}; not needed for tests)")
    return ok, lines


def _cmd_doctor(_: argparse.Namespace) -> int:
    ok, lines = doctor_report()
    print("\n".join(lines))
    print("ready" if ok else "not ready")
    return 0 if ok else EXIT_ERROR


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="ggh",
        description="Gemma Grounded Health: build, evaluate and release a grounded Gemma model.",
    )
    ap.add_argument("--version", action="version", version=f"ggh {__version__}")
    sub = ap.add_subparsers(dest="command")
    doctor = sub.add_parser("doctor", help="check this machine is ready")
    doctor.set_defaults(func=_cmd_doctor)
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:
        ap.print_help()
        return 0
    return func(args)


if __name__ == "__main__":
    sys.exit(main())
