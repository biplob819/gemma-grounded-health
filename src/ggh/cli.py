"""Command-line interface: `ggh`.

stdlib argparse keeps the install light. `ggh lineage verify` walks a manifest
back through its parents (Tech Spec §2.1). Later nodes add `data`, `serve`, `bench`, `eval`,
`export` and `release` (Tech Spec §2.2). Exit codes: 0 success, 2 configuration or runtime error.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
from pathlib import Path

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


def _cmd_lineage_verify(args: argparse.Namespace) -> int:
    from ggh.lineage import verify

    problems = verify(Path(args.manifest), Path(args.root))
    if problems:
        print("lineage: FAIL")
        for p in problems:
            print(f"  - {p}")
        return EXIT_ERROR
    print("lineage: OK")
    return 0


def _cmd_data(args: argparse.Namespace) -> int:
    from ggh.data import build, fetch

    try:
        m = build.build(
            Path(args.root),
            seed=args.seed,
            download=not args.offline,
            refreeze_offscope=args.refreeze_offscope,
        )
    except (fetch.DataError, ValueError, OSError) as exc:
        print(f"data: FAIL: {exc}")
        return EXIT_ERROR
    for split, counts in m.counts.items():
        print(f"{split:<14} {sum(counts.values()):>7}  {dict(counts)}")
    print(f"dropped for PQA-L overlap: {dict(m.drops) or 'none'}")
    print(f"manifest: {build.MANIFEST_PATH}")
    return 0


def _cmd_targets(args: argparse.Namespace) -> int:
    from ggh.lineage import LineageError
    from ggh.targets import make

    def scorer():  # the real model; needs `uv sync --group nli`
        from ggh.targets.nli import HFScorer

        return HFScorer()

    try:
        report = make.build(
            Path(args.root),
            scorer,
            seed=args.seed,
            progress=lambda m: print(m, file=sys.stderr, flush=True),
        )
    except ImportError as exc:
        print(f"targets: FAIL: {exc}; install the NLI model's dependencies: uv sync --group nli")
        return EXIT_ERROR
    except (ValueError, OSError, LineageError) as exc:
        print(f"targets: FAIL: {exc}")
        return EXIT_ERROR
    print(f"train {report['train']}  val {report['val']}  (cap {report['cap_train']})")
    print(f"quotas: {report['quotas']}")
    print(f"report: {make.REPORT_COPY}")
    return 0


def _cmd_audit_serve(args: argparse.Namespace) -> int:
    from ggh.targets import audit

    try:
        srv = audit.server(Path(args.root), args.port)
    except (ValueError, OSError) as exc:
        print(f"audit: FAIL: {exc}")
        return EXIT_ERROR
    host, port = srv.server_address[:2]
    print(f"audit page: http://{host}:{port}/  (Ctrl-C to stop; labels save as you go)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


def _cmd_audit_score(args: argparse.Namespace) -> int:
    import json

    from ggh.targets import audit

    try:
        result = audit.score(Path(args.root))
    except (ValueError, OSError) as exc:
        print(f"audit: FAIL: {exc}")
        return EXIT_ERROR
    print(json.dumps(result, indent=2))
    print("audit: PASS" if result["passed"] else "audit: NOT PASSED")
    return 0 if result["passed"] else EXIT_ERROR


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="ggh",
        description="Gemma Grounded Health: build, evaluate and release a grounded Gemma model.",
    )
    ap.add_argument("--version", action="version", version=f"ggh {__version__}")
    sub = ap.add_subparsers(dest="command")
    doctor = sub.add_parser("doctor", help="check this machine is ready")
    doctor.set_defaults(func=_cmd_doctor)
    data = sub.add_parser("data", help="build the training pool and dev splits (N1.2)")
    data.add_argument("--root", default=".", help="repo root")
    data.add_argument("--seed", type=int, default=7)
    data.add_argument("--offline", action="store_true", help="use the cache only; never download")
    data.add_argument(
        "--refreeze-offscope",
        action="store_true",
        help="accept a changed off-scope set (only after the owner approves it)",
    )
    data.set_defaults(func=_cmd_data)
    targets = sub.add_parser("targets", help="build training targets from the data splits (N1.3)")
    targets.add_argument("--root", default=".", help="repo root")
    targets.add_argument("--seed", type=int, default=7)
    targets.set_defaults(func=_cmd_targets)
    aud = sub.add_parser("audit", help="hand audit of 50 training examples (N1.3)")
    asub = aud.add_subparsers(dest="audit_command", required=True)
    aserve = asub.add_parser("serve", help="open the local labelling page")
    aserve.add_argument("--root", default=".", help="repo root")
    aserve.add_argument("--port", type=int, default=8765)
    aserve.set_defaults(func=_cmd_audit_serve)
    ascore = asub.add_parser("score", help="apply the pass bar to the saved labels")
    ascore.add_argument("--root", default=".", help="repo root")
    ascore.set_defaults(func=_cmd_audit_score)
    lineage = sub.add_parser("lineage", help="check artifact lineage manifests")
    lsub = lineage.add_subparsers(dest="lineage_command", required=True)
    lverify = lsub.add_parser("verify", help="walk a manifest back through its parents")
    lverify.add_argument("manifest", help="path to a stage's manifest.json")
    lverify.add_argument("--root", default=".", help="repo root that holds every manifest")
    lverify.set_defaults(func=_cmd_lineage_verify)
    from ggh.cli_serve import register  # E3.1: llama, export, serve, checks, tasks, behaviour

    register(sub)
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
