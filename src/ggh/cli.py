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


def _cmd_doctor(args: argparse.Namespace) -> int:
    ok, lines = doctor_report()
    if getattr(args, "kaggle", False):
        from ggh.train.kaggle import kaggle_report

        k_ok, k_lines = kaggle_report(Path(args.work))
        ok, lines = ok and k_ok, lines + k_lines
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

    root = Path(args.root)

    def scorer():  # the real model; needs `uv sync --group nli`
        if args.replay:  # the committed scores: same bytes on any machine, no model (D-10)
            return make.replay_scorer(root)
        from ggh.targets.nli import HFScorer

        return HFScorer()

    try:
        expected = make.expected_files(root) if args.replay else {}
        report = make.build(
            root,
            scorer,
            seed=args.seed,
            progress=lambda m: print(m, file=sys.stderr, flush=True),
        )
        problems = make.replay_problems(root, expected) if args.replay else []
    except ImportError as exc:
        print(f"targets: FAIL: {exc}; install the NLI model's dependencies: uv sync --group nli")
        return EXIT_ERROR
    except (ValueError, OSError, LineageError) as exc:
        print(f"targets: FAIL: {exc}")
        return EXIT_ERROR
    if problems:
        print("targets: FAIL: the replayed targets differ from the committed ones")
        for p in problems:
            print(f"  {p}")
        return EXIT_ERROR
    print(f"train {report['train']}  val {report['val']}  (cap {report['cap_train']})")
    print(f"quotas: {report['quotas']}")
    print(f"report: {make.REPORT_COPY}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    import json

    from ggh.train.run import RecipeError, run  # light: heavy imports happen inside run()

    try:
        report = run(
            Path(args.config), Path(args.root), Path(args.out),
            max_steps=args.max_steps, tool=args.tool,
        )  # fmt: skip
    except ImportError as exc:
        print(f"train: FAIL: {exc}; needs the training packages (uv sync --group train)")
        return EXIT_ERROR
    except (RecipeError, ValueError, OSError) as exc:
        print(f"train: FAIL: {exc}")
        return EXIT_ERROR
    keys = ("backend", "steps", "tokens_per_s", "peak_vram_gb", "non_finite", "seconds")
    print(json.dumps({k: report[k] for k in keys}))
    bad = report["non_finite"] + report.get("non_finite_steps", 0)
    if bad:
        print("train: FAIL: a loss or gradient norm was not finite")
        return EXIT_ERROR
    return 0


def _cmd_dev_eval(args: argparse.Namespace) -> int:
    import json

    from ggh.train.devrun import dev_eval

    try:
        rec = dev_eval(Path(args.root), Path(args.run), Path(args.out), Path(args.config),
                       limit=args.limit)  # fmt: skip
    except ImportError as exc:
        print(f"dev-eval: FAIL: {exc}; needs the Kaggle training packages")
        return EXIT_ERROR
    except (ValueError, OSError) as exc:
        print(f"dev-eval: FAIL: {exc}")
        return EXIT_ERROR
    for t in rec["tried"]:
        print(json.dumps({k: t[k] for k in ("step", "eval_loss", "passed", "failed")}))
    if rec["selected_step"] is None:
        print("dev-eval: FAIL: no checkpoint passed the §5.2 bars")
        return EXIT_ERROR
    print(f"selected step {rec['selected_step']}, adapter sha256 {rec['adapter_sha256']}")
    return 0


def _cmd_freeze(args: argparse.Namespace) -> int:
    import json

    from ggh.train.devrun import FreezeError, freeze, write_records
    from ggh.versions import version

    root, v = Path(args.root), version(args.model)
    folder = Path(args.out) if args.out else v.measurements  # relative to --root
    # v1 was frozen before the recipe check existed; every later version checks its recipe
    config = args.config or (None if args.model == "ft" else v.recipe)
    try:
        report = json.loads(Path(args.report).read_text(encoding="utf-8"))
        selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
        run = freeze(
            root,
            report,
            selection,
            Path(args.adapter),
            kernel_version=args.kernel_version,
            repo_commit=args.repo_commit,
            out=folder / "train_run.json",
            config=None if config is None else root / config,
        )
        kept = []
        if args.generations:
            log = Path(args.log) if args.log else None
            kept = write_records(root / folder, report, selection, Path(args.generations), log)
    except (FreezeError, ValueError, OSError, KeyError) as exc:
        print(f"freeze: FAIL: {exc}")
        return EXIT_ERROR
    print(f"froze step {run.selected_step} -> {folder / 'train_run.json'}")
    for path in kept:
        print(f"kept {path.relative_to(root)}")
    return 0


def _cmd_audit_serve(args: argparse.Namespace) -> int:
    from ggh.targets import audit, audit_v2

    try:
        if args.v2:  # AUD-R4: per cited sentence, its own files and port
            port = audit_v2.DEFAULT_PORT if args.port is None else args.port
            srv = audit_v2.server(Path(args.root), port)
        else:
            srv = audit.server(Path(args.root), 8765 if args.port is None else args.port)
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

    from ggh.targets import audit, audit_v2

    try:
        result = audit_v2.write_result(Path(args.root)) if args.v2 else audit.score(Path(args.root))
    except (ValueError, OSError) as exc:
        print(f"audit: FAIL: {exc}")
        return EXIT_ERROR
    print(json.dumps(result, indent=2))
    if args.v2:
        print(f"result: {audit_v2.RESULT}")
    print("audit: PASS" if result["passed"] else "audit: NOT PASSED")
    return 0 if result["passed"] else EXIT_ERROR


def _cmd_audit_sample(args: argparse.Namespace) -> int:
    from ggh.targets import audit_v2

    try:
        sample = audit_v2.draw(Path(args.root))
    except (ValueError, OSError) as exc:
        print(f"audit: FAIL: {exc}")
        return EXIT_ERROR
    print(f"{len(sample)} examples drawn into {audit_v2.SAMPLE}")
    return 0


def _cmd_audit_refresh(args: argparse.Namespace) -> int:
    import json

    from ggh.targets import audit_v2

    try:
        out = audit_v2.refresh(Path(args.root))
    except (ValueError, OSError, StopIteration) as exc:
        print(f"audit: FAIL: {exc!r}")
        return EXIT_ERROR
    print(json.dumps(out, indent=2))
    print(f"{len(out['relabel'])} examples need labels ({out['relabel_answers']} answers); "
          f"list in {audit_v2.RELABEL}")  # fmt: skip
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="ggh",
        description="Gemma Grounded Health: build, evaluate and release a grounded Gemma model.",
    )
    ap.add_argument("--version", action="version", version=f"ggh {__version__}")
    sub = ap.add_subparsers(dest="command")
    doctor = sub.add_parser("doctor", help="check this machine is ready")
    doctor.add_argument("--kaggle", action="store_true", help="also check GPU, disk, internet")
    doctor.add_argument("--work", default=".", help="folder whose free disk is checked")
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
    targets.add_argument(
        "--replay",
        action="store_true",
        help="use the committed NLI scores (data/nli_scores.jsonl.gz) and check the hashes",
    )
    targets.set_defaults(func=_cmd_targets)
    tr = sub.add_parser("train", help="train from the recipe (T2.1, T2.2)")
    tr.add_argument("--config", default="configs/recipe.yaml")
    tr.add_argument("--root", default=".", help="repo root holding artifacts/targets")
    tr.add_argument("--out", required=True, help="folder for checkpoints and report.json")
    tr.add_argument("--max-steps", type=int, default=None, help="stop early (the pilot: 300)")
    tr.add_argument("--tool", choices=["unsloth", "trl"], default=None, help="override the recipe")
    tr.set_defaults(func=_cmd_train)
    de = sub.add_parser("dev-eval", help="dev generations, metrics and checkpoint choice (T2.2)")
    de.add_argument("--run", required=True, help="the ggh train output folder")
    de.add_argument("--out", required=True, help="folder for selection.json and the adapter")
    de.add_argument("--config", default="configs/recipe.yaml")
    de.add_argument("--root", default=".", help="repo root holding artifacts/data")
    de.add_argument("--limit", type=int, default=None, help="first N dev prompts (tests)")
    de.set_defaults(func=_cmd_dev_eval)
    from ggh.versions import VERSIONS

    fz = sub.add_parser("freeze", help="check the downloaded adapter and write TrainRun (T2.2)")
    fz.add_argument("--report", required=True)
    fz.add_argument("--selection", required=True)
    fz.add_argument("--adapter", required=True)
    fz.add_argument("--kernel-version", required=True)
    fz.add_argument("--repo-commit", required=True)
    fz.add_argument("--root", default=".")
    fz.add_argument("--model", default="ft", choices=sorted(VERSIONS),
                    help="ft (v1) or ft2 (v2, its records in docs/measurements/v2)")  # fmt: skip
    fz.add_argument("--out", default=None, help="folder for train_run.json (default: the model's)")
    fz.add_argument("--config", default=None, help="recipe whose hash the run must carry "
                    "(default: the model's; v1 skips the check)")  # fmt: skip
    fz.add_argument("--generations", default=None, help="dev/generations.jsonl: also keep the "
                    "selection, report and generations beside train_run.json")  # fmt: skip
    fz.add_argument("--log", default=None, help="the Kaggle session log (with --generations)")
    fz.set_defaults(func=_cmd_freeze)
    aud = sub.add_parser("audit", help="hand audit of 50 training examples (N1.3)")
    asub = aud.add_subparsers(dest="audit_command", required=True)
    aserve = asub.add_parser("serve", help="open the local labelling page")
    aserve.add_argument("--root", default=".", help="repo root")
    aserve.add_argument("--port", type=int, default=None, help="default 8765, or 8766 with --v2")
    aserve.add_argument("--v2", action="store_true", help="the v2 per-sentence audit (AUD-R4)")
    aserve.set_defaults(func=_cmd_audit_serve)
    ascore = asub.add_parser("score", help="apply the pass bar to the saved labels")
    ascore.add_argument("--root", default=".", help="repo root")
    ascore.add_argument("--v2", action="store_true", help="score the v2 audit, write its result")
    ascore.set_defaults(func=_cmd_audit_score)
    asample = asub.add_parser("sample", help="draw the v2 audit sample from the v2 train split")
    asample.add_argument("--root", default=".", help="repo root")
    asample.add_argument("--v2", action="store_true", required=True, help="(v2 only)")
    asample.set_defaults(func=_cmd_audit_sample)
    arefresh = asub.add_parser(
        "refresh", help="after a targets rebuild: same v2 ids, re-read; list who needs labels"
    )
    arefresh.add_argument("--root", default=".", help="repo root")
    arefresh.add_argument("--v2", action="store_true", required=True, help="(v2 only)")
    arefresh.set_defaults(func=_cmd_audit_refresh)
    lineage = sub.add_parser("lineage", help="check artifact lineage manifests")
    lsub = lineage.add_subparsers(dest="lineage_command", required=True)
    lverify = lsub.add_parser("verify", help="walk a manifest back through its parents")
    lverify.add_argument("manifest", help="path to a stage's manifest.json")
    lverify.add_argument("--root", default=".", help="repo root that holds every manifest")
    lverify.set_defaults(func=_cmd_lineage_verify)
    from ggh.cli_serve import register  # E3.1: llama, export, serve, checks, tasks, behaviour

    register(sub)
    from ggh.cli_results import register as register_results  # E3.3: bundle, sample, plans

    register_results(sub)
    from ggh.cli_release import register as register_release  # R4: card, release, space

    register_release(sub)
    from ggh.cli_calibrate import register as register_calibrate  # E3.3: OWNER D-24 check

    register_calibrate(sub)
    from ggh.cli_rl import register as register_rl  # V3.1: v3 GRPO (OWNER D-66)

    register_rl(sub)
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
