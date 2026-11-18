"""`ggh` commands for the release (R4.1, R4.2, R4.3, R4.5): `card`, `readme-table`, `release`
(dry run unless `--push`), `space`, `space-samples` and `passkey set`.

Every document is generated from the results bundle (D-20); `--bundle` and the other path flags
let v2 point at its own files. `release` never pushes without `--push`, a token in the
environment (read by name, never printed) and a plan with no problems.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path

EXIT_ERROR = 2


def _fail(cmd: str, exc: BaseException | str) -> int:
    print(f"{cmd}: FAIL: {exc}")
    return EXIT_ERROR


def _paths(args: argparse.Namespace):  # -> sources.Paths
    from ggh.release import sources

    d = sources.Paths()
    return sources.Paths(
        bundle=Path(args.bundle) if args.bundle else d.bundle,
        data_manifest=Path(args.data_manifest) if args.data_manifest else d.data_manifest,
        targets_report=Path(args.targets_report) if args.targets_report else d.targets_report,
        targets_lineage=Path(args.targets_lineage) if args.targets_lineage else d.targets_lineage,
        recipe=Path(args.recipe) if args.recipe else d.recipe,
        base_model=args.base_model or d.base_model,
        ft_model=args.ft_model or d.ft_model,
    )


def _sources(args: argparse.Namespace):
    from ggh.release import sources

    return sources.load(Path(args.root), _paths(args))


def _cmd_card(args: argparse.Namespace) -> int:
    from ggh.release import card, sources

    try:
        text = card.render(_sources(args))
        problems = card.metadata_problems(card.split_front_matter(text)[0])
    except (sources.SourceError, ValueError, OSError) as exc:
        return _fail("card", exc)
    if problems:
        return _fail("card", "; ".join(problems))
    if args.stdout:
        sys.stdout.write(text)
        return 0
    out = Path(args.root) / args.out
    out.write_text(text, encoding="utf-8")
    print(f"card: wrote {args.out}")
    return 0


def _cmd_readme_table(args: argparse.Namespace) -> int:
    from ggh.release import readme, sources

    try:
        src = _sources(args)
        block = readme.results_block(src)
        if args.write:
            target = Path(args.root) / args.write
            text = readme.update(target.read_text(encoding="utf-8"), src)
            target.write_text(text, encoding="utf-8")
            print(f"readme-table: updated {args.write}")
            return 0
    except (sources.SourceError, readme.SpliceError, ValueError, OSError) as exc:
        return _fail("readme-table", exc)
    sys.stdout.write(block)
    return 0


def _cmd_release(args: argparse.Namespace) -> int:
    from ggh.release import card, hub, sources

    root = Path(args.root)
    try:
        src = _sources(args)
        p = hub.plan(
            root,
            src.paths.bundle,
            card.render(src),
            owner=args.owner,
            export_manifest=Path(args.export_manifest),
            train_manifest=Path(args.train_manifest),
        )
    except (sources.SourceError, ValueError, OSError) as exc:
        return _fail("release", exc)
    print("\n".join(hub.describe(p)))
    if not args.push:
        print("release: dry run, nothing pushed (add --push, with the owner's go-ahead)")
        return 0 if not p.problems else EXIT_ERROR
    token = hub.token_from_env(os.environ)
    if token is None:
        return _fail("release", f"--push needs {hub.TOKEN_ENV} in the environment")
    if not p.ready:
        return _fail("release", "the plan has problems or no --owner; nothing pushed")
    try:  # pragma: no cover - network; only with the owner's explicit go-ahead
        problems = hub.push(
            p, root, hub.HfHubApi(token), tag=args.tag, staging=root / "artifacts" / "release"
        )
    except (hub.ReleaseError, ImportError, OSError) as exc:  # pragma: no cover
        return _fail("release", exc)
    for x in problems:  # pragma: no cover
        print(f"  - {x}")
    return EXIT_ERROR if problems else 0  # pragma: no cover


def _cmd_space(args: argparse.Namespace) -> int:  # pragma: no cover - starts a server
    from ggh.serve import llama
    from ggh.space import app

    root = Path(args.root)
    path = root / args.samples
    if not args.local:
        return _fail("space", "only --local runs here; the Space itself runs space/app.py")
    try:
        gens = app.local_generators(
            f"http://{llama.HOST}:{llama.PORTS['base']}", f"http://{llama.HOST}:{llama.PORTS['ft']}"
        )
        space = app.make_space(os.environ, gens, path)
        print(f"space: http://127.0.0.1:{args.port}/ against ggh serve (Ctrl-C to stop)")
        app.launch(space, path, port=args.port)
    except ImportError as exc:
        return _fail(
            "space", f"{exc}; the page needs gradio: uv run --with gradio ggh space --local"
        )
    except (ValueError, OSError) as exc:
        return _fail("space", exc)
    return 0


def _cmd_space_samples(args: argparse.Namespace) -> int:
    from ggh.space import samples

    root = Path(args.root)
    try:
        picked = samples.build(
            root / args.run,
            base_model=args.base_model or samples.BASE_MODEL,
            ft_model=args.ft_model or samples.FT_MODEL,
        )
    except (ValueError, OSError, KeyError) as exc:
        return _fail("space-samples", exc)
    (root / args.out).parent.mkdir(parents=True, exist_ok=True)
    (root / args.out).write_text(samples.dump(picked), encoding="utf-8")
    print(f"space-samples: wrote {len(picked)} samples to {args.out}")
    return 0


def _cmd_passkey_set(_: argparse.Namespace) -> int:
    from ggh.space import passkey

    code = getpass.getpass("New numeric passcode: ")
    again = getpass.getpass("Again: ")
    if code != again:
        return _fail("passkey set", "the two entries differ")
    try:
        value = passkey.make_hash(code)
    except passkey.PasskeyError as exc:
        return _fail("passkey set", exc)
    del code, again
    print(f"Set the Space secret {passkey.SECRET_ENV} to this salted hash (not the passcode):")
    print(value)
    return 0


def _source_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--root", default=".", help="repo root")
    p.add_argument("--bundle", default=None, help="results_bundle.json (default: docs/results/)")
    p.add_argument("--data-manifest", default=None)
    p.add_argument("--targets-report", default=None)
    p.add_argument("--targets-lineage", default=None)
    p.add_argument("--recipe", default=None)
    p.add_argument("--base-model", default=None, help="EvalKit name of the base model")
    p.add_argument("--ft-model", default=None, help="EvalKit name of the released fine-tune")


def register(sub: argparse._SubParsersAction) -> None:  # pyright: ignore[reportPrivateUsage]
    from ggh.release.hub import EXPORT_MANIFEST, OWNER_PLACEHOLDER, TRAIN_MANIFEST
    from ggh.space.samples import RUN, SAMPLES

    cd = sub.add_parser("card", help="generate MODEL_CARD.md from the results bundle (R4.1)")
    _source_flags(cd)
    cd.add_argument("--out", default="MODEL_CARD.md")
    cd.add_argument("--stdout", action="store_true", help="print instead of writing")
    cd.set_defaults(func=_cmd_card)
    rt = sub.add_parser("readme-table", help="the README results block from the bundle (R4.5)")
    _source_flags(rt)
    rt.add_argument("--write", default=None, help="splice the block into this file's markers")
    rt.set_defaults(func=_cmd_readme_table)
    rl = sub.add_parser("release", help="Hub release plan; dry run unless --push (R4.2)")
    _source_flags(rl)
    rl.add_argument("--dry-run", action="store_true", help="the default; listed for clarity")
    rl.add_argument("--push", action="store_true", help="push (needs HF_TOKEN and a clean plan)")
    rl.add_argument("--owner", default=OWNER_PLACEHOLDER, help="Hub user or organisation")
    rl.add_argument("--tag", default="v1.0.0")
    rl.add_argument("--export-manifest", default=str(EXPORT_MANIFEST))
    rl.add_argument("--train-manifest", default=str(TRAIN_MANIFEST))
    rl.set_defaults(func=_cmd_release)
    sp = sub.add_parser("space", help="run the demo locally against ggh serve (R4.3, E7)")
    sp.add_argument("--local", action="store_true", help="use ggh serve's llama-servers")
    sp.add_argument("--root", default=".")
    sp.add_argument("--samples", default=str(SAMPLES))
    sp.add_argument("--port", type=int, default=7860)
    sp.set_defaults(func=_cmd_space)
    ss = sub.add_parser("space-samples", help="pick the six demo samples from a dev run (R4.3)")
    ss.add_argument("--root", default=".")
    ss.add_argument("--run", default=str(RUN), help="EvalKit run folder over dev_eval")
    ss.add_argument("--out", default=str(SAMPLES))
    ss.add_argument("--base-model", default=None)
    ss.add_argument("--ft-model", default=None)
    ss.set_defaults(func=_cmd_space_samples)
    pk = sub.add_parser("passkey", help="the demo passcode (OWNER D-39)")
    pks = pk.add_subparsers(dest="passkey_command", required=True)
    pset = pks.add_parser("set", help="hash a new passcode for the Space secret")
    pset.set_defaults(func=_cmd_passkey_set)
