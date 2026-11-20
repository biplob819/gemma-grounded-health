"""`ggh` commands for the release (R4.1, R4.2, R4.3, R4.5): `card`, `readme-table`,
`release-notes`, `release` (dry run unless `--push`), `space`, `space-samples` and `passkey set`.

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
        src = _sources(args)
        text = card.render(src)
        problems = card.metadata_problems(card.split_front_matter(text)[0])
    except (sources.SourceError, ValueError, OSError) as exc:
        return _fail("card", exc)
    if problems:
        return _fail("card", "; ".join(problems))
    terms = (src.licence or {}).get("terms_page") or {}
    if not terms.get("reread_at"):  # CRD-05: the owner re-reads the page before publishing
        print(
            f"card: note: the Gemma terms page {terms.get('url', '')} is not re-read yet "
            f"({sources.LICENCE_RECORD}: terms_page.reread_at)",
            file=sys.stderr,
        )
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


def _cmd_release_notes(args: argparse.Namespace) -> int:
    from ggh.release import notes, sources

    root = Path(args.root)
    try:
        text = notes.render(_sources(args), root)
    except (sources.SourceError, ValueError, OSError, KeyError) as exc:
        return _fail("release-notes", exc)
    if args.stdout:
        sys.stdout.write(text)
        return 0
    out = root / args.out
    if args.check:
        if not out.is_file() or out.read_text(encoding="utf-8") != text:
            return _fail("release-notes", f"{args.out} is not the generator's output")
        print(f"release-notes: {args.out} is current")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"release-notes: wrote {args.out}")
    return 0


def _cmd_release(args: argparse.Namespace) -> int:
    import json

    from ggh.release import card, hub, sources

    root = Path(args.root)
    if args.ollama_check:
        return _ollama(root, args.owner, args.record_json)
    try:
        src = _sources(args)
        layout = args.release or src.bundle.shipped.version
        shipped = layout == src.bundle.shipped.version
        card_path = args.card or ("MODEL_CARD.md" if shipped else None)
        p = hub.plan(
            root,
            src.paths.bundle,
            card.render(src),
            owner=args.owner,
            layout=layout,
            export_manifest=Path(args.export_manifest) if args.export_manifest else None,
            train_manifest=Path(args.train_manifest) if args.train_manifest else None,
            committed_card=Path(card_path) if card_path else None,
        )
    except (sources.SourceError, ValueError, OSError) as exc:
        return _fail("release", exc)
    print("\n".join(hub.describe(p)))
    if args.plan_json:
        out = root / args.plan_json
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(hub.plan_json(p), indent=2) + "\n", encoding="utf-8")
        print(f"release: wrote the plan to {args.plan_json}")
    if not args.push:
        print("release: dry run, nothing pushed (add --push, with the owner's go-ahead)")
        return 0 if not p.problems else EXIT_ERROR
    token = hub.token_from_env(os.environ)
    if token is None:
        return _fail("release", f"--push needs {hub.TOKEN_ENV} in the environment")
    if not p.ready:
        return _fail("release", "the plan has problems or no --owner; nothing pushed")
    staging = root / "artifacts" / "release"  # pragma: no cover - network from here on
    record: dict[str, object] = {}  # pragma: no cover

    def check(ref: str) -> str | None:  # pragma: no cover
        r = hub.ollama_check(ref, root)
        record["ollama"] = {k: v for k, v in vars(r).items() if k != "answer"}
        return r.answer if r.ok else None

    try:  # pragma: no cover - network; only with the owner's explicit go-ahead
        problems = hub.push(
            p,
            root,
            hub.HfHubApi(token, staging / "download"),
            tag=args.tag or f"{p.release}.0.0",
            staging=staging,
            ollama=None if args.no_ollama else check,
            record=record,
        )
    except (hub.ReleaseError, ImportError, OSError) as exc:  # pragma: no cover
        return _fail("release", exc)
    if args.record_json:  # pragma: no cover
        (root / args.record_json).write_text(json.dumps(record, indent=2) + "\n", "utf-8")
    for x in problems:  # pragma: no cover
        print(f"  - {x}")
    return EXIT_ERROR if problems else 0  # pragma: no cover


def _ollama(root: Path, owner: str, record_json: str | None) -> int:  # pragma: no cover
    """HUB-05 on its own: the published GGUF answers one demo sample through Ollama."""
    import json

    from ggh.release import hub

    if owner == hub.OWNER_PLACEHOLDER:
        return _fail("release", "--ollama-check needs --owner")
    ref = f"hf.co/{owner}/{hub.GGUF_REPO}:{hub.OLLAMA_QUANT}"
    r = hub.ollama_check(ref, root)
    print(f"ollama: {ref} sample {r.sample_id}: ok={r.ok} label={r.label} cites={r.citations}")
    print(r.answer if r.ok else f"ollama: FAIL: {r.error}")
    if record_json:
        (root / record_json).write_text(json.dumps(vars(r), indent=2) + "\n", "utf-8")
    return 0 if r.ok else EXIT_ERROR


def _cmd_space(args: argparse.Namespace) -> int:  # pragma: no cover - starts a server
    import logging

    # SPC-05: the demo's own outcome lines (an outcome word and a latency) go to stderr, so a
    # run's log shows what was logged; nothing a visitor types is ever in them
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    from ggh.release import sources
    from ggh.serve import llama
    from ggh.space import app, passkey, ui

    root = Path(args.root)
    path = root / args.samples
    if not args.local:
        return _fail("space", "only --local runs here (OWNER D-63: the demo is the local app)")
    ft_url = args.ft_url or f"http://{llama.HOST}:{llama.PORTS['ft2']}"
    try:
        env = app.passkey_env(os.environ, Path(args.env_file) if args.env_file else None)
        space = app.make_space(env, app.local_generators(ft_url, args.base_url), path)
        bundle = Path(args.bundle) if args.bundle else sources.BUNDLE
        summary = ui.results_summary(sources.load(root, sources.Paths(bundle=bundle)))
    except (sources.SourceError, passkey.PasskeyError, ValueError, OSError) as exc:
        return _fail("space", exc)
    if not env[passkey.SECRET_ENV].strip():
        print(f"space: note: {passkey.SECRET_ENV} is not set; live answers stay off (OWNER D-36)")
    print(f"space: http://127.0.0.1:{args.port}/ against {ft_url} (Ctrl-C to stop)")
    try:
        app.launch(space, path, port=args.port, summary=summary)
    except ImportError as exc:
        return _fail(
            "space", f"{exc}; the page needs gradio: uv run --group demo ggh space --local"
        )
    except OSError as exc:
        return _fail("space", exc)
    return 0


def _cmd_space_samples(args: argparse.Namespace) -> int:
    from ggh.space import samples

    root = Path(args.root)
    try:
        picked = samples.build(
            root / args.run,
            base_run=root / args.base_run if args.base_run else None,
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
    from ggh.release.hub import LAYOUTS, OWNER_PLACEHOLDER
    from ggh.space.samples import BASE_RUN, RUN, SAMPLES

    cd = sub.add_parser("card", help="generate MODEL_CARD.md from the results bundle (R4.1)")
    _source_flags(cd)
    cd.add_argument("--out", default="MODEL_CARD.md")
    cd.add_argument("--stdout", action="store_true", help="print instead of writing")
    cd.set_defaults(func=_cmd_card)
    rt = sub.add_parser("readme-table", help="the README results block from the bundle (R4.5)")
    _source_flags(rt)
    rt.add_argument("--write", default=None, help="splice the block into this file's markers")
    rt.set_defaults(func=_cmd_readme_table)
    from ggh.release.notes import OUT as NOTES_OUT

    rn = sub.add_parser("release-notes", help="GitHub release notes from the bundle (R4.5)")
    _source_flags(rn)
    rn.add_argument("--out", default=str(NOTES_OUT))
    rn.add_argument("--stdout", action="store_true", help="print instead of writing")
    rn.add_argument("--check", action="store_true", help="fail if --out is not current")
    rn.set_defaults(func=_cmd_release_notes)
    rl = sub.add_parser("release", help="Hub release plan; dry run unless --push (R4.2)")
    _source_flags(rl)
    rl.add_argument("--dry-run", action="store_true", help="the default; listed for clarity")
    rl.add_argument("--push", action="store_true", help="push (needs HF_TOKEN and a clean plan)")
    rl.add_argument("--owner", default=OWNER_PLACEHOLDER, help="Hub user or organisation")
    rl.add_argument("--tag", default=None, help="Hub tag (default: the release, e.g. v2.0.0)")
    rl.add_argument(
        "--release", choices=sorted(LAYOUTS), default=None, help="default: the shipped release"
    )
    rl.add_argument("--export-manifest", default=None, help="default: the release's own")
    rl.add_argument("--train-manifest", default=None, help="default: the release's own")
    rl.add_argument(
        "--card", default=None, help="committed card README.md must equal (shipped: MODEL_CARD.md)"
    )
    rl.add_argument("--plan-json", default=None, help="write the plan as JSON (evidence)")
    rl.add_argument("--record-json", default=None, help="after --push: what was checked")
    rl.add_argument("--no-ollama", action="store_true", help="skip the Ollama check after push")
    rl.add_argument("--ollama-check", action="store_true", help="only run the Ollama check")
    rl.set_defaults(func=_cmd_release)
    sp = sub.add_parser(
        "space", help="run the demo locally against `ggh serve ft2` (R4.3, OWNER D-63)"
    )
    sp.add_argument("--local", action="store_true", help="use ggh serve's llama-server")
    sp.add_argument("--root", default=".")
    sp.add_argument("--samples", default=str(SAMPLES))
    sp.add_argument("--port", type=int, default=7860)
    sp.add_argument("--ft-url", default=None, help="v2 GGUF server (default: ggh serve ft2)")
    sp.add_argument(
        "--base-url", default=None, help="also answer live with base (e.g. ggh serve base)"
    )
    sp.add_argument(
        "--env-file", default=None, help="read GGH_SPACE_PASSKEY from this file if unset"
    )
    sp.add_argument("--bundle", default=None, help="results_bundle.json (default: docs/results/)")
    sp.set_defaults(func=_cmd_space)
    ss = sub.add_parser("space-samples", help="pick the six demo samples from a dev run (R4.3)")
    ss.add_argument("--root", default=".")
    ss.add_argument("--run", default=str(RUN), help="EvalKit run folder over dev_eval")
    ss.add_argument(
        "--base-run", default=str(BASE_RUN), help="run with base's replies ('': --run's)"
    )
    ss.add_argument("--out", default=str(SAMPLES))
    ss.add_argument("--base-model", default=None)
    ss.add_argument("--ft-model", default=None)
    ss.set_defaults(func=_cmd_space_samples)
    pk = sub.add_parser("passkey", help="the demo passcode (OWNER D-39)")
    pks = pk.add_subparsers(dest="passkey_command", required=True)
    pset = pks.add_parser("set", help="hash a new passcode for the Space secret")
    pset.set_defaults(func=_cmd_passkey_set)
