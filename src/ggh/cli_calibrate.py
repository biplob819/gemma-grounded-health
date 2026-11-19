"""`ggh calibrate ...` (E3.3, RES-03, OWNER D-24): the 40-claim Gemma-family check. `sample`
draws the blind sheet, `label` opens EvalKit's labelling page on it (127.0.0.1), `score` joins
the labels to the verdicts already in the run's records. No command here calls an API.

EvalKit's own `pubmedqa` calibration (100 answers plus a 20-answer relabel) is EvalKit's
`evalkit calibrate export|label|score` on EvalKit's published run; see
specs/E3.3-results/CALIBRATION-PREP.md."""

from __future__ import annotations

import argparse
from pathlib import Path

EXIT_ERROR = 2


def _cmd_sample(args: argparse.Namespace) -> int:
    from ggh.results import family_check as F
    from ggh.results.runs import ResultsError, Run

    root = Path(args.root)
    results_root = Path(args.results_root) if args.results_root else root
    try:
        alloc = F.parse_allocation(args.allocation) if args.allocation else None
        sheet = F.draw(Run(results_root, args.run), n=args.n, seed=args.seed, alloc=alloc)
        path = F.write_sheet(root, sheet, force=args.force)
    except (ResultsError, FileExistsError) as exc:
        print(f"calibrate sample: FAIL: {exc}")
        return EXIT_ERROR
    split = ", ".join(f"{m} {k}" for m, k in sheet.allocation.items())
    print(f"wrote {path}: {len(sheet.items)} claims from {sheet.run} ({split}), seed {sheet.seed}")
    print("label them: ggh calibrate label")
    return 0


def _cmd_label(args: argparse.Namespace) -> int:
    from ggh.results import family_check as F
    from ggh.results.runs import ResultsError

    try:
        srv = F.serve(Path(args.root), port=args.port)
    except (ResultsError, ValueError, OSError) as exc:
        print(f"calibrate label: FAIL: {exc}")
        return EXIT_ERROR
    print(f"labelling page: http://127.0.0.1:{srv.server_address[1]}/  (Ctrl-C to stop; autosave)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    from ggh.results import family_check as F
    from ggh.results.runs import ResultsError

    root = Path(args.root)
    try:
        report = F.score(root, results_root=Path(args.results_root) if args.results_root else None)
    except ResultsError as exc:
        print(f"calibrate score: FAIL: {exc}")
        return EXIT_ERROR
    for p in F.write_report(root, report).values():
        print(f"wrote {p}")
    for j in report.judges:
        k = j.kappa
        print(
            f"{j.judge}: agreement {j.agreement:.3f} on {j.n_claims} claims, kappa "
            f"{k['point']:.2f} [{k['lo']:.2f}, {k['hi']:.2f}]"
        )
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    cal = sub.add_parser("calibrate", help="the 40-claim Gemma-family judge check (OWNER D-24)")
    csub = cal.add_subparsers(dest="calibrate_command", required=True)
    s = csub.add_parser("sample", help="draw the blind 40-claim sheet from a committed run")
    s.add_argument("--root", default=".", help="repo root (the sheet is written here)")
    s.add_argument("--results-root", default=None, help="repo root holding docs/results/<run> "
                   "(default --root)")  # fmt: skip
    s.add_argument("--run", default="main_v2", help="a folder under docs/results/")
    s.add_argument("--n", type=int, default=40)
    s.add_argument("--seed", type=int, default=24)
    s.add_argument("--allocation", default=None, help="model=count,...; default half base, the "
                   "rest split over the fine-tunes")  # fmt: skip
    s.add_argument("--force", action="store_true", help="redraw an unlabelled sheet")
    s.set_defaults(func=_cmd_sample)
    lab = csub.add_parser("label", help="open EvalKit's labelling page on the sheet (127.0.0.1)")
    lab.add_argument("--root", default=".", help="repo root")
    lab.add_argument("--port", type=int, default=0, help="default: any free port")
    lab.set_defaults(func=_cmd_label)
    sc = csub.add_parser("score", help="score each judge's recorded verdicts against the labels")
    sc.add_argument("--root", default=".", help="repo root")
    sc.add_argument("--results-root", default=None, help="repo root holding docs/results/<run>")
    sc.set_defaults(func=_cmd_score)
