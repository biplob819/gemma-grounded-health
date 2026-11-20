"""`ggh results ...` (E3.3, V2.2): bundle, sample, estimate, plan-check,
aim-check. Committed files only;
nothing here calls an API or reads artifacts/ (except `bundle --full-lineage`)."""

from __future__ import annotations

import argparse
from pathlib import Path

EXIT_ERROR = 2


def _cmd_bundle(args: argparse.Namespace) -> int:
    from ggh.results import bundle as B
    from ggh.results.runs import ResultsError

    root = Path(args.root)
    try:
        built = B.build(root, full_lineage=args.full_lineage)
    except (ResultsError, ValueError, OSError) as exc:
        print(f"results bundle: FAIL: {exc}")
        return EXIT_ERROR
    for p in B.write(root, built, Path(args.out)):
        print(f"wrote {p}")
    b = built.bundle
    print(
        f"releases: {', '.join(r.version for r in b.releases)}; runs: {len(b.runs)}; "
        f"lineage ({b.lineage_scope}): {'OK' if b.lineage_ok else 'FAIL'}"
    )
    for x in b.publish_blockers:
        print(f"  blocker: {x}")
    return 0 if b.lineage_ok else EXIT_ERROR


def _cmd_sample(args: argparse.Namespace) -> int:
    from ggh.results import sample as S
    from ggh.results.runs import Run

    root = Path(args.root)
    try:
        path = S.write_sheet(root, Run(root, args.run), force=args.force)
    except (FileExistsError, ValueError) as exc:
        print(f"results sample: FAIL: {exc}")
        return EXIT_ERROR
    print(f"wrote {path}: fill finding (ok, model_fault, pipeline_fault, unclear), note, read_by")
    return 0


def _cmd_estimate(args: argparse.Namespace) -> int:
    from ggh.results import estimate as E
    from ggh.results.plan import load_plan

    root = Path(args.root)
    est = E.estimate(root, load_plan(root, args.plan), cpa=args.claims_per_answer)
    print(E.dumps(est) if args.json else E.text(est))
    return 0


def _cmd_plan_check(args: argparse.Namespace) -> int:
    from ggh.results.plan import all_plans, plan_problems, preregistration_problems

    root = Path(args.root)
    bad = 0
    for plan in all_plans(root):
        problems = plan_problems(root, plan) + preregistration_problems(root, plan)
        print(f"{plan.path} ({plan.status}): {'OK' if not problems else 'FAIL'}")
        for p in problems:
            print(f"  - {p}")
        bad += bool(problems)
    return EXIT_ERROR if bad else 0


def _cmd_aim_check(args: argparse.Namespace) -> int:
    from ggh.results import aim_check as A
    from ggh.results.runs import ResultsError

    root = Path(args.root)
    try:
        result = A.build(root)
    except (ResultsError, ValueError, OSError) as exc:
        print(f"results aim-check: FAIL: {exc}")
        return EXIT_ERROR
    path = A.write(root, result, Path(args.out))
    print(f"wrote {path} (post-hoc, OWNER D-62; published numbers unchanged)")
    for model, per in result["citation"][A.RUN].items():
        for j, row in per.items():
            p, c = row["published"], row["corrected"]
            print(
                f"  {model} [{j}] precision {p['citation_precision']:.3f} -> "
                f"{c['citation_precision']:.3f}, aim rate {p['aim_citation_rate']:.3f} -> "
                f"{c['aim_citation_rate']:.3f}"
            )
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    rs = sub.add_parser("results", help="results bundle, sample read, plans and estimates (E3.3)")
    rsub = rs.add_subparsers(dest="results_command", required=True)
    b = rsub.add_parser("bundle", help="write results_bundle.json and the behaviour table")
    b.add_argument("--root", default=".", help="repo root")
    b.add_argument("--out", default="docs/results", help="folder for the outputs")
    b.add_argument("--full-lineage", action="store_true", help="also verify artifacts/ manifests")
    b.set_defaults(func=_cmd_bundle)
    s = rsub.add_parser("sample", help="draw the 20-per-model sample-read sheet for a run (§7.6)")
    s.add_argument("--run", required=True, help="a folder under docs/results/")
    s.add_argument("--root", default=".", help="repo root")
    s.add_argument("--force", action="store_true", help="redraw even if findings exist")
    s.set_defaults(func=_cmd_sample)
    e = rsub.add_parser("estimate", help="cache-aware spend estimate for a plan (no API calls)")
    e.add_argument("--plan", default="configs/eval_plan_v2.yaml")
    e.add_argument("--root", default=".", help="repo root")
    e.add_argument("--claims-per-answer", type=float, default=None)
    e.add_argument("--json", action="store_true")
    e.set_defaults(func=_cmd_estimate)
    c = rsub.add_parser("plan-check", help="every plan's hash and pre-registration order")
    c.add_argument("--root", default=".", help="repo root")
    c.set_defaults(func=_cmd_plan_check)
    a = rsub.add_parser("aim-check", help="post-hoc aim-citation rule check (OWNER D-62)")
    a.add_argument("--root", default=".", help="repo root")
    a.add_argument("--out", default="docs/measurements/aim_rule_check.json")
    a.set_defaults(func=_cmd_aim_check)
