"""`ggh grpo` and `ggh rl-dev` (V3.1, OWNER D-66): the v3 GRPO run and its dev check. Both need
the Kaggle GPU packages (notebooks/requirements-kaggle-grpo.txt); no command here calls an API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

EXIT_ERROR = 2


def _cmd_grpo(args: argparse.Namespace) -> int:
    from ggh.config import ConfigError
    from ggh.rl.grpo import InitError, run

    try:
        report = run(Path(args.config), Path(args.root), Path(args.out), max_steps=args.max_steps)
    except ImportError as exc:
        print(f"grpo: FAIL: {exc}; needs the Kaggle GRPO packages")
        return EXIT_ERROR
    except (ConfigError, InitError, OSError) as exc:
        print(f"grpo: FAIL: {exc}")
        return EXIT_ERROR
    short = {k: v for k, v in report.items() if k not in ("reward_trace", "trl_args")}
    print(json.dumps(short, indent=2))
    return 0


def _cmd_rl_dev(args: argparse.Namespace) -> int:
    from ggh.config import ConfigError
    from ggh.rl.grpo import InitError, rl_dev

    try:
        rl_dev(Path(args.config), Path(args.run), Path(args.root), Path(args.out), n=args.n)
    except ImportError as exc:
        print(f"rl-dev: FAIL: {exc}; needs the Kaggle GRPO packages")
        return EXIT_ERROR
    except (ConfigError, InitError, ValueError, OSError) as exc:
        print(f"rl-dev: FAIL: {exc}")
        return EXIT_ERROR
    return 0


def _cmd_rl_dev_one(args: argparse.Namespace) -> int:
    from ggh.config import ConfigError
    from ggh.rl.grpo import rl_dev_one

    try:
        rl_dev_one(Path(args.config), Path(args.adapter), args.name, Path(args.root),
                   Path(args.out), n=args.n)  # fmt: skip
    except ImportError as exc:
        print(f"rl-dev-one: FAIL: {exc}; needs the Kaggle GRPO packages")
        return EXIT_ERROR
    except (ConfigError, ValueError, OSError) as exc:
        print(f"rl-dev-one: FAIL: {exc}")
        return EXIT_ERROR
    return 0


def _cmd_grpo_b(args: argparse.Namespace) -> int:
    from ggh.config import ConfigError
    from ggh.rl.grpo import InitError
    from ggh.rl.grpo_b import CheckerError, run

    try:
        report = run(Path(args.config), Path(args.root), Path(args.out), max_steps=args.max_steps)
    except ImportError as exc:
        print(f"grpo-b: FAIL: {exc}; needs the Kaggle GRPO packages")
        return EXIT_ERROR
    except (ConfigError, InitError, CheckerError, OSError) as exc:
        print(f"grpo-b: FAIL: {exc}")
        return EXIT_ERROR
    short = {k: v for k, v in report.items() if k not in ("reward_trace", "trl_args")}
    print(json.dumps(short, indent=2))
    return 0


def _cmd_rl_dev_b(args: argparse.Namespace) -> int:
    from ggh.config import ConfigError
    from ggh.rl.grpo import InitError
    from ggh.rl.grpo_b import rl_dev

    try:
        rl_dev(Path(args.config), Path(args.run), Path(args.root), Path(args.out), n=args.n)
    except ImportError as exc:
        print(f"rl-dev-b: FAIL: {exc}; needs the Kaggle GRPO packages")
        return EXIT_ERROR
    except (ConfigError, InitError, ValueError, OSError) as exc:
        print(f"rl-dev-b: FAIL: {exc}")
        return EXIT_ERROR
    return 0


def _cmd_rl_dev_one_b(args: argparse.Namespace) -> int:
    from ggh.config import ConfigError
    from ggh.rl.grpo_b import rl_dev_one

    try:
        rl_dev_one(Path(args.config), Path(args.adapter), args.name, Path(args.root),
                   Path(args.out), n=args.n)  # fmt: skip
    except ImportError as exc:
        print(f"rl-dev-one-b: FAIL: {exc}; needs the Kaggle GRPO packages")
        return EXIT_ERROR
    except (ConfigError, ValueError, OSError) as exc:
        print(f"rl-dev-one-b: FAIL: {exc}")
        return EXIT_ERROR
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    g = sub.add_parser("grpo", help="v3: GRPO from v2's adapter with the programmatic reward")
    g.add_argument("--config", default="configs/recipe_v3_grpo.yaml")
    g.add_argument("--root", default=".", help="repo root holding artifacts/targets")
    g.add_argument("--out", required=True, help="folder for checkpoints and report.json")
    g.add_argument("--max-steps", type=int, default=None, help="stop early (the dev trial)")
    g.set_defaults(func=_cmd_grpo)
    d = sub.add_parser("rl-dev", help="v3: score v2 and each GRPO checkpoint on dev prompts")
    d.add_argument("--config", default="configs/recipe_v3_grpo.yaml")
    d.add_argument("--run", required=True, help="the ggh grpo output folder")
    d.add_argument("--root", default=".", help="repo root holding artifacts/data")
    d.add_argument("--out", required=True, help="folder for rl_dev.json and the replies")
    d.add_argument("--n", type=int, default=200, help="first N dev prompts")
    d.set_defaults(func=_cmd_rl_dev)
    o = sub.add_parser("rl-dev-one", help="v3: score one adapter on dev prompts (run by rl-dev)")
    o.add_argument("--config", default="configs/recipe_v3_grpo.yaml")
    o.add_argument("--adapter", required=True, help="the adapter folder")
    o.add_argument("--name", required=True, help="its name in rl_dev.json")
    o.add_argument("--root", default=".", help="repo root holding artifacts/data")
    o.add_argument("--out", required=True, help="folder for dev_<name>.json and the replies")
    o.add_argument("--n", type=int, required=True, help="first N dev prompts")
    o.set_defaults(func=_cmd_rl_dev_one)
    gb = sub.add_parser("grpo-b", help="v3b: GRPO from v2's adapter with the checker reward")
    gb.add_argument("--config", default="configs/recipe_v3b_grpo.yaml")
    gb.add_argument("--root", default=".", help="repo root holding artifacts/targets")
    gb.add_argument("--out", required=True, help="folder for checkpoints and report.json")
    gb.add_argument("--max-steps", type=int, default=None, help="stop after this many steps")
    gb.set_defaults(func=_cmd_grpo_b)
    db = sub.add_parser("rl-dev-b", help="v3b: score v2 and each checkpoint on held-out dev")
    db.add_argument("--config", default="configs/recipe_v3b_grpo.yaml")
    db.add_argument("--run", required=True, help="the ggh grpo-b output folder")
    db.add_argument("--root", default=".", help="repo root holding artifacts/data")
    db.add_argument("--out", required=True, help="folder for rl_dev.json and the replies")
    db.add_argument("--n", type=int, default=200, help="first N held-out dev prompts")
    db.set_defaults(func=_cmd_rl_dev_b)
    ob = sub.add_parser("rl-dev-one-b", help="v3b: score one adapter (run by rl-dev-b)")
    ob.add_argument("--config", default="configs/recipe_v3b_grpo.yaml")
    ob.add_argument("--adapter", required=True, help="the adapter folder")
    ob.add_argument("--name", required=True, help="its name in rl_dev.json")
    ob.add_argument("--root", default=".", help="repo root holding artifacts/data")
    ob.add_argument("--out", required=True, help="folder for dev_<name>.json and the replies")
    ob.add_argument("--n", type=int, required=True, help="first N held-out dev prompts")
    ob.set_defaults(func=_cmd_rl_dev_one_b)
