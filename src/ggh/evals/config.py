"""EVK-05: `configs/evalkit.yaml` against Tech Spec §7.1, the EvalKit pin, and the cache-reuse
check that must pass before any reuse of EvalKit's reference run is claimed.

EvalKit's store reuses a cached answer, claim set or verdict only when its cache key matches,
so reuse is never wrong; it can only be *claimed* wrongly (an estimate that counts on hits
that will not happen). `reuse_problems` lists every condition that fails; an empty list is
the only state in which a cost estimate may assume reuse.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any

from evalkit.config import load_run_config
from evalkit.pipeline.extract import EXTRACTOR_VERSION
from evalkit.pipeline.judge import JUDGE_PROMPT_HASH
from evalkit.schemas import RunConfig
from evalkit.tasks import builtin_task

from ggh.data.build import EVALKIT_COMMIT

MAIN = Path("configs") / "evalkit.yaml"
DEV = Path("configs") / "evalkit_dev.yaml"

# Tech Spec §7.1, field by field
SPEC: dict[str, Any] = {
    "models": ("gemma-e2b-base-q4", "gemma-e2b-grounded-q4", "claude-haiku-5-5", "gpt-6-luna"),
    "endpoints": {
        "gemma-e2b-base-q4": "http://127.0.0.1:8080/v1",
        "gemma-e2b-grounded-q4": "http://127.0.0.1:8081/v1",
    },
    "judges": ("claude-opus-5-5", "gpt-6.1-sol"),
    "extractor": "claude-haiku-5-5",
    "structured_output": True,
    "max_output_tokens": 2048,
    "concurrency": {"anthropic": 8, "openai": 8, "openai_compatible": 1},
    "seed": 7,
    "default_effort": "medium",
    "judge_effort": "medium",
    "extractor_effort": "medium",
    "max_cost_usd": 70.0,
    "efforts": {},
    "thresholds": (),
}
# the dev config may differ from the main one only here
DEV_MAY_DIFFER = frozenset({"models", "endpoints", "max_cost_usd"})


def _value(cfg: RunConfig, field: str) -> Any:
    v = getattr(cfg, field)
    return dict(v) if field in ("endpoints", "concurrency", "efforts") else v


def config_problems(root: Path) -> list[str]:
    problems = []
    main = load_run_config(root / MAIN)
    for field, want in SPEC.items():
        got = _value(main, field)
        if got != want:
            problems.append(f"{MAIN}: {field} is {got!r}, §7.1 says {want!r}")
    dev = load_run_config(root / DEV)
    for field in SPEC:
        if field not in DEV_MAY_DIFFER and _value(dev, field) != _value(main, field):
            problems.append(f"{DEV}: {field} differs from {MAIN}")
    for m in dev.models:
        if m not in main.models or dev.endpoints.get(m) != main.endpoints.get(m):
            problems.append(f"{DEV}: model {m!r} or its endpoint is not in {MAIN}")
    return problems


def pinned_commit(root: Path) -> str:
    """The EvalKit commit pinned in pyproject.toml."""
    deps = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "dependencies"
    ]
    pins = [d for d in deps if d.startswith("evalkit-harness")]
    if len(pins) != 1 or "@" not in pins[0]:
        raise ValueError("pyproject.toml must pin evalkit-harness to one git commit")
    return pins[0].rsplit("@", 1)[1].strip()


def installed_commit() -> str | None:
    """The commit the installed EvalKit was built from (PEP 610 direct_url.json)."""
    from importlib.metadata import distribution

    text = distribution("evalkit-harness").read_text("direct_url.json")
    if not text:
        return None
    return json.loads(text).get("vcs_info", {}).get("commit_id")


def pin_problems(root: Path) -> list[str]:
    problems = []
    pin = pinned_commit(root)
    if pin != EVALKIT_COMMIT:
        problems.append(f"pyproject pins EvalKit {pin[:12]}, D-28 says {EVALKIT_COMMIT[:12]}")
    got = installed_commit()
    if got != pin:
        problems.append(f"installed EvalKit is {str(got)[:12]}, pyproject pins {pin[:12]}")
    return problems


def prompt_hash(task: str = "pubmedqa") -> str:
    """As EvalKit stamps it into every record."""
    return hashlib.sha256(builtin_task(task).prompt_template.encode()).hexdigest()[:16]


def ours(cfg: RunConfig, evalkit_sha: str) -> dict[str, Any]:
    """What this project's runs will stamp, for comparison with a reference run."""
    return {
        "evalkit": evalkit_sha,
        "prompt_hash": prompt_hash(),
        "rubric_hash": JUDGE_PROMPT_HASH,
        "extractor_version": EXTRACTOR_VERSION,
        "seed": cfg.seed,
        "structured_output": cfg.structured_output,
        "max_output_tokens": cfg.max_output_tokens,
        "judges": sorted(cfg.judges),
        "extractor": cfg.extractor,
        "judge_effort": cfg.judge_effort,
        "extractor_effort": cfg.extractor_effort,
        "effort": {m: cfg.effort_for(m) for m in cfg.models if m not in cfg.endpoints},
    }


def reference(run_dir: Path, ref_config: Path) -> dict[str, Any]:
    """The same fields read back from a finished EvalKit run folder and its config."""
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (run_dir / "records.jsonl").read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]
    cfg = load_run_config(ref_config)

    def one(values: set[Any], what: str) -> Any:
        return values.pop() if len(values) == 1 else f"<{len(values)} values of {what}>"

    return {
        "evalkit": one({r["git_sha"] for r in records}, "git_sha"),
        "prompt_hash": one({r["prompt_hash"] for r in records}, "prompt_hash"),
        "rubric_hash": one(
            {v["judge_prompt_hash"] for r in records for v in r["verdicts"]}, "rubric_hash"
        ),
        "extractor_version": one(
            {c["extractor_version"] for r in records for c in r["claims"]}, "extractor_version"
        ),
        "seed": run["seed"],
        "structured_output": one(
            {r["completion"]["params"].get("structured_output") for r in records},
            "structured_output",
        ),
        "max_output_tokens": cfg.max_output_tokens,
        "judges": sorted({v["judge"] for r in records for v in r["verdicts"]}),
        "extractor": cfg.extractor,
        "judge_effort": cfg.judge_effort,
        "extractor_effort": cfg.extractor_effort,
        "effort": {
            r["completion"]["model"]: r["completion"]["params"].get("effort") for r in records
        },
    }


COMPARED = (
    "evalkit", "prompt_hash", "rubric_hash", "extractor_version", "seed", "structured_output",
    "max_output_tokens", "judges", "extractor", "judge_effort", "extractor_effort",
)  # fmt: skip


def reuse_problems(
    mine: dict[str, Any], ref: dict[str, Any], store: Path | None = None
) -> list[str]:
    """Every reason the reference run's cache cannot be counted on; empty means it can.

    These are the inputs of EvalKit's cache keys that a run folder records. Case content is
    covered by the prompt hash and the shared PQA-L cases; the store itself must also be
    present, since a matching key is worth nothing without the entry behind it.
    """
    problems = [
        f"{field}: ours {mine[field]!r}, reference {ref[field]!r}"
        for field in COMPARED
        if mine[field] != ref[field]
    ]
    if store is None or not store.is_dir() or not any(store.iterdir()):
        problems.append(f"no EvalKit store with the reference run's entries at {store}")
    shared = set(mine["effort"]) & set(ref["effort"])
    if not shared:
        problems.append("no generator model is shared with the reference run")
    for m in sorted(shared):
        if mine["effort"][m] != ref["effort"][m]:
            problems.append(
                f"effort for {m}: ours {mine['effort'][m]!r}, reference {ref['effort'][m]!r}"
            )
    return problems
