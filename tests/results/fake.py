"""A throwaway copy of the committed results, and a synthetic v2 built from v1's records, so the
bundle's v2 path is tested before V2.2 has run. Nothing here is a real v2 number."""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V1 = "gemma-e2b-grounded-q4"
V2 = "gemma-e2b-grounded-v2-q4"
RUNS = (
    "main",
    "main_citation",
    "offscope",
    "probe_original",
    "probe_paraphrase",
    "parity",
    "dev_ground_v1",
)
PLAN_COMMITTED = "2026-11-30T00:00:00Z"
V2_STARTED = "2026-12-01T00:00:00Z"


def copy_repo(dst: Path) -> Path:
    shutil.copytree(ROOT / "configs", dst / "configs")
    shutil.copytree(ROOT / "docs/measurements", dst / "docs/measurements")
    for r in RUNS:
        shutil.copytree(ROOT / "docs/results" / r, dst / "docs/results" / r)
    shutil.copy(ROOT / "pyproject.toml", dst / "pyproject.toml")
    return dst


def _gz(rows: list[dict]) -> bytes:
    text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    return gzip.compress(text.encode("utf-8"), mtime=0)


def _records(folder: Path) -> list[dict]:
    raw = gzip.decompress((folder / "records.jsonl.gz").read_bytes()).decode("utf-8")
    return [json.loads(x) for x in raw.splitlines() if x]


def _as_v2(obj: dict) -> dict:
    text = json.dumps(obj).replace(V1, V2).replace("ft-Q4_K_M.gguf", "ft-v2-Q4_K_M.gguf")
    return json.loads(text)


def _v2_run(
    root: Path,
    src: str,
    dst: str,
    *,
    keep_others: bool,
    started: str = V2_STARTED,
    run_id: str | None = None,
) -> None:
    s_dir, d_dir = root / "docs/results" / src, root / "docs/results" / dst
    d_dir.mkdir(parents=True)
    shutil.copy(s_dir / "cases.jsonl.gz", d_dir / "cases.jsonl.gz")
    recs = _records(s_dir)
    v2 = [_as_v2(r) for r in recs if r["completion"]["model"] == V1]
    rows = (recs if keep_others else []) + v2
    (d_dir / "records.jsonl.gz").write_bytes(_gz(rows))
    summary = json.loads((s_dir / "summary.json").read_text(encoding="utf-8"))
    v1s = [m for m in summary["models"] if m["model"] == V1]
    if keep_others:
        summary["models"] = summary["models"] + [_as_v2(m) for m in v1s]
        extra = [_as_v2(c) for c in summary["comparisons"] if V1 in (c["model_a"], c["model_b"])]
        tie = [
            {
                "model_a": V2,
                "model_b": V1,
                "metric": c["metric"],
                "judge": c["judge"],
                "diff": {"point": 0.0, "lo": -0.03, "hi": 0.03},
                "p_value": 1.0,
                "separated": False,
            }
            for c in summary["comparisons"]
            if c["model_a"] == "gemma-e2b-base-q4" and c["model_b"] == V1
        ]
        summary["comparisons"] = summary["comparisons"] + extra + tie
    else:
        summary["models"] = [_as_v2(m) for m in v1s]
        summary["comparisons"] = []
    run = json.loads((s_dir / "run.json").read_text(encoding="utf-8"))
    run["models"] = [m["model"] for m in summary["models"]]
    run["started_at"] = summary["started_at"] = started
    run["run_id"] = summary["run_id"] = run_id or f"r-fake-{dst}"
    (d_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (d_dir / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")


def add_v2(root: Path, *, started: str = V2_STARTED, finalise: bool = True) -> None:
    """main_v2 (five models, from the v1 citation rescore), the four v2-only runs, the v2
    measurements, and, unless `finalise` is False, the plan finalised and pre-registered."""
    _v2_run(root, "main_citation", "main_v2", keep_others=True, started=started)
    for src in ("offscope", "probe_original", "probe_paraphrase"):
        _v2_run(root, src, f"{src}_v2", keep_others=False, started=started)
    shutil.copytree(root / "docs/results/parity", root / "docs/results/parity_v2")
    p = json.loads((root / "docs/results/parity_v2/run.json").read_text(encoding="utf-8"))
    p["started_at"] = started
    (root / "docs/results/parity_v2/run.json").write_text(json.dumps(p), encoding="utf-8")
    m = root / "docs/measurements"
    for a, b in (
        ("train_run", "train_run_v2"),
        ("train_selection", "train_selection_v2"),
        ("parity_report", "parity_report_v2"),
        ("parity_headline", "parity_headline_v2"),
    ):
        shutil.copy(m / f"{a}.json", m / f"{b}.json")
    art = json.loads((m / "export_artifacts.json").read_text(encoding="utf-8"))
    (m / "export_artifacts_v2.json").write_text(json.dumps([_as_v2(a) for a in art]), "utf-8")
    probe = json.loads((m / "probe_report.json").read_text(encoding="utf-8"))
    probe["models"] = {V2: probe["models"][V1]}
    (m / "probe_report_v2.json").write_text(json.dumps(probe), encoding="utf-8")
    for name in ("audit_v2_result", "audit_v2r1_result", "audit_v2r2_result"):
        res = {"labelled": 50, "overall": {"ok": 49}, "passed": False}
        (m / f"{name}.json").write_text(json.dumps(res), encoding="utf-8")
    for a in ("train", "parity"):
        shutil.copy(m / "lineage" / f"{a}.json", m / "lineage" / f"{a}_v2.json")
    if finalise:
        finalise_plan(root)


def finalise_plan(root: Path, committed_at: str = PLAN_COMMITTED) -> str:
    plan = root / "configs/eval_plan_v2.yaml"
    plan.write_text(
        plan.read_text(encoding="utf-8").replace("status: draft", "status: final"), encoding="utf-8"
    )
    sha = hashlib.sha256(plan.read_bytes()).hexdigest()
    (root / "docs/measurements/eval_plan_v2.sha256").write_text(sha + "\n", encoding="utf-8")
    row = {
        "commit": "0" * 40,
        "committed_at": committed_at,
        "hash_file": "docs/measurements/eval_plan_v2.sha256",
        "plan": "configs/eval_plan_v2.yaml",
        "sha256": sha,
    }
    with (root / "docs/measurements/preregistrations.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
    return sha
