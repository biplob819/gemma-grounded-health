"""RLS-01, RLS-02 (V3.5): the committed main_v3 run, as configs/eval_plan_v3.yaml registers it, and
v3's primary test, guardrails, stretch goal and label and length report."""

import gzip
import json
import subprocess
from datetime import datetime
from pathlib import Path

import yaml

from ggh.results import v3 as R

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "docs/results"
V3, V2, BASE = (f"gemma-e2b-{m}-q4" for m in ("grounded-v3", "grounded-v2", "base"))
PLAN = yaml.safe_load((ROOT / "configs/eval_plan_v3.yaml").read_text(encoding="utf-8"))


def records(run):
    raw = gzip.decompress((RES / run / "records.jsonl.gz").read_bytes()).decode("utf-8")
    return [json.loads(x) for x in raw.splitlines() if x]


def run_json(run="main_v3"):
    return json.loads((RES / run / "run.json").read_text(encoding="utf-8"))


def test_the_run_is_the_registered_main_v3():
    run, cfg = run_json(), yaml.safe_load((ROOT / "configs/evalkit_v3.yaml").read_text("utf-8"))
    (spec,) = PLAN["runs"]
    assert spec["name"] == "main_v3" and spec["results"] == "docs/results/main_v3"
    assert run["n"] == PLAN["n_cases"] == 1000 and run["seed"] == PLAN["seed"] == 7
    assert run["judges"] == PLAN["judges"] == cfg["judges"] == ["gpt-6.1-sol"]
    assert cfg["extractor"] == PLAN["extractor"] and run["task_ref"] == PLAN["task_file"]
    assert sorted(run["models"]) == sorted(spec["new_models"] + spec["cached_models"])
    assert cfg["max_cost_usd"] == 15 and len(records("main_v3")) == 3000


def test_base_and_v2_answers_come_from_the_v2_2_cache():
    new = {(r["completion"]["model"], r["case_id"]): r for r in records("main_v3")}
    old = {(r["completion"]["model"], r["case_id"]): r for r in records("main_v2")}
    for m in (BASE, V2):
        keys = [k for k in new if k[0] == m]
        assert len(keys) == 1000
        assert all(new[k]["completion"]["raw_text"] == old[k]["completion"]["raw_text"]
                   for k in keys)  # fmt: skip
    assert {c for m, c in new if m == V3} == {c for m, c in old if m == V2}  # the same cases
    result = json.loads((ROOT / R.RESULT).read_text(encoding="utf-8"))
    assert (result["headline"][BASE], result["headline"][V2]) == (0.404, 0.329)  # V2.2's values


def test_v3_answers_are_the_frozen_q4_k_m():
    cfg = (ROOT / "configs/evalkit_v3.yaml").read_text(encoding="utf-8")
    gguf = json.loads((ROOT / "docs/measurements/v3/gguf_ft3_manifest.json").read_text("utf-8"))
    sel = json.loads((ROOT / "docs/measurements/v3/train_selection.json").read_text("utf-8"))
    assert gguf["source"]["revision"] == sel["adapter_sha256"]
    q4 = gguf["files"]["ft3-Q4_K_M.gguf"]["sha256"]
    assert q4[:8] in cfg and sel["adapter_sha256"][:8] in cfg and "8085" in cfg
    v3 = [r for r in records("main_v3") if r["completion"]["model"] == V3]
    assert len(v3) == 1000 and not any(r["completion"]["error"] for r in v3)


def test_the_run_started_after_the_plan_and_the_frozen_hashes():
    started = datetime.fromisoformat(run_json()["started_at"].replace("Z", "+00:00"))

    def committed(path: str) -> datetime:
        out = subprocess.run(["git", "log", "-1", "--format=%cI", "--", path], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout.strip()  # fmt: skip
        return datetime.fromisoformat(out)

    paths = (
        "configs/eval_plan_v3.yaml",
        "docs/measurements/v3/train_selection.json",
        "docs/measurements/v3/gguf_ft3_manifest.json",
        "configs/evalkit_v3.yaml",
    )
    for path in paths:
        assert committed(path) < started, path


def test_spend_is_logged_and_within_the_owner_cap():
    rows = [json.loads(x) for x in (ROOT / "docs/measurements/spend-ledger.jsonl")
            .read_text(encoding="utf-8").splitlines() if x]  # fmt: skip
    (row,) = [r for r in rows if r.get("node") == "V3.5"]
    assert row["run_id"] == run_json()["run_id"] and row["n_cases"] == 1000
    assert row["spent_usd"] == round(run_json()["spent_usd"], 6) <= 15
    assert "OWNER D-69" in row["note"] and row["evalkit"] == PLAN["evalkit"]


def test_the_primary_test_follows_the_plan_rule():
    result = R.result(ROOT)
    p = result["primary"]
    assert (p["model_a"], p["model_b"], p["family_size"]) == (V3, V2, 1)
    assert (p["point"], p["lo"], p["hi"]) == (0.038, 0.014, 0.063) and p["separated"]
    assert result["headline"][V3] > result["headline"][V2]
    assert result["verdict"] == "improves on v2"
    assert all(g["ok"] for g in result["guardrails"].values())
    assert result["stretch"] == {"reach": 0.404, "value": 0.367, "reached": False}
    assert R.verdict(0.367, 0.329, False) == "tie"  # an interval through zero is a tie
    assert R.verdict(0.320, 0.329, True) == "tie"  # a separated drop is not an improvement
    a = result["answers"]
    assert a[V3]["labels"] == {"yes": 453, "no": 525, "maybe": 22} and a[V3]["unparsed"] == 0
    assert (a[V3]["mean_words"], a[V2]["mean_words"]) == (17.23, 15.03)


def test_the_result_file_matches_the_run():
    committed = json.loads((ROOT / R.RESULT).read_text(encoding="utf-8"))
    assert committed == R.result(ROOT)
    assert committed["run_id"] == run_json()["run_id"] == "r-63dbbdfd4b9e"


def test_the_published_bundle_leaves_wave_7_spend_out():
    from ggh import ledger
    from ggh.results.bundle import _in_scope

    path = ROOT / "docs/measurements/spend-ledger.jsonl"
    cfg = yaml.safe_load((ROOT / "configs/results.yaml").read_text(encoding="utf-8"))
    keep = _in_scope(cfg)
    assert keep is not None and not keep({"node": "V3.5"}) and keep({"node": "V2.2"})
    bundle = json.loads((RES / "results_bundle.json").read_text(encoding="utf-8"))
    assert bundle["spend_usd"] == round(ledger.total(path, "spent_usd", keep), 6) == 118.035054
    assert round(ledger.total(path, "spent_usd"), 6) == round(118.035054 + 1.815194, 6)
