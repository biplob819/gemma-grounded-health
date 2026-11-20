"""VEV-04 to VEV-06 on the committed V2.2 runs: the cache rule for base and v1, v2's off-scope
run, and the V2.2 spend (OWNER D-57, D-53, D-43)."""

import gzip
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "docs/results"
M = ROOT / "docs/measurements"
BASE_V1 = ("gemma-e2b-base-q4", "gemma-e2b-grounded-q4")
V2 = "gemma-e2b-grounded-v2-q4"
V22_RUNS = ("main_v2", "offscope_v2", "probe_original_v2", "probe_paraphrase_v2", "parity_v2")


def records(run):
    raw = gzip.decompress((RES / run / "records.jsonl.gz").read_bytes()).decode("utf-8")
    return [json.loads(x) for x in raw.splitlines() if x]


def run_json(run):
    return json.loads((RES / run / "run.json").read_text(encoding="utf-8"))


def test_main_v2_base_and_v1_answers_are_e32s_byte_for_byte_on_the_same_cases():
    new = {(r["completion"]["model"], r["case_id"]): r for r in records("main_v2")}
    old = {(r["completion"]["model"], r["case_id"]): r for r in records("main_citation")}
    assert {m for m, _ in new} == {*BASE_V1, V2}
    for m in BASE_V1:
        keys = [k for k in new if k[0] == m]
        assert len(keys) == 1000
        assert all(new[k]["completion"]["raw_text"] == old[k]["completion"]["raw_text"]
                   for k in keys)  # fmt: skip
    assert {c for m, c in new if m == V2} == {c for m, c in new if m == BASE_V1[0]}


def test_v2_offscope_covers_the_20_frozen_prompts():
    run = run_json("offscope_v2")
    assert run["n"] == 20 and run["models"] == [V2] and run["judges"] == ["gpt-6.1-sol"]
    ids = {json.loads(x)["id"] for x in (ROOT / "data/offscope_task.jsonl")
           .read_text(encoding="utf-8").splitlines() if x}  # fmt: skip
    assert {r["case_id"] for r in records("offscope_v2")} == ids


def test_every_v22_run_is_in_the_ledger_within_the_cap_and_labelled_built_after_v1():
    rows = [json.loads(x) for x in (M / "spend-ledger.jsonl").read_text(encoding="utf-8")
            .splitlines() if x]  # fmt: skip
    v22 = [r for r in rows if r.get("node") == "V2.2"]
    assert sorted(r["run_id"] for r in v22) == sorted(run_json(n)["run_id"] for n in V22_RUNS)
    for r in v22:
        assert r["spent_usd"] == round(run_json_by_id(r["run_id"])["spent_usd"], 6)
        assert "built after v1's test results were seen" in r["note"]
    plan = yaml.safe_load((ROOT / "configs/eval_plan_v2.yaml").read_text(encoding="utf-8"))
    assert sum(r["spent_usd"] for r in v22) <= 25 and plan["status"] == "final"


def run_json_by_id(run_id):
    (hit,) = [run_json(n) for n in V22_RUNS if run_json(n)["run_id"] == run_id]
    return hit
