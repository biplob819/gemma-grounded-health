"""RBS-01..03 (V3.7): the committed main_v3b run, as configs/eval_plan_v3b.yaml registers it with
the owner's Q8_0 extra (OWNER D-73, D-74), v3b's primary test and reported signals, and the Q8_0
parity check."""

import functools
import gzip
import json
import subprocess
from datetime import datetime
from pathlib import Path

import yaml

from ggh.results import v3b as R

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "docs/results"
V3B = ROOT / "docs/measurements/v3b"
Q4, Q8 = "gemma-e2b-grounded-v3b-q4", "gemma-e2b-grounded-v3b-q8"
V2, BASE = "gemma-e2b-grounded-v2-q4", "gemma-e2b-base-q4"
PLAN = yaml.safe_load((ROOT / "configs/eval_plan_v3b.yaml").read_text(encoding="utf-8"))


@functools.cache
def computed():
    """v3b's result recomputed from the committed run once per session."""
    return R.result(ROOT)


def records(run):
    raw = gzip.decompress((RES / run / "records.jsonl.gz").read_bytes()).decode("utf-8")
    return [json.loads(x) for x in raw.splitlines() if x]


def run_json(run="main_v3b"):
    return json.loads((RES / run / "run.json").read_text(encoding="utf-8"))


def test_the_run_is_the_registered_main_v3b():
    run, cfg = run_json(), yaml.safe_load((ROOT / "configs/evalkit_v3b.yaml").read_text("utf-8"))
    (spec,) = PLAN["runs"]
    assert spec["name"] == "main_v3b" and spec["results"] == "docs/results/main_v3b"
    assert run["n"] == PLAN["n_cases"] == 1000 and run["seed"] == PLAN["seed"] == 7
    assert run["judges"] == PLAN["judges"] == cfg["judges"] == ["gpt-6.1-sol"]
    assert cfg["extractor"] == PLAN["extractor"] and run["task_ref"] == PLAN["task_file"]
    registered = spec["new_models"] + spec["cached_models"]
    assert sorted(run["models"]) == sorted([*registered, Q8])  # the owner's extra (D-73)
    assert computed()["exploratory_models"] == [Q8] and Q8 not in PLAN["primary"][0]
    assert cfg["max_cost_usd"] == 5 and len(records("main_v3b")) == 4000


def test_base_and_v2_answers_come_from_the_v2_2_cache():
    new = {(r["completion"]["model"], r["case_id"]): r for r in records("main_v3b")}
    old = {(r["completion"]["model"], r["case_id"]): r for r in records("main_v2")}
    for m in (BASE, V2):
        keys = [k for k in new if k[0] == m]
        assert len(keys) == 1000
        assert all(
            new[k]["completion"]["raw_text"] == old[k]["completion"]["raw_text"] for k in keys
        )
    assert {c for m, c in new if m == Q4} == {c for m, c in old if m == V2}  # the same cases
    assert (computed()["headline"][BASE], computed()["headline"][V2]) == (0.404, 0.329)


def test_v3b_answers_are_the_frozen_ggufs():
    cfg = (ROOT / "configs/evalkit_v3b.yaml").read_text(encoding="utf-8")
    gguf = json.loads((V3B / "gguf_ft3b_manifest.json").read_text("utf-8"))
    sel = json.loads((V3B / "train_selection.json").read_text("utf-8"))
    assert gguf["source"]["revision"] == sel["adapter_sha256"]
    for name in ("ft3b-Q4_K_M.gguf", "ft3b-Q8_0.gguf"):
        assert gguf["files"][name]["sha256"][:8] in cfg
    assert sel["adapter_sha256"][:8] in cfg and "8086" in cfg and "8087" in cfg
    rows = records("main_v3b")
    for model, snap in ((Q4, "ft3b-Q4_K_M.gguf"), (Q8, "ft3b-Q8_0.gguf")):
        mine = [r for r in rows if r["completion"]["model"] == model]
        assert len(mine) == 1000 and not any(r["completion"]["error"] for r in mine)
        assert {r["completion"]["model_snapshot"] for r in mine} == {f"artifacts/gguf/{snap}"}


def test_the_run_started_after_the_plan_and_the_frozen_hashes():
    started = datetime.fromisoformat(run_json()["started_at"].replace("Z", "+00:00"))

    def committed(path: str) -> datetime:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%cI", "--", path],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return datetime.fromisoformat(out)

    paths = (
        "configs/eval_plan_v3b.yaml",
        "docs/measurements/v3b/train_selection.json",
        "docs/measurements/v3b/gguf_ft3b_manifest.json",
        "configs/evalkit_v3b.yaml",
        "docs/measurements/v3b/parity_q8_0.json",
    )
    for path in paths:
        assert committed(path) < started, path


def test_spend_is_logged_and_within_the_owner_cap():
    rows = [
        json.loads(x)
        for x in (ROOT / "docs/measurements/spend-ledger.jsonl").read_text("utf-8").splitlines()
        if x
    ]
    (row,) = [r for r in rows if r.get("node") == "V3.7"]
    assert row["run_id"] == run_json()["run_id"] and row["n_cases"] == 1000
    assert row["spent_usd"] == round(run_json()["spent_usd"], 6) <= 8  # OWNER D-74
    assert "OWNER D-73" in row["note"] and "OWNER D-74" in row["note"]
    decisions = (ROOT / "docs/decisions.md").read_text(encoding="utf-8")
    assert "| OWNER D-73 |" in decisions and "| OWNER D-74 |" in decisions


def test_the_primary_test_follows_the_plan_rule():
    result = computed()
    p = result["primary"]
    assert (p["model_a"], p["model_b"], p["family_size"]) == (Q4, V2, 1)
    assert (p["point"], p["lo"], p["hi"]) == (0.174, 0.147, 0.202) and p["separated"]
    assert result["verdict"] == "improves on v2"
    assert all(g["ok"] for g in result["guardrails"].values())
    assert result["stretch"] == {"reach": 0.404, "value": 0.503, "reached": True}
    a = result["answers"]
    assert a[Q4]["labels"] == {"yes": 420, "no": 558, "maybe": 22} and a[Q4]["unparsed"] == 0


def test_the_non_copy_headline_and_reported_signals():
    c = computed()["copies"]
    assert (c[Q4]["copied"], c[V2]["copied"], c[BASE]["copied"]) == (0.191, 0.107, 0.271)
    assert c[Q4]["non_copy_headline"] > c[BASE]["non_copy_headline"] > c[V2]["non_copy_headline"]
    assert (c[Q4]["non_copy_n"], c[V2]["non_copy_n"]) == (809, 893)
    ex = computed()["exploratory"]
    assert ex[f"{Q4} minus {BASE}"]["headline"]["point"] == 0.099
    assert ex[f"{Q8} minus {Q4}"]["headline"]["point"] == 0.038
    assert computed()["unjudged_citations"] == {BASE: 0, V2: 0, Q4: 2, Q8: 2}


def test_the_result_file_matches_the_run():
    committed = json.loads((ROOT / R.RESULT).read_text(encoding="utf-8"))
    assert committed == computed()
    assert committed["run_id"] == run_json()["run_id"] == "r-66616600de17"


def _q8_rows():
    raw = gzip.decompress((V3B / "parity_q8_0_generations.jsonl.gz").read_bytes())
    return [json.loads(x) for x in raw.decode("utf-8").splitlines() if x]


def test_the_q8_0_parity_check_rebuilds_from_its_generations():
    from ggh.export import parity

    rep = json.loads((V3B / "parity_q8_0.json").read_text(encoding="utf-8"))
    rows = _q8_rows()
    rebuilt = parity.build_report([r["bf16"] for r in rows], [r["q8_0"] for r in rows])
    assert json.loads(rebuilt.model_dump_json()) == rep["q8_0_vs_bf16"]["report"]
    assert rebuilt.label_agreement == 0.99 and parity.failures(rebuilt) == []
    assert rep["q8_0_vs_bf16"]["label_accuracy"] == 0.904 and "exploratory" in rep["note"]


def test_the_q8_0_parity_check_reuses_the_registered_bf16_replies():
    raw = gzip.decompress((V3B / "parity_generations.jsonl.gz").read_bytes()).decode("utf-8")
    registered = [json.loads(x) for x in raw.splitlines() if x]
    rows = _q8_rows()
    assert [r["pmid"] for r in rows] == [r["pmid"] for r in registered]
    assert [r["bf16"] for r in rows] == [r["bf16"] for r in registered]
    rep = json.loads((V3B / "parity_q8_0.json").read_text(encoding="utf-8"))
    q4 = json.loads((V3B / "parity.json").read_text(encoding="utf-8"))
    assert rep["q4_k_m_vs_bf16"]["report"]["label_agreement"] == q4["label_agreement"] == 0.924
