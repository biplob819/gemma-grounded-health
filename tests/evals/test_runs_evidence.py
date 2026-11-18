"""RUN-01 to RUN-07 on the committed E3.2 evidence: the pre-registered plan, the estimates and
spend, the main, parity, off-scope and probe runs."""

import gzip
import hashlib
import json
from pathlib import Path

import yaml

from ggh.evals import probe as P

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "docs/results"
M = ROOT / "docs/measurements"
RUN_STATE = (ROOT / "specs/E3.2-runs/RUN-STATE.md").read_text(encoding="utf-8")
GEMMA = ("gemma-e2b-base-q4", "gemma-e2b-grounded-q4")
FOUR = (*GEMMA, "claude-haiku-5-5", "gpt-6-luna")
JUDGES = ["claude-opus-5-5", "gpt-6.1-sol"]


def load(run, name):
    return json.loads((RES / run / name).read_text(encoding="utf-8"))


def records(run):
    raw = gzip.decompress((RES / run / "records.jsonl.gz").read_bytes()).decode("utf-8")
    return [json.loads(x) for x in raw.splitlines() if x]


def ledger():
    return [json.loads(x) for x in (M / "spend-ledger.jsonl").read_text(encoding="utf-8")
            .splitlines() if x]  # fmt: skip


# RUN-01


def test_the_plan_has_the_committed_hash_and_the_three_primary_comparisons():
    text = (ROOT / "configs/eval_plan.yaml").read_bytes()
    assert (
        hashlib.sha256(text).hexdigest()
        == (M / "eval_plan.sha256").read_text(encoding="utf-8").strip()
    )
    plan = yaml.safe_load(text)
    assert plan["metric"] == "headline" and plan["judges"] == JUDGES and plan["n_cases"] == 1000
    ft = "gemma-e2b-grounded-q4"
    assert plan["primary"] == [
        [ft, "gemma-e2b-base-q4"],
        [ft, "claude-haiku-5-5"],
        [ft, "gpt-6-luna"],
    ]
    assert "Holm" in plan["rule"] and "tie" in plan["rule"]


def test_the_plan_was_committed_before_the_main_run_started():
    import datetime as dt
    import re

    m = re.search(r"9bc0f97 \(plan and its hash\) committed at (\S+?Z)", RUN_STATE)
    assert m, "RUN-STATE must record when the plan was committed"
    committed = dt.datetime.fromisoformat(m[1].replace("Z", "+00:00"))
    started = dt.datetime.fromisoformat(
        load("main", "run.json")["started_at"].replace("Z", "+00:00")
    )
    assert committed < started
    import subprocess

    out = subprocess.run(["git", "log", "-1", "--format=%cI", "9bc0f97"], cwd=ROOT,
                         capture_output=True, text=True)  # fmt: skip
    if out.returncode == 0 and out.stdout.strip():  # absent in a shallow CI clone
        assert dt.datetime.fromisoformat(out.stdout.strip()) == committed


# RUN-02 and RUN-07


def test_every_paid_run_is_in_the_ledger_with_its_own_cost():
    rows = {r.get("run_id"): r for r in ledger() if r.get("node") == "E3.2" and r.get("run_id")}
    for run in ("main", "offscope", "probe_original", "probe_paraphrase", "parity"):
        r = load(run, "run.json")
        assert rows[r["run_id"]]["spent_usd"] == round(r["spent_usd"], 6), run
    main = rows[load("main", "run.json")["run_id"]]
    assert main["estimate_usd"] == 63.65 and main["spent_usd"] < 70  # max_cost_usd, OWNER D-17
    for est in ("main $63.65", "off-scope $1.27", "probe original $3.16", "parity $3.16"):
        assert est in RUN_STATE


def test_only_the_main_run_crossed_the_owner_approval_line():
    big = [r for r in ledger() if r.get("node") == "E3.2" and r["spent_usd"] > 25]
    assert [r["kind"] for r in big] == ["main"]  # covered by OWNER D-17


# RUN-03


def test_main_run_is_four_models_on_all_1000_cases_with_both_judges():
    run, s = load("main", "run.json"), load("main", "summary.json")
    assert run["n"] == 1000 and run["seed"] == 7 and run["task"] == "pubmedqa"
    assert run["models"] == list(FOUR) and run["judges"] == JUDGES and run["judge_mode"] == "dual"
    assert s["judging_status"] == "complete" and [m["model"] for m in s["models"]] == list(FOUR)
    recs = records("main")
    assert len(recs) == 4000 and len({r["case_id"] for r in recs}) == 1000
    assert all({v["judge"] for v in r["verdicts"]} <= set(JUDGES) for r in recs)
    assert all(m["n_errors"] == {} for m in s["models"])


def test_the_recorded_headline_numbers_are_the_summary():
    s = {m["model"]: m for m in load("main", "summary.json")["models"]}
    ft = s["gemma-e2b-grounded-q4"]
    assert round(ft["accuracy"]["point"], 3) == 0.728
    assert round(ft["headline"]["claude-opus-5-5"]["point"], 3) == 0.388
    assert round(s["gemma-e2b-base-q4"]["headline"]["claude-opus-5-5"]["point"], 3) == 0.418
    assert "fine-tuned 0.388 / 0.330" in RUN_STATE


# RUN-05


def test_offscope_run_covers_the_20_frozen_prompts_for_four_models():
    run = load("offscope", "run.json")
    assert run["n"] == 20 and run["models"] == list(FOUR) and run["judges"] == JUDGES
    recs = records("offscope")
    ids = {json.loads(x)["id"] for x in (ROOT / "data/offscope_task.jsonl")
           .read_text(encoding="utf-8").splitlines() if x}  # fmt: skip
    assert len(recs) == 80 and {r["case_id"] for r in recs} == ids


# RUN-06


def test_the_probe_ran_both_wordings_on_the_same_100_cases():
    o, p = records("probe_original"), records("probe_paraphrase")
    assert len(o) == len(p) == 200
    assert {r["case_id"] for r in o} == {r["case_id"] for r in p}
    kept = [json.loads(x) for x in (ROOT / P.PROBE).read_text(encoding="utf-8").splitlines() if x]
    assert len(kept) == 100 and {f"probe-{r['pubid']}" for r in kept} == {r["case_id"] for r in o}


def test_the_probe_report_recomputes_and_shows_no_contamination_signal():
    rep = json.loads((M / "probe_report.json").read_text(encoding="utf-8"))
    o, p = records("probe_original"), records("probe_paraphrase")
    for m in GEMMA:
        got = P.paired_drop(P.correct_by_case(o, m), P.correct_by_case(p, m))
        assert got == rep["models"][m] and not got["contamination_signal"]


# RUN-04


def test_parity_ran_the_bf16_finetune_on_200_cases_of_the_main_run():
    run, s = load("parity", "run.json"), load("parity", "summary.json")
    assert (
        run["n"] == 200 and run["models"] == ["gemma-e2b-grounded-bf16"] and run["judges"] == JUDGES
    )
    assert s["judging_status"] == "complete"
    par = records("parity")
    main_ids = {r["case_id"] for r in records("main")}
    assert len(par) == 200 and {r["case_id"] for r in par} <= main_ids


def test_the_headline_difference_against_q4_recomputes_and_is_reported():
    rep = json.loads((M / "parity_headline.json").read_text(encoding="utf-8"))
    par = records("parity")
    q4 = {
        r["case_id"]: r
        for r in records("main")
        if r["completion"]["model"] == "gemma-e2b-grounded-q4"
    }
    for j in JUDGES:
        bf = sum(bool(r["scores"]["correct_and_grounded"][j]) for r in par) / 200
        qq = sum(bool(q4[r["case_id"]]["scores"]["correct_and_grounded"][j]) for r in par) / 200
        assert (
            round(bf, 3) == rep["bf16"][f"headline_{j}"]
            and round(qq, 3) == rep["q4_k_m"][f"headline_{j}"]
        )
        d = rep["bf16_minus_q4_headline"][j]
        assert d["lo"] <= d["point"] <= d["hi"] and d["point"] == round(bf - qq, 3)
    assert "Reported, not gated" in RUN_STATE


KNOWN_MISSING_VERDICTS = {
    ("main", "pqa-23792130:claude-haiku-5-5:1", "claude-opus-5-5"),
    ("parity", "pqa-26852225:gemma-e2b-grounded-bf16:1", "gpt-6.1-sol"),
}


def test_every_claim_has_both_verdicts_except_the_two_disclosed():
    missing = set()
    for run in ("main", "parity", "offscope", "probe_original", "probe_paraphrase"):
        for r in records(run):
            have = {(v["claim_id"], v["judge"]) for v in r["verdicts"]}
            for c in r["claims"]:
                for j in JUDGES:
                    if (c["id"], j) not in have:
                        missing.add((run, c["id"], j))
    assert missing == KNOWN_MISSING_VERDICTS
    assert "Two verdicts are missing" in RUN_STATE
