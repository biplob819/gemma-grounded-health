"""RUN-01 to RUN-07 on the committed E3.2 evidence: the pre-registered plan, the estimates and
spend, the main, parity, off-scope and probe runs."""

import gzip
import hashlib
import json
from pathlib import Path

import pytest
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


RUN_DIRS = {"main": "main", "offscope": "offscope", "probe_original": "probe_original",
            "probe_paraphrase": "probe_paraphrase", "parity": "parity",
            "main_citation": "main_citation", "dev_ground_v1": "dev_ground_v1",
            "dev_ground_v2_sol": "dev_ground_v2_sol",
            "dev_ground_v2_dual_partial": "dev_ground_v2_dual_partial"}  # fmt: skip


def test_every_paid_run_is_in_the_ledger_with_its_own_cost():
    rows = {r["run_id"]: r for r in ledger() if r.get("run_id")}
    seen = set()
    for run in RUN_DIRS.values():
        r = load(run, "run.json")
        assert rows[r["run_id"]]["spent_usd"] == round(r["spent_usd"], 6), run
        if "live_judges" in r:
            assert rows[r["run_id"]]["live_judges"] == r["live_judges"], run
        seen.add(r["run_id"])
    assert set(rows) - {"r-044ca4838ee9"} == seen  # every logged run (E3.1's smoke aside) is here
    main = rows[load("main", "run.json")["run_id"]]
    assert main["estimate_usd"] == 63.65 and main["spent_usd"] < 70  # max_cost_usd, OWNER D-17
    for est in ("main $63.65", "off-scope $1.27", "probe original $3.16", "parity $3.16"):
        assert est in RUN_STATE


def test_only_approved_runs_crossed_the_owner_approval_line():
    big = sorted(r["kind"] for r in ledger() if r["spent_usd"] > 25)
    assert big == ["main", "rescore"]  # main: OWNER D-17; rescore: owner approval, cap $65
    assert "Owner approved all four models" in RUN_STATE and "raised the cap to $65" in RUN_STATE


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


KNOWN_CITATION_GAPS = 9  # main_citation, all GPT-6.1 Sol (EvalKit 'complete but missing')


def test_citation_support_covers_every_cited_sentence_except_the_disclosed_gaps():
    gaps = {}
    for run in ("main_citation", "dev_ground_v1"):
        for r in records(run):
            cs = r.get("citation_support") or {}
            for c in r["claims"]:
                for j in JUDGES:
                    got = {e["n"] for e in cs.get(j, []) if e["claim_id"] == c["id"]}
                    if set(c["cited"]) - got:
                        gaps.setdefault(run, set()).add((c["id"], j))
    assert set(gaps) == {"main_citation"} and len(gaps["main_citation"]) == KNOWN_CITATION_GAPS
    assert all(j == "gpt-6.1-sol" for _, j in gaps["main_citation"])
    assert "9 claim-judge pairs" in RUN_STATE


# VTG-05: the v2 dev check, judged by GPT-6.1 Sol only (OWNER D-54)


def test_v2_dev_check_on_sol_is_complete_and_scored_against_the_d52_bars():
    import yaml

    bars = yaml.safe_load((ROOT / "configs/v2_dev_bars.yaml").read_text(encoding="utf-8"))
    s = load("dev_ground_v2_sol", "summary.json")
    assert s["judging_status"] == "complete" and s["task"] == "ggh_dev_ground"
    (m,) = s["models"]
    assert m["model"] == "gemma-e2b-grounded-v2-q4" and m["n"] == 200
    j = "gpt-6.1-sol"
    prec, aim = m["citation_precision"][j]["point"], m["aim_citation_rate"][j]["point"]
    head, acc = m["headline"][j]["point"], m["accuracy"]["point"]
    assert prec >= bars["bars"]["citation_precision_min"]  # 0.768, v1 0.404
    assert aim <= bars["bars"]["aim_citation_rate_max"]  # 0.067, v1 0.590
    v1 = load("dev_ground_v1", "summary.json")["models"]
    (v1m,) = [x for x in v1 if x["model"] == "gemma-e2b-grounded-q4"]
    assert head - v1m["headline"][j]["point"] < bars["bars"]["headline_min_vs_v1"]  # disclosed
    assert acc < bars["bars"]["accuracy_min"]  # 0.795 vs 0.80: disclosed miss
    v21 = (ROOT / "specs/V2.1-retrain/RUN-STATE.md").read_text(encoding="utf-8")
    assert "v2 dev check on Sol" in v21


def test_v2_dev_check_has_citation_support_for_every_cited_sentence():
    for r in records("dev_ground_v2_sol"):
        cs = r.get("citation_support") or {}
        for c in r["claims"]:
            got = {e["n"] for e in cs.get("gpt-6.1-sol", []) if e["claim_id"] == c["id"]}
            assert set(c["cited"]) <= got, (r.get("case_id"), c["id"])


def test_v2_opus_headline_is_bounded_below_v1_by_the_partial_verdicts():
    j = "claude-opus-5-5"
    lo = hi = 0
    unjudged = 0
    for r in records("dev_ground_v2_dual_partial"):
        if not r["scores"]["label_correct"] or not r["claims"]:
            continue
        got = {v["claim_id"]: v["verdict"] for v in r["verdicts"] if v["judge"] == j}
        ids = [c["id"] for c in r["claims"]]
        unjudged += sum(i not in got for i in ids)
        lo += all(got.get(i) == "supported" for i in ids)
        hi += all(got.get(i, "supported") == "supported" for i in ids)
    v1 = load("dev_ground_v1", "summary.json")["models"]
    (v1m,) = [x for x in v1 if x["model"] == "gemma-e2b-grounded-q4"]
    assert unjudged == 2 and (lo / 200, hi / 200) == (0.44, 0.445)
    assert hi / 200 < v1m["headline"][j]["point"] == 0.465
    assert "0.440 (the 2 unjudged claims" in (ROOT / "specs/V2.1-retrain/RUN-STATE.md").read_text(
        encoding="utf-8"
    )


def test_v2_dev_bars_were_committed_before_either_v2_dev_run_started():
    import datetime as dt
    import subprocess

    # 8ff61b9 added the bars file; a shallow CI clone lacks it, so the check runs locally
    cmd = ["git", "show", "-s", "--format=%cI", "8ff61b9"]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0 or not out.stdout.strip():
        pytest.skip("8ff61b9 not in this clone (shallow)")
    files = subprocess.run(["git", "show", "--name-status", "--format=", "8ff61b9"], cwd=ROOT,
                           capture_output=True, text=True).stdout  # fmt: skip
    assert "A\tconfigs/v2_dev_bars.yaml" in files
    added = dt.datetime.fromisoformat(out.stdout.strip())
    for run in ("dev_ground_v2_dual_partial", "dev_ground_v2_sol"):
        started = load(run, "run.json")["started_at"].replace("Z", "+00:00")
        assert added < dt.datetime.fromisoformat(started), run
