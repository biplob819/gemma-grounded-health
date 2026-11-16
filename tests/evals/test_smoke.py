import json
from pathlib import Path

from evalkit.run import read_records

from ggh.serve import checks

ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "docs" / "measurements" / "dev_smoke_base.json"
LEDGER = ROOT / "docs" / "measurements" / "spend-ledger.jsonl"
SPEC = ROOT / "Gemma Grounded Health Tech Spec.md"
THROUGHPUT = ROOT / "docs" / "measurements" / "serving_throughput.json"


def smoke():
    return json.loads(SMOKE.read_text(encoding="utf-8"))


def test_smoke_measurement_is_in_the_spend_ledger():
    s = smoke()
    entries = [json.loads(x) for x in LEDGER.read_text(encoding="utf-8").splitlines() if x]
    (e,) = [x for x in entries if x.get("run_id") == s["run_id"]]
    assert e["kind"] == "dev_smoke" and e["spent_usd"] == s["spent_usd"] and e["node"] == "E3.1"
    assert s["usd_per_answer_extract_and_dual_judge"] == round(s["spent_usd"] / s["n"], 5)


def test_cost_model_in_section_11_quotes_the_measured_numbers():
    s = smoke()
    tp = json.loads(THROUGHPUT.read_text(encoding="utf-8"))
    sec = SPEC.read_text(encoding="utf-8").split("## 11. Cost and compute model")[1]
    sec = sec.split("\n## ")[0]
    assert f"{s['claims_per_answer']:.2f} claims per answer" in sec
    assert f"${s['spent_usd']:.3f}" in sec
    assert f"about {tp['seconds_per_answer']:.2f} seconds per answer" in sec
    assert f"about {round(2 * 1000 * tp['seconds_per_answer'] / 60)} minutes" in sec
    four = next(r for r in tp["runs"] if r["in_flight"] == 4)
    assert f"{four['latency_mean_s']:.1f} s mean latency" in sec
    assert f"{s['seconds_per_answer']['mean']:.1f} seconds" in sec and "queue-inflated" in sec


def test_throughput_is_flat_across_in_flight_levels_and_latency_is_the_queue():
    tp = json.loads(THROUGHPUT.read_text(encoding="utf-8"))
    runs = {r["in_flight"]: r for r in tp["runs"]}
    assert (
        set(runs) == set(checks.IN_FLIGHT)
        and tp["seconds_per_answer"] == runs[1]["seconds_per_answer"]
    )
    spa = [r["seconds_per_answer"] for r in runs.values()]
    assert max(spa) < 1.2 * min(spa)  # one slot: throughput does not depend on the queue
    assert runs[1]["latency_mean_s"] < runs[4]["latency_mean_s"] < runs[20]["latency_mean_s"]
    for r in runs.values():
        assert r["seconds_per_answer"] == round(r["wall_s"] / 20, 3)


def test_smoke_behaviour_shows_no_b1_or_b4_failures_for_the_constrained_base():
    b = smoke()["behaviour"]["gemma-e2b-base-q4"]
    assert b["B1"]["failures"] == 0 and b["B4"]["over_80_words"] == 0 and b["B1"]["n"] == 20


RUN = ROOT / "docs" / "results" / "dev_smoke_base"  # the EvalKit run folder, committed


def test_smoke_numbers_recompute_from_the_committed_run():
    from evalkit.run import read_cases

    from ggh.evals import behaviour

    s = smoke()
    assert json.loads((RUN / "run.json").read_text(encoding="utf-8"))["run_id"] == s["run_id"]
    rs = read_records(RUN)
    assert len(rs) == s["n"] == 20
    assert round(sum(len(r.claims) for r in rs) / len(rs), 3) == s["claims_per_answer"]
    lat = sorted(r.completion.latency_ms / 1000 for r in rs)
    assert round(sum(lat) / len(lat), 3) == s["seconds_per_answer"]["mean"]
    rep = json.loads(json.dumps(behaviour.report(rs, read_cases(RUN))))  # tuples become lists
    assert rep == s["behaviour"]


def test_evalkit_run_outputs_equal_the_serve_check_outputs():
    """Determinism across request paths: the dev smoke went through EvalKit's runner with 4 in
    flight, the serve check one at a time; greedy decoding on one slot gives the same bytes."""
    rec = {r.case_id: r.completion.raw_text for r in read_records(RUN)}
    sc = json.loads((ROOT / checks.OUT).read_text(encoding="utf-8"))
    served = {r["id"]: r["constrained"]["raw_text"] for r in sc["structured"]}
    assert rec == served and len(rec) == 20
