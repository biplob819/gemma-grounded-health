"""RES-01, RES-04, RES-06 on the committed v1 data: `ggh results bundle`, the behaviour table, and
the v2 path on a synthetic v2 (tests/results/fake.py). No artifacts/, no network, no torch."""

import json

import pytest

from ggh.cli import main
from ggh.results import bundle as B
from ggh.results.runs import ResultsError
from ggh.schemas import ResultsBundle
from tests.results import fake

ROOT = fake.ROOT
JUDGES = ("claude-opus-5-5", "gpt-6.1-sol")
FOUR = ("gemma-e2b-base-q4", "gemma-e2b-grounded-q4", "claude-haiku-5-5", "gpt-6-luna")


@pytest.fixture(scope="module")
def v1():
    return B.build(ROOT)


@pytest.fixture(scope="module")
def v2(tmp_path_factory):
    root = fake.copy_repo(tmp_path_factory.mktemp("v2"))
    fake.add_v2(root)
    return B.build(root)


def test_today_the_bundle_has_v1_only_with_every_committed_run(v1):
    b = v1.bundle
    assert [r.version for r in b.releases] == ["v1"]
    assert [r.name for r in b.runs] == sorted(
        ["main", "main_citation", "offscope", "parity", "probe_original", "probe_paraphrase"]
    )
    main = json.loads((ROOT / "docs/results/main/run.json").read_text(encoding="utf-8"))
    assert main["run_id"] in b.evalkit_run_ids and b.judge_mode == "dual"
    assert b.calibration_status == "uncalibrated" and b.gemma_family_check is None
    rel = b.releases[0]
    assert rel.plan_status == "final" and rel.built_after_test_seen is False
    assert rel.train.selected_step == 527 and rel.adapter_sha256 and rel.audits[0].passed
    assert rel.probe and rel.probe.contamination_signal is False
    assert rel.parity and rel.parity.label_agreement == 0.956


def test_primary_comparisons_are_the_plans_oriented_as_written(v1):
    prim = v1.bundle.primary_comparisons
    assert len(prim) == 6 and all(c.model_a == "gemma-e2b-grounded-q4" for c in prim)
    by = {(c.model_b, c.judge): c for c in prim}
    base_opus = by[("gemma-e2b-base-q4", "claude-opus-5-5")]
    # 0.388 - 0.418: EvalKit stores base minus v1, the plan says v1 minus base
    assert base_opus.diff.point == pytest.approx(-0.03) and base_opus.verdict == "tie"
    assert by[("gemma-e2b-base-q4", "gpt-6.1-sol")].verdict == "b_better"
    assert by[("claude-haiku-5-5", "gpt-6.1-sol")].verdict == "a_better"
    assert {c.plan for c in prim} == {"configs/eval_plan.yaml"}
    assert {c.plan for c in v1.bundle.secondary} == {"secondary"}


def test_numbers_are_copied_from_the_summaries(v1):
    res = {(r.run, r.model): r for r in v1.bundle.results}
    ft = res[("main", "gemma-e2b-grounded-q4")]
    assert round(ft.accuracy.point, 3) == 0.728 and ft.headline["claude-opus-5-5"].point == 0.388
    cit = res[("main_citation", "gemma-e2b-grounded-q4")]
    assert (
        cit.citation_precision
        and round(cit.citation_precision["claude-opus-5-5"].point, 3) == 0.338
    )
    assert ft.citation_precision is None  # the E3.2 main run had no citation_support


def test_behaviour_table_covers_b1_to_b8_for_every_model_with_sources(v1):
    rows = v1.bundle.behaviour
    for m in FOUR:
        assert {r.rule for r in rows if r.model == m} == {f"B{i}" for i in range(1, 9)}, m
    b6 = {(r.model, r.metric): r for r in rows if r.rule == "B6"}
    # RUN-STATE: the primary judge marks 525 of v1's 1,174 claims unsupported
    assert b6[("gemma-e2b-grounded-q4", "unsupported_claim_share:claude-opus-5-5")].value == (
        pytest.approx(525 / 1174)
    )
    pending = {(r.rule, r.model) for r in rows if r.value is None}
    assert pending == {(rule, m) for rule in ("B5", "B8") for m in FOUR}  # owner reviews to come
    assert all(r.source == "owner_review" for r in rows if r.value is None)
    md = v1.behaviour_md
    assert "| B7 | citation_precision:claude-opus-5-5 |" in md and md.count("### ") == 3


def test_lineage_is_checked_on_the_committed_manifests(v1):
    b = v1.bundle
    assert b.lineage_ok and b.lineage_scope == "committed" and b.lineage_problems == ()


def test_publish_blockers_name_what_is_still_owed(v1):
    text = "\n".join(v1.bundle.publish_blockers)
    assert "uncalibrated" in text and "B8 gives_advice_count" in text and "B5" in text


def test_spend_is_the_ledger_total(v1):
    rows = [
        json.loads(x)
        for x in (ROOT / "docs/measurements/spend-ledger.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if x
    ]
    assert v1.bundle.spend_usd == pytest.approx(sum(r["spent_usd"] for r in rows))


def test_the_bundle_is_deterministic_and_round_trips(v1, tmp_path):
    again = B.build(ROOT)
    assert B.digest(again) == B.digest(v1)
    paths = B.write(ROOT, v1, tmp_path)
    loaded = ResultsBundle.model_validate_json(paths[0].read_text(encoding="utf-8"))
    assert loaded == v1.bundle
    assert json.loads(paths[1].read_text(encoding="utf-8"))[0]["rule"] == "B1"


def test_cli_writes_the_bundle(tmp_path, capsys):
    assert main(["results", "bundle", "--root", str(ROOT), "--out", str(tmp_path)]) == 0
    assert (tmp_path / "results_bundle.json").is_file()
    assert "releases: v1" in capsys.readouterr().out


# v2 (synthetic, from v1's records)


def test_v2_appears_when_its_main_run_exists(v2):
    b = v2.bundle
    assert [r.version for r in b.releases] == ["v1", "v2"]
    r2 = b.releases[1]
    assert r2.model == fake.V2 and r2.plan_status == "final" and r2.plan_sha256
    assert r2.built_after_test_seen is True and len(r2.audits) == 3
    assert {"main_v2", "offscope_v2", "parity_v2"} <= {r.name for r in b.runs}


def test_v2_primary_family_is_the_owners_d53_family_under_one_judge(v2):
    p2 = [c for c in v2.bundle.primary_comparisons if c.plan == "configs/eval_plan_v2.yaml"]
    # 2 pairs (v1, base) x 3 outcomes x Sol alone (OWNER D-57), Holm over those 6
    assert len(p2) == 6 and {c.family_size for c in p2} == {6}
    assert {c.judge for c in p2} == {"gpt-6.1-sol"}
    assert {c.model_b for c in p2} == {fake.V1, "gemma-e2b-base-q4"}
    assert all(
        c.model_a == fake.V2 and c.run == "main_v2" and c.method == "plan_family" for c in p2
    )
    assert {(c.metric, c.better) for c in p2} == {
        ("headline", "higher"),
        ("citation_precision", "higher"),
        ("aim_citation_rate", "lower"),
    }
    # the fake v2 is v1's records, so v2 minus v1 is exactly zero: a tie on every outcome
    vs_v1 = [c for c in p2 if c.model_b == fake.V1]
    assert all(c.diff.point == 0 and c.verdict == "tie" for c in vs_v1)
    # against base, v1's citations are worse on both citation outcomes, whatever the direction
    by = {(c.metric, c.judge): c for c in p2 if c.model_b == "gemma-e2b-base-q4"}
    aim = by[("aim_citation_rate", "gpt-6.1-sol")]
    assert aim.diff.point == pytest.approx(0.5671 - 0.1417, abs=1e-3) and aim.verdict == "b_better"
    assert by[("citation_precision", "gpt-6.1-sol")].verdict == "b_better"
    key = f"aim_citation_rate: {fake.V2} minus gemma-e2b-base-q4 (gpt-6.1-sol)"
    assert v2.bundle.primary[key] == aim.diff


def test_v2_is_never_compared_with_haiku_or_luna(v2):
    # OWNER D-57: they are not re-scored, so no v2 comparison with them exists anywhere
    every = [*v2.bundle.primary_comparisons, *v2.bundle.secondary]
    v2_pairs = {
        c.model_b for c in every if c.run == "main_v2" and fake.V2 in (c.model_a, c.model_b)
    }
    assert not v2_pairs & {"claude-haiku-5-5", "gpt-6-luna"}
    sec = [c for c in v2.bundle.secondary if c.run == "main_v2" and c.model_a == fake.V2]
    assert all(c.method == "evalkit" and c.plan == "secondary" for c in sec)


def test_r4_flat_fields_read_from_the_shipped_release(v1, v2):
    for built in (v1, v2):
        b = built.bundle
        assert b.train == b.releases[-1].train and b.artifacts == b.releases[-1].artifacts
        assert b.audit_pass_rate == b.releases[-1].audits[-1].pass_rate
        assert b.bench.quant == "Q4_K_M" and {"ft-Q4_K_M", "base-Q4_K_M"} <= set(b.bench_variants)
    p = v1.bundle.parity
    assert p and p.judged_headline_delta
    assert p.judged_headline_delta["claude-opus-5-5"].point == 0.025
    assert set(v1.bundle.primary) == {
        f"headline: gemma-e2b-grounded-q4 minus {m} ({j})"
        for m in ("gemma-e2b-base-q4", "claude-haiku-5-5", "gpt-6-luna")
        for j in JUDGES
    }


def test_v2_behaviour_rows_include_its_citation_metrics_and_offscope(v2):
    rows = [r for r in v2.bundle.behaviour if r.model == fake.V2]
    assert {r.rule for r in rows} == {f"B{i}" for i in range(1, 9)}
    assert any(r.metric == "aim_citation_rate:gpt-6.1-sol" and r.run == "main_v2" for r in rows)
    assert {r.run for r in rows if r.rule == "B8"} == {"offscope_v2"}


def test_v2_lineage_checks_the_gguf_its_answers_came_from(v2):
    assert v2.bundle.lineage_ok, v2.bundle.lineage_problems


def test_a_v2_main_run_without_its_side_runs_is_refused(tmp_path):
    root = fake.copy_repo(tmp_path)
    fake.add_v2(root)
    import shutil

    shutil.rmtree(root / "docs/results/probe_paraphrase_v2")
    with pytest.raises(ResultsError, match="probe_paraphrase_v2 has no results"):
        B.build(root)


def test_answers_from_another_gguf_break_lineage(tmp_path):
    root = fake.copy_repo(tmp_path)
    fake.add_v2(root)
    m = root / "docs/measurements/export_artifacts_v2.json"
    m.write_text(m.read_text(encoding="utf-8").replace("ft-v2-Q4_K_M", "ft-v3-Q4_K_M"), "utf-8")
    b = B.build(root).bundle
    assert not b.lineage_ok
    assert any(
        "ft-v2-Q4_K_M.gguf is not in the release's export records" in p for p in b.lineage_problems
    )
