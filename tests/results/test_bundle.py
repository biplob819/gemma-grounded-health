"""RES-01, RES-04, RES-06 on the committed data (v1 from E3.2, the real v2 from V2.1 and V2.2):
`ggh results bundle`, the behaviour table, and the v2 path again on a synthetic v2 whose answers
are known (tests/results/fake.py). No artifacts/, no network, no torch."""

import json

import pytest
import yaml

from ggh.cli import main
from ggh.results import bundle as B
from ggh.results.runs import ResultsError
from ggh.schemas import ResultsBundle
from tests.results import fake

ROOT = fake.ROOT
JUDGES = ("claude-opus-5-5", "gpt-6.1-sol")
FOUR = ("gemma-e2b-base-q4", "gemma-e2b-grounded-q4", "claude-haiku-5-5", "gpt-6-luna")
SOL = "gpt-6.1-sol"
V1_RUNS = ("main", "main_citation", "offscope", "parity", "probe_original", "probe_paraphrase")
V2_RUNS = ("main_v2", "offscope_v2", "parity_v2", "probe_original_v2", "probe_paraphrase_v2")
V3B_RUNS = ("main_v3b", "offscope_v3b", "probe_original_v3b", "probe_paraphrase_v3b")
V3B, V3B_Q8 = "gemma-e2b-grounded-v3b-q4", "gemma-e2b-grounded-v3b-q8"


@pytest.fixture(scope="module")
def real():
    """The committed repo: v1, the real v2 (OWNER D-57: its runs judged by Sol alone) and v3b
    (V3.8, OWNER D-75)."""
    return B.build(ROOT)


@pytest.fixture(scope="module")
def v2(tmp_path_factory):
    root = fake.copy_repo(tmp_path_factory.mktemp("v2"))
    fake.add_v2(root)
    return B.build(root)


def test_the_bundle_has_v1_and_the_real_v2_with_every_committed_run(real):
    b = real.bundle
    assert [r.version for r in b.releases] == ["v1", "v2", "v3b"]
    assert [r.name for r in b.runs] == sorted([*V1_RUNS, *V2_RUNS, *V3B_RUNS])
    for name in ("main", "main_v2", "main_v3b"):
        run = json.loads((ROOT / f"docs/results/{name}/run.json").read_text(encoding="utf-8"))
        assert run["run_id"] in b.evalkit_run_ids
    by = {r.name: r for r in b.runs}
    assert {by[n].judge_mode for n in V1_RUNS} == {"dual"}
    assert {by[n].judge_mode for n in (*V2_RUNS, *V3B_RUNS)} == {"single"}
    assert b.judge_mode == "single"  # dual only when every run is
    # RES-03 (OWNER D-59, D-60): v2 relies on Sol, stamped; v1 also on Opus, never stamped
    assert b.calibration_status == "uncalibrated" and b.gemma_family_check is None
    assert [r.calibration_status for r in b.releases] == [
        "uncalibrated",
        "calibrated",
        "calibrated",  # v3b relies on Sol alone too
    ]
    rel = b.releases[0]
    assert rel.plan_status == "final" and rel.built_after_test_seen is False
    assert rel.train.selected_step == 527 and rel.adapter_sha256 and rel.audits[0].passed
    assert rel.probe and rel.probe.contamination_signal is False
    assert rel.parity and rel.parity.label_agreement == 0.956
    assert rel.parity.judged_headline_delta
    assert rel.parity.judged_headline_delta["claude-opus-5-5"].point == 0.025


def test_the_real_v2_release_reads_v21s_records_and_v22s_runs(real):
    r2 = real.bundle.releases[1]
    sel = json.loads(
        (ROOT / "docs/measurements/v2/train_selection.json").read_text(encoding="utf-8")
    )
    assert r2.model == "gemma-e2b-grounded-v2-q4" and sorted(r2.runs) == sorted(V2_RUNS)
    assert r2.plan == "configs/eval_plan_v2.yaml" and r2.plan_status == "final" and r2.plan_sha256
    assert r2.built_after_test_seen is True and len(r2.audits) == 3
    assert r2.train.selected_step == 340 and r2.adapter_sha256 == sel["adapter_sha256"]
    # OWNER D-55: 0.92 label agreement misses the 0.95 bar; disclosed, kept as measured
    assert r2.parity and r2.parity.label_agreement == 0.92
    assert r2.parity.judged_headline_delta and set(r2.parity.judged_headline_delta) == {SOL}
    assert r2.parity.judged_headline_delta[SOL].point == -0.03
    assert r2.probe and r2.probe.model == r2.model and r2.probe.n == 100
    assert r2.probe.drop.point == 0.0 and r2.probe.contamination_signal is False
    q4 = [a for a in r2.artifacts if a.path == "artifacts/gguf/ft2-Q4_K_M.gguf"]
    assert len(q4) == 1 and q4[0].kind == "gguf" and q4[0].quant == "Q4_K_M"


def test_primary_comparisons_are_the_plans_oriented_as_written(real):
    every = real.bundle.primary_comparisons
    prim = [c for c in every if c.plan == "configs/eval_plan.yaml"]
    assert len(prim) == 6 and all(c.model_a == "gemma-e2b-grounded-q4" for c in prim)
    by = {(c.model_b, c.judge): c for c in prim}
    base_opus = by[("gemma-e2b-base-q4", "claude-opus-5-5")]
    # 0.388 - 0.418: EvalKit stores base minus v1, the plan says v1 minus base
    assert base_opus.diff.point == pytest.approx(-0.03) and base_opus.verdict == "tie"
    assert by[("gemma-e2b-base-q4", "gpt-6.1-sol")].verdict == "b_better"
    assert by[("claude-haiku-5-5", "gpt-6.1-sol")].verdict == "a_better"
    assert {c.plan for c in every} == {
        "configs/eval_plan.yaml",
        "configs/eval_plan_v2.yaml",
        "configs/eval_plan_v3b.yaml",
    }
    assert {c.plan for c in real.bundle.secondary} == {"secondary"}


def test_the_real_v2_primary_family_is_the_six_sol_tests_from_main_v2(real):
    p2 = [c for c in real.bundle.primary_comparisons if c.plan == "configs/eval_plan_v2.yaml"]
    assert len(p2) == 6 and {c.family_size for c in p2} == {6} and {c.judge for c in p2} == {SOL}
    assert all(c.model_a == "gemma-e2b-grounded-v2-q4" and c.run == "main_v2" for c in p2)
    assert {c.model_b for c in p2} == {"gemma-e2b-grounded-q4", "gemma-e2b-base-q4"}
    assert all(c.method == "plan_family" for c in p2)


# the published v2 numbers (README, changelog, RUN-STATE, headline chart), pinned to the bundle
V2_FAMILY = {
    ("gemma-e2b-grounded-q4", "headline"): (-0.002, -0.034, 0.030, "tie"),
    ("gemma-e2b-grounded-q4", "citation_precision"): (0.374, 0.344, 0.403, "a_better"),
    ("gemma-e2b-grounded-q4", "aim_citation_rate"): (-0.476, -0.508, -0.443, "a_better"),
    ("gemma-e2b-base-q4", "headline"): (-0.075, -0.113, -0.037, "b_better"),
    ("gemma-e2b-base-q4", "citation_precision"): (-0.011, -0.046, 0.023, "tie"),
    ("gemma-e2b-base-q4", "aim_citation_rate"): (-0.054, -0.080, -0.028, "a_better"),
}


def test_the_real_v2_primary_family_has_the_published_numbers_and_verdicts(real):
    p2 = [c for c in real.bundle.primary_comparisons if c.plan == "configs/eval_plan_v2.yaml"]
    got = {
        (c.model_b, c.metric): (
            round(c.diff.point, 3),
            round(c.diff.lo, 3),
            round(c.diff.hi, 3),
            c.verdict,
        )
        for c in p2
    }
    assert got == V2_FAMILY
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for key, (point, *_) in V2_FAMILY.items():
        if key != ("gemma-e2b-base-q4", "citation_precision"):  # the README says "a tie"
            assert f"{point:+.3f}".replace("-", "−") in readme, key


def test_numbers_are_copied_from_the_summaries(real):
    res = {(r.run, r.model): r for r in real.bundle.results}
    ft = res[("main", "gemma-e2b-grounded-q4")]
    assert round(ft.accuracy.point, 3) == 0.728 and ft.headline["claude-opus-5-5"].point == 0.388
    cit = res[("main_citation", "gemma-e2b-grounded-q4")]
    assert (
        cit.citation_precision
        and round(cit.citation_precision["claude-opus-5-5"].point, 3) == 0.338
    )
    assert ft.citation_precision is None  # the E3.2 main run had no citation_support


def test_behaviour_table_covers_b1_to_b8_for_every_model_with_sources(real):
    rows = real.bundle.behaviour
    for m in FOUR:
        assert {r.rule for r in rows if r.model == m} == {f"B{i}" for i in range(1, 9)}, m
    b6 = {(r.model, r.metric): r for r in rows if r.rule == "B6"}
    # RUN-STATE: the primary judge marks 525 of v1's 1,174 claims unsupported
    assert b6[("gemma-e2b-grounded-q4", "unsupported_claim_share:claude-opus-5-5")].value == (
        pytest.approx(525 / 1174)
    )
    assert not [r for r in rows if r.value is None]  # B5 and B8 reviewed (OWNER D-61)
    assert all(r.source == "model_review" for r in rows if r.rule == "B5")
    advice = {
        (r.model, r.run): r.value for r in rows if r.metric == "gives_advice_count"
    }  # docs/measurements/b8_review.jsonl, a blind Claude Opus 5.5 review
    assert advice == {
        ("gemma-e2b-base-q4", "offscope"): 0.0,
        ("gemma-e2b-grounded-q4", "offscope"): 2.0,
        ("claude-haiku-5-5", "offscope"): 1.0,
        ("gpt-6-luna", "offscope"): 0.0,
        ("gemma-e2b-grounded-v2-q4", "offscope_v2"): 0.0,
        (V3B, "offscope_v3b"): 0.0,  # V3.8: both shipped GGUFs, reviewed blind
        (V3B_Q8, "offscope_v3b"): 0.0,
    }
    v2_models = ("gemma-e2b-base-q4", "gemma-e2b-grounded-q4", "gemma-e2b-grounded-v2-q4")
    assert {r.rule for r in rows if r.model == "gemma-e2b-grounded-v2-q4"} == {
        f"B{i}" for i in range(1, 9)
    }
    b5 = {r.model for r in rows if r.rule == "B5" and r.run == "main_v2"}
    assert b5 == set(v2_models)
    for m in (V3B, V3B_Q8):
        assert {r.rule for r in rows if r.model == m} == {f"B{i}" for i in range(1, 9)}, m
    b5_v3b = {r.model for r in rows if r.rule == "B5" and r.run == "main_v3b"}
    assert b5_v3b == {"gemma-e2b-base-q4", "gemma-e2b-grounded-v2-q4", V3B, V3B_Q8}
    # OWNER D-61: every model-review row names its reviewer and committed prompt (R4.1)
    for r in rows:
        if r.source == "model_review":
            assert r.reviewer and r.reviewer.startswith("claude-opus-5-5 (blind subagent"), r
            want = (
                "configs/sample_read_prompt.md"
                if r.rule == "B5"
                else ("configs/offscope_review_prompt.md")
            )
            assert r.review_prompt == want, r
        else:
            assert r.reviewer is None and r.review_prompt is None, r
    md = real.behaviour_md
    assert "| B7 | citation_precision:claude-opus-5-5 |" in md
    heads = [x for x in md.splitlines() if x.startswith("### ")]
    assert heads == [
        f"### {r}"
        for r in (
            "main",
            "main_citation",
            "main_v2",
            "main_v3b",
            "offscope",
            "offscope_v2",
            "offscope_v3b",
        )
    ]
    assert "| B7 | citation_precision:gpt-6.1-sol |" in md


def test_lineage_is_checked_on_the_committed_manifests(real):
    b = real.bundle
    assert b.lineage_ok and b.lineage_scope == "committed" and b.lineage_problems == ()


def test_publish_blockers_name_what_is_still_owed(real):
    text = "\n".join(real.bundle.publish_blockers)
    # the calibration input is committed and valid: a judge under the bar is disclosed in
    # `calibration`, not a blocker (OWNER D-59); tests/results/test_calibration.py breaks it
    assert not any(
        b.startswith("calibration") or "uncalibrated" in b for b in real.bundle.publish_blockers
    )
    # B5, B8 reviewed and every sample-read pipeline fault explained (sample_read_faults.md)
    assert text == ""


def test_spend_is_the_ledger_total(real):
    rows = [
        json.loads(x)
        for x in (ROOT / "docs/measurements/spend-ledger.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if x
    ]
    in_scope = [r for r in rows if not str(r.get("node", "")).startswith("V3.")]  # Wave 7 out
    assert [r.version for r in real.bundle.releases][-1] == "v3b"
    # OWNER D-75: with v3b a release, every paid run counts, V3.5's v3 scoring included
    assert real.bundle.spend_usd == pytest.approx(sum(r["spent_usd"] for r in rows))
    assert len(in_scope) < len(rows)
    # D-76: the v1 + v2 bundle leaves Wave 7 out, as v1.0.0 published it
    v1_v2 = B.build(ROOT, versions=("v1", "v2")).bundle
    assert v1_v2.spend_usd == pytest.approx(sum(r["spent_usd"] for r in in_scope)) == 118.035054
    cfg = yaml.safe_load((ROOT / B.CONFIG).read_text(encoding="utf-8"))
    assert B._in_scope(cfg, ("v1", "v2")) is not None
    assert B._in_scope(cfg, ("v1", "v2", "v3b")) is None


def test_the_bundle_is_deterministic_and_round_trips(real, tmp_path):
    again = B.build(ROOT)
    assert B.digest(again) == B.digest(real)
    paths = B.write(ROOT, real, tmp_path)
    loaded = ResultsBundle.model_validate_json(paths[0].read_text(encoding="utf-8"))
    assert loaded == real.bundle
    assert json.loads(paths[1].read_text(encoding="utf-8"))[0]["rule"] == "B1"


def test_cli_writes_the_bundle(tmp_path, capsys):
    assert main(["results", "bundle", "--root", str(ROOT), "--out", str(tmp_path)]) == 0
    assert (tmp_path / "results_bundle.json").is_file()
    assert "releases: v1, v2, v3b;" in capsys.readouterr().out


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


def test_every_run_names_its_extractor_and_judges(real, v2):
    # E3.2: Haiku extracts, Opus and Sol judge; V2.2 (OWNER D-57): Luna extracts, Sol alone
    real_by = {r.name: r for r in real.bundle.runs}
    assert {real_by[n].extractor for n in V1_RUNS} == {"claude-haiku-5-5"}
    for name in V1_RUNS:
        assert real_by[name].judges == JUDGES
    for name in V2_RUNS:
        r = real_by[name]
        assert r.extractor == "gpt-6-luna" and r.judges == (SOL,) and r.judge_mode == "single"
    assert set(real_by["main_v2"].models) == {"gemma-e2b-base-q4", fake.V1, fake.V2}
    by = {r.name: r for r in v2.bundle.runs}
    for name in ("main", "main_citation", "offscope"):
        assert by[name].extractor == "claude-haiku-5-5" and by[name].judges == JUDGES
    for name in ("main_v2", "offscope_v2", "probe_original_v2", "parity_v2"):
        assert by[name].extractor == "gpt-6-luna" and by[name].judges == ("gpt-6.1-sol",)
        assert by[name].judge_mode == "single"
    assert set(by["main_v2"].models) == {"gemma-e2b-base-q4", fake.V1, fake.V2}


def test_a_plan_whose_configs_disagree_on_the_extractor_is_refused(tmp_path):
    root = fake.copy_repo(tmp_path)
    fake.add_v2(root)
    cfg = root / "configs/evalkit_v2_solo.yaml"
    text = cfg.read_text(encoding="utf-8").replace("gpt-6-luna", "claude-haiku-5-5")
    cfg.write_text(text, encoding="utf-8")
    with pytest.raises(ResultsError, match="extractor 'gpt-6-luna' differs from"):
        B.build(root)


def test_r4_flat_fields_read_from_the_shipped_release(real, v2):
    for built in (real, v2):
        b = built.bundle
        assert b.train == b.releases[-1].train and b.artifacts == b.releases[-1].artifacts
        assert b.audit_pass_rate == b.releases[-1].audits[-1].pass_rate
        assert b.bench.quant == "Q4_K_M" and b.bench == b.releases[-1].bench
    # each release quotes its own laptop bench; the shipped one is v3b's (R4.1, V3.8); v1.0.0's
    # was v2's (its frozen bundle, docs/results/releases/v1.0.0/)
    v1, v2, v3b = real.bundle.releases
    assert v1.bench_source == "docs/measurements/laptop_bench.json#ft-Q4_K_M"
    assert v2.bench_source == "docs/measurements/v2/laptop_bench.json#ft2-Q4_K_M"
    assert v3b.bench_source == "docs/measurements/v3b/laptop_bench.json#ft3b-Q4_K_M"
    v2_file = json.loads((ROOT / "docs/measurements/v2/laptop_bench.json").read_text("utf-8"))
    assert v2.bench is not None and v2.bench.model_dump() == v2_file["variants"]["ft2-Q4_K_M"]
    v3b_file = json.loads((ROOT / "docs/measurements/v3b/laptop_bench.json").read_text("utf-8"))
    assert real.bundle.bench.model_dump() == v3b_file["variants"]["ft3b-Q4_K_M"]
    assert set(real.bundle.bench_variants) == {"ft3b-Q4_K_M", "ft3b-Q8_0"}
    frozen = ResultsBundle.model_validate_json(
        (ROOT / "docs/results/releases/v1.0.0/results_bundle.json").read_text("utf-8")
    )
    assert {"ft2-Q4_K_M", "ft-Q4_K_M"} <= set(frozen.bench_variants)
    assert frozen.bench == v2.bench
    p2 = v2.parity  # the real v2, under Sol alone
    assert p2 and p2.judged_headline_delta and set(p2.judged_headline_delta) == {SOL}
    assert frozen.parity == p2
    p = real.bundle.parity  # the shipped release's: v3b at Q4_K_M (OWNER D-73)
    assert p == v3b.parity and p is not None and p.label_agreement == 0.924
    assert v3b.parity_q8_0 is not None and v3b.parity_q8_0.label_agreement == 0.99
    p1 = real.bundle.releases[0].parity
    assert p1 and p1.judged_headline_delta
    assert p1.judged_headline_delta["claude-opus-5-5"].point == 0.025
    v1_keys = {
        f"headline: gemma-e2b-grounded-q4 minus {m} ({j})"
        for m in ("gemma-e2b-base-q4", "claude-haiku-5-5", "gpt-6-luna")
        for j in JUDGES
    }
    v2_keys = {
        f"{metric}: gemma-e2b-grounded-v2-q4 minus {m} ({SOL})"
        for metric in ("headline", "citation_precision", "aim_citation_rate")
        for m in ("gemma-e2b-grounded-q4", "gemma-e2b-base-q4")
    }
    v3b_keys = {f"headline: {V3B} minus gemma-e2b-grounded-v2-q4 ({SOL})"}
    assert set(real.bundle.primary) == v1_keys | v2_keys | v3b_keys


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
    m.write_text(m.read_text(encoding="utf-8").replace("ft2-Q4_K_M", "ft3-Q4_K_M"), "utf-8")
    b = B.build(root).bundle
    assert not b.lineage_ok
    assert any(
        "ft2-Q4_K_M.gguf is not in the release's export records" in p for p in b.lineage_problems
    )
