"""RES-03 (OWNER D-24, D-59, D-60): the judges' calibration stamp, read from the committed copy of
EvalKit's calibration.json, the Gemma-family report and the labeller record. On the real data,
then on copies of those files broken one way at a time. No API call, no artifacts/."""

import json
import shutil
from pathlib import Path

import pytest
import yaml
from evalkit.pipeline.judge import JUDGE_PROMPT_HASH
from pydantic import ValidationError

from ggh.results import bundle as B
from ggh.results import calibration as C
from ggh.results.runs import Run
from ggh.schemas import JudgeCalibration, ResultsBundle
from tests.results import fake

ROOT = fake.ROOT
OPUS, SOL = "claude-opus-5-5", "gpt-6.1-sol"
CFG = yaml.safe_load((ROOT / B.CONFIG).read_text(encoding="utf-8"))
CAL = CFG["calibration"]
CHECK = CFG["gemma_family_check"]
EXEMPT = (
    "offscope",
    "offscope_v2",
    "probe_original",
    "probe_original_v2",
    "probe_paraphrase",
    "probe_paraphrase_v2",
)
FILES = (CAL["evalkit"], CAL["labeller"], CAL["gemma_family_report"], CHECK, CAL["labeller_prompt"])


@pytest.fixture(scope="module")
def real():
    return B.build(ROOT).bundle


@pytest.fixture(scope="module")
def runs():
    names = [p.parent.name for p in (ROOT / "docs/results").glob("*/summary.json")]
    cfg_runs = {n for r in CFG["releases"] for n in r["runs"]}
    return {n: Run(ROOT, n) for n in sorted(names) if n in cfg_runs}


@pytest.fixture
def files(tmp_path):
    """A root holding only copies of the calibration inputs, to break one at a time."""
    for rel in FILES:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, tmp_path / rel)
    return tmp_path


def _edit(root: Path, rel: str, change) -> None:
    p = root / rel
    d = json.loads(p.read_text(encoding="utf-8"))
    change(d)
    p.write_text(json.dumps(d), encoding="utf-8")


def _stamp(root: Path, runs, cfg=None) -> C.Stamp:
    return C.stamp(root, CAL if cfg is None else cfg, runs, CHECK)


# the real, committed calibration


def test_sol_is_stamped_and_opus_is_reported_never_stamped(real):
    by = {c.judge: c for c in real.calibration}
    assert set(by) == {OPUS, SOL}
    sol, opus = by[SOL], by[OPUS]
    assert sol.status == "calibrated" and sol.meets_bar and sol.kappa.lo >= 0.4
    assert (
        sol.task == "pubmedqa" and sol.judge_prompt_hash == JUDGE_PROMPT_HASH == "9a3a8246049f6f85"
    )
    assert sol.n_claims == 231 and sol.bar == 0.4
    assert "agreement with a Claude Opus 5.5 reference labeller" in sol.wording
    assert "not a human expert's" in sol.wording
    assert (
        sol.gemma_family and sol.gemma_family.agreement == 0.95 and sol.gemma_family.n_claims == 40
    )
    # OWNER D-60: the labels are Opus's own model's, so it is never stamped; and it misses the bar
    assert opus.status == "not_stampable_self_reference" and not opus.meets_bar
    assert opus.kappa.lo == pytest.approx(0.263, abs=1e-3) and opus.gemma_family is None
    assert "agreeing with itself" in opus.wording and "under the 0.4 bar" in opus.wording
    for c in (sol, opus):
        assert c.reference_model == OPUS
        assert c.reference_labeller == "claude-opus-5-5 (configs/labeller_prompt.md, OWNER D-60)"


def test_v2_is_calibrated_and_v1s_opus_numbers_stay_uncalibrated(real):
    v1, v2 = real.releases
    assert v1.calibration_judges == (OPUS, SOL) and v1.calibration_status == "uncalibrated"
    assert v2.calibration_judges == (SOL,) and v2.calibration_status == "calibrated"
    # the bundle as a whole holds v1's Opus-judged comparisons, so it is not calibrated
    assert real.calibration_status == "uncalibrated" and real.gemma_family_check is None
    # the summaries' own stamps are EvalKit's, never rewritten (OWNER D-59)
    assert {r.calibration_status for r in real.runs} == {"uncalibrated"}


def test_the_labeller_relabel_is_reported(real):
    lc = real.labeller_consistency
    assert lc and lc.labeller.startswith("claude-opus-5-5") and lc.prompt == CAL["labeller_prompt"]
    assert lc.claims == 88 and lc.answers == 20 and lc.kappa.lo == pytest.approx(0.662, abs=1e-3)


def test_offscope_and_probe_runs_are_exempt_and_say_so(real):
    by = {r.name: r for r in real.runs}
    for name in EXEMPT:
        note = by[name].calibration_exempt
        assert note and note.startswith("exempt:") and by[name].task in note and "D-59" in note
    for name in ("main", "main_citation", "main_v2", "parity", "parity_v2"):
        assert by[name].task == "pubmedqa" and by[name].calibration_exempt is None


def test_no_calibration_blocker_on_the_real_data_and_a_miss_is_not_one(real):
    assert not [b for b in real.publish_blockers if b.startswith("calibration")]
    assert not any("uncalibrated" in b for b in real.publish_blockers)


# broken copies, one way at a time


def test_the_committed_copies_stamp_like_the_real_bundle(files, runs, real):
    s = _stamp(files, runs)
    assert s.problems == () and s.judges == real.calibration


def test_a_calibration_for_another_task_is_refused(files, runs):
    _edit(files, CAL["evalkit"], lambda d: d["current"].update(task="ggh_offscope"))
    _edit(
        files,
        CAL["evalkit"],
        lambda d: [a.update(task="ggh_offscope") for a in d["current"]["agreements"]],
    )
    s = _stamp(files, runs)
    assert s.judges == () and any(
        "for task 'ggh_offscope', not 'pubmedqa'" in p for p in s.problems
    )
    assert s.release_status([SOL]) == "uncalibrated"


def test_a_calibration_under_another_rubric_is_refused(files, runs):
    def rehash(d):
        d["current"]["judge_prompt_hash"] = "0000000000000000"
        for a in d["current"]["agreements"]:
            a["judge_prompt_hash"] = "0000000000000000"

    _edit(files, CAL["evalkit"], rehash)
    s = _stamp(files, runs)
    assert s.judges == () and any("under rubric 0000000000000000" in p for p in s.problems)
    assert s.release_status([SOL]) == "uncalibrated"


def test_a_run_judged_under_another_rubric_is_a_blocker(files, runs, monkeypatch):
    monkeypatch.setattr(C, "verdict_hashes", lambda r: {"1111111111111111"})
    s = _stamp(files, runs)
    assert any("main_v2's verdicts use rubric" in p for p in s.problems)
    assert s.release_status([SOL]) == "uncalibrated"


def test_a_judge_missing_from_the_calibration_is_refused(files, runs):
    _edit(
        files,
        CAL["evalkit"],
        lambda d: d["current"].update(
            agreements=[a for a in d["current"]["agreements"] if a["judge"] != SOL]
        ),
    )
    s = _stamp(files, runs)
    assert any(f"no held-out agreement for judge {SOL}" in p for p in s.problems)
    assert SOL not in s.by_judge() and s.release_status([SOL]) == "uncalibrated"


def test_a_judge_below_the_bar_is_disclosed_not_blocked(files, runs):
    def lower(d):
        for a in d["current"]["agreements"]:
            if a["judge"] == SOL:
                a["kappa"] = {"point": 0.5, "lo": 0.399, "hi": 0.6}

    _edit(files, CAL["evalkit"], lower)
    s = _stamp(files, runs)
    sol = s.by_judge()[SOL]
    assert s.problems == ()
    assert sol.status == "below_bar" and not sol.meets_bar
    assert "under the 0.4 bar" in sol.wording and "stay published" in sol.wording
    assert s.release_status([SOL]) == "uncalibrated"


def test_the_reference_labellers_own_model_is_never_stamped(files, runs):
    # had Sol made the labels, Sol agreeing with them would be Sol agreeing with itself
    def relabel(d):
        d["labeller"] = d["labeller"].replace(OPUS, SOL)

    _edit(files, CAL["labeller"], relabel)
    _edit(files, CAL["gemma_family_report"], relabel)
    s = _stamp(files, runs, {**CAL, "reference_labeller": SOL})
    by = s.by_judge()
    assert s.problems == ()
    assert by[SOL].status == "not_stampable_self_reference" and by[SOL].meets_bar
    assert by[OPUS].status == "below_bar"  # no longer self-reference, but under the bar
    assert s.release_status([SOL]) == "uncalibrated"


def test_a_missing_or_invalid_calibration_file_is_a_blocker(files, runs):
    (files / CAL["evalkit"]).write_text("{not json", encoding="utf-8")
    s = _stamp(files, runs)
    assert s.judges == () and any("is not valid JSON" in p for p in s.problems)
    (files / CAL["evalkit"]).write_text(json.dumps({"current": {"task": "pubmedqa"}}), "utf-8")
    assert any("not an EvalKit calibration report" in p for p in _stamp(files, runs).problems)
    (files / CAL["evalkit"]).unlink()
    s = _stamp(files, runs)
    assert s.judges == () and any(f"{CAL['evalkit']} is missing" in p for p in s.problems)
    assert C.stamp(files, None, runs, CHECK).problems  # not configured at all


def test_a_missing_gemma_family_report_blocks_the_stamp(files, runs):
    (files / CAL["gemma_family_report"]).unlink()
    s = _stamp(files, runs)
    assert s.by_judge()[SOL].gemma_family is None and s.release_status([SOL]) == "uncalibrated"
    assert any("gemma_family_report.json is missing" in p for p in s.problems)


def test_a_labeller_record_that_does_not_match_the_prompt_is_a_blocker(files, runs):
    (files / CAL["labeller_prompt"]).write_text("another prompt\n", encoding="utf-8")
    s = _stamp(files, runs)
    assert any("no longer has the sha256" in p for p in s.problems)


def test_a_gemma_check_that_disagrees_with_its_report_is_a_blocker(files, runs):
    _edit(files, CHECK, lambda d: d.update({SOL: 0.5}))
    assert any("disagrees with" in p for p in _stamp(files, runs).problems)


def test_the_full_bundle_lists_a_broken_calibration_as_a_blocker(tmp_path):
    root = fake.copy_repo(tmp_path)
    (root / CAL["evalkit"]).unlink()
    b = B.build(root).bundle
    assert any(p.startswith("calibration:") and "is missing" in p for p in b.publish_blockers)
    assert b.calibration == () and {r.calibration_status for r in b.releases} == {"uncalibrated"}


# the schema keeps the rule


def test_the_schema_refuses_a_stamp_the_rule_does_not_give(real):
    sol = next(c for c in real.calibration if c.judge == SOL).model_dump()
    with pytest.raises(ValidationError, match="contradicts OWNER D-59/D-60"):
        JudgeCalibration.model_validate({**sol, "reference_model": SOL})
    low = {**sol, "kappa": {"point": 0.5, "lo": 0.3, "hi": 0.6}}
    with pytest.raises(ValidationError, match="meets_bar"):
        JudgeCalibration.model_validate(low)
    with pytest.raises(ValidationError, match="contradicts"):
        JudgeCalibration.model_validate({**low, "meets_bar": False})
    assert JudgeCalibration.model_validate({**low, "meets_bar": False, "status": "below_bar"})


def test_the_schema_refuses_a_calibrated_release_on_an_unstamped_judge(real):
    d = json.loads(real.model_dump_json())
    d["releases"][0]["calibration_status"] = "calibrated"  # v1 relies on Opus
    with pytest.raises(ValidationError, match="claude-opus-5-5 is not stamped"):
        ResultsBundle.model_validate(d)
    d = json.loads(real.model_dump_json())
    d["calibration_status"] = "calibrated"
    d["gemma_family_check"] = {SOL: 0.95}
    with pytest.raises(ValidationError, match="the bundle is calibrated but v1 is not"):
        ResultsBundle.model_validate(d)
    d = json.loads(real.model_dump_json())
    d["calibration"] = [c for c in d["calibration"] if c["judge"] != SOL]
    with pytest.raises(ValidationError, match=r"gpt-6\.1-sol is not stamped"):
        ResultsBundle.model_validate(d)
