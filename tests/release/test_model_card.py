"""CRD-01..06 on the real card of v1.0.0 (v2): the card v1.0.0 published, frozen under
docs/results/releases/v1.0.0/ (D-76) and still generated from v1.0.0's frozen results bundle
(D-20), with the stamps and disclosures of OWNER D-55, D-59, D-60, D-61 and D-62 word for word
and none of the claims they rule out. The live MODEL_CARD.md is checked against the live bundle
in tests/release/test_release_v3b.py once it ships v3b; until then it is this file."""

import json
import re
from pathlib import Path

import pytest
from huggingface_hub import ModelCard, ModelCardData, metadata_load

from ggh.release import card, readme, sources
from tests.release import build_v1_bundle as B
from tests.release.test_card import DECIMAL, SECTIONS, _numbers

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "docs" / "results" / "releases" / "v1.0.0"
CARD = FROZEN / "MODEL_CARD.md"  # v1.0.0's published card (v2)
PATHS = sources.Paths(bundle=(FROZEN / "results_bundle.json").relative_to(ROOT))
README = ROOT / "README.md"
LIVE_IS_V1_0_0 = (ROOT / "docs/results/results_bundle.json").read_bytes() == (
    FROZEN / "results_bundle.json"
).read_bytes()
BASE, V1, V2 = "gemma-e2b-base-q4", "gemma-e2b-grounded-q4", "gemma-e2b-grounded-v2-q4"
SOL, OPUS, LUNA, HAIKU = "gpt-6.1-sol", "claude-opus-5-5", "gpt-6-luna", "claude-haiku-5-5"
ORDER = (
    "## Summary",
    "## Intended use",
    "## Out-of-scope uses",
    "## How to use",
    "## Training data",
    "## Training procedure",
    "## Evaluation",
    "### Benchmark",
    "### Pre-registered comparisons",
    "#### Citation results and the aim rule (OWNER D-62)",
    "### Secondary: label accuracy",
    "### Behaviour checks (B1–B8)",
    "### Parity",
    "### Laptop speed",
    "## Limits",
    "## Licence and credits",
    "## Research only, not medical advice",
)
# a standalone integer of two or more digits: not part of an id, hash, path, date or version
INTEGER = re.compile(r"(?<![\w§.\-/:`#@])\d{2,}(?![\w.\-/:`%])")


@pytest.fixture(scope="module")
def src():
    return sources.load(ROOT, PATHS)


@pytest.fixture(scope="module")
def text():
    return CARD.read_text(encoding="utf-8")


def _section(text: str, start: str, end: str) -> str:
    i = text.index(start)
    return text[i : text.index(end, i + len(start))]


def test_the_committed_card_is_the_generator_output(src, text):  # CRD-01
    assert text == card.render(src), "run: uv run ggh card"
    assert src.bundle.shipped.version == "v2" and src.ft_model == V2


def test_every_section_in_order(text):  # CRD-01, Tech Spec §10.1
    at = [text.index(s) for s in ORDER]
    assert at == sorted(at)
    assert all(s in text for s in SECTIONS)


def test_every_number_comes_from_the_bundle_or_its_sources(src, text):  # CRD-02
    allowed: set[str] = set()
    _numbers(src.raw, allowed)
    unsigned = {a.lstrip("-") for a in allowed}
    found = DECIMAL.findall(text)
    assert len(found) > 200
    stray = sorted({x for x in found if x.lstrip("-") not in unsigned})
    assert not stray, f"decimals not in the bundle or its inputs: {stray}"
    for v in json.dumps(src.raw).split('"'):  # integers inside committed text ("..., 2019")
        unsigned.update(INTEGER.findall(v))
    # the one derived count: the constructed `maybe` examples, summed over the targets' splits
    unsigned.add(str(sum(int(q.get("insufficient", 0)) for q in src.targets["quotas"].values())))
    ints = set(INTEGER.findall(text))
    stray = sorted(x for x in ints if x not in unsigned)
    assert not stray, f"integers not in the bundle or its inputs: {stray}"


def test_front_matter_validates_with_huggingface_hub(src, text):  # CRD-04
    meta = metadata_load(CARD)
    assert meta is not None
    assert card.metadata_problems(meta) == []
    for key in card.REQUIRED_META:
        assert meta.get(key), key
    assert meta["license"] == "apache-2.0" and meta["language"] == ["en"]
    assert meta["base_model"] == src.recipe.base_model == "google/gemma-4-E2B-it"
    assert meta["datasets"] == [src.data.dataset] and meta["pipeline_tag"] == "text-generation"
    data = ModelCard.load(CARD).data
    assert isinstance(data, ModelCardData)
    assert data.eval_results and data.model_name == card.MODEL_NAME


def test_model_index_equals_the_bundle(src):  # CRD-04
    meta = metadata_load(CARD)
    assert meta is not None
    res = meta["model-index"][0]["results"][0]
    run = src.ref(src.main())
    row = next(r for r in src.bundle.results if r.run == run.name and r.model == V2)
    got = {(m["type"], m["name"]): m["value"] for m in res["metrics"]}
    assert row.citation_precision and row.aim_citation_rate
    want = {
        ("accuracy", "Label accuracy"): row.accuracy.point,
        ("grounded_accuracy", f"Correct and fully grounded ({SOL})"): row.headline[SOL].point,
        ("citation_validity", "Citation validity"): row.citation_validity.point,
        ("citation_precision", f"Citation precision ({SOL})"): row.citation_precision[SOL].point,
        (
            "aim_citation_rate",
            f"Aim-citation rate, lower is better ({SOL})",
        ): row.aim_citation_rate[SOL].point,
    }
    assert got == want
    assert res["source"]["name"] == f"EvalKit run {run.run_id}" == "EvalKit run r-715fc213a8a5"
    assert res["dataset"]["revision"] == src.data.dataset_revision


def test_the_shipped_calibration_stamp_word_for_word(src, text):  # CRD-03, OWNER D-59, D-60
    rel = src.bundle.shipped
    assert rel.calibration_status == "calibrated" and rel.calibration_judges == (SOL,)
    ev = _section(text, "## Evaluation", "### Benchmark")
    assert "- Judge mode: **single**." in ev
    assert "- Calibration of v2: **calibrated**." in ev
    by = {c.judge: c for c in src.bundle.calibration}
    assert f"  - {by[SOL].wording}." in ev
    assert (
        "agreement with a Claude Opus 5.5 reference labeller (a model's labels, not a human "
        "expert's; OWNER D-60)" in by[SOL].wording
    )
    assert f"  - {by[OPUS].wording} (not a judge of v2's runs)." in ev  # reported, never stamped
    assert by[OPUS].status == "not_stampable_self_reference"
    assert ev.index(by[SOL].wording) < ev.index(by[OPUS].wording)
    lc = src.bundle.labeller_consistency
    assert lc and f"kappa {card.iv(lc.kappa)} (OWNER D-60)" in ev


def test_evalkits_per_run_stamp_is_never_the_releases_status(text):  # OWNER D-59
    assert "uncalibrated" not in text
    assert "owner-labelled" not in text and "owner-labelled" not in card.TEMPLATE.read_text("utf-8")
    assert "- ⚠ single judge: self-preference not measured" in text
    assert "- ⚠ judge not calibrated" not in text
    hits = [line for line in text.splitlines() if "judge not calibrated for this task" in line]
    assert len(hits) == 1 and "keep EvalKit's own per-run stamp" in hits[0]
    for run in ("offscope_v2", "probe_original_v2", "probe_paraphrase_v2"):
        assert f"- `{run}`: exempt: task " in text


def test_b5_and_b8_name_the_model_reviewer(src, text):  # CRD-03, OWNER D-61
    beh = _section(text, "### Behaviour checks", "### Parity")
    assert "a blind review by Claude Opus 5.5, not the owner (OWNER D-61)" in beh
    assert "B5 is the sample read of 20 answers per model from `main_v2` under " in beh
    assert "`configs/sample_read_prompt.md`" in beh
    assert (
        "`configs/offscope_review_prompt.md`, the reviewers seeing the question and answer only"
        in beh
    )
    rows = [r for r in src.bundle.behaviour if r.source == "model_review" and r.run]
    assert {r.run for r in rows if r.run in src.bundle.shipped.runs} == {"main_v2", "offscope_v2"}
    for r in rows:
        assert r.reviewer and f"`{r.reviewer}`" in beh
    assert "owner_review" not in text


def test_laptop_speed_is_v2s_own_bench(src, text):
    raw = json.loads((ROOT / "docs/measurements/v2/laptop_bench.json").read_text("utf-8"))
    v2 = raw["variants"]["ft2-Q4_K_M"]
    sec = _section(text, "### Laptop speed", "## Limits")
    assert "variant `ft2-Q4_K_M` in `docs/measurements/v2/laptop_bench.json`" in sec
    assert f"| Generation (p50 / p95) | {v2['gen_tps_p50']:.1f} / {v2['gen_tps_p95']:.1f}" in sec
    v1 = src.bundle.releases[0].bench
    assert v1 is not None and f"{v1.gen_tps_p50:.1f} / " not in sec  # not v1's 105.1


def test_aim_rule_caveat_and_no_v2_beats_base_claim(src, text):  # CRD-03, OWNER D-62
    aim = _section(text, "#### Citation results and the aim rule", "### Secondary")
    assert "The v2-vs-base citation results depend on how the judge applies" in aim
    assert "`docs/measurements/aim_rule_check.json`" in aim
    assert "aim-citation rate from 0.138 to 0.054" in aim
    assert f"`{BASE}` is then better than `{V2}` on citation precision and aim-citation rate" in aim
    assert "So this card does not claim that v2 beats base on citations." in aim
    assert f"The results of `{V2}` against `{V1}` hold under both counts." in aim
    for line in text.splitlines():
        low = line.lower()
        if "beats base" in low or "beat base" in low or "beat it on citations" in low:
            assert " not " in low, line
        if f"{V2} minus {BASE}" in line and f"{V2} better" in line:
            assert "depends on the aim rule, OWNER D-62" in line, line
    summary = _section(text, "## Summary", "## Intended use")
    assert "v2 is not claimed to beat it on citations (OWNER D-62)" in summary
    assert "better on citation precision, better on aim-citation rate" in summary  # v2 vs v1


def test_no_line_claims_v1_beats_luna_under_sol(text):  # CRD-03, known issue E3.3
    for line in text.splitlines():
        if line.startswith("| ") and " minus " in line:
            assert LUNA not in line and HAIKU not in line, line
        assert not (f"{V1} better" in line and LUNA in line), line
    # earlier releases appear only as published numbers, labelled, never compared
    pub = _section(text, "### Published v1 numbers (not compared with v2)", "### Pre-registered")
    assert "not as a v2 comparison" in pub
    assert " better" not in pub.replace("lower is better", "")  # no verdict, only numbers


def test_the_v1_golden_fixture_is_historical_only():
    """The v1-only fixture still prints E3.2's v1-beats-Luna verdict under Sol (its own plan's
    published comparison); the real card above cannot print it."""
    golden = B.CARD.read_text(encoding="utf-8")
    assert f"| {V1} minus {LUNA} | headline | {SOL} |" in golden
    assert f"| {V1} minus {LUNA} |" not in CARD.read_text(encoding="utf-8")


def test_release_check_misses_are_disclosed(text):  # OWNER D-55, D-50
    par = _section(text, "### Parity", "### Laptop speed")
    assert "0.920 of prompts" in par and "under the plan's bar of 0.950" in par
    assert "(OWNER D-55)" in par
    lim = _section(text, "## Limits", "## Licence and credits")
    assert "missed on headline 0.350 against v1's 0.355" in lim
    assert "label accuracy 0.795 (bar 0.800)" in lim and "(OWNER D-55)" in lim
    train = _section(text, "## Training data", "## Training procedure")
    assert "the audit did not pass its bar" in train


def test_licence_notice_and_credits(text):  # CRD-05
    lic = _section(text, "## Licence and credits", "## Research only")
    assert "Apache-2.0 (`LICENSE`)" in lic and "`NOTICE`" in lic
    assert "https://ai.google.dev/gemma/docs/gemma_4_license" in lic
    assert "recorded in `docs/licence_check.json`" in lic
    assert "PubMedQA (Jin et al., 2019), MIT licence" in lic
    assert "EvalKit" in lic and "MIT licence, https://github.com/biplob819/evalkit" in lic
    rec = json.loads((ROOT / sources.LICENCE_RECORD).read_text("utf-8"))
    assert rec["base_model"]["read_at"] in lic and rec["base_model"]["model_card"]["sha256"] in lic
    notice = (ROOT / "NOTICE").read_text("utf-8")
    assert "Gemma 4" in notice and "PubMedQA" in notice and "Apache License" in notice


def test_research_only_in_summary_and_footer(text):  # CRD-06
    summary = _section(text, "## Summary", "## Intended use")
    assert f"{card.BANNER}." in summary
    footer = text[text.index("## Research only, not medical advice") :]
    assert "not medical advice" in footer
    assert text.count(card.BANNER) >= 3


def test_readme_block_carries_the_same_stamps():  # OSS-02 with R4.1's disclosures
    src = sources.load(ROOT, PATHS)
    block = readme.results_block(src)
    if LIVE_IS_V1_0_0:  # the README carries the live bundle's block (test_readme checks it)
        assert block in README.read_text(encoding="utf-8")
    assert "calibration of v2: calibrated" in block and "uncalibrated" not in block
    sol = next(c for c in src.bundle.calibration if c.judge == SOL)
    assert sol.wording in block
    assert "B5 and B8 are a blind review by Claude Opus 5.5, not the owner (OWNER D-61)" in block
    assert "so v2 is not claimed to beat base on citations" in block
    assert "v2-vs-v1 holds under both counts" in block
    assert "generates 73.1 tokens/s (p50)" in block


def test_the_cards_audit_is_the_one_on_the_targets_the_release_trained_on():
    """The last audit listed for the shipped release audited the sample its targets lineage
    recorded (configs/results.yaml once rotated the v2 audit files)."""
    import json as _json
    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parents[2]
    bundle = _json.loads((FROZEN / "results_bundle.json").read_text("utf-8"))
    shipped = bundle["releases"][-1]
    last = shipped["audits"][-1]
    result = _json.loads((root / last["path"]).read_text("utf-8"))
    lineage = _json.loads((root / "docs/measurements/lineage/targets_v2.json").read_text("utf-8"))
    assert result["sample_sha256"] == lineage["outputs"]["artifacts/targets/audit_sample.jsonl"]
    ok = result["overall"]["ok"] / result["labelled"]
    card = CARD.read_text("utf-8")
    assert f"{ok:.3f} judged ok overall" in card and "blind Claude reviewer" in card


def test_the_targets_line_describes_the_rule_not_a_guarantee():
    """The audit found weak and aim/method answer citations (OWNER D-50), so the card may not
    say every citation supports its answer."""
    for text in (CARD.read_text("utf-8"), (ROOT / "MODEL_CARD.md").read_text("utf-8")):
        assert "cites only sentences that support it" not in text
        assert "Citations are chosen by rule" in text
        assert "A selection rule is not a guarantee" in text
