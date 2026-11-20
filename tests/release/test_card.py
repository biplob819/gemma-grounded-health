"""CRD-01..06: the model card is generated from the results bundle and nothing else (D-20)."""

import json
import re
import shutil
from pathlib import Path

import pytest
import yaml

from ggh.release import card, sources
from tests.release import build_v1_bundle as B

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = B.BUNDLE.relative_to(ROOT)
PATHS = sources.Paths(bundle=FIXTURE)
SECTIONS = (
    "## Summary",
    "## Intended use",
    "## Out-of-scope uses",
    "## Training data",
    "## Training procedure",
    "## Evaluation",
    "### Behaviour checks (B1–B8)",
    "### Parity",
    "### Laptop speed",
    "## Limits",
    "## Licence and credits",
    "## Research only, not medical advice",
)
DECIMAL = re.compile(r"(?<![\w§.-])-?\d+\.\d+(?![\w.])")


@pytest.fixture(scope="module")
def src():
    return sources.load(ROOT, PATHS)


@pytest.fixture(scope="module")
def text(src):
    return card.render(src)


def test_fixture_bundle_is_rebuilt_from_committed_results():
    assert json.loads(B.BUNDLE.read_text(encoding="utf-8")) == B.bundle()


@pytest.mark.golden
def test_card_matches_the_golden_file(text):
    assert text == B.CARD.read_text(encoding="utf-8")


def test_every_section_in_order(text):
    at = [text.index(s) for s in SECTIONS]
    assert at == sorted(at)


def test_template_types_no_number():
    tpl = card.TEMPLATE.read_text(encoding="utf-8")
    assert not DECIMAL.search(tpl)
    assert not re.search(r"\d+\s?%", tpl)


def _numbers(value, out):
    if isinstance(value, bool):
        return
    if isinstance(value, int | float):
        out.add(str(value))
        for fmt in ("{:.3f}", "{:.2f}", "{:.1f}", "{:.0f}"):
            out.add(fmt.format(value))
    elif isinstance(value, str):
        out.update(DECIMAL.findall(value))
    elif isinstance(value, dict):
        for v in value.values():
            _numbers(v, out)
    elif isinstance(value, list | tuple):
        for v in value:
            _numbers(v, out)


def test_every_number_in_the_card_comes_from_the_inputs(src, text):  # CRD-02
    allowed: set[str] = set()
    _numbers(src.raw, allowed)
    found = DECIMAL.findall(text)
    assert found, "the card shows no numbers"
    unsigned = {a.lstrip("-") for a in allowed}  # a comparison may be shown re-oriented (b - a)
    stray = sorted({x for x in found if x.lstrip("-") not in unsigned})
    assert not stray, f"numbers not traceable to the bundle or its inputs: {stray}"


def test_a_changed_bundle_changes_the_card(tmp_path, src):
    data = json.loads(B.BUNDLE.read_text(encoding="utf-8"))
    data["releases"][-1]["parity"]["label_agreement"] = 0.987654
    data["releases"][-1]["bench"]["gen_tps_p50"] = 321.0  # the release's own bench
    bundle = tmp_path / "bundle.json"
    bundle.write_text(json.dumps(data), encoding="utf-8")
    other = card.render(sources.load(ROOT, sources.Paths(bundle=bundle)))
    assert "0.988 of prompts" in other and "321.0 / " in other
    assert "0.956 of prompts" in card.render(src)


def test_stamps_word_for_word(src, text):  # CRD-03
    from evalkit.pipeline.aggregate import stamp_lines

    lines = stamp_lines(src.main())
    assert lines, "the v1 run is uncalibrated, so EvalKit stamps it"
    for line in lines:
        assert f"- ⚠ {line}\n" in text
    assert f"Judge mode: **{src.bundle.judge_mode}**" in text
    rel = src.bundle.shipped
    assert f"Calibration of {rel.version}: **{rel.calibration_status}**" in text
    for c in src.bundle.calibration:  # each judge as the bundle words it (OWNER D-59, D-60)
        assert f"  - {c.wording}" in text
    assert "owner-labelled" not in text  # OWNER D-60: the labels are a model's


def test_hub_metadata_validates(text):  # CRD-04
    meta, _ = card.split_front_matter(text)
    assert card.metadata_problems(meta) == []
    assert meta["license"] == "apache-2.0" and meta["base_model"] == "google/gemma-4-E2B-it"
    metrics = meta["model-index"][0]["results"][0]["metrics"]
    assert {m["type"] for m in metrics} >= {"accuracy", "grounded_accuracy"}


def test_metadata_problems_catch_a_broken_index():
    bad = {"license": "apache-2.0", "model-index": [{"results": [{"metrics": [{"value": "x"}]}]}]}
    problems = card.metadata_problems(bad)
    assert "missing base_model" in problems
    assert any("bad metric" in p for p in problems)
    assert any("task.type" in p for p in problems)


def test_licence_credits_and_banner(text):  # CRD-05, CRD-06
    assert text.count(card.BANNER) >= 2
    assert "Apache-2.0" in text and "NOTICE" in text and "PubMedQA (Jin et al., 2019), MIT" in text


def test_v2_model_name_is_a_flag_not_a_constant(tmp_path):
    data = json.loads(B.BUNDLE.read_text(encoding="utf-8"))
    bundle = tmp_path / "b.json"
    bundle.write_text(json.dumps(data), encoding="utf-8")
    src = sources.load(ROOT, sources.Paths(bundle=bundle, ft_model="claude-haiku-5-5"))
    meta, _ = card.split_front_matter(card.render(src))
    acc = meta["model-index"][0]["results"][0]["metrics"][0]["value"]
    haiku = next(m for m in src.main().models if m.model == "claude-haiku-5-5")
    assert acc == haiku.accuracy.point


def test_an_input_that_does_not_match_its_hash_is_refused(tmp_path):
    for rel in (
        FIXTURE,
        *B.SUMMARIES,
        sources.DATA_MANIFEST,
        sources.TARGETS_REPORT,
        "docs/measurements/targets_drop_report_v1.json",  # v1's kept copy (lineage.KEPT_COPIES)
        sources.TARGETS_LINEAGE,
        sources.RECIPE,
        "configs/eval_plan.yaml",  # the shipped release's plan, checked against its hash
    ):
        dest = tmp_path / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, dest)
    sources.load(tmp_path, PATHS)  # the copy is intact
    recipe = tmp_path / sources.RECIPE
    recipe.write_text(recipe.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(sources.SourceError, match="recipe"):
        sources.load(tmp_path, PATHS)


def test_a_summary_from_another_run_is_refused(tmp_path):
    data = json.loads(B.BUNDLE.read_text(encoding="utf-8"))
    data["evalkit_run_ids"] = ["r-someother"]
    bundle = tmp_path / "b.json"
    bundle.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(sources.SourceError, match="not one the bundle lists"):
        sources.load(ROOT, sources.Paths(bundle=bundle))


def test_front_matter_is_yaml_first(text):
    assert text.startswith("---\n")
    yaml.safe_load(text[4:].split("\n---\n", 1)[0])
