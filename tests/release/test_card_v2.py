"""OWNER D-57 on the synthetic v2 (tests/results/fake.py): the card and README report v2 under
GPT-6.1 Sol with GPT-6 Luna extracting, and show v1's published E3.2 numbers (Haiku-extracted,
Opus and Sol judged) beside them, labelled, with Haiku 5.5 and GPT-6 Luna never compared with v2.
Nothing here is a real v2 number."""

import pytest

from ggh.release import card, readme, sources
from ggh.results import bundle as B
from tests.release.test_card import DECIMAL, _numbers
from tests.results import fake

OPUS, SOL = "claude-opus-5-5", "gpt-6.1-sol"
HAIKU, LUNA = "claude-haiku-5-5", "gpt-6-luna"
PUBLISHED = "Published v1 numbers (not compared with v2)"


@pytest.fixture(scope="module")
def src(tmp_path_factory):
    root = fake.copy_repo(tmp_path_factory.mktemp("v2card"))
    fake.add_v2(root)
    B.write(root, B.build(root))
    # the post-hoc aim-rule count (OWNER D-62) read the real main_v2; on the synthetic v2 it is
    # stale and sources.load refuses it, so it is left out here (test_model_card covers it)
    (root / sources.AIM_CHECK).unlink()
    # the synthetic v2 copies v1's training run (tests/results/fake.py), so it trained on v1's
    # recipe and targets; the real v2's own inputs are sources.SHIPPED_INPUTS["v2"]
    paths = sources.Paths(
        recipe=sources.RECIPE,
        targets_lineage=sources.TARGETS_LINEAGE,
        targets_report=sources.TARGETS_REPORT,
    )
    return sources.load(root, paths)


@pytest.fixture(scope="module")
def text(src):
    return card.render(src)


def _not_as_judge(text: str) -> str:
    """The text less the lines where Claude Opus 5.5 is named as the reference labeller or the
    B5/B8 reviewer (OWNER D-60, D-61), which every card must say; elsewhere it must not appear
    as a v2 judge."""
    keep = [x for x in text.splitlines() if "OWNER D-60" not in x and "OWNER D-61" not in x]
    return "\n".join(keep)


def _section(text: str, start: str, end: str) -> str:
    return text[text.index(start) : text.index(end, text.index(start))]


def test_the_release_defaults_to_the_shipped_model_not_v1(src):
    assert src.ft_model == fake.V2


def test_v2_benchmark_names_luna_and_sol_and_has_no_opus_column(text):
    v2 = _section(text, "### Benchmark", f"### {PUBLISHED}")
    assert f"claims extracted by `{LUNA}`, judged by `{SOL}` alone" in v2
    header = next(line for line in v2.splitlines() if line.startswith("| Model"))
    assert f"Correct and grounded ({SOL})" in header and f"Citation precision ({SOL})" in header
    assert OPUS not in v2 and HAIKU not in v2
    rows = [line.split(" | ")[0].strip("|* ") for line in v2.splitlines() if line.startswith("| ")]
    assert rows[1:] == ["gemma-e2b-base-q4", fake.V1, fake.V2]  # base, v1, v2 only


def test_haiku_and_luna_appear_only_as_published_numbers_not_compared(text):
    pub = _section(text, f"### {PUBLISHED}", "### Pre-registered comparisons")
    assert "not as a v2 comparison" in pub and "not comparable" in pub
    assert f"`{HAIKU}` and `{LUNA}` were not re-scored for v2" in pub
    assert f"claims extracted by `{HAIKU}`, judged by `{OPUS}` and `{SOL}`" in pub
    assert f"| {HAIKU} |" in pub and f"| {LUNA} |" in pub
    rest = _not_as_judge(text.replace(pub, ""))
    assert f"| {HAIKU} |" not in rest and f"| {LUNA} |" not in rest
    assert OPUS not in rest and HAIKU not in rest  # Luna is still named, as v2's extractor


def test_v1s_published_numbers_stand_as_they_are(src, text):
    pub = _section(text, f"### {PUBLISHED}", "### Pre-registered comparisons")
    v1_main = next(r for r in src.bundle.results if r.run == "main" and r.model == fake.V1)
    want = card.iv(v1_main.headline[OPUS])
    assert want == "0.388 [0.358, 0.419]" and want in pub  # E3.2's v1 headline under Opus
    assert "Run `main` (`r-f0f3b432a413`)" in pub and "Run `main_citation`" in pub


def test_primary_table_is_the_six_test_sol_family_with_its_scoring(text):
    prim = _section(text, "### Pre-registered comparisons", "### Secondary")
    assert "`configs/eval_plan_v2.yaml`" in prim and "Holm runs over the plan's 6 tests" in prim
    assert f"run `main_v2`, claims extracted by `{LUNA}`, judged by `{SOL}` alone" in prim
    rows = [line for line in prim.splitlines() if line.startswith(f"| {fake.V2} minus")]
    assert len(rows) == 6 and all(f"| {SOL} |" in r for r in rows)
    assert HAIKU not in prim and OPUS not in prim


def test_behaviour_shows_v2s_runs_only_each_labelled(text):
    beh = _section(text, "### Behaviour checks", "### Parity")
    assert f"`main_v2`: claims extracted by `{LUNA}`, judged by `{SOL}` alone" in beh
    runs = {line.split(" | ")[5] for line in beh.splitlines() if line.startswith("| B")}
    assert runs == {"`main_v2`", "`offscope_v2`"}
    assert OPUS not in _not_as_judge(beh) and HAIKU not in beh


def test_model_index_reports_v2_under_sol_only(src, text):
    meta, _ = card.split_front_matter(text)
    assert card.metadata_problems(meta) == []
    metrics = meta["model-index"][0]["results"][0]["metrics"]
    names = [m["name"] for m in metrics]
    assert f"Correct and fully grounded ({SOL})" in names
    assert not any(OPUS in n for n in names)
    v2 = next(r for r in src.bundle.results if r.run == "main_v2" and r.model == fake.V2)
    assert metrics[0]["value"] == v2.accuracy.point


def test_every_number_in_the_v2_card_comes_from_the_inputs(src, text):
    allowed: set[str] = set()
    _numbers(src.raw, allowed)
    unsigned = {a.lstrip("-") for a in allowed}
    stray = sorted({x for x in DECIMAL.findall(text) if x.lstrip("-") not in unsigned})
    assert not stray, stray


def test_readme_block_labels_v2_and_keeps_v1_published_apart(src):
    block = readme.results_block(src)
    head, pub = block.split(f"#### {PUBLISHED}")
    assert f"claims extracted by `{LUNA}`, judged by `{SOL}` alone" in head
    assert f"**{fake.V2}**" in head and OPUS not in head and HAIKU not in head
    assert f"`{HAIKU}` and `{LUNA}` were not re-scored for v2" in pub
    assert f"Correct and grounded ({OPUS})" in pub
