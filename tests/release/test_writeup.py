"""PUB-01..04: the write-up is generated from the results bundle and committed evidence (D-20).

The note and the project page come from number-free templates; the decisions summary from a list
whose ids, dates and quotes are checked against docs/decisions.md; the video script from timed
segments. Every committed file must equal `ggh writeup`'s output, every decimal in it must trace
to the bundle or a committed evidence file, and no file may make a claim the bundle or the cards
do not. No network."""

import json
import re
from pathlib import Path

import pytest

from ggh.cli import main
from ggh.release import links, sources, writeup
from tests.release.test_card import DECIMAL, _numbers
from tests.release.test_links import link_problems

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / writeup.OUT_DIR
GENERATED = tuple(writeup.FILES)
ALL_FILES = (*GENERATED, writeup.PRD_SUMMARY)
PRD = ROOT / "Gemma Grounded Health PRD.md"
# names whose digits are part of a name, not a result
NAMES = re.compile(
    r"\$\{[^}]+\}"  # a placeholder
    r"|\]\([^)]*\)"  # a link target
    r"|`[^`]*`"  # code
    r"|Gemma 4 E2B|Gemma 4|GPT-6\.1 Sol|GPT-6 Luna|Claude (?:Opus|Haiku) 5\.5"
    r"|Q4_K_M|Q8_0|bf16|\b(?:OWNER )?D-\d+\b|\bB[1-8]\b|\b[A-Z]\d\.\d\b|\bv\d\b"
)
NUMBER_WORDS = re.compile(
    r"\b(two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty|fifty|"
    r"hundred|thousand|half|dozen|twice|thrice|percent)\b",
    re.I,
)
FORBIDDEN = (
    "v2 beats base on citations",
    "v2 beats the base model on citations",
    "calibrated by a human",
    "human-calibrated",
    "runs on ZeroGPU",
    "ZeroGPU Space serves",
    "v1 beats Luna",
    "v1 beats GPT-6 Luna",
)


@pytest.fixture(scope="module")
def src():
    return sources.load(ROOT)


@pytest.fixture(scope="module")
def story(src):
    """The write-up tells the v1-to-v2 story: the bundle as it stood at v2 (V3.8)."""
    return writeup.story_view(src, ROOT)


@pytest.fixture(scope="module")
def facts(src):
    return writeup.facts(src, ROOT)


@pytest.fixture(scope="module")
def rendered(src):
    return writeup.render_all(src, ROOT)


def _read(name: str) -> str:
    return (OUT / name).read_text(encoding="utf-8")


def _allowed(src) -> set[str]:
    allowed: set[str] = set()
    _numbers(src.raw, allowed)
    story = writeup.story_view(src, ROOT)  # v2's dev check and aim-rule count (V3.8)
    _numbers([story.dev_check, story.aim_check, story.plan], allowed)
    for rel in writeup.EVIDENCE:
        _numbers(json.loads((ROOT / rel).read_text(encoding="utf-8")), allowed)
    ledger = (ROOT / writeup.GPU_LEDGER).read_text(encoding="utf-8").splitlines()
    _numbers([json.loads(ln) for ln in ledger if ln.strip()], allowed)
    return {a.lstrip("-") for a in allowed}


# --- PUB-02 (and every generated file): committed equals the generator --------------------


@pytest.mark.parametrize("name", GENERATED)
def test_committed_files_are_the_generator_output(name, rendered):
    assert _read(name) == rendered[name], "run: uv run ggh writeup"


def test_cli_check_and_write(tmp_path, capsys):
    assert main(["writeup", "--check"]) == 0
    out = tmp_path / "w"
    args = ["writeup", "--root", str(ROOT), "--out-dir", str(out)]
    assert main(args) == 0 and main([*args, "--check"]) == 0
    assert sorted(p.name for p in out.iterdir()) == sorted(GENERATED)
    (out / "project-page.md").write_text("stale\n", encoding="utf-8")
    assert main([*args, "--check"]) == 2
    assert "writeup:" in capsys.readouterr().out


def test_templates_and_prose_carry_no_typed_number():
    texts = {
        "how_i_built_it.md": (writeup.TEMPLATES / "how_i_built_it.md").read_text(encoding="utf-8"),
        "project_page.md": (writeup.TEMPLATES / "project_page.md").read_text(encoding="utf-8"),
    }
    texts.update({f"segment {i}": s.narration for i, s in enumerate(writeup.SEGMENTS)})
    texts.update({did: line for did, line in writeup.DECISIONS})
    for name, text in texts.items():
        stripped = NAMES.sub("", text)
        assert not re.search(r"\d", stripped), (name, re.findall(r".{20}\d.{20}", stripped))
    for name in ("how_i_built_it.md", "project_page.md"):
        words = NUMBER_WORDS.findall(NAMES.sub("", texts[name]))
        assert not words, (name, words)


@pytest.mark.parametrize("name", GENERATED)
def test_every_decimal_traces_to_the_bundle_or_evidence(name, src, rendered):
    allowed = _allowed(src)
    found = DECIMAL.findall(rendered[name])
    if name != "decisions-summary.md":
        assert found, name
    assert sorted({x for x in found if x.lstrip("-") not in allowed}) == []


def test_a_changed_input_changes_the_output(src, facts):
    changed = {**facts, "v2_head_v2run": "0.987654", "gen_tps": "321.0"}
    note = writeup.how_i_built_it(src, ROOT, changed)
    page = writeup.project_page(src, ROOT, changed)
    assert "0.987654" in note and "321.0" in note and "321.0" in page
    with pytest.raises(writeup.WriteupError):
        writeup.fill("${no_such_fact}", facts)


def test_facts_read_the_bundle(src, story, facts):
    b = src.bundle
    v2 = {r.version: r for r in b.releases}["v2"]
    assert story.bundle.shipped == v2 and story.bundle.releases == b.releases[:2]
    main_v2 = {r.model: r for r in b.results if r.run == "main_v2"}
    assert facts["v2_head_v2run"] == f"{main_v2[v2.model].headline['gpt-6.1-sol'].point:.3f}"
    assert facts["n"] == f"{main_v2[v2.model].n:,}"
    # the spend is the full bundle's: with v3b a release, every paid run (OWNER D-75)
    assert facts["total_spend"] == f"${b.spend_usd:.2f}"
    assert facts["paid_runs"] == "All paid runs" and b.shipped.version == "v3b"
    assert writeup.paid_runs(["v1", "v2"]) == "v1's and v2's paid runs"
    assert facts["parity"] == f"{v2.parity.label_agreement:.3f}"  # type: ignore[union-attr]
    assert int(facts["nodes"]) >= 19 and int(facts["gates"]) == 13
    assert int(facts["verifier_rounds"]) >= int(facts["verified_nodes"])


def test_signed_formatting():
    assert writeup.signed(0.373662) == "+0.374"
    assert writeup.signed(-0.475783) == "−0.476"
    assert writeup.signed(-0.0001) == "0.000"


# --- PUB-01: the note -------------------------------------------------------------------------


def _prose(text: str) -> str:
    body = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    return body.split("\n## Links", 1)[0]


def test_note_length():
    words = len(_prose(_read("how-i-built-it.md")).split())
    assert 1200 <= words <= 1800, words


def test_note_covers_each_stage_and_every_caveat():
    text = _read("how-i-built-it.md")
    for heading in (
        "## The problem",
        "## The harness",
        "## Data and targets",
        "## Training on Kaggle",
        "## Export and the laptop",
        "## Evaluation with EvalKit",
        "## v1: the failure",
        "## v2: the retrain",
        "## The caveats",
        "## What the verifiers caught",
    ):
        assert heading in text, heading
    for need in (
        "OWNER D-55",
        "OWNER D-57",
        "OWNER D-59, D-60",
        "OWNER D-61",
        "OWNER D-62",
        "OWNER D-63",
        "lean Kova harness",
        "skeptical verifier",
        "--think=false",
        "not claimed to beat base on citations",
        "Research only, not medical advice",
    ):
        assert need in text, need


def test_note_claims_hold_in_the_bundle(story):
    """The verdicts the prose states in words are the bundle's verdicts (the v1-to-v2 story)."""
    src = story
    b = src.bundle
    base, v1, v2, sol = src.paths.base_model, b.releases[0].model, b.shipped.model, "gpt-6.1-sol"
    verdicts = {(c.model_a, c.model_b, c.metric, c.judge): c.verdict for c in b.primary_comparisons}
    assert verdicts[(v2, v1, "headline", sol)] == "tie"  # "the headline did not move"
    assert verdicts[(v2, v1, "citation_precision", sol)] == "a_better"  # "citations are fixed"
    assert verdicts[(v2, v1, "aim_citation_rate", sol)] == "a_better"
    assert verdicts[(v2, base, "headline", sol)] == "b_better"  # "base is still better"
    worse, held = writeup.aim_outcome(src)
    assert worse and v1 in held  # D-62: base better post hoc; v2 vs v1 holds
    assert all(not a.passed for a in b.shipped.audits)  # "none passed the bar"
    assert b.releases[0].audits[0].passed  # v1's audit passed


# --- PUB-03: decisions -------------------------------------------------------------------------


def test_decision_ids_exist_dates_and_quotes_match():
    rows = writeup.decision_rows((ROOT / writeup.DECISIONS_FILE).read_text(encoding="utf-8"))
    assert 12 <= len(writeup.DECISIONS) <= 18
    assert len({d for d, _ in writeup.DECISIONS}) == len(writeup.DECISIONS)
    text = _read("decisions-summary.md")
    for did, line in writeup.DECISIONS:
        assert did in rows, did
        assert f"- **{did}** ({rows[did].date}): {line}" in text
        for q in writeup.quotes(line):
            assert q in rows[did].row, (did, q)
    # no line names a decision id without its OWNER mark, or the reverse
    for did in re.findall(r"\*\*((?:OWNER )?D-\d+)\*\*", text):
        assert did in rows, did


def test_decisions_generator_refuses_a_bad_id_or_quote(monkeypatch):
    monkeypatch.setattr(writeup, "DECISIONS", (("D-999", "No such decision."),))
    with pytest.raises(writeup.WriteupError, match="D-999"):
        writeup.decisions_summary(ROOT)
    monkeypatch.setattr(writeup, "DECISIONS", (("D-15", 'Owner said "never".'),))
    with pytest.raises(writeup.WriteupError, match="D-15"):  # D-15 is "OWNER D-15"
        writeup.decisions_summary(ROOT)
    monkeypatch.setattr(writeup, "DECISIONS", (("OWNER D-57", 'It said "use Anthropic".'),))
    with pytest.raises(writeup.WriteupError, match="does not say"):
        writeup.decisions_summary(ROOT)


def test_decisions_link_the_prd_and_its_summary():
    text = _read("decisions-summary.md")
    assert "](prd-one-page.md)" in text
    assert "](../../Gemma%20Grounded%20Health%20PRD.md)" in text
    assert "](../decisions.md)" in text


def test_prd_summary_cites_the_prd_and_quotes_only_its_numbers():
    text = _read(writeup.PRD_SUMMARY)
    prd = PRD.read_text(encoding="utf-8")
    assert "](../../Gemma%20Grounded%20Health%20PRD.md)" in text
    for heading in ("## Goal", "## Users", "## Must-haves", "## Success metrics"):
        assert heading in text, heading
    for us in ("US1", "US2", "US3", "US4", "US5", "US6"):
        assert us in text and f"| {us} |" in prd, us
    ids = re.compile(r"\b(?:OWNER )?D-\d+\b|\b(?:US|NFR|FR)\d+\b")
    for num in re.findall(r"\d[\d,]*(?:\.\d+)?", ids.sub("", text)):
        assert num in prd, num
    for q in re.findall(r'"([^"]+)"', text):
        assert q.lower() in prd.lower(), q


# --- PUB-04: the video script ------------------------------------------------------------------


def test_video_is_about_ninety_seconds_and_timed():
    total = sum(s.seconds for s in writeup.SEGMENTS)
    assert 80 <= total <= 100, total
    text = _read("video-script.md")
    stamps = re.findall(r"^\| (\d+):(\d{2})–(\d+):(\d{2}) \|", text, re.M)
    assert len(stamps) == len(writeup.SEGMENTS)
    t = 0
    for m1, s1, m2, s2 in stamps:
        assert int(m1) * 60 + int(s1) == t
        t = int(m2) * 60 + int(s2)
    assert t == total


def test_video_has_the_offline_laptop_segment_and_real_assets():
    text = _read("video-script.md")
    assert "ollama run --think=false" in text and "docs/assets/demo.gif" in text
    assert "Offline" in text
    for path in set(re.findall(r"`(docs/assets/[^`]+)`", text)):
        assert (ROOT / path).is_file(), path
    for need in ("headline_v2.svg", "demo_answer.png", "v1_vs_v2_citation.png"):
        assert need in text, need


@pytest.mark.parametrize("name", ALL_FILES)
def test_no_forbidden_claims(name):
    text = _read(name).lower()
    for claim in FORBIDDEN:
        assert claim.lower() not in text, (name, claim)


# --- links ------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ALL_FILES)
def test_every_link_in_the_writeup_resolves(name):
    text = _read(name)
    assert link_problems(text, OUT) == []


def test_writeup_link_is_live_in_the_readme_but_not_in_the_published_notes():
    assert links.LINKS["writeup"][1] == "docs/writeup/how-i-built-it.md"
    assert (ROOT / "docs/writeup/how-i-built-it.md").is_file()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "- [Write-up](docs/writeup/how-i-built-it.md)" in readme
    notes = (ROOT / "docs" / "release_notes_v1.0.0.md").read_text(encoding="utf-8")
    assert "- Write-up (link pending)" in notes  # v1.0.0's notes, exactly as published
    pending = links.md_abs("writeup", as_released=True, version="1.0.0")
    assert pending == "Write-up (link pending)"
    assert links.md_abs("writeup").startswith("[Write-up](https://github.com/")
    # v1.1.0 shipped after the write-up, so its notes link it
    assert links.md_abs("writeup", as_released=True).startswith("[Write-up](https://github.com/")
    v1_1 = (ROOT / "docs" / "release_notes_v1.1.0.md").read_text(encoding="utf-8")
    assert f"- {links.md_abs('writeup')}" in v1_1


def test_frontier_sentence_matches_the_bundle_verdicts():
    """The note's v1-vs-frontier words follow the bundle's primary verdicts (model_a = v1)."""
    import json as _json

    b = _json.loads((ROOT / "docs/results/results_bundle.json").read_text(encoding="utf-8"))
    found = {}

    def walk(o):
        if isinstance(o, dict):
            if o.get("metric") == "headline" and o.get("model_a") == "gemma-e2b-grounded-q4":
                found[(o.get("model_b"), o.get("judge"))] = o.get("verdict")
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(b)
    assert found[("claude-haiku-5-5", "claude-opus-5-5")] == "tie"
    assert found[("claude-haiku-5-5", "gpt-6.1-sol")] == "a_better"
    assert found[("gpt-6-luna", "claude-opus-5-5")] == "tie"
    note = (ROOT / "docs/writeup/how-i-built-it.md").read_text(encoding="utf-8")
    assert (
        "level with Claude Haiku 5.5 under Claude Opus 5.5 and ahead of it under GPT-6.1 Sol, "
        "and level with GPT-6 Luna under Claude Opus 5.5" in note
    )
