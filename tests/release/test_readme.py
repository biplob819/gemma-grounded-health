"""OSS-02, OSS-06: the README results block and links come from the bundle and the links registry;
the release draft has its sections, screenshot slots and both generated blocks in sync."""

import re
from pathlib import Path

import pytest

from ggh.cli import main
from ggh.release import links, readme, sources
from tests.release import build_v1_bundle as B

ROOT = Path(__file__).resolve().parents[2]
DRAFT = ROOT / "docs" / "README_release_draft.md"
SCREENSHOTS = ("docs/assets/screenshot-space.png", "docs/assets/screenshot-evalkit-report.png")


def _bundle_paths() -> sources.Paths:
    """The real bundle once E3.3 has written it; the v1 fixture until then."""
    if (ROOT / sources.BUNDLE).is_file():
        return sources.Paths()
    return sources.Paths(bundle=B.BUNDLE.relative_to(ROOT))


@pytest.fixture(scope="module")
def fixture_src():
    return sources.load(ROOT, sources.Paths(bundle=B.BUNDLE.relative_to(ROOT)))


@pytest.mark.golden
def test_block_matches_the_golden_file(fixture_src):
    assert readme.results_block(fixture_src) == B.README_BLOCK.read_text(encoding="utf-8")


def test_block_has_every_model_and_the_stamps(fixture_src):
    block = readme.results_block(fixture_src)
    for m in fixture_src.main().models:
        assert m.model in block
    assert f"**{fixture_src.ft_model}**" in block
    assert "judge not calibrated" in block and "Research only, not medical advice" in block


def test_draft_is_in_sync_with_the_bundle():
    src = sources.load(ROOT, _bundle_paths())
    text = DRAFT.read_text(encoding="utf-8")
    assert readme.update(text, src) == text, "run: uv run ggh readme-table --write " + str(
        DRAFT.relative_to(ROOT)
    )


def test_draft_sections_and_screenshot_slots():
    text = DRAFT.read_text(encoding="utf-8")
    for h in (
        "## Results",
        "## Try it",
        "## The EvalKit report",
        "## Reproduce from scratch",
        "## Links",
        "## Licence and credits",
    ):
        assert f"\n{h}\n" in text
    shots = re.findall(r"!\[[^\]]+\]\(([^)]+)\)", text)
    assert sorted(shots) == sorted(SCREENSHOTS)
    assert all(s.startswith("docs/assets/") for s in shots)
    assert "Research only, not medical advice" in text


def test_links_block_lists_every_link_and_marks_the_pending_ones():
    block = readme.links_block()
    for name, (label, url) in links.LINKS.items():
        assert (f"[{label}]({url})" if url else f"{label} {links.PENDING}") in block, name
    assert set(links.pending()) >= {"hf_model", "hf_gguf", "space", "release"}


def test_splice_refuses_a_file_without_markers():
    with pytest.raises(readme.SpliceError):
        readme.splice("# README\n", "block")


def test_cli_prints_and_writes(tmp_path, capsys):
    rel = B.BUNDLE.relative_to(ROOT)
    assert main(["readme-table", "--root", str(ROOT), "--bundle", str(rel)]) == 0
    assert readme.START in capsys.readouterr().out
    target = tmp_path / "R.md"
    target.write_text(f"# T\n\n{readme.START}\nold\n{readme.END}\n", encoding="utf-8")
    assert (
        main(["readme-table", "--root", str(ROOT), "--bundle", str(rel), "--write", str(target)])
        == 0
    )
    assert "old" not in target.read_text(encoding="utf-8")
    bad = tmp_path / "bad.md"
    bad.write_text("# no markers\n", encoding="utf-8")
    assert (
        main(["readme-table", "--root", str(ROOT), "--bundle", str(rel), "--write", str(bad)]) == 2
    )


def test_card_cli_writes_a_valid_card(tmp_path):
    rel = B.BUNDLE.relative_to(ROOT)
    out = tmp_path / "MODEL_CARD.md"
    assert main(["card", "--root", str(ROOT), "--bundle", str(rel), "--out", str(out)]) == 0
    assert out.read_text(encoding="utf-8") == B.CARD.read_text(encoding="utf-8")
    assert main(["card", "--root", str(ROOT), "--bundle", "nope.json"]) == 2
