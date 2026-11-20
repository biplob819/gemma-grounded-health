"""OSS-02, OSS-06: the README results block and links come from the bundle and the links registry;
the README (which replaced docs/README_release_draft.md at v1.0.0) has its sections, its real
screenshots and both generated blocks in sync."""

import re
from pathlib import Path

import pytest

from ggh.cli import main
from ggh.release import links, readme, sources
from tests.release import build_v1_bundle as B

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"
# OWNER D-37: real screenshots of the release (the demo is local, OWNER D-63)
SCREENSHOTS = (
    "docs/assets/headline_v2.svg",
    "docs/assets/screenshots/v1_vs_v2_citation.png",
    "docs/assets/demo.gif",
    "docs/assets/screenshots/hf_model_page.png",
    "docs/assets/screenshots/demo_answer.png",
    "docs/assets/screenshots/demo_passcode_gate.png",
    "docs/assets/screenshots/evalkit_report_main_v2.png",
)


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


def test_readme_is_in_sync_with_the_bundle():
    src = sources.load(ROOT, _bundle_paths())
    text = README.read_text(encoding="utf-8")
    assert readme.update(text, src) == text, "run: uv run ggh readme-table --write README.md"
    assert readme.LINKS_START in text and readme.START in text


def test_readme_sections_and_screenshots():
    text = README.read_text(encoding="utf-8")
    for h in (
        "## v2 result: the citations are fixed, the headline is not",
        "## Results",
        "## Run it offline",
        "## Step-by-step guide",
        "## Screenshots",
        "## Links",
        "## Licence and credits",
    ):
        assert f"\n{h}\n" in text
    shots = re.findall(r"!\[[^\]]+\]\(([^)]+)\)", text)
    local = [s for s in shots if not s.startswith("https://")]
    assert sorted(local) == sorted(SCREENSHOTS)
    assert text.count("Research only, not medical advice") >= 2


def test_links_block_lists_every_link_and_marks_the_pending_ones():
    block = readme.links_block()
    for name, (label, url) in links.LINKS.items():
        assert (f"[{label}]({url})" if url else f"{label} {links.PENDING}") in block, name
    # R4.2 published both Hub repos; R4.3 recorded the local demo (OWNER D-63)
    assert not {"hf_model", "hf_gguf", "space", "parity"} & set(links.pending())
    # R4.5 filled the release (the v1.0.0 tag) and the EvalKit report page; only the write-up
    # (P5.1) may stay pending
    assert set(links.pending()) <= {"writeup"}


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


def test_the_readme_says_the_demo_serves_the_benchmarked_gguf():
    """OWNER D-63: the local demo serves the Q4_K_M GGUF that was benchmarked, not bf16."""
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "demo runs bf16" not in text and "The demo runs" not in text
    assert "The local demo serves the benchmarked Q4_K_M GGUF" in text
