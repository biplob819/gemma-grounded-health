"""OSS-05, OSS-06 (OWNER D-37): every link in the README, the release notes and the links
registry is well formed and every local path exists; the screenshots are real image files; and
every number the README's prose quotes comes from the bundle or its sources (D-20). Offline:
an https link is checked for its host and, when it points into this repo, for the file it
names. Whether a page answers is the builder's check after the tag (two pages go live then:
the v1.0.0 release and EvalKit's /gemma/ page)."""

import re
import tomllib
from pathlib import Path
from urllib.parse import unquote, urlparse

import pytest

from ggh.release import links, notes, readme, sources
from tests.release.test_card import DECIMAL, _numbers

ROOT = Path(__file__).resolve().parents[2]
DOCS = (
    ROOT / "README.md",
    ROOT / notes.OUT,  # v1.1.0's notes
    ROOT / "docs" / "release_notes_v1.0.0.md",
    ROOT / "docs" / "reproduce.md",
)
ALLOWED_HOSTS = {
    "github.com",
    "raw.githubusercontent.com",
    "huggingface.co",
    "biplob819.github.io",
    "pubmedqa.github.io",
    "docs.astral.sh",
    "ollama.com",
}
OWN = ("biplob819/gemma-grounded-health",)
BRANCH_PATHS = (["blob", "main"], ["tree", "main"], ["raw", "main"])
LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)\)")
SCREENSHOT_FILES = {
    "docs/assets/screenshots/hf_model_page.png": b"\x89PNG\r\n\x1a\n",
    "docs/assets/screenshots/demo_answer.png": b"\x89PNG\r\n\x1a\n",
    "docs/assets/screenshots/demo_passcode_gate.png": b"\x89PNG\r\n\x1a\n",
    "docs/assets/screenshots/v1_vs_v2_citation.png": b"\x89PNG\r\n\x1a\n",
    "docs/assets/screenshots/evalkit_report_main_v2.png": b"\x89PNG\r\n\x1a\n",
    "docs/assets/demo.gif": b"GIF89a",
    "docs/assets/walkthrough.gif": b"GIF89a",
}


def _local_target(url: str, base: Path) -> Path | None:
    """The file a link names in this repo, or None for an external page."""
    u = urlparse(url)
    if not u.scheme:
        return (base / unquote(u.path)).resolve() if u.path else None
    parts = u.path.strip("/").split("/")
    own = "/".join(parts[:2]) in OWN
    if u.netloc == "github.com" and own and len(parts) > 4 and parts[2:4] in BRANCH_PATHS:
        return ROOT / unquote("/".join(parts[4:]))
    if u.netloc == "raw.githubusercontent.com" and "/".join(parts[:2]) in OWN:
        return ROOT / unquote("/".join(parts[3:]))
    return None


def link_problems(text: str, base: Path) -> list[str]:
    out = []
    for url in LINK.findall(text):
        if url.startswith("#"):
            continue
        u = urlparse(url)
        if u.scheme and u.scheme != "https":
            out.append(f"{url}: not https")
        elif u.scheme and u.netloc not in ALLOWED_HOSTS:
            out.append(f"{url}: host {u.netloc} is not an expected host")
        target = _local_target(url, base)
        if target is not None and not target.exists():
            out.append(f"{url}: {target} does not exist")
    return out


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_every_link_is_well_formed_and_every_local_path_exists(doc):
    text = doc.read_text(encoding="utf-8")
    assert LINK.findall(text)
    assert link_problems(text, doc.parent) == []


def test_link_checker_catches_bad_links(tmp_path):
    bad = (
        "[a](http://github.com/x) [b](https://evil.example/x) [c](missing.md) "
        "[d](https://github.com/biplob819/gemma-grounded-health/blob/main/nope.md)"
    )
    assert len(link_problems(bad, tmp_path)) == 4
    assert link_problems("[ok](#links) [ok](https://huggingface.co/x)", tmp_path) == []


def test_registry_links_and_the_readme_carries_every_one():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert set(links.pending()) <= {"writeup"}  # P5.1 may stay pending
    for name, (_, url) in links.LINKS.items():
        assert links.md(name) in text, name
        if url:
            assert link_problems(f"[x]({url})", ROOT) == [], name
    assert links.LINKS["release"][1] == f"{links.REPO}/releases/tag/v1.1.0"  # V3.8: v3b
    assert links.LINKS["release_notes"][1] == "docs/release_notes_v1.1.0.md"
    # v1.0.0's links as published, for its notes (docs/release_notes_v1.0.0.md)
    old = links.links_at("1.0.0")
    assert old["release"][1] == f"{links.REPO}/releases/tag/v1.0.0"
    assert old["release_notes"][1] == "docs/release_notes_v1.0.0.md"
    assert old["parity"][1] == f"{links.REPO}/blob/main/docs/measurements/v2/parity.json"
    for name, (_, url) in old.items():
        if url:
            assert link_problems(f"[x]({url})", ROOT) == [], name
    assert links.LINKS["evalkit_report"][1] == "https://biplob819.github.io/evalkit/gemma/"
    assert links.absolute("docs/x.md") == f"{links.REPO}/blob/main/docs/x.md"


def test_readme_has_every_important_link():  # OWNER D-37
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for url in (
        "https://huggingface.co/biplob123/gemma-4-e2b-grounded-health",
        "https://huggingface.co/biplob123/gemma-4-e2b-grounded-health-GGUF",
        "https://biplob819.github.io/evalkit/gemma/",
        "https://github.com/biplob819/evalkit/pull/1",
        "docs/reproduce.md",
        "docs/release_notes_v1.0.0.md",
        "https://github.com/biplob819/gemma-grounded-health/releases/tag/v1.1.0",  # V3.8: v3b
        "docs/release_notes_v1.1.0.md",
        "uv run --group demo ggh space --local --env-file .env",
        "ollama run --think=false hf.co/biplob123/gemma-4-e2b-grounded-health-GGUF:Q4_K_M",
    ):
        assert url in text, url


@pytest.mark.parametrize("rel", sorted(SCREENSHOT_FILES))
def test_screenshot_files_are_real_images(rel):
    data = (ROOT / rel).read_bytes()
    assert data.startswith(SCREENSHOT_FILES[rel]) and len(data) > 20_000, rel


def test_headline_svg_is_an_svg():
    text = (ROOT / "docs/assets/headline_v2.svg").read_text(encoding="utf-8")
    assert text.lstrip().startswith("<svg") and "</svg>" in text


def test_readme_prose_numbers_come_from_the_bundle():  # D-20
    src = sources.load(ROOT)
    allowed: set[str] = set()
    _numbers(src.raw, allowed)
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    allowed.add(pyproject["project"]["requires-python"].lstrip(">="))  # "Python 3.11+"
    # V3.8: the v3b section names v3's headline; v3 is no release, so its committed result file
    # (docs/results/main_v3/result.json, V3.5) is the source, not the bundle
    import json

    _numbers(json.loads((ROOT / "docs/results/main_v3/result.json").read_text("utf-8")), allowed)
    unsigned = {a.lstrip("-") for a in allowed}
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    head, rest = text.split(readme.START, 1)
    prose = head + rest.split(readme.END, 1)[1]
    found = DECIMAL.findall(prose)
    assert len(found) > 15
    assert sorted({x for x in found if x.lstrip("-") not in unsigned}) == []
