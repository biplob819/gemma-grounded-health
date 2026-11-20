"""OSS-04: the v1.0.0 release notes are the generator's output, from the bundle (results and
comparisons with the OWNER D-62 caveat), the Hub release plan (artifacts and sha256) and the
known issues (limits). No network."""

import json
import re
from pathlib import Path

import pytest

from ggh.cli import main
from ggh.release import links, notes, sources
from tests.release.test_card import DECIMAL, _numbers

ROOT = Path(__file__).resolve().parents[2]
NOTES = ROOT / notes.OUT


@pytest.fixture(scope="module")
def src():
    return sources.load(ROOT)


@pytest.fixture(scope="module")
def text():
    return NOTES.read_text(encoding="utf-8")


def test_committed_notes_are_the_generator_output(src, text):
    assert text == notes.render(src, ROOT), "run: uv run ggh release-notes"
    assert notes.OUT.as_posix() == "docs/release_notes_v1.0.0.md"


def test_cli_check_and_write(tmp_path, capsys):
    assert main(["release-notes", "--check"]) == 0
    out = tmp_path / "n.md"
    rel = Path("docs") / "results" / "results_bundle.json"
    args = ["release-notes", "--root", str(ROOT), "--bundle", str(rel), "--out", str(out)]
    assert main(args) == 0 and main([*args, "--check"]) == 0
    out.write_text("stale\n", encoding="utf-8")
    assert main([*args, "--check"]) == 2
    assert "release-notes:" in capsys.readouterr().out


def test_results_numbers_come_from_the_bundle(src, text):
    results = text[text.index("## Results") : text.index("## Artifacts")]
    allowed: set[str] = set()
    _numbers(src.raw, allowed)
    unsigned = {a.lstrip("-") for a in allowed}
    found = DECIMAL.findall(results)
    assert len(found) > 40
    assert sorted({x for x in found if x.lstrip("-") not in unsigned}) == []
    for r in (r for r in src.bundle.results if r.run == src.ref(src.main()).name):
        assert r.model in results


def test_d62_caveat_and_no_v2_beats_base_citation_claim(text):
    assert "#### Citation results and the aim rule (OWNER D-62)" in text
    assert "so v2 is not claimed to beat base on citations" in text
    assert "v2-vs-v1 holds under both counts" in text
    assert "So this release does not claim that v2 beats base on citations" in text
    assert "Research only, not medical advice" in text


def test_every_planned_file_is_listed_with_its_sha256(text):
    plan = json.loads((ROOT / notes.PLAN).read_text(encoding="utf-8"))
    for f in plan["files"]:
        assert f"| `{f['repo']}/{f['path_in_repo']}` |" in text
        assert f"`{f['sha256']}`" in text
    assert f"{plan['total_bytes']:,} bytes" in text


def test_hub_status_follows_the_download_back_record():
    plan = {
        "release": "v2",
        "model": "m",
        "files": [
            {"repo": "o/r", "path_in_repo": "a", "size": 1, "sha256": "x"},
            {"repo": "o/r", "path_in_repo": "b", "size": 2000, "sha256": "y"},
        ],
        "total_bytes": 2001,
        "lineage_verified_from": "p",
        "lineage_chain": ["s"],
        "problems": [],
    }
    downloads = [
        {"file": "o/r/a", "sha256": "x", "ok": True},
        {"file": "o/r/b", "sha256": "old", "ok": True},
    ]
    rec = {"tag": "v2.0.0", "downloads": downloads}
    out = notes.artifacts(plan, rec)
    assert "| `o/r/a` | 1 B | `x` | published |" in out
    assert "| `o/r/b` | 2.00 KB | `y` | re-upload pending |" in out
    assert "1 regenerated card(s)" in out


def test_known_limits_are_the_open_and_disclosed_release_rows(text):
    table = (ROOT / notes.KNOWN_ISSUES).read_text(encoding="utf-8")
    issues = notes.known_limits(table)
    assert issues and all(i.node in notes.RELEASE_NODES for i in issues)
    assert all(i.status.startswith(notes.LIMIT_STATUS) for i in issues)
    for i in issues:
        assert f"- **{i.node}.** {i.finding} *Status:* {i.status}" in text
    accepted = [ln for ln in table.splitlines() if ln.startswith("| R4.3 ") and "Accepted" in ln]
    assert accepted and not any(ln.split(" | ")[1] in text for ln in accepted)
    fake = "\n".join(
        (
            "| R4.2 | a | P | L | Open; x |",
            "| W0.1 | b | P | L | Open; y |",
            "| R4.3 | c | P | L | Ok |",
        )
    )
    assert [i.finding for i in notes.known_limits(fake)] == ["a"]


def test_links_are_absolute(text):
    body = text[text.index("## Links") :]
    for label, url in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", body):
        assert url.startswith("https://"), (label, url)
    assert links.LINKS["release"][1] in body
