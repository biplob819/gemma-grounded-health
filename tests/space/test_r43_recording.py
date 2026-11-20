"""SPC-03, SPC-06, SPC-07 on the committed R4.3 evidence: the recorded local demo (OWNER D-63)."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "docs/assets"


def test_the_local_demo_is_recorded_as_screenshots_and_a_looping_gif():
    for name in ("screenshots/demo_answer.png", "screenshots/demo_passcode_gate.png"):
        data = (ASSETS / name).read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) > 50_000, name
    gif = (ASSETS / "demo.gif").read_bytes()
    assert gif[:6] in (b"GIF89a", b"GIF87a") and b"NETSCAPE2.0" in gif  # loops
    assert len(gif) < 8_000_000  # small enough for a README


def test_the_six_samples_were_read_on_the_running_demo():
    text = (ROOT / "specs/R4.3-demo/RUN-STATE.md").read_text(encoding="utf-8")
    assert "six samples on the running local demo" in text
    for sid in ("dev-1471704", "dev-16716359", "dev-17683644", "dev-21857142",
                "dev-23251408", "dev-26383970"):  # fmt: skip
        assert sid in text


def test_the_first_sample_is_shown_on_arrival():
    pytest.importorskip("gradio")
    from ggh.space import ui

    src = Path(ui.__file__).read_text(encoding="utf-8")
    assert "demo.load(on_pick, pick" in src and "value=first" in src
