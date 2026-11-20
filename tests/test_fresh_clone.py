"""OSS-05: the fresh-clone check. Its plan is tested always; the clone itself (several minutes,
uv may reach the network) runs only with GGH_FRESH_CLONE=1."""

import os
from pathlib import Path

import pytest

import fresh_clone_check as fc

ROOT = Path(__file__).resolve().parents[1]


def test_steps_clone_head_then_doctor_then_fast_tests(tmp_path):
    s = fc.steps("abc123", tmp_path / "c")
    assert [label for label, _, _ in s] == [
        "clone",
        "checkout",
        "uv sync",
        "ggh doctor",
        "fast tests",
    ]
    assert s[1][1][-1] == "abc123" and s[3][1] == ["uv", "run", "ggh", "doctor"]
    assert "--offline" in fc.steps("x", tmp_path, offline=True)[2][1]
    for rel in fc.FAST_TESTS:
        assert (ROOT / rel).exists(), rel


def test_clean_env_drops_credentials():
    env = fc.clean_env(
        {"PATH": "/bin", "HF_TOKEN": "x", "OPENAI_API_KEY": "x", "GGH_SPACE_PASSKEY": "x"}
    )
    assert env == {"PATH": "/bin"}


@pytest.mark.skipif(os.environ.get("GGH_FRESH_CLONE") != "1", reason="set GGH_FRESH_CLONE=1")
def test_fresh_clone_passes():  # pragma: no cover - slow, local
    assert fc.main([]) == 0
