import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

import ggh
from ggh import cli

ROOT = Path(__file__).resolve().parent.parent


def test_cli_version_flag_prints_version(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == "ggh 1.0.0"


@pytest.mark.contract
def test_pyproject_distribution_metadata():
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert meta["name"] == "gemma-grounded-health"
    assert meta["version"] == ggh.__version__ == "1.0.0"
    assert meta["requires-python"] == ">=3.11"
    assert meta["license"] == "Apache-2.0"
    assert meta["license-files"] == ["LICENSE", "NOTICE"]
    assert meta["scripts"] == {"ggh": "ggh.cli:main"}


def test_licence_is_apache_and_notice_credits_sources():
    lic = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "Apache License" in lic and "Version 2.0, January 2004" in lic
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
    assert "Gemma 4" in notice and "PubMedQA" in notice and "MIT" in notice
    assert "not medical advice" in notice.lower()


def test_help_lists_doctor(capsys):
    assert cli.main([]) == 0
    assert "doctor" in capsys.readouterr().out


def test_doctor_ready_on_supported_python(capsys):
    assert cli.main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "[ok] python" in out and out.strip().endswith("ready")


def test_doctor_fails_below_python_floor(capsys, monkeypatch):
    monkeypatch.setattr(cli, "MIN_PYTHON", (99, 0))
    assert cli.main(["doctor"]) == cli.EXIT_ERROR
    out = capsys.readouterr().out
    assert "[FAIL] python" in out and out.strip().endswith("not ready")


def test_cli_runs_as_module():
    out = subprocess.run(
        [sys.executable, "-m", "ggh.cli", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "ggh 1.0.0"


@pytest.mark.safety
def test_doctor_names_credentials_but_never_prints_values(capsys, monkeypatch):
    secret = "hf_" + "Zz9" * 12
    monkeypatch.setenv("HF_TOKEN", secret)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    assert cli.main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "HF_TOKEN set" in out and "KAGGLE_KEY not set" in out
    assert secret not in out and secret[:8] not in out
