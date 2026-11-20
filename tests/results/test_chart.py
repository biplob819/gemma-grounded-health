"""The V2.2 headline chart is generated from the committed main_v2 summary and never edited."""

import json
import re
from pathlib import Path

import pytest

from ggh.results import chart

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "docs/results/main_v2"
SVG = ROOT / "docs/assets/headline_v2.svg"

pytestmark = pytest.mark.skipif(not (RUN / "summary.json").is_file(), reason="no main_v2 yet")


def _summary() -> dict:
    return json.loads((RUN / "summary.json").read_text(encoding="utf-8"))


def test_committed_chart_is_exactly_what_the_summary_generates():
    run_id = json.loads((RUN / "run.json").read_text(encoding="utf-8"))["run_id"]
    assert SVG.read_text(encoding="utf-8") == chart.svg(_summary(), run_id=run_id)


def test_every_plotted_number_is_the_summary_value_under_sol():
    text = SVG.read_text(encoding="utf-8")
    vals = chart.values(_summary(), "gpt-6.1-sol")
    found = re.search(r"<desc[^>]*>(.*?)</desc>", text, re.S)
    assert found
    desc = found.group(1)
    for model, label in chart.MODELS:
        for metric, _, _ in chart.PANELS:
            v = vals[model][metric]
            assert f"{v['point']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]" in desc, (label, metric)
            assert f">{v['point']:.3f}</text>" in text  # printed beside its mark
    assert "GPT-6 Luna" in text and "GPT-6.1 Sol" in text and "Opus" not in text  # OWNER D-57
    assert "prefers-color-scheme:dark" in text  # a selected dark mode
