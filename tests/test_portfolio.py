"""portfolio.yml: the images exist, the priorities list follows its rules (4-7 built, 2-5 cut,
2-5 later; plain words, decision codes only in `source`), and every decimal it quotes comes
from the results bundle or its sources (D-20)."""

import json
import re
from pathlib import Path

import pytest
import yaml

from ggh.release import sources
from tests.release.test_card import DECIMAL, _numbers

ROOT = Path(__file__).resolve().parents[1]
CODE = re.compile(r"\b(?:OWNER )?D-\d+\b|\b[A-Z]\d\.\d\b")
DECISION = re.compile(r"^(?:OWNER )?D-\d+$")
COUNTS = {"built": (4, 7), "cut": (2, 5), "later": (2, 5)}


@pytest.fixture(scope="module")
def port():
    return yaml.safe_load((ROOT / "portfolio.yml").read_text(encoding="utf-8"))


def _decisions() -> set[str]:
    text = (ROOT / "docs" / "decisions.md").read_text(encoding="utf-8")
    text += (ROOT / "Gemma Grounded Health Tech Spec.md").read_text(encoding="utf-8")
    return set(re.findall(r"^\| ((?:OWNER )?D-\d+) \|", text, re.M))


def test_images_exist_with_alt_text(port):
    srcs = [i["src"] for i in port["images"]]
    for img in port["images"]:
        assert (ROOT / img["src"]).is_file(), img["src"]
        assert len(img["alt"]) > 20 and img["kind"]
    for need in (
        "docs/assets/walkthrough.gif",
        "docs/assets/demo.gif",
        "docs/assets/headline_v2.svg",
        "docs/assets/screenshots/hf_model_page.png",
        "docs/assets/screenshots/demo_answer.png",
        "docs/assets/screenshots/demo_passcode_gate.png",
        "docs/assets/screenshots/v1_vs_v2_citation.png",
        "docs/assets/screenshots/evalkit_report_main_v2.png",
    ):
        assert need in srcs, need


def test_priorities_follow_the_rules(port):
    items = port["priorities"]
    decisions = _decisions()
    for call, (lo, hi) in COUNTS.items():
        n = sum(p["call"] == call for p in items)
        assert lo <= n <= hi, (call, n)
    for p in items:
        assert set(p) == {"item", "call", "why", "source"}, p
        assert p["call"] in COUNTS
        assert len(p["item"].split()) <= 15, p["item"]
        why = p["why"].strip()
        assert why.endswith(".") and ". " not in why[:-1], why  # one sentence
        assert not CODE.search(p["item"]) and not CODE.search(why), p
        src = p["source"]
        assert (ROOT / src).exists() or (DECISION.match(src) and src in decisions), src
    cut = {p["source"] for p in items if p["call"] == "cut"}
    assert {"OWNER D-58", "OWNER D-63"} <= cut  # the DPO and ZeroGPU cuts


def test_quoted_numbers_come_from_the_bundle(port):
    src = sources.load(ROOT)
    allowed: set[str] = set()
    _numbers(src.raw, allowed)
    for rel in ("docs/measurements/v2/parity.json",):
        _numbers(json.loads((ROOT / rel).read_text(encoding="utf-8")), allowed)
    text = json.dumps(
        {k: port[k] for k in ("metrics", "story", "cost")}, ensure_ascii=False
    ).replace("≥ 0.95", "")  # the parity bar is a target, not a result
    found = DECIMAL.findall(text)
    assert len(found) > 10
    assert sorted({x for x in found if x not in allowed}) == []
