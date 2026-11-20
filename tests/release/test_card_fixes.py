"""Known issue R4.2, fixed in R4.5: the main card says its weights are a fine-tune of the base
(not an adapter), and both cards say a JSON answer needs EvalKit's PubMedQA prompt. Offline;
the re-upload to the Hub is the builder's outward step."""

from pathlib import Path

import pytest

from ggh.release import card, hub, sources
from ggh.space.engine import build_prompt

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def src():
    return sources.load(ROOT)


def test_main_card_front_matter_says_finetune():
    meta, _ = card.split_front_matter((ROOT / "MODEL_CARD.md").read_text(encoding="utf-8"))
    assert meta["base_model_relation"] == "finetune"
    assert meta["base_model"] == "google/gemma-4-E2B-it"
    assert card.metadata_problems(meta) == []
    assert card.metadata_problems({**meta, "base_model_relation": "adapter"})


def test_main_card_shows_the_benchmark_prompt():
    text = (ROOT / "MODEL_CARD.md").read_text(encoding="utf-8")
    prompt = build_prompt("<question>", card.EXAMPLE_SENTENCES).rstrip()
    assert "## How to use" in text and prompt in text
    assert "ollama run --think=false hf.co/biplob123/gemma-4-e2b-grounded-health-GGUF" in text


def test_gguf_card_points_at_the_prompt_on_the_main_card(src):
    text = hub.gguf_card("biplob123", [], src.bundle)
    assert "**Prompt format.**" in text and "EvalKit's PubMedQA prompt" in text
    assert "(https://huggingface.co/biplob123/gemma-4-e2b-grounded-health)" in text
