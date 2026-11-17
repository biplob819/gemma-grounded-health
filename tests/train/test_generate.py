import json

import pytest
from evalkit.pipeline.answer import parse_answer

from ggh.train import generate as G
from tests.train.test_run import tiny  # noqa: F401  (fixture)


def test_left_pad_aligns_every_row_to_the_right():
    ids, mask = G.left_pad([[5, 6, 7], [8]], pad_id=0)
    assert ids == [[5, 6, 7], [0, 0, 8]] and mask == [[1, 1, 1], [0, 0, 1]]


def test_schema_and_separators_match_evalkit_and_the_training_targets():
    from evalkit.schemas import Answer

    from ggh.schemas import TrainExample  # noqa: F401
    from ggh.train.format import target_text

    s = json.loads(G.schema_json())
    assert s["properties"]["label"]["enum"] == ["yes", "no", "maybe"]
    assert s["required"] == ["label", "answer", "citations"]
    text = target_text(Answer(label="yes", answer="A.", citations=(1, 2)))
    assert text == '{"label": "yes", "answer": "A.", "citations": [1, 2]}'
    assert G.SEPARATORS == (", ", ": ")  # exactly the separators in that text


class Tok:
    pad_token_id = 0
    eos_token_id = 1

    def convert_tokens_to_ids(self, t):
        return 106

    def apply_chat_template(self, conv, **kw):
        return {"input_ids": [2, *range(10, 10 + len(conv[0]["content"]))]}


def test_stop_ids_are_end_of_turn_and_eos():
    assert G.stop_ids(Tok()) == [106, 1]
    assert G.prompt_ids(Tok(), "abc") == [2, 10, 11, 12]


@pytest.mark.recipe
def test_constrained_generation_on_a_tiny_gemma4_gives_evalkit_valid_json(tiny):  # noqa: F811
    xgr = pytest.importorskip("xgrammar")
    import torch  # pyright: ignore[reportMissingImports]

    tok, make = tiny
    torch.manual_seed(0)
    model = make().eval()
    info = xgr.TokenizerInfo.from_huggingface(tok, vocab_size=len(tok))
    schema = json.loads(G.schema_json())
    schema["properties"]["answer"]["maxLength"] = 40  # a random model never ends a string
    grammar = xgr.GrammarCompiler(info).compile_json_schema(
        json.dumps(schema), any_whitespace=False, separators=G.SEPARATORS
    )
    prompts = ["Does it work?", "A longer question about outcomes and citations here?"]
    out = G.generate(model, tok, prompts, grammar=grammar, batch_size=2, max_new_tokens=160)
    assert len(out) == 2
    for text in out:
        a = parse_answer(text, ("yes", "no", "maybe"))  # valid in EvalKit's own parser
        assert text.startswith('{"label": "') and a.label in ("yes", "no", "maybe")
    free = G.generate(model, tok, prompts, batch_size=1, max_new_tokens=8)
    assert len(free) == 2  # unconstrained: whatever it says, in order
    assert G.generate(model, tok, prompts[:1], grammar=grammar, max_new_tokens=160) == out[:1]
