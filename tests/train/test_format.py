import json

import pytest
from evalkit.pipeline.answer import parse_answer

from ggh.schemas import TrainExample
from ggh.train import format as fmt

EX = TrainExample.model_validate(
    {
        "id": "pqa-a-1001", "pmid": "1001", "kind": "answer", "split": "train",
        "question": "Does it work?", "context": ["It worked [x].", "Other."],
        "prompt": "PROMPT TEXT", "source_label": "yes",
        "target": {"label": "yes", "answer": "It worked [1].", "citations": [1]},
        "rule_ids": ["TGT-R1"], "support": 1.0, "nli_entail": 0.9,
    }
)  # fmt: skip


class FakeTok:
    """A chat template shaped like Gemma 4's: characters are tokens."""

    EOT = 9000

    def __init__(self, break_prefix=False, no_eot=False):
        self.break_prefix = break_prefix
        self.no_eot = no_eot

    def apply_chat_template(self, conversation, **kwargs):
        conv, add_generation_prompt = conversation, kwargs.get("add_generation_prompt", False)
        text = "<bos>" + "".join(
            f"<{m['role']}>{m['content']}" + ("" if self.no_eot and m["role"] == "assistant"
                                              else "|E|") + "\n"
            for m in conv
        )  # fmt: skip
        if add_generation_prompt:
            text += "<assistant>"
        if self.break_prefix and len(conv) == 2:
            text = "X" + text
        # characters are tokens; the three "|E|" characters become one end-of-turn token
        out, i = [], 0
        while i < len(text):
            if text[i : i + 3] == "|E|":
                out.append(self.EOT)
                i += 3
            else:
                out.append(ord(text[i]))
                i += 1
        return {"input_ids": out}

    def convert_tokens_to_ids(self, tokens):
        assert tokens == fmt.END_OF_TURN
        return self.EOT


def test_target_is_compact_json_that_evalkit_parses_back():
    text = fmt.target_text(EX.target)
    assert text == '{"label": "yes", "answer": "It worked [1].", "citations": [1]}'
    assert parse_answer(text, ("yes", "no", "maybe")) == EX.target
    assert fmt.messages(EX) == [
        {"role": "user", "content": "PROMPT TEXT"},
        {"role": "assistant", "content": text},
    ]


def test_prompt_and_generation_prompt_are_masked_and_the_answer_is_kept():
    enc = fmt.encode(FakeTok(), EX)
    prompt = FakeTok().apply_chat_template(fmt.messages(EX, with_answer=False),
                                          add_generation_prompt=True)["input_ids"]  # fmt: skip
    assert enc.input_ids[: len(prompt)] == prompt and enc.n_prompt == len(prompt)
    assert enc.labels[: len(prompt)] == [fmt.IGNORE] * len(prompt)
    kept = enc.labels[len(prompt) :]
    assert fmt.IGNORE not in kept and kept[-1] == FakeTok.EOT  # the end-of-turn is learned
    assert "".join(chr(i) for i in kept[:-1]) == fmt.target_text(EX.target)
    assert enc.input_ids[-1] == FakeTok.EOT  # nothing after the end-of-turn token
    assert enc.n_target == len(kept)


def test_a_template_whose_prompt_is_not_a_prefix_is_refused():
    with pytest.raises(fmt.FormatError, match="not a token prefix"):
        fmt.encode(FakeTok(break_prefix=True), EX)
    with pytest.raises(fmt.FormatError, match="no <turn"):
        fmt.encode(FakeTok(no_eot=True), EX)


def test_long_examples_are_dropped_and_counted_never_truncated():
    n = len(fmt.encode(FakeTok(), EX).input_ids)
    ds = fmt.build(FakeTok(), [EX, EX], max_len=n)
    assert len(ds.rows) == 2 and ds.dropped_too_long == 0 and ds.tokens == 2 * n
    ds = fmt.build(FakeTok(), [EX], max_len=n - 1)
    assert ds.rows == [] and ds.dropped_too_long == 1


def test_collate_right_pads_with_ignored_labels_and_zero_attention():
    a = fmt.Encoded([1, 2, 3], [-100, 2, 3], 1, 2)
    b = fmt.Encoded([4, 5], [-100, 5], 1, 1)
    out = fmt.collate([a, b], pad_id=0)
    assert out["input_ids"] == [[1, 2, 3], [4, 5, 0]]
    assert out["labels"] == [[-100, 2, 3], [-100, 5, -100]]
    assert out["attention_mask"] == [[1, 1, 1], [1, 1, 0]]


@pytest.fixture(scope="module")
def gemma_tok():
    tr = pytest.importorskip("transformers")
    try:
        return tr.AutoTokenizer.from_pretrained(
            "google/gemma-4-E2B-it",
            revision="4adf15748de9e1a0097b6274cee9c15836c41c37",
            local_files_only=True,
        )
    except OSError:
        pytest.skip("the pinned Gemma 4 tokenizer is not in the local Hugging Face cache")


@pytest.mark.recipe
def test_masking_with_the_real_gemma_template(gemma_tok):
    enc = fmt.encode(gemma_tok, EX)
    prompt = gemma_tok.apply_chat_template(
        fmt.messages(EX, with_answer=False), tokenize=True, add_generation_prompt=True
    )["input_ids"]
    assert enc.input_ids[: len(prompt)] == list(prompt)
    assert gemma_tok.decode(enc.input_ids[: len(prompt)]).endswith("<|turn>model\n")
    kept = [t for t in enc.labels if t != fmt.IGNORE]
    assert gemma_tok.convert_ids_to_tokens(kept[-1]) == "<turn|>"
    answer = gemma_tok.decode(kept[:-1])
    assert json.loads(answer) == json.loads(fmt.target_text(EX.target))
    assert parse_answer(answer, ("yes", "no", "maybe")) == EX.target
