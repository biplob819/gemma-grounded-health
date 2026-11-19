"""SPC-01, SPC-02: compare() with EvalKit's splitter, template and Answer parsing; cited sentences
highlighted, invalid citations flagged, schema failures shown."""

import json

import pytest
from evalkit.schemas import Case
from evalkit.tasks import builtin_task, render_prompt
from evalkit.text import number_sentences

from ggh.space import engine as E

Q = "Does drug X lower blood pressure?"
ABSTRACT = (
    "Drug X was given to 40 adults. Blood pressure fell by 10 mmHg.\n\nNo side effects were seen."
)


def reply(label="yes", cites=(2,), answer="It did.") -> str:
    return json.dumps({"label": label, "answer": answer, "citations": list(cites)})


def test_split_and_prompt_are_evalkits():
    sents = E.split_abstract(ABSTRACT)
    assert sents == number_sentences(tuple(ABSTRACT.split("\n\n")))
    case = Case(id="x", question=Q, context=sents, source="s")
    assert E.build_prompt(Q, sents) == render_prompt(builtin_task("pubmedqa"), case)


def test_compare_gives_one_card_per_model_on_the_same_prompt():
    seen: list[str] = []

    def gen(text):
        def g(prompt):
            seen.append(prompt)
            return text

        return g

    cards = E.compare(Q, ABSTRACT, {"base": gen(reply("no", (1,))), "ft": gen(reply())})
    assert seen[0] == seen[1]
    assert cards["ft"].label == "yes" and cards["ft"].cited[0].text.startswith("Blood pressure")
    assert cards["base"].label == "no"


def test_invalid_citation_is_kept_and_flagged():
    card = E.to_card("ft", reply(cites=(3, 2, 7)), E.split_abstract(ABSTRACT))
    assert card.citations == (2, 3, 7) and card.invalid_citations == (7,)
    assert [c.n for c in card.cited] == [2, 3]
    assert "Invalid citation: sentence 7" in E.card_html("ft", card)


def test_schema_failure_is_shown_not_hidden():
    card = E.to_card("base", "Sure! The answer is yes.", ("a",))
    assert card.error == "schema" and card.label is None
    html = E.card_html("base", card)
    assert "Schema failure" in html and "Sure! The answer is yes." in html


def test_a_failing_model_does_not_take_the_other_down():
    def bad(_):
        raise RuntimeError("cuda")

    cards = E.compare(Q, ABSTRACT, {"base": bad, "ft": lambda _: reply()})
    assert cards["base"].error == "generation" and cards["ft"].label == "yes"


@pytest.mark.parametrize(
    ("q", "a"), [("", ABSTRACT), (Q, ""), ("x" * 501, ABSTRACT), (Q, "A long sentence here. " * 70)]
)
def test_inputs_are_bounded(q, a):
    with pytest.raises(E.InputError):
        E.compare(q, a, {"ft": lambda _: reply()})


def test_highlight_marks_cited_sentences_and_escapes():
    sents = ("<b>one</b>.", "two.")
    html = E.highlight(sents, E.to_card("ft", reply(cites=(1,)), sents))
    assert "<span class='cited'><sup>1</sup> &lt;b&gt;one&lt;/b&gt;.</span>" in html
    assert "<span><sup>2</sup> two.</span>" in html


def test_every_card_carries_the_banner():
    assert E.BANNER in E.card_html("ft", E.to_card("ft", reply(), ("a", "b")))
