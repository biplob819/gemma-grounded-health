import pytest

from ggh.train import dev

ROWS = [
    {"sentences": ["a", "b", "c"], "label": "yes"},
    {"sentences": ["a", "b"], "label": "no"},
    {"sentences": ["a"], "label": "yes"},
    {"sentences": ["a", "b", "c", "d"], "label": "no"},
]


def ans(label, citations, words=3):
    cites = ", ".join(map(str, citations))
    text = " ".join(["word"] * words)
    return f'{{"label": "{label}", "answer": "{text}", "citations": [{cites}]}}'


def test_metrics_are_computed_from_texts_with_evalkits_parser():
    constrained = [
        ans("yes", [1, 3]),  # valid, correct
        ans("yes", [3]),  # citation 3 out of range for 2 sentences; wrong label
        "not json at all",  # schema failure counts against everything
        ans("no", [], words=81),  # no citation, over 80 words, correct
    ]
    unconstrained = [ans("yes", [1]), '```json\n{"label": "yes"}\n```', "x", ans("no", [1])]
    m = dev.dev_metrics(ROWS, constrained, unconstrained)
    assert m.n == 4
    assert m.schema_valid == 3 / 4 and m.schema_valid_unconstrained == 2 / 4
    assert m.has_citation == 2 / 4 and m.citation_valid == 1 / 4
    assert dict(m.label_accuracy) == {"no": 1 / 2, "yes": 1 / 2}
    assert m.mean_words == pytest.approx((3 + 3 + 81) / 3) and m.over_80_words == 1 / 3


def test_string_citations_are_schema_failures_as_in_the_benchmark():
    text = '{"label": "yes", "answer": "x", "citations": ["1"]}'
    m = dev.dev_metrics(ROWS[:1], [text], [text])
    assert m.schema_valid == 0 and m.has_citation == 0 and m.mean_words == 0


def test_labels_match_case_insensitively_like_evalkit():
    m = dev.dev_metrics(ROWS[:1], [ans("YES", [1])], [ans("yes", [1])])
    assert dict(m.label_accuracy) == {"yes": 1.0}


def test_inputs_must_be_aligned():
    with pytest.raises(ValueError, match="aligned"):
        dev.dev_metrics(ROWS, [ans("yes", [1])], [ans("yes", [1])])
    with pytest.raises(ValueError):
        dev.dev_metrics([], [], [])


def test_selection_bar_needs_all_three_at_098():
    good = dev.dev_metrics(ROWS[:1], [ans("yes", [1])], [ans("yes", [1])])
    assert dev.passes_selection(good)
    for field in ("schema_valid", "has_citation", "citation_valid"):
        assert not dev.passes_selection(good.model_copy(update={field: 0.97})), field
    assert dev.passes_selection(good.model_copy(update={"schema_valid": 0.98}))
