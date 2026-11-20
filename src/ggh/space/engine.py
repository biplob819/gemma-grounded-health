"""SPC-01, SPC-02: `compare(question, abstract)` for the demo (Tech Spec §9).

The abstract is split with EvalKit's splitter and the prompt rendered with EvalKit's PubMedQA
template, exactly as in training and the benchmark. Each model's reply is parsed with EvalKit's
`Answer` schema; a parse failure becomes a card with `error="schema"`, never a hidden one. A
citation outside 1..n is kept on the card and flagged as invalid (B7 on screen).

Generation is injected (`Generate`: prompt -> raw reply text), so tests use fakes and the
Space, the local fallback and the precomputed samples share this one code path. Nothing here
logs or stores the question or the abstract.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from html import escape

from evalkit.pipeline.answer import SchemaError, parse_answer
from evalkit.schemas import Case
from evalkit.tasks import builtin_task, render_prompt
from evalkit.text import number_sentences

from ggh.schemas._base import Model, NonNegInt

LABELS = ("yes", "no", "maybe")
MAX_NEW_TOKENS = 256  # Tech Spec §9
MAX_QUESTION_CHARS = 500
MAX_ABSTRACT_CHARS = 12_000
MAX_SENTENCES = 60
BANNER = "Research only, not medical advice."
TASK = "pubmedqa"

Generate = Callable[[str], str]


class InputError(ValueError):
    """A question or abstract the demo will not send to a model."""


class Cited(Model):
    n: NonNegInt
    text: str


class AnswerCard(Model):
    model: str
    label: str | None = None
    answer: str | None = None
    citations: tuple[int, ...] = ()
    cited: tuple[Cited, ...] = ()  # valid citations with their sentence text
    invalid_citations: tuple[int, ...] = ()  # outside 1..n: flagged on screen (B7)
    error: str | None = None  # "schema", "generation"
    raw_text: str = ""
    latency_ms: NonNegInt = 0


def split_abstract(abstract: str) -> tuple[str, ...]:
    """EvalKit's splitter over the pasted paragraphs (blank lines separate paragraphs)."""
    paragraphs = [p.strip() for p in abstract.replace("\r\n", "\n").split("\n\n") if p.strip()]
    return number_sentences(paragraphs)


def check_inputs(question: str, sentences: Sequence[str]) -> None:
    if not question.strip():
        raise InputError("Enter a question.")
    if len(question) > MAX_QUESTION_CHARS:
        raise InputError(f"The question is longer than {MAX_QUESTION_CHARS} characters.")
    if not sentences:
        raise InputError("Paste an abstract.")
    if len(sentences) > MAX_SENTENCES or sum(len(s) for s in sentences) > MAX_ABSTRACT_CHARS:
        raise InputError("The abstract is too long for the demo.")


def build_prompt(question: str, sentences: Sequence[str]) -> str:
    case = Case(id="demo", question=question.strip(), context=tuple(sentences), source="demo")
    return render_prompt(builtin_task(TASK), case)


def to_card(model: str, raw: str, sentences: Sequence[str], latency_ms: int = 0) -> AnswerCard:
    try:
        ans = parse_answer(raw, LABELS)
    except SchemaError:
        return AnswerCard(model=model, error="schema", raw_text=raw, latency_ms=latency_ms)
    n = len(sentences)
    cites = tuple(sorted(set(ans.citations)))
    return AnswerCard(
        model=model,
        label=ans.label,
        answer=ans.answer,
        citations=cites,
        cited=tuple(Cited(n=c, text=sentences[c - 1]) for c in cites if 1 <= c <= n),
        invalid_citations=tuple(c for c in cites if not 1 <= c <= n),
        raw_text=raw,
        latency_ms=latency_ms,
    )


def _run(
    model: str,
    generate: Generate,
    prompt: str,
    sentences: Sequence[str],
    clock: Callable[[], float],
) -> AnswerCard:
    t0 = clock()
    try:
        raw = generate(prompt)
    except Exception:  # a failed model is shown as a failed card, never as the other's answer
        return AnswerCard(model=model, error="generation")
    return to_card(model, raw, sentences, int((clock() - t0) * 1000))


def compare(
    question: str,
    abstract: str | Sequence[str],
    generators: dict[str, Generate],
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, AnswerCard]:
    """One card per model (`base`, `ft`) for the same EvalKit prompt."""
    sentences = split_abstract(abstract) if isinstance(abstract, str) else tuple(abstract)
    check_inputs(question, sentences)
    prompt = build_prompt(question, sentences)
    return {name: _run(name, gen, prompt, sentences, clock) for name, gen in generators.items()}


def highlight(sentences: Sequence[str], card: AnswerCard) -> str:
    """The abstract as HTML with this card's cited sentences marked (escaped)."""
    cited = set(card.citations)
    parts = []
    for i, s in enumerate(sentences, 1):
        cls = " class='cited'" if i in cited else ""
        parts.append(f"<span{cls}><sup>{i}</sup> {escape(s)}</span>")
    return "<p class='abstract'>" + " ".join(parts) + "</p>"


def card_html(title: str, card: AnswerCard) -> str:
    out = [f"<div class='card'><h4>{escape(title)}</h4>"]
    if card.error == "schema":
        out.append("<p class='err'>Schema failure: the reply was not a valid answer.</p>")
        out.append(f"<pre>{escape(card.raw_text)}</pre>")
    elif card.error:
        out.append(f"<p class='err'>No answer ({escape(card.error)}).</p>")
    else:
        out.append(f"<p><b>Label:</b> {escape(card.label or '')}</p>")
        out.append(f"<p>{escape(card.answer or '')}</p>")
        if card.cited:
            out.append("<ul class='cited'>")
            out += [f"<li>[{c.n}] {escape(c.text)}</li>" for c in card.cited]
            out.append("</ul>")
        if card.invalid_citations:
            bad = ", ".join(str(c) for c in card.invalid_citations)
            out.append(f"<p class='err'>Invalid citation: sentence {bad} does not exist.</p>")
        if not card.citations:
            out.append("<p class='err'>No sentence cited.</p>")
    out.append(f"<p class='notice'>{escape(BANNER)}</p></div>")
    return "".join(out)
