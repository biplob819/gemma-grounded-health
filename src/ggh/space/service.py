"""The demo's logic without the UI (Tech Spec §9): samples without a passcode, live answers and
the optional grounding check behind the passcode gate.

Privacy (SPC-05): the only log line is an outcome word and a latency. The question, the
abstract, the passcode and a visitor's API key are never logged, stored on the service or
written to disk. A visitor's key is passed to the grounding check for that one request and goes
out of scope with it; it never replaces the passcode (OWNER D-36).
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from ggh.space.engine import AnswerCard, Generate, InputError, compare, split_abstract, to_card
from ggh.space.passkey import Gate, Refused
from ggh.space.samples import Sample

LOG = logging.getLogger("ggh.space")
MODELS = ("base", "ft")

# question, sentences, the card to check, the visitor's key (None: the owner's configured key)
GroundingCheck = Callable[[str, Sequence[str], AnswerCard, "str | None"], str]


@dataclass(frozen=True)
class Result:
    sentences: tuple[str, ...] = ()
    cards: Mapping[str, AnswerCard] = field(default_factory=dict[str, AnswerCard])
    grounding: Mapping[str, str] = field(default_factory=dict[str, str])
    message: str = ""  # shown to the visitor: a refusal, an input problem or a note


def log_event(outcome: str, latency_ms: int) -> str:
    """One line per action: an outcome word and a latency, nothing the visitor typed."""
    line = json.dumps({"event": "space", "outcome": outcome, "latency_ms": int(latency_ms)})
    LOG.info(line)
    return line


class Space:
    def __init__(
        self,
        gate: Gate,
        generators: Mapping[str, Generate],
        samples: Sequence[Sample],
        *,
        grounding: GroundingCheck | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._gate = gate
        self._generators = dict(generators)
        self._samples = {s.id: s for s in samples}
        self._grounding = grounding
        self._clock = clock

    @property
    def sample_ids(self) -> list[str]:
        return list(self._samples)

    def sample(self, sample_id: str) -> Result:
        """A precomputed sample: no passcode, no model call."""
        s = self._samples.get(sample_id)
        if s is None:
            return Result(message="Unknown sample.")
        cards = {
            "base": to_card("base", s.base_raw, s.context),
            "ft": to_card("ft", s.ft_raw, s.context),
        }
        log_event("sample", 0)
        return Result(sentences=s.context, cards=cards)

    def live(
        self,
        question: str,
        abstract: str | Sequence[str],
        passcode: str,
        session: str,
        judge_key: str | None = None,
        check: bool = False,
    ) -> Result:
        """Both models on the visitor's input, after the passcode; then, if asked, the grounding
        check with the visitor's key for this request only."""
        t0 = self._clock()
        try:
            self._gate.check(passcode, session)
        except Refused as exc:
            log_event("refused", int((self._clock() - t0) * 1000))
            return Result(message=str(exc))
        sentences = split_abstract(abstract) if isinstance(abstract, str) else tuple(abstract)
        try:
            cards = compare(question, sentences, self._generators, clock=self._clock)
        except InputError as exc:
            log_event("bad_input", int((self._clock() - t0) * 1000))
            return Result(message=str(exc))
        grounding: dict[str, str] = {}
        note = ""
        if check:
            if self._grounding is None:
                note = "The live grounding check is not available on this demo."
            else:
                key = judge_key.strip() if judge_key and judge_key.strip() else None
                for name, card in cards.items():
                    if card.error:
                        continue
                    try:
                        grounding[name] = self._grounding(question, sentences, card, key)
                    except Exception:  # the message may echo the key: never show or log it
                        grounding[name] = "The grounding check failed."
                del key
        log_event("live", int((self._clock() - t0) * 1000))
        return Result(sentences=sentences, cards=cards, grounding=grounding, message=note)
