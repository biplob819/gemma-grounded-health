"""SPC-03, SPC-05, SPC-08: samples without a passcode; every live action behind it, also with a
visitor's own key; nothing typed is logged or kept; the visitor's key is used once."""

import json
import logging
from pathlib import Path

import pytest

from ggh.space import app, passkey, samples
from ggh.space.service import Space

ROOT = Path(__file__).resolve().parents[2]
CODE = "482916"
SECRET_Q = "Does zanthoxylin-77 cure gout?"
SECRET_A = "Zanthoxylin-77 was tested in 12 people. Gout pain fell in 9 of them."
VISITOR_KEY = "visitor-judge-key-never-log-me"


def reply(cites=(2,)) -> str:
    return json.dumps({"label": "yes", "answer": "It helped most.", "citations": list(cites)})


@pytest.fixture
def space():
    calls: list[str] = []
    gens = {
        "base": lambda p: calls.append("base") or reply((1,)),
        "ft": lambda p: calls.append("ft") or reply(),
    }
    gate = passkey.Gate(passkey.StoredHash.parse(passkey.make_hash(CODE, iterations=100_000)))
    s = Space(gate, gens, samples.load(ROOT / samples.SAMPLES))
    s.calls = calls  # type: ignore[attr-defined]
    return s


def test_six_samples_need_no_passcode_and_no_model(space):
    assert len(space.sample_ids) == samples.N_SAMPLES
    for sid in space.sample_ids:
        r = space.sample(sid)
        assert set(r.cards) == {"base", "ft"} and r.sentences
    assert space.calls == []


def test_live_needs_the_passcode_even_with_a_visitor_key(space):
    r = space.live(SECRET_Q, SECRET_A, "000000", "s1", judge_key=VISITOR_KEY, check=True)
    assert r.message == "Wrong passcode." and not r.cards
    assert space.calls == []
    r = space.live(SECRET_Q, SECRET_A, CODE, "s1")
    assert r.cards["ft"].label == "yes" and space.calls == ["base", "ft"]


def test_the_visitor_key_reaches_the_check_once_and_is_not_kept(space):
    seen: list[str | None] = []

    def check(q, sents, card, key):
        seen.append(key)
        return "4 of 4 claims supported"

    space._grounding = check
    r = space.live(SECRET_Q, SECRET_A, CODE, "s1", judge_key=f"  {VISITOR_KEY} ", check=True)
    assert seen == [VISITOR_KEY, VISITOR_KEY]  # one call per card, this request only
    assert r.grounding == {"base": "4 of 4 claims supported", "ft": "4 of 4 claims supported"}
    assert VISITOR_KEY not in repr(vars(space))
    space.live(SECRET_Q, SECRET_A, CODE, "s1", check=True)
    assert seen[-1] is None  # no key on a later request: nothing was remembered


def test_a_failing_check_never_shows_its_message(space):
    def check(*_):
        raise RuntimeError(f"401 bad key {VISITOR_KEY}")

    space._grounding = check
    r = space.live(SECRET_Q, SECRET_A, CODE, "s", judge_key=VISITOR_KEY, check=True)
    assert VISITOR_KEY not in json.dumps(dict(r.grounding))


def test_nothing_typed_is_logged(space, caplog):  # SPC-05
    caplog.set_level(logging.DEBUG)
    space._grounding = lambda *_: "ok"
    space.live(SECRET_Q, SECRET_A, "111111", "s9", judge_key=VISITOR_KEY, check=True)
    space.live(SECRET_Q, SECRET_A, CODE, "s9", judge_key=VISITOR_KEY, check=True)
    space.live("", SECRET_A, CODE, "s9")
    space.sample(space.sample_ids[0])
    text = caplog.text
    assert text, "the service logs outcomes"
    for secret in (SECRET_Q, "Zanthoxylin", CODE, "111111", VISITOR_KEY, "s9"):
        assert secret not in text
    for rec in caplog.records:  # the service's own lines (a library may log too, never the input)
        if rec.name != "ggh.space":
            continue
        assert set(json.loads(rec.getMessage())) == {"event", "outcome", "latency_ms"}


def test_no_grounding_hook_says_so(space):
    r = space.live(SECRET_Q, SECRET_A, CODE, "s", check=True)
    assert "not available" in r.message and r.cards


def test_make_space_reads_the_hash_from_the_environment():
    env = {passkey.SECRET_ENV: passkey.make_hash(CODE, iterations=100_000)}
    s = app.make_space(
        env, {"base": lambda p: reply(), "ft": lambda p: reply()}, ROOT / samples.SAMPLES
    )
    assert s.live(SECRET_Q, SECRET_A, CODE, "s").cards
    off = app.make_space({}, {}, ROOT / samples.SAMPLES)
    assert "switched off" in off.live(SECRET_Q, SECRET_A, CODE, "s").message
