"""SPC-08: one numeric passcode held as a salted hash, constant-time compare, rate-limited per
session; a visitor's own API key never bypasses it (OWNER D-36, D-39)."""

import inspect

import pytest

from ggh.cli import main
from ggh.space import passkey as P

FAST = 100_000  # the floor StoredHash accepts; production hashes use P.ITERATIONS
CODE = "482916"


def stored(code: str = CODE) -> P.StoredHash:
    return P.StoredHash.parse(P.make_hash(code, iterations=FAST))


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def test_hash_is_salted_and_never_contains_the_code():
    a, b = P.make_hash(CODE, iterations=FAST), P.make_hash(CODE, iterations=FAST)
    assert a != b and CODE not in a and a.startswith("pbkdf2_sha256$100000$")


def test_matches_only_the_right_code():
    s = stored()
    assert s.matches(CODE)
    assert not s.matches("482917") and not s.matches("") and not s.matches("48291６")


def test_compare_is_constant_time():
    src = inspect.getsource(P.StoredHash.matches)
    assert "hmac.compare_digest" in src and "==" not in src


@pytest.mark.parametrize("bad", ["12a4", "123", "1" * 13, "", "１２３４"])
def test_only_numeric_codes_can_be_set(bad):
    with pytest.raises(P.PasskeyError):
        P.make_hash(bad, iterations=FAST)


@pytest.mark.parametrize(
    "value", ["482916", "md5$1$aa$bb", "pbkdf2_sha256$10$" + "00" * 16 + "$" + "00" * 32]
)
def test_a_plain_code_or_weak_hash_in_the_secret_is_refused(value):
    with pytest.raises(P.PasskeyError):
        P.StoredHash.parse(value)


def test_unset_secret_switches_live_off():
    assert P.from_env({}) is None
    with pytest.raises(P.Refused, match="switched off"):
        P.Gate(None).check(CODE, "s1")


def test_failures_lock_the_session_not_others():
    clock = Clock()
    gate = P.Gate(stored(), P.RateLimiter(clock=clock, max_failures=3, window_s=60))
    for _ in range(3):
        with pytest.raises(P.Refused, match="Wrong passcode"):
            gate.check("000000", "s1")
    with pytest.raises(P.Refused, match="Too many"):
        gate.check(CODE, "s1")  # even the right code, while locked
    gate.check(CODE, "s2")
    clock.t = 61
    gate.check(CODE, "s1")


def test_live_requests_are_capped_per_session():
    gate = P.Gate(stored(), P.RateLimiter(clock=Clock(), max_live=2))
    gate.check(CODE, "s")
    gate.check(CODE, "s")
    with pytest.raises(P.Refused, match="used its live answers"):
        gate.check(CODE, "s")


def test_refusal_messages_never_echo_the_input():
    gate = P.Gate(stored())
    with pytest.raises(P.Refused) as e:
        gate.check("999999", "s")
    assert "999999" not in str(e.value)


def test_passkey_set_prints_only_the_hash(monkeypatch, capsys):
    answers = iter([CODE, CODE])
    monkeypatch.setattr("getpass.getpass", lambda _prompt="": next(answers))
    monkeypatch.setattr(P, "ITERATIONS", FAST)
    monkeypatch.setattr(P.make_hash, "__kwdefaults__", {"salt": None, "iterations": FAST})
    assert main(["passkey", "set"]) == 0
    out = capsys.readouterr().out
    assert CODE not in out
    value = out.strip().splitlines()[-1]
    assert P.StoredHash.parse(value).matches(CODE)


def test_passkey_set_refuses_a_mismatch(monkeypatch, capsys):
    answers = iter([CODE, "111111"])
    monkeypatch.setattr("getpass.getpass", lambda _prompt="": next(answers))
    assert main(["passkey", "set"]) == 2
    assert CODE not in capsys.readouterr().out
