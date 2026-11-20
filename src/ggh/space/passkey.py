"""SPC-08: the demo passcode gate (OWNER D-36, D-39).

One numeric passcode that only the owner knows. The Space holds only its salted PBKDF2 hash, in
the Space secret named by `SECRET_ENV`; `ggh passkey set` makes a new hash. Input is compared in
constant time (`hmac.compare_digest` on the derived keys) and rate-limited per session. Neither
the passcode nor the input is ever logged, stored or echoed; refusals carry no detail about why
a code was wrong.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

SECRET_ENV = "GGH_SPACE_PASSKEY"  # holds the salted hash, never the passcode
SCHEME = "pbkdf2_sha256"
ITERATIONS = 600_000
MIN_DIGITS, MAX_DIGITS = 4, 12
MAX_FAILURES = 5  # per session per window
MAX_LIVE = 30  # live requests per session per window
WINDOW_S = 3600.0


class PasskeyError(ValueError):
    """A stored hash that cannot be used, or a new passcode that is not acceptable."""


class Refused(Exception):
    """A live action the gate refused; the message is safe to show the visitor."""


def _derive(code: str, salt: bytes, iterations: int) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", code.encode("utf-8"), salt, iterations)


def is_numeric_code(code: str) -> bool:
    return code.isascii() and code.isdigit() and MIN_DIGITS <= len(code) <= MAX_DIGITS


def make_hash(code: str, *, salt: bytes | None = None, iterations: int = ITERATIONS) -> str:
    if not is_numeric_code(code):
        raise PasskeyError(f"the passcode must be {MIN_DIGITS} to {MAX_DIGITS} digits")
    salt = salt if salt is not None else secrets.token_bytes(16)
    digest = _derive(code, salt, iterations)
    return f"{SCHEME}${iterations}${salt.hex()}${digest.hex()}"


@dataclass(frozen=True)
class StoredHash:
    iterations: int
    salt: bytes
    digest: bytes

    @classmethod
    def parse(cls, value: str) -> StoredHash:
        try:
            scheme, iters, salt, digest = value.strip().split("$")
            if scheme != SCHEME:
                raise ValueError
            out = cls(int(iters), bytes.fromhex(salt), bytes.fromhex(digest))
        except ValueError:
            raise PasskeyError(
                f"{SECRET_ENV} is not a {SCHEME} hash from `ggh passkey set`"
            ) from None
        if out.iterations < 100_000 or len(out.salt) < 16 or len(out.digest) != 32:
            raise PasskeyError(f"{SECRET_ENV} is too weak a hash; run `ggh passkey set`")
        return out

    def matches(self, code: str) -> bool:
        # derive even for malformed input, so the time taken does not say which check failed
        candidate = _derive(code, self.salt, self.iterations)
        return hmac.compare_digest(candidate, self.digest) and is_numeric_code(code)


def from_env(env: Mapping[str, str]) -> StoredHash | None:
    """The stored hash, or None when the secret is unset (live actions then stay off)."""
    value = env.get(SECRET_ENV, "")
    return StoredHash.parse(value) if value.strip() else None


@dataclass
class RateLimiter:
    """Per-session sliding windows for failed codes and for live requests."""

    clock: Callable[[], float] = time.monotonic
    max_failures: int = MAX_FAILURES
    max_live: int = MAX_LIVE
    window_s: float = WINDOW_S
    _failures: dict[str, deque[float]] = field(default_factory=dict[str, "deque[float]"])
    _live: dict[str, deque[float]] = field(default_factory=dict[str, "deque[float]"])

    def _recent(self, book: dict[str, deque[float]], session: str) -> deque[float]:
        q = book.setdefault(session, deque())
        now = self.clock()
        while q and now - q[0] > self.window_s:
            q.popleft()
        return q

    def locked(self, session: str) -> bool:
        return len(self._recent(self._failures, session)) >= self.max_failures

    def fail(self, session: str) -> None:
        self._recent(self._failures, session).append(self.clock())

    def take_live(self, session: str) -> bool:
        q = self._recent(self._live, session)
        if len(q) >= self.max_live:
            return False
        q.append(self.clock())
        return True


@dataclass
class Gate:
    stored: StoredHash | None
    limiter: RateLimiter = field(default_factory=RateLimiter)

    def check(self, code: str, session: str) -> None:
        """Raise `Refused` unless this session may run one live action with this code. A
        visitor's own API key does not change anything here (OWNER D-36)."""
        if self.stored is None:
            raise Refused("Live answers are switched off on this demo. The samples still work.")
        if self.limiter.locked(session):
            raise Refused("Too many wrong passcodes. Try again later.")
        if not self.stored.matches(code):
            self.limiter.fail(session)
            raise Refused("Wrong passcode.")
        if not self.limiter.take_live(session):
            raise Refused("This session has used its live answers for now. Try again later.")
