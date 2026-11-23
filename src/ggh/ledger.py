"""Spend and GPU ledgers (Tech Spec D-22, HAR-08).

Every real API dollar goes into docs/measurements/spend-ledger.jsonl and every Kaggle GPU hour
into docs/measurements/gpu-ledger.jsonl, one JSON object per line. Amounts must be finite and
non-negative: a NaN or negative value is an error, never silently recorded (EvalKit L-12).
"""

from __future__ import annotations

import contextlib
import datetime as dt
import json
import math
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

SPEND_FIELD = "spent_usd"
NAME_MAX = 255  # bytes per path component on Linux and macOS
GPU_FIELD = "gpu_hours"


class LedgerError(ValueError):
    """Raised for an invalid ledger entry or a malformed ledger file."""


def _show(value: object) -> str:
    """A short description of a value; never repr an int past Python's digit limit."""
    if isinstance(value, int) and not isinstance(value, bool) and value.bit_length() > 256:
        return f"an integer of {value.bit_length()} bits"
    text = repr(value)
    return text if len(text) <= 80 else text[:77] + "..."


def _check_amount(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise LedgerError(f"{name} must be a number, got {_show(value)}")
    try:
        v = float(value)
    except OverflowError as exc:
        raise LedgerError(f"{name} is too large: {_show(value)}") from exc
    if not math.isfinite(v) or v < 0:
        raise LedgerError(f"{name} must be finite and non-negative, got {_show(value)}")
    return v + 0.0  # normalise -0.0 to 0.0


def append(path: Path, kind: str, amount_field: str, amount: float, **fields: Any) -> dict:
    """Append one entry and return it. `amount_field` is spent_usd or gpu_hours."""
    path = Path(path)
    _check_amount_field(amount_field)
    if not isinstance(kind, str) or not kind.strip():
        raise LedgerError("kind must be a non-empty string")
    reserved = {"at", "kind", SPEND_FIELD, GPU_FIELD} & set(fields)  # exactly one amount
    if reserved:
        raise LedgerError(f"reserved fields passed as extras: {sorted(reserved)}")
    entry = {  # everything is validated before anything touches the filesystem
        "at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "kind": kind,
        amount_field: _check_amount(amount_field, amount),
        **fields,
    }
    try:
        line = json.dumps(entry, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError, RecursionError) as exc:  # NaN, non-JSON, huge or deep values
        raise LedgerError(f"entry is not valid JSON: {exc}") from exc
    _check_path(path)
    missing = [p for p in reversed(path.parents) if not p.exists()]
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except (OSError, ValueError) as exc:
        for d in reversed(missing):  # leave no directories behind from a failed append
            with contextlib.suppress(OSError):
                d.rmdir()
        raise LedgerError(f"cannot write {path}: {getattr(exc, 'strerror', None) or exc}") from exc
    return entry


def _check_amount_field(amount_field: object) -> None:
    if not isinstance(amount_field, str) or amount_field not in (SPEND_FIELD, GPU_FIELD):
        raise LedgerError(f"amount field must be {SPEND_FIELD} or {GPU_FIELD}")


def _check_path(path: Path) -> bool:
    """True if the ledger file exists. Absent is fine; a directory, a broken symlink, or any OS
    error while looking (permissions, over-long names) is a LedgerError."""
    if "\x00" in str(path):  # pathlib's exists() silently says False for these
        raise LedgerError(f"invalid ledger path {str(path)!r}: embedded NUL byte")
    try:
        # exists() hides ENOTDIR and ELOOP, so walk up the parents: the nearest one that is
        # there must be a directory, and none on the way may be a broken or looping symlink
        for parent in path.parents:
            if parent.is_symlink() and not parent.exists():
                raise LedgerError(f"{parent} is a broken or looping symlink")
            if parent.exists():
                if not parent.is_dir():
                    raise LedgerError(f"{parent} is not a directory")
                break
        for part in path.parts:  # reject over-long names before any directory is created
            if len(os.fsencode(part)) > NAME_MAX:
                raise LedgerError(f"path component longer than {NAME_MAX} bytes in {path}")
        if path.is_symlink() and not path.exists():
            raise LedgerError(f"{path} is a broken or looping symlink")
        if path.exists() and not path.is_file():
            raise LedgerError(f"{path} is not a file")
        return path.exists()
    except LedgerError:
        raise
    except ValueError as exc:  # e.g. an unencodable path
        raise LedgerError(f"invalid ledger path {str(path)!r}: {exc}") from exc
    except OSError as exc:
        raise LedgerError(f"cannot access {path}: {exc.strerror or exc}") from exc


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict:
    keys = [k for k, _ in pairs]
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    if dupes:
        raise LedgerError(f"duplicate keys {dupes}")
    return dict(pairs)


def _entries(path: Path) -> list[tuple[int, dict]]:
    if not _check_path(path):
        return []
    try:
        text = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LedgerError(f"{path.name}: not UTF-8 ({exc.reason} at byte {exc.start})") from exc
    except OSError as exc:
        raise LedgerError(f"cannot read {path}: {exc.strerror or exc}") from exc
    out: list[tuple[int, dict]] = []
    # split on "\n" only: U+2028 or NEL inside a JSON string is valid, not a line break
    for lineno, line in enumerate(text.split("\n"), start=1):
        line = line.removesuffix("\r")
        if not line.strip():
            continue
        try:
            entry = json.loads(line, object_pairs_hook=_no_duplicate_keys)
        except LedgerError as exc:
            raise LedgerError(f"{path.name} line {lineno}: {exc}") from exc
        except (ValueError, RecursionError) as exc:  # bad JSON, huge ints, absurd nesting
            raise LedgerError(f"{path.name} line {lineno}: not valid JSON ({exc})") from exc
        if not isinstance(entry, dict):
            raise LedgerError(f"{path.name} line {lineno}: not a JSON object")
        if SPEND_FIELD in entry and GPU_FIELD in entry:
            raise LedgerError(f"{path.name} line {lineno}: holds both amount fields")
        out.append((lineno, entry))
    return out


def count(path: Path) -> int:
    """Number of entries; 0 for a missing file. Raises LedgerError on a malformed line."""
    path = Path(path)
    return len(_entries(path))


def total(path: Path, amount_field: str, keep: Callable[[dict], bool] | None = None) -> float:
    """Sum of one amount field across the ledger (the rows `keep` accepts, if given); 0.0 for a
    missing file. Every row is checked, kept or not."""
    path = Path(path)
    _check_amount_field(amount_field)
    values: list[float] = []
    for lineno, entry in _entries(path):
        if amount_field not in entry:
            raise LedgerError(f"{path.name} line {lineno}: no {amount_field}")
        value = _check_amount(f"line {lineno} {amount_field}", entry[amount_field])
        if keep is None or keep(entry):
            values.append(value)
    try:
        result = math.fsum(values)  # EvalKit L-27
    except OverflowError as exc:
        raise LedgerError(f"{path.name}: {amount_field} total overflows") from exc
    if not math.isfinite(result):
        raise LedgerError(f"{path.name}: {amount_field} total is not finite")
    return result
