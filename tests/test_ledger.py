import json
import math
import os
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ggh import ledger


def test_append_writes_one_sorted_json_line(tmp_path):
    path = tmp_path / "m" / "spend-ledger.jsonl"
    entry = ledger.append(path, "smoke", ledger.SPEND_FIELD, 0.25, run_id="r-1")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == entry
    assert list(json.loads(lines[0])) == sorted(entry)
    assert entry["kind"] == "smoke" and entry["spent_usd"] == 0.25 and entry["run_id"] == "r-1"
    assert "gpu_hours" not in entry and entry["at"].endswith("+00:00")


@pytest.mark.parametrize("bad", [math.nan, math.inf, -0.01, True, "1.0", None])
def test_bad_amounts_are_errors_not_entries(tmp_path, bad):
    path = tmp_path / "l.jsonl"
    with pytest.raises(ledger.LedgerError):
        ledger.append(path, "x", ledger.SPEND_FIELD, bad)
    assert not path.exists()


def test_bad_kind_field_or_extras_rejected(tmp_path):
    path = tmp_path / "l.jsonl"
    with pytest.raises(ledger.LedgerError):
        ledger.append(path, " ", ledger.SPEND_FIELD, 1.0)
    with pytest.raises(ledger.LedgerError):
        ledger.append(path, "x", "dollars", 1.0)
    with pytest.raises(ledger.LedgerError):
        ledger.append(path, "x", ledger.GPU_FIELD, 1.0, at="2020-01-01")
    with pytest.raises(ledger.LedgerError):
        ledger.append(path, "x", ledger.GPU_FIELD, 1.0, spent_usd=2.0)
    assert not path.exists()


def test_total_missing_file_is_zero(tmp_path):
    assert ledger.total(tmp_path / "none.jsonl", ledger.SPEND_FIELD) == 0.0


def test_total_raises_on_malformed_lines(tmp_path):
    path = tmp_path / "l.jsonl"
    path.write_text("not json\n", encoding="utf-8")
    with pytest.raises(ledger.LedgerError):
        ledger.total(path, ledger.SPEND_FIELD)
    path.write_text(json.dumps({"gpu_hours": 1}) + "\n", encoding="utf-8")
    with pytest.raises(ledger.LedgerError):
        ledger.total(path, ledger.SPEND_FIELD)
    path.write_text(json.dumps({"spent_usd": -1}) + "\n", encoding="utf-8")
    with pytest.raises(ledger.LedgerError):
        ledger.total(path, ledger.SPEND_FIELD)


@given(st.lists(st.floats(min_value=0, max_value=1e4, allow_nan=False), max_size=20))
def test_total_is_exact_fsum_of_appended_amounts(amounts):
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "gpu-ledger.jsonl"
        for a in amounts:
            ledger.append(path, "train", ledger.GPU_FIELD, a)
        assert ledger.total(path, ledger.GPU_FIELD) == math.fsum(amounts)


def test_negative_zero_is_normalised(tmp_path):
    entry = ledger.append(tmp_path / "l.jsonl", "x", ledger.SPEND_FIELD, -0.0)
    assert math.copysign(1.0, entry["spent_usd"]) == 1.0
    assert '"spent_usd": 0.0' in (tmp_path / "l.jsonl").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "call",
    [
        lambda p: ledger.append(p, "x", ledger.SPEND_FIELD, 1.0, tokens=math.nan),
        lambda p: ledger.append(p, "x", ledger.SPEND_FIELD, 1.0, blob=object()),
        lambda p: ledger.append(p, 5, ledger.SPEND_FIELD, 1.0),  # type: ignore[arg-type]
        lambda p: ledger.append(p, "x", ledger.SPEND_FIELD, 10**400),
    ],
)
def test_invalid_entries_raise_ledger_error_and_write_nothing(tmp_path, call):
    path = tmp_path / "l.jsonl"
    with pytest.raises(ledger.LedgerError):
        call(path)
    assert not path.exists()


@pytest.mark.parametrize("line", ["5", '"x"', "[1]", '{"spent_usd": 1, "gpu_hours": 2}'])
def test_total_and_count_reject_non_entries(tmp_path, line):
    path = tmp_path / "l.jsonl"
    path.write_text(line + "\n", encoding="utf-8")
    with pytest.raises(ledger.LedgerError):
        ledger.total(path, ledger.SPEND_FIELD)
    with pytest.raises(ledger.LedgerError):
        ledger.count(path)


@pytest.mark.parametrize(
    "raw",
    [
        b'{"spent_usd": ' + b"9" * 5000 + b"}\n",
        b'{"spent_usd": 1}\n\xff\n',
        b'{"spent_usd": 1e308}\n{"spent_usd": 1e308}\n',
        b'{"spent_usd": -5, "spent_usd": 1}\n',
    ],
)
def test_exotic_ledgers_raise_ledger_error(tmp_path, raw):
    path = tmp_path / "spend-ledger.jsonl"
    path.write_bytes(raw)
    with pytest.raises(ledger.LedgerError):
        ledger.total(path, ledger.SPEND_FIELD)


def test_directory_ledger_is_an_error(tmp_path):
    d = tmp_path / "spend-ledger.jsonl"
    d.mkdir()
    with pytest.raises(ledger.LedgerError):
        ledger.total(d, ledger.SPEND_FIELD)
    with pytest.raises(ledger.LedgerError):
        ledger.append(d, "x", ledger.SPEND_FIELD, 1.0)


def test_invalid_entry_creates_no_directory(tmp_path):
    path = tmp_path / "new" / "l.jsonl"
    with pytest.raises(ledger.LedgerError):
        ledger.append(path, "x", ledger.SPEND_FIELD, math.nan)
    assert not (tmp_path / "new").exists()


def test_deep_nesting_raises_ledger_error(tmp_path):
    path = tmp_path / "l.jsonl"
    path.write_text(
        '{"spent_usd": 1, "x": ' + "[" * 100000 + "]" * 100000 + "}\n", encoding="utf-8"
    )
    with pytest.raises(ledger.LedgerError):
        ledger.total(path, ledger.SPEND_FIELD)
    deep: list = []
    for _ in range(100000):
        deep = [deep]
    with pytest.raises(ledger.LedgerError):
        ledger.append(tmp_path / "m.jsonl", "x", ledger.SPEND_FIELD, 1.0, blob=deep)


def test_os_errors_become_ledger_errors(tmp_path):
    blocker = tmp_path / "afile"
    blocker.write_text("x", encoding="utf-8")
    with pytest.raises(ledger.LedgerError):
        ledger.append(blocker / "l.jsonl", "x", ledger.SPEND_FIELD, 1.0)
    unreadable = tmp_path / "u.jsonl"
    unreadable.write_text('{"spent_usd": 1}\n', encoding="utf-8")
    unreadable.chmod(0)
    try:
        if not os.access(unreadable, os.R_OK):  # root can read anything
            with pytest.raises(ledger.LedgerError):
                ledger.total(unreadable, ledger.SPEND_FIELD)
    finally:
        unreadable.chmod(0o644)


def test_looping_symlink_is_an_error_not_zero(tmp_path):
    loop = tmp_path / "loop.jsonl"
    loop.symlink_to(loop)
    with pytest.raises(ledger.LedgerError):
        ledger.total(loop, ledger.SPEND_FIELD)


def test_bad_amount_field_argument_to_total(tmp_path):
    with pytest.raises(ledger.LedgerError):
        ledger.total(tmp_path / "x.jsonl", ["spent_usd"])  # type: ignore[arg-type]


def test_inaccessible_directory_and_long_names_raise_ledger_error(tmp_path):
    long = tmp_path / ("x" * 300) / "l.jsonl"
    with pytest.raises(ledger.LedgerError):
        ledger.total(long, ledger.SPEND_FIELD)
    with pytest.raises(ledger.LedgerError):
        ledger.append(long, "x", ledger.SPEND_FIELD, 1.0)
    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / "l.jsonl").write_text('{"spent_usd": 1}\n', encoding="utf-8")
    locked.chmod(0)
    try:
        if not os.access(locked, os.X_OK):  # root ignores permissions
            for call in (
                lambda: ledger.total(locked / "l.jsonl", ledger.SPEND_FIELD),
                lambda: ledger.count(locked / "l.jsonl"),
                lambda: ledger.append(locked / "l.jsonl", "x", ledger.SPEND_FIELD, 1.0),
            ):
                with pytest.raises(ledger.LedgerError):
                    call()
    finally:
        locked.chmod(0o755)


def test_enormous_int_and_nul_path_raise_ledger_error(tmp_path):
    with pytest.raises(ledger.LedgerError):
        ledger.append(tmp_path / "l.jsonl", "x", ledger.SPEND_FIELD, 10**5000)
    bad = Path(str(tmp_path / "l") + "\x00.jsonl")
    with pytest.raises(ledger.LedgerError):
        ledger.append(bad, "x", ledger.SPEND_FIELD, 1.0)
    with pytest.raises(ledger.LedgerError):
        ledger.total(bad, ledger.SPEND_FIELD)


def test_parent_that_is_a_file_is_an_error_not_zero(tmp_path):
    afile = tmp_path / "measurements"
    afile.write_text("not a directory", encoding="utf-8")
    with pytest.raises(ledger.LedgerError, match="not a directory"):
        ledger.total(afile / "spend-ledger.jsonl", ledger.SPEND_FIELD)
    with pytest.raises(ledger.LedgerError):
        ledger.count(afile / "spend-ledger.jsonl")


def test_str_paths_are_accepted(tmp_path):
    path = str(tmp_path / "l.jsonl")
    ledger.append(path, "x", ledger.SPEND_FIELD, 0.5)  # type: ignore[arg-type]
    assert ledger.total(path, ledger.SPEND_FIELD) == 0.5  # type: ignore[arg-type]


def test_error_messages_are_not_double_wrapped(tmp_path):
    d = tmp_path / "dd"
    d.mkdir()
    with pytest.raises(ledger.LedgerError) as info:
        ledger.total(d, ledger.SPEND_FIELD)
    assert "invalid ledger path" not in str(info.value)


def test_symlinked_parents_that_loop_or_dangle_are_errors(tmp_path):
    loop = tmp_path / "loopdir"
    loop.symlink_to(loop)
    dangling = tmp_path / "gone"
    dangling.symlink_to(tmp_path / "does-not-exist")
    for parent in (loop, dangling):
        with pytest.raises(ledger.LedgerError):
            ledger.total(parent / "spend.jsonl", ledger.SPEND_FIELD)
        with pytest.raises(ledger.LedgerError):
            ledger.count(parent / "spend.jsonl")


def test_dangling_file_symlink_and_fifo_are_errors(tmp_path):
    dangling = tmp_path / "d.jsonl"
    dangling.symlink_to(tmp_path / "nowhere.jsonl")
    with pytest.raises(ledger.LedgerError):
        ledger.total(dangling, ledger.SPEND_FIELD)
    fifo = tmp_path / "f.jsonl"
    os.mkfifo(fifo)
    import signal

    def hung(*_):
        raise AssertionError("reading or writing the FIFO blocked")

    old = signal.signal(signal.SIGALRM, hung)
    signal.alarm(5)  # a regression that opens the pipe fails here instead of hanging the suite
    try:
        with pytest.raises(ledger.LedgerError):
            ledger.total(fifo, ledger.SPEND_FIELD)
        with pytest.raises(ledger.LedgerError):
            ledger.append(fifo, "x", ledger.SPEND_FIELD, 1.0)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def test_failure_after_mkdir_cleans_up_created_directories(tmp_path):
    target = tmp_path / "a" / "b" / ".."  # a/b gets created, then opening a directory fails
    with pytest.raises(ledger.LedgerError):
        ledger.append(target, "x", ledger.SPEND_FIELD, 1.0)
    assert list(tmp_path.iterdir()) == []


def test_over_long_component_is_rejected_before_any_mkdir(tmp_path, monkeypatch):
    made: list = []
    real_mkdir = Path.mkdir

    def spy(self, *a, **k):
        made.append(self)
        return real_mkdir(self, *a, **k)

    monkeypatch.setattr(Path, "mkdir", spy)
    with pytest.raises(ledger.LedgerError, match="longer than 255 bytes"):
        ledger.append(tmp_path / "new" / ("y" * 300) / "l.jsonl", "x", ledger.SPEND_FIELD, 1.0)
    assert made == [] and list(tmp_path.iterdir()) == []


def test_unicode_line_separators_inside_json_strings_are_valid(tmp_path):
    path = tmp_path / "l.jsonl"
    path.write_text('{"spent_usd": 1, "note": "a\u2028b\x85c"}\n', encoding="utf-8")
    assert ledger.total(path, ledger.SPEND_FIELD) == 1.0
