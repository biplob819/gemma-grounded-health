"""The terminal-GIF recorder (scripts/termgif.py) and the two GIFs it made for the video."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _termgif():
    pytest.importorskip("PIL")  # Pillow is a dev tool here, absent from the minimal envs
    spec = importlib.util.spec_from_file_location("termgif", ROOT / "scripts/termgif.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_clean_drops_ansi_escapes_and_keeps_text():
    termgif = _termgif()
    raw = '\x1b[?25l\x1b[1G⠙ \x1b[K\x1b[?25h{"label": "no"}\x1b[?2026l'
    assert termgif._clean(raw) == '⠙ {"label": "no"}'


def test_record_and_render_replay_a_real_command(tmp_path):
    termgif = _termgif()
    steps = tmp_path / "steps.json"
    steps.write_text(
        json.dumps([{"comment": "# hi"}, {"show": "echo ok", "run": "echo ok"}]), encoding="utf-8"
    )
    cast, gif = tmp_path / "cast.json", tmp_path / "out.gif"
    termgif.record(str(steps), str(cast))
    events = json.loads(cast.read_text(encoding="utf-8"))
    assert [e["kind"] for e in events] == ["comment", "cmd", "out", "exit"]
    assert events[2]["text"] == "ok\n" and events[3]["code"] == 0
    termgif.render(str(cast), str(gif), "t", cols=40, rows=6, max_gap=1.0)
    data = gif.read_bytes()
    assert data[:6] == b"GIF89a" and b"NETSCAPE2.0" in data


def test_the_offline_ollama_and_bench_gifs_are_committed_and_loop():
    for name in ("ollama_offline.gif", "laptop_bench.gif"):
        path = ROOT / "docs/assets" / name
        data = path.read_bytes()
        assert data[:6] == b"GIF89a" and b"NETSCAPE2.0" in data, name
        assert 50_000 < len(data) < 2_000_000, name
        assert data.count(b"\x21\xf9\x04") > 20, name  # one graphic control block per frame


def test_the_offline_capture_blocks_the_internet_and_uses_greedy_decoding():
    sb = (ROOT / "scripts/offline/offline.sb").read_text(encoding="utf-8")
    assert "(deny network-outbound)" in sb and '"localhost:*"' in sb
    mf = (ROOT / "scripts/offline/Modelfile").read_text(encoding="utf-8")
    assert "gemma-4-e2b-grounded-health-GGUF:Q4_K_M" in mf
    assert "temperature 0" in mf and "top_k 1" in mf
    steps = json.loads((ROOT / "scripts/offline/ollama_steps.json").read_text(encoding="utf-8"))
    runs = [s["run"] for s in steps if "run" in s and "ollama" in s["run"]]
    assert runs and all(r.startswith("sandbox-exec -f scripts/offline/offline.sb") for r in runs)


def test_the_bench_rerun_is_kept_apart_and_says_the_machine_was_busy():
    rerun = json.loads(
        (ROOT / "docs/measurements/v2/laptop_bench_rerun_2026-11-21.json").read_text("utf-8")
    )
    assert "NOT a quiet machine" in rerun["rerun_note"]
    assert rerun["load_average_1m"]["end"] > rerun["load_average_1m"]["start"]
