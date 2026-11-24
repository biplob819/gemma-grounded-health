"""The walkthrough GIF's pure parts: screen list, rendering from committed data, scrubbing.

Capture (headless Chrome) and encoding (ffmpeg) are not run here.
"""

import json
from pathlib import Path

import pytest

import walkthrough as wt

ROOT = Path(__file__).resolve().parents[1]
FAKE_HOME = "/Users/someone"
FAKE_USER = "someone"


def ctx(audit_url: str = "about:blank") -> wt.Ctx:
    return wt.Ctx(ROOT, wt.Paths(), num=1, total=len(wt.screens()), audit_url=audit_url)


def fake_ggh(root: Path, args: list[str]) -> str:
    return f"usage: ggh\n  {{a,b}}\n    doctor   check\nran {' '.join(args)} in {root}/x\n"


def test_screen_list_fits_the_spec():
    ss = wt.screens()
    assert 6 <= len(ss) <= 10
    assert len({s.slug for s in ss}) == len(ss)
    assert [s.slug for s in ss if s.live] == ["audit"]


def test_every_input_is_committed():
    for rel in vars(wt.Paths()).values():
        assert (ROOT / rel).exists(), rel


def test_results_screen_reads_the_committed_summary():
    summary = json.loads((ROOT / wt.Paths().summary).read_text(encoding="utf-8"))
    page = wt.screen_results(ctx())
    for m in summary["models"]:
        assert wt.name(m["model"]) in page
        assert wt.f3(m["accuracy"]["point"]) in page
        for j, h in m["headline"].items():
            assert wt.f3(h["point"]) in page, (m["model"], j)


def test_laptop_and_checks_screens_read_the_committed_measurements():
    bench = json.loads((ROOT / wt.Paths().laptop_bench).read_text(encoding="utf-8"))
    page = wt.screen_laptop(ctx())
    for name, v in bench["variants"].items():
        assert name in page
        assert f"{v['answer_latency_ms_p50']} / {v['answer_latency_ms_p95']} ms" in page
    parity = json.loads((ROOT / wt.Paths().parity_report).read_text(encoding="utf-8"))
    assert wt.f3(parity["label_agreement"]) in wt.screen_checks(ctx())


def test_failure_stats_are_counted_from_the_records():
    fs = wt.failure_stats(ctx())
    for st in fs["stats"].values():
        assert 0 <= st["only_s1"] <= st["cite_s1"] <= st["unsupported"]
    ex = fs["example"]
    assert ex and ex["question"] and ex["sentence1"]
    assert wt.AIM_RATIONALE.search(ex["rationale"])


def test_title_and_audit_screens_render():
    assert "Research only, not medical advice" in wt.screen_title(ctx())
    page = wt.screen_audit(ctx("http://127.0.0.1:1/"))
    assert 'src="http://127.0.0.1:1/"' in page and "of 50 ok" in page


def test_terminal_screens_scrub_paths(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(wt, "run_ggh", fake_ggh)
    for render in (wt.screen_cli_help, wt.screen_cli_doctor):
        page = render(ctx())
        assert str(ROOT) not in page
        assert wt.NEUTRAL_REPO in page
    assert "usage:" not in wt.screen_cli_help(ctx())


def test_render_all_writes_every_screen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(wt, "run_ggh", fake_ggh)
    done = wt.render_all(ROOT, wt.Paths(), tmp_path)
    assert [s.slug for s, _ in done] == [s.slug for s in wt.screens()]
    for _, p in done:
        assert p.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_render_all_refuses_a_leaking_page(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(wt, "run_ggh", lambda root, args: "contact me at a.person@example.com")
    with pytest.raises(SystemExit, match="email address"):
        wt.render_all(ROOT, wt.Paths(), tmp_path)


def test_scrub_replaces_repo_home_and_user():
    text = f"{ROOT}/src and {FAKE_HOME}/.cache, /home/other/x, C:\\Users\\other\\y, {FAKE_USER}!"
    out = wt.scrub(text, ROOT, home=FAKE_HOME, user=FAKE_USER)
    assert out == (f"{wt.NEUTRAL_REPO}/src and ~/.cache, ~/x, ~\\y, {wt.NEUTRAL_USER}!")
    assert wt.find_leaks(out, home=FAKE_HOME, user=FAKE_USER) == []


def test_find_leaks_flags_paths_users_emails_and_keys():
    key = "sk-ant-" + "a" * 30  # split so the repo's secret scan does not flag this file
    assert "home path" in wt.find_leaks(f"{FAKE_HOME}/x", home=FAKE_HOME, user="")
    assert "local username" in wt.find_leaks("hi someone", home="", user=FAKE_USER)
    assert "email address" in wt.find_leaks("x@y.org", home="", user="")
    assert wt.find_leaks(key, home="", user="")
    assert wt.find_leaks("0.728 [0.700, 0.755]", home="", user="") == []


def test_trim_help_keeps_the_command_table():
    raw = (
        "usage: ggh [-h]\n    {a,b,\nc}\n    ...\n\nDesc.\n\npositional:\n  {a,b,\nc}\n    a   x\n"
    )
    assert wt.trim_help(raw) == "Desc.\n\npositional:\n    a   x"


def test_cli_env_has_no_credentials(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("HF_TOKEN", "x")
    env = wt.cli_env()
    assert not any(h in k for k in env for h in wt.CREDENTIAL_HINTS)
    assert env["HOME"] != str(Path.home())


def test_v2_screens_show_the_release():
    """R4.5: the default walkthrough is v2's run (v1.0.0's frozen bundle, D-76), with the
    pre-registered verdicts and the aim-rule caveat, the Hugging Face page and the demo
    (screenshots inlined, no file path)."""
    assert wt.Paths().summary == "docs/results/main_v2/summary.json"
    assert wt.Paths().bundle == "docs/results/releases/v1.0.0/results_bundle.json"
    results = wt.screen_results(ctx())
    assert "Gemma fine-tuned v2 minus Gemma fine-tuned v1" in results
    assert results.count("depends on the aim rule (OWNER D-62)") == 2
    for render in (wt.screen_hf, wt.screen_demo):
        page = render(ctx())
        assert "data:image/png;base64," in page and "file://" not in page
        assert wt.find_leaks(wt.DATA_URI.sub("", page)) == []
    title = wt.screen_title(ctx())
    assert "Cited sentences that support a claim" in title and "r-715fc213a8a5" in title
    assert {"hf", "demo", "results"} <= {s.slug for s in wt.screens()}


def test_v1_summary_still_renders():
    """--summary docs/results/main/summary.json: v1's two-judge run with its own plan's
    comparisons, which the aim-rule caveat does not touch."""
    paths = wt.Paths(summary="docs/results/main/summary.json", run="docs/results/main/run.json")
    c = wt.Ctx(ROOT, paths, num=1, total=10)
    results = wt.screen_results(c)
    assert "Gemma fine-tuned v1 minus" in results and "aim rule" not in results
    assert "Claims supported by cited sentences" in wt.screen_title(c)
