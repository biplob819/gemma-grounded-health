"""SPC-03, SPC-04, E7: the committed samples are rebuilt from the v2 dev run (base's replies from
v1's); the local backend sends EvalKit's schema request to ggh serve; the local footer and the
bundle summary (OWNER D-63, D-14); the answer view; the Gradio page builds (skipped when gradio
is not installed, as in CI)."""

import gzip
import json
from pathlib import Path

import pytest

from ggh.release import sources
from ggh.space import app, backends, passkey, samples, ui
from ggh.space.engine import BANNER, MAX_NEW_TOKENS, to_card
from ggh.space.service import Result

ROOT = Path(__file__).resolve().parents[2]


def test_committed_samples_are_rebuilt_from_the_v2_dev_run():
    built = samples.build(ROOT / samples.RUN, base_run=ROOT / samples.BASE_RUN)
    assert samples.dump(built) == (ROOT / samples.SAMPLES).read_text(encoding="utf-8")
    assert len(built) == samples.N_SAMPLES
    assert {s.reference_label for s in built} == {"yes", "no"}
    assert all(s.id.startswith("dev-") for s in built)  # dev_eval, never the PQA-L benchmark
    # v2's replies from v2's dev run, base's from v1's (same cases), both named in the source
    assert samples.FT_MODEL == "gemma-e2b-grounded-v2-q4" == app.LOCAL_FT_MODEL
    assert all(s.source.endswith(f"|evalkit:r-4b7e3a0c3d19:{samples.FT_MODEL}") for s in built)
    assert all(s.source.startswith("evalkit:r-67b40f1ce1f5:gemma-e2b-base-q4|") for s in built)
    for s in built:  # every v2 sample reply is a valid answer citing real sentences
        card = to_card("ft", s.ft_raw, s.context)
        assert card.error is None and card.cited and not card.invalid_citations


def test_two_runs_over_different_cases_are_refused(tmp_path):
    other = tmp_path / "run"
    other.mkdir()
    src = ROOT / samples.BASE_RUN
    (other / "records.jsonl.gz").write_bytes((src / "records.jsonl.gz").read_bytes())
    lines = gzip.decompress((src / "cases.jsonl.gz").read_bytes()).decode().splitlines()
    first = json.loads(lines[0])
    first["question"] = "A different question?"
    lines[0] = json.dumps(first)
    (other / "cases.jsonl.gz").write_bytes(gzip.compress("\n".join(lines).encode()))
    with pytest.raises(ValueError, match="same cases"):
        samples.build(ROOT / samples.RUN, base_run=other)


def test_too_small_a_run_is_refused():
    with pytest.raises(ValueError, match="only"):
        samples.build(ROOT / samples.RUN, base_run=ROOT / samples.BASE_RUN, n=10_000)


def test_local_generators_serve_v2_and_base_only_when_asked():
    assert list(app.local_generators("http://127.0.0.1:8083")) == ["ft"]
    both = app.local_generators("http://127.0.0.1:8083", "http://127.0.0.1:8080")
    assert list(both) == ["base", "ft"]


def test_passkey_env_prefers_the_environment_and_reads_only_the_hash(tmp_path):
    h = passkey.make_hash("1234", iterations=100_000)
    f = tmp_path / ".env"
    f.write_text(f"OTHER_SECRET=nope\nexport {passkey.SECRET_ENV}='{h}'\n", encoding="utf-8")
    assert app.passkey_env({}, f) == {passkey.SECRET_ENV: h}
    assert app.passkey_env({passkey.SECRET_ENV: "x"}, f) == {passkey.SECRET_ENV: "x"}
    assert app.passkey_env({}, None) == {passkey.SECRET_ENV: ""}
    f.write_text("OTHER_SECRET=nope\n", encoding="utf-8")
    assert app.passkey_env({}, f) == {passkey.SECRET_ENV: ""}


def test_endpoint_backend_sends_the_schema_request():
    sent: list[tuple[str, dict]] = []

    def post(url, body):
        sent.append((url, json.loads(body)))
        return json.dumps({"choices": [{"message": {"content": '{"label":"no"}'}}]}).encode()

    gen = backends.endpoint("http://127.0.0.1:8081/", post=post)
    assert gen("PROMPT") == '{"label":"no"}'
    url, body = sent[0]
    assert url == "http://127.0.0.1:8081/v1/chat/completions"
    assert body["messages"] == [{"role": "user", "content": "PROMPT"}]
    assert body["temperature"] == 0 and body["max_tokens"] == MAX_NEW_TOKENS
    assert body["response_format"]["type"] == "json_schema"


def test_local_footer_says_the_demo_serves_the_benchmarked_gguf():
    f = ui.footer()
    assert "ZeroGPU" not in f and "bf16" not in f and BANNER in f  # OWNER D-63
    assert "benchmark numbers are for the Q4_K_M GGUF" in f and "exactly that GGUF" in f
    assert app.LOCAL_FT_MODEL in f and "Benchmark results" in f
    assert (
        "Parity (Q4_K_M against the merged weights)](https://github.com/" in f
    )  # SPC-04, absolute
    space_footer = ui.footer(local=False)
    assert (
        "ZeroGPU" in space_footer and "Parity (Q4_K_M against the merged weights)" in space_footer
    )


def test_results_summary_comes_from_the_committed_bundle():
    src = sources.load(ROOT)
    text = ui.results_summary(src)
    main = src.main()
    assert main.run_id in text and src.bundle.shipped.version in text
    assert "exactly that GGUF" in text and f"**{app.LOCAL_FT_MODEL}**" in text
    v2 = next(m for m in main.models if m.model == app.LOCAL_FT_MODEL)
    assert f"{v2.accuracy.point:.3f} [{v2.accuracy.lo:.3f}, {v2.accuracy.hi:.3f}]" in text
    assert "OWNER D-62" in text  # the aim-rule caveat travels with the numbers
    with pytest.raises(ValueError, match="demo serves"):
        ui.results_summary(src, served_model="gemma-e2b-grounded-q4")


def test_render_shows_label_answer_and_highlights_the_cited_sentences():
    sentences = ("Aspirin was given.", "Pain fell in 9 of 12.", "No harm <b>seen</b>.")
    raw = json.dumps({"label": "yes", "answer": "Pain fell.", "citations": [2]})
    msg, abstract, base, ft = ui.render(
        Result(sentences=sentences, cards={"ft": to_card("ft", raw, sentences)})
    )
    assert msg == "" and base == ""
    assert "<span class='cited'><sup>2</sup> Pain fell in 9 of 12.</span>" in abstract
    assert "<b>seen</b>" not in abstract  # escaped
    assert ui.TITLES["ft"] in ft and "<b>Label:</b> yes" in ft and "Pain fell." in ft
    assert "[2] Pain fell in 9 of 12." in ft


def test_render_escapes_messages():
    msg, *_ = ui.render(Result(message="<script>x</script>"))
    assert "<script>" not in msg


def test_gradio_page_builds(monkeypatch):
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")  # no network from a test
    pytest.importorskip("gradio")
    from ggh.space import app

    space = app.make_space({}, {}, ROOT / samples.SAMPLES)
    assert ui.build(space, app.sample_labels(samples.load(ROOT / samples.SAMPLES))) is not None
