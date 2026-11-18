"""SPC-03, SPC-04, E7: the committed samples are rebuilt from the dev run; the local backend sends
EvalKit's schema request to ggh serve; footer links and banner; the Gradio page builds (skipped
when gradio is not installed, as in CI)."""

import json
from pathlib import Path

import pytest

from ggh.space import backends, samples, ui
from ggh.space.engine import BANNER, MAX_NEW_TOKENS
from ggh.space.service import Result

ROOT = Path(__file__).resolve().parents[2]


def test_committed_samples_are_rebuilt_from_the_dev_run():
    built = samples.build(ROOT / samples.RUN)
    assert samples.dump(built) == (ROOT / samples.SAMPLES).read_text(encoding="utf-8")
    assert len(built) == samples.N_SAMPLES
    assert {s.reference_label for s in built} == {"yes", "no"}
    assert all(s.id.startswith("dev-") for s in built)  # dev_eval, never the PQA-L benchmark


def test_too_small_a_run_is_refused():
    with pytest.raises(ValueError, match="only"):
        samples.build(ROOT / samples.RUN, n=10_000)


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


def test_footer_states_quota_weights_and_links():
    f = ui.footer()
    assert "ZeroGPU" in f and "bf16" in f and "Q4_K_M" in f and BANNER in f
    assert "Parity figure" in f and "Benchmark results" in f


def test_render_escapes_messages():
    msg, *_ = ui.render(Result(message="<script>x</script>"))
    assert "<script>" not in msg


def test_gradio_page_builds(tmp_path):
    pytest.importorskip("gradio")
    from ggh.space import app

    space = app.make_space({}, {}, ROOT / samples.SAMPLES)
    assert ui.build(space, app.sample_labels(samples.load(ROOT / samples.SAMPLES))) is not None
