"""RUN-04: the bf16 server speaks the chat-completions shape EvalKit's adapter reads, passes the
schema and token cap through, is greedy only, and serves one request at a time."""

import json
import threading
import urllib.error
import urllib.request

import pytest

from ggh.serve import hf


def test_parse_takes_messages_cap_and_schema():
    rf = {"type": "json_schema", "json_schema": {"name": "answer", "schema": {"type": "object"}}}
    body = {"messages": [{"role": "user", "content": "Q?"}], "max_completion_tokens": 2048,
            "response_format": rf}  # fmt: skip
    assert hf.parse_request(body) == ([{"role": "user", "content": "Q?"}], 2048, {"type": "object"})
    assert hf.parse_request({"messages": body["messages"], "max_tokens": 5})[1:] == (5, None)
    assert hf.parse_request({"messages": body["messages"]})[1] == hf.DEFAULT_MAX


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"messages": []},
        {"messages": [{"role": "user"}]},
        {"messages": [{"role": "user", "content": "x"}], "max_tokens": 0},
        {"messages": [{"role": "user", "content": "x"}], "temperature": 0.7},
        {
            "messages": [{"role": "user", "content": "x"}],
            "response_format": {"type": "json_object"},
        },
    ],
)
def test_bad_requests_are_refused(body):
    with pytest.raises(hf.BadRequest):
        hf.parse_request(body)


def test_completion_has_the_fields_evalkits_adapter_reads():
    out = hf.completion(hf.Reply('{"label": "yes"}', 500, 12, "stop"))
    assert out["choices"][0]["message"] == {"role": "assistant", "content": '{"label": "yes"}'}
    assert out["choices"][0]["finish_reason"] == "stop" and out["model"] == hf.MODEL_ID
    assert out["usage"] == {"prompt_tokens": 500, "completion_tokens": 12, "total_tokens": 512}
    assert out["object"] == "chat.completion" and out["id"].startswith("chatcmpl-")


@pytest.fixture
def server():
    seen, active, peak = [], [0], [0]
    lock = threading.Lock()

    def gen(msgs, max_tokens, schema):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        threading.Event().wait(0.02)
        seen.append((msgs, max_tokens, schema))
        with lock:
            active[0] -= 1
        return hf.Reply("ok", 3, 1, "stop")

    srv = hf.serve(gen, port=0)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://{hf.HOST}:{srv.server_address[1]}", seen, peak
    srv.shutdown()


def post(url, body):
    req = urllib.request.Request(
        url, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def test_http_round_trip_health_models_and_chat(server):
    base, seen, _ = server
    with urllib.request.urlopen(f"{base}/health") as r:
        assert json.loads(r.read()) == {"status": "ok"}
    with urllib.request.urlopen(f"{base}/v1/models") as r:
        assert json.loads(r.read())["data"][0]["id"] == hf.MODEL_ID
    sch = {"type": "object"}
    rf = {"type": "json_schema", "json_schema": {"schema": sch}}
    msg = [{"role": "user", "content": "Q"}]
    out = post(f"{base}/v1/chat/completions",
               {"messages": msg, "max_completion_tokens": 7, "response_format": rf})  # fmt: skip
    assert out["choices"][0]["message"]["content"] == "ok"
    assert seen == [([{"role": "user", "content": "Q"}], 7, sch)]
    with pytest.raises(urllib.error.HTTPError) as e:
        post(f"{base}/v1/chat/completions", {"messages": []})
    assert e.value.code == 400


def test_requests_are_served_one_at_a_time(server):
    base, seen, peak = server
    body = {"messages": [{"role": "user", "content": "Q"}]}
    ts = [
        threading.Thread(target=post, args=(f"{base}/v1/chat/completions", body)) for _ in range(4)
    ]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(seen) == 4 and peak[0] == 1


def test_the_server_binds_loopback_only():
    assert hf.HOST == "127.0.0.1" and hf.PORT not in (8080, 8081)


def test_grammar_keeps_the_schemas_property_order():
    from evalkit.pipeline.answer import answer_schema

    sch = dict(answer_schema(("yes", "no", "maybe")))
    src = json.loads(hf.grammar_source(sch))
    assert list(src["properties"]) == ["label", "answer", "citations"]  # not sorted
    assert next(iter(json.loads(json.dumps(sch, sort_keys=True))["properties"])) == "answer"
