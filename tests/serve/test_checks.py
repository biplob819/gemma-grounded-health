import hashlib
import json
from pathlib import Path

from evalkit.adapters.base import RawCompletion

from ggh.evals import tasks
from ggh.serve import checks, llama

ROOT = Path(__file__).resolve().parents[2]
GOOD = '{"label": "yes", "answer": "It helped [1].", "citations": [1]}'
STRINGY = '```json\n{"label": "yes", "answer": "It helped.", "citations": ["1"]}\n```'


def raw(text, latency=10):
    return RawCompletion(
        model="m", model_snapshot="", params={}, raw_text=text, tokens_in=1, tokens_out=1,
        tokens_reasoning=0, tokens_cached=0, latency_ms=latency, cost_usd=0.0,
        request_id=None, stop_reason="stop", refusal=False, batch=False,
    )  # fmt: skip


class Server:
    """A fake llama-server: the template wraps the user turn; tokens are character codes."""

    def __init__(self, drift_on=None):
        self.drift_on = drift_on
        self.seen = []

    def __call__(self, url, body):
        self.seen.append(url)
        if url.endswith("/apply-template"):
            (msg,) = body["messages"]
            assert msg["role"] == "user"
            return {"prompt": f"<u>{msg['content']}</u>"}
        assert url.endswith("/tokenize") and body["add_special"] and body["parse_special"]
        ids = [ord(c) for c in body["content"]]
        if self.drift_on and self.drift_on in body["content"]:
            ids[3] += 1
        return {"tokens": ids}


def hf(prompt):
    return [ord(c) for c in f"<u>{prompt}</u>"]


PROMPTS = [("a", "first prompt"), ("b", "second prompt")]


def test_parity_compares_token_ids_and_reports_the_first_mismatch():
    rows = checks.parity("http://s", PROMPTS, Server(drift_on="second"), hf)
    assert [r["equal"] for r in rows] == [True, False]
    assert rows[1]["first_mismatch"] == 3 and rows[0]["first_mismatch"] is None
    assert rows[0]["server_sha256"] == rows[0]["hf_sha256"] != rows[1]["server_sha256"]
    assert checks.first_mismatch([1, 2], [1, 2, 3]) == 2


def test_structured_uses_the_schema_only_when_constrained_and_parses_like_evalkit():
    reqs = []

    def complete(req):
        reqs.append(req)
        return raw(GOOD if req.json_schema is not None else STRINGY)

    rows = checks.structured(PROMPTS, "gemma-e2b-base-q4", complete)
    assert all(r["constrained"]["valid"] and not r["unconstrained"]["valid"] for r in rows)
    assert "citations.0" in rows[0]["unconstrained"]["error"]
    constrained = [r for r in reqs if r.json_schema is not None]
    assert len(constrained) == 2 and len(reqs) == 4
    schema = dict(constrained[0].json_schema)
    assert list(schema["required"]) == ["label", "answer", "citations"]
    assert all(r.max_output_tokens == 2048 and r.model_id == "gemma-e2b-base-q4" for r in reqs)


def test_parity_catches_a_mismatch_in_the_last_token_only():
    def tail(prompt):
        ids = hf(prompt)
        return [*ids[:-1], ids[-1] + 1]

    (row,) = checks.parity("http://s", [("a", "x")], Server(), tail)
    assert not row["equal"] and row["first_mismatch"] == row["n_tokens"] - 1


def test_determinism_needs_byte_identical_replies():
    first = {"a": GOOD, "b": GOOD}
    replies = iter([GOOD, GOOD + " "])
    rows = checks.determinism(PROMPTS, "m", lambda req: raw(next(replies)), first)
    assert [r["identical"] for r in rows] == [True, False]


def test_pass_bar_gates_parity_constrained_json_and_reruns_not_unconstrained():
    s = {"n": 20, "parity_equal": 20, "constrained_valid": 20, "unconstrained_valid": 0,
         "identical_reruns": 20}  # fmt: skip
    assert checks.passed(s)
    for key in ("parity_equal", "constrained_valid", "identical_reruns"):
        assert not checks.passed({**s, key: 19}), key
    assert not checks.passed({**s, "n": 19, "parity_equal": 19, "constrained_valid": 19,
                              "identical_reruns": 19})  # fmt: skip


def test_run_checks_the_twenty_dev_smoke_prompts(tmp_path, monkeypatch):
    for rel in (tasks.SMOKE_TASK, tasks.SMOKE_DATA):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes((ROOT / rel).read_bytes())
    gguf = llama.model_path(tmp_path, "base", "Q4_K_M")
    gguf.parent.mkdir(parents=True)
    gguf.write_bytes(b"GGUF")
    reqs = []

    def complete(req):
        reqs.append(req)
        return raw(GOOD)

    rep = checks.run(tmp_path, "base", "Q4_K_M", post=Server(), complete=complete, hf=hf)
    assert rep["passed"] and rep["summary"]["n"] == 20
    ids = [c for c, _ in tasks.prompts(ROOT, tasks.SMOKE_TASK)]
    assert [r["id"] for r in rep["parity"]] == ids
    assert rep["llama_tag"] == llama.TAG and rep["gguf"] == "base-Q4_K_M.gguf"
    assert rep["gguf_sha256"] == hashlib.sha256(b"GGUF").hexdigest()
    args = rep["server_args"]
    assert args[args.index("--port") + 1] == "8080" and "--temp" in args
    # what EvalKit sends an endpoint model: no effort or other params, its own model name
    assert len(reqs) == 60
    assert all(dict(r.params) == {} and r.model_id == "gemma-e2b-base-q4" for r in reqs)
    # 20 constrained + 20 unconstrained, then 20 constrained determinism reruns
    assert [r.json_schema is not None for r in reqs[:2]] == [True, False]
    assert all(r.json_schema is not None for r in reqs[40:])
    assert all(r["n_tokens"] == len(hf(p)) for r, (_, p) in zip(rep["parity"], tasks.prompts(
        ROOT, tasks.SMOKE_TASK), strict=True))  # fmt: skip


def test_run_fails_on_a_drifting_rerun_and_compares_against_the_constrained_reply(tmp_path):
    for rel in (tasks.SMOKE_TASK, tasks.SMOKE_DATA):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes((ROOT / rel).read_bytes())
    gguf = llama.model_path(tmp_path, "base", "Q4_K_M")
    gguf.parent.mkdir(parents=True)
    gguf.write_bytes(b"GGUF")
    calls = {"n": 0}

    def complete(req):
        calls["n"] += 1
        if req.json_schema is None:
            return raw(STRINGY)  # the unconstrained reply differs from the constrained one
        return raw(GOOD + (" " if calls["n"] == 59 else ""))  # one rerun drifts

    rep = checks.run(tmp_path, "base", "Q4_K_M", post=Server(), complete=complete, hf=hf)
    assert rep["summary"]["identical_reruns"] == 19 and not rep["passed"]
    assert rep["summary"]["unconstrained_valid"] == 0


def test_recorded_checks_passed_on_the_exported_base_model():
    rep = json.loads((ROOT / checks.OUT).read_text(encoding="utf-8"))
    man_path = ROOT / "docs/measurements/gguf_base_manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    assert rep["passed"] and rep["llama_tag"] == llama.TAG
    assert rep["gguf_sha256"] == man["files"]["base-Q4_K_M.gguf"]["sha256"]
    s = rep["summary"]
    assert (
        s["n"] == 20 and s["parity_equal"] == s["constrained_valid"] == s["identical_reruns"] == 20
    )
    assert s == checks.summary(rep)  # the summary matches the per-case rows
    ids = [c for c, _ in tasks.prompts(ROOT, tasks.SMOKE_TASK)]
    for part in ("parity", "structured", "determinism"):
        assert [r["id"] for r in rep[part]] == ids
    for r in rep["structured"]:
        assert checks.valid(r["constrained"]["raw_text"])[0]
        assert checks.valid(r["unconstrained"]["raw_text"])[0] == r["unconstrained"]["valid"]


def test_throughput_measures_wall_time_at_each_in_flight_level(tmp_path):
    import asyncio

    for rel in (tasks.SMOKE_TASK, tasks.SMOKE_DATA):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes((ROOT / rel).read_bytes())
    peak = {"now": 0, "max": []}
    first_prompt = tasks.prompts(tmp_path, tasks.SMOKE_TASK)[0][1]

    def new_client():
        peak["now"], seen = 0, []
        peak["max"].append(seen)

        async def call(req):
            assert req.json_schema is not None and dict(req.params) == {}
            peak["now"] += 1
            seen.append(peak["now"])
            await asyncio.sleep(0.001)
            peak["now"] -= 1
            return raw(GOOD, latency=300 if req.prompt == first_prompt else 100)

        return call

    rep = checks.throughput(tmp_path, "base", "Q4_K_M", new_client, host="h")
    assert [r["in_flight"] for r in rep["runs"]] == list(checks.IN_FLIGHT) == [1, 4, 20]
    assert [max(s) for s in peak["max"]] == [1, 4, 20]  # the semaphore really limits
    for r in rep["runs"]:
        assert r["latency_mean_s"] == 0.11 and r["latency_max_s"] == 0.3
    assert rep["seconds_per_answer"] == rep["runs"][0]["seconds_per_answer"] and rep["host"] == "h"


def test_timed_divides_wall_time_by_the_prompt_count_and_reports_the_max(monkeypatch):
    import asyncio
    import time

    ticks = iter([100.0, 110.0])  # 10 s of wall time for 20 prompts
    monkeypatch.setattr(time, "perf_counter", lambda: next(ticks))

    async def call(req):
        return raw(GOOD, latency=100 + 50 * len(req.prompt) % 7)

    prompts = [(str(i), "p" * i) for i in range(20)]
    out = asyncio.run(checks._timed(prompts, "m", call, 4))
    lats = [(100 + 50 * i % 7) / 1000 for i in range(20)]
    assert out["seconds_per_answer"] == 0.5 and out["wall_s"] == 10.0
    assert out["latency_max_s"] == round(max(lats), 3) and out["in_flight"] == 4
