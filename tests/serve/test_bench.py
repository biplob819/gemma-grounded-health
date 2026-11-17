"""BEN-01 to BEN-05: the §8 protocol on fixed numbers, the memory sampler, the hardware record,
llama-bench parsing, the four variants, and the guards."""

import json
import threading
from pathlib import Path

import pytest

from ggh import cli
from ggh.schemas import LaptopBench
from ggh.serve import bench as B
from ggh.serve import llama

HW = {"chip": "Apple M3 Max", "ram_gb": 64, "os": "macOS 26.5.2", "power": "ac"}


def timings(prompt_n=400, prompt_ms=100.0, pred_n=40, pred_ms=400.0):
    return {"timings": {"prompt_n": prompt_n, "prompt_ms": prompt_ms,
                        "prompt_per_second": prompt_n / prompt_ms * 1000,
                        "predicted_n": pred_n, "predicted_ms": pred_ms,
                        "predicted_per_second": pred_n / pred_ms * 1000}}  # fmt: skip


# ---- BEN-01: protocol ----


def test_prompts_are_seeded_and_include_three_warmups():
    rows = [{"prompt": f"p{i}"} for i in range(500)]
    a, b = B.select_prompts(rows), B.select_prompts(list(reversed(rows))[::-1])
    assert a == b and len(a) == B.N_PROMPTS + B.WARMUP == 53 and len(set(a)) == 53
    assert B.select_prompts(rows, seed=8) != a
    with pytest.raises(B.BenchError, match="at least 53"):
        B.select_prompts(rows[:52])


def test_request_is_evalkits_constrained_chat_with_256_tokens_and_no_prompt_cache():
    body = B.request_body("Q?")
    assert body["messages"] == [{"role": "user", "content": "Q?"}]
    assert body["max_tokens"] == 256 and body["temperature"] == 0
    assert body["cache_prompt"] is False
    from evalkit.pipeline.answer import answer_schema

    assert body["response_format"]["json_schema"]["schema"] == dict(answer_schema(B.LABELS))


def test_measure_uses_server_timings_and_wall_clock():
    m = B.measure(timings(), 612.0)
    assert m == {"prompt_tps": 4000.0, "gen_tps": 100.0, "ttft_ms": 110.0, "latency_ms": 612.0,
                 "prompt_tokens": 400.0, "answer_tokens": 40.0}  # fmt: skip
    with pytest.raises(B.BenchError, match="no work"):
        B.measure(timings(pred_n=0), 1.0)


def test_percentile_is_linear_interpolation():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert B.pct(xs, 50) == 3.0 and B.pct(xs, 95) == pytest.approx(4.8) and B.pct([7.0], 95) == 7.0
    assert B.pct(list(range(1, 51)), 95) == pytest.approx(47.55)


def test_aggregate_is_the_median_of_the_per_repetition_percentiles():
    def rep(scale):
        return [{"prompt_tps": scale * i, "gen_tps": scale * i, "ttft_ms": scale * i,
                 "latency_ms": scale * i} for i in range(1, 6)]  # fmt: skip

    agg = B.aggregate([rep(1.0), rep(10.0), rep(2.0)])
    assert agg["gen_tps_p50"] == 6.0  # p50s are 3, 30, 6: median 6 (not the pooled median)
    assert agg["latency_ms_p95"] == pytest.approx(9.6)  # p95s 4.8, 48, 9.6
    with pytest.raises(B.BenchError):
        B.aggregate([rep(1.0), rep(2.0)])


def test_workload_discards_three_warmups_then_runs_three_repetitions():
    sent = []

    def post(url, body):
        sent.append(body["messages"][0]["content"])
        return timings()

    ticks = iter(range(10_000))
    reps = B.run_workload([f"p{i}" for i in range(53)], post, "http://h:1",
                          clock=lambda: next(ticks) / 1000)  # fmt: skip
    assert sent[:3] == ["p0", "p1", "p2"] and len(sent) == 3 + 3 * 50
    assert sent[3:53] == sent[53:103] == sent[103:] == [f"p{i}" for i in range(3, 53)]
    assert [len(r) for r in reps] == [50, 50, 50] and reps[0][0]["latency_ms"] == pytest.approx(1)


def test_the_server_gets_ggh_serve_arguments_on_its_own_port(tmp_path, monkeypatch):
    gguf = tmp_path / llama.GGUF_DIR / "ft-Q4_K_M.gguf"
    gguf.parent.mkdir(parents=True)
    gguf.write_bytes(b"x" * 2**20)
    started = {}

    class Proc:
        pid = 99
        terminated = False

        def terminate(self):
            Proc.terminated = True

        def wait(self, timeout=None):
            return 0

    def start(root, model, quant, port, *, log):
        started.update(model=model, quant=quant, port=port)
        return Proc()

    class NoSampler:
        peak = 3000.0
        samples = 12

        def __init__(self, pid):
            assert pid == 99

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    lb = json.dumps([{"n_prompt": 512, "n_gen": 0, "avg_ts": 2000.0},
                     {"n_prompt": 0, "n_gen": 128, "avg_ts": 90.0}])  # fmt: skip
    r, det = B.bench_variant(tmp_path, "ft", "Q4_K_M", [f"p{i}" for i in range(53)], HW,
                        post=lambda u, b: timings(), start=start, sampler=NoSampler,  # pyright: ignore[reportArgumentType]
                        run=lambda cmd: lb)  # fmt: skip
    assert started == {"model": "ft", "quant": "Q4_K_M", "port": B.PORT} and Proc.terminated
    assert B.PORT not in llama.PORTS.values()
    assert r.n_prompts == 50 and r.repetitions == 3 and r.gen_tps_p50 == 100.0
    assert r.model_file_mb == 1.0 and r.peak_rss_mb == 3000.0 and r.llama_cpp_commit == llama.TAG
    assert dict(r.llama_bench) == {"pp512": 2000.0, "tg128": 90.0}
    assert det["answer_tokens_p50"] == 40.0 and det["prompt_tokens_p50"] == 400.0
    assert det["llama_bench_stddev"] == {"pp512": 0.0, "tg128": 0.0}  # its own single call


# ---- BEN-02: memory and hardware ----


def test_the_sampler_keeps_the_peak_and_samples_until_stopped():
    vals = iter([100.0, None, 900.0, 300.0] + [200.0] * 1000)
    done = threading.Event()

    def read(pid):
        v = next(vals)
        if v == 300.0:
            done.set()
        return v

    with B.RssSampler(1, interval=0.001, read=read) as s:
        assert done.wait(2)
    assert s.peak == 900.0 and s.samples >= 3
    n = s.samples
    assert not s._thread.is_alive()  # joined on exit
    threading.Event().wait(0.02)
    assert s.samples == n  # and it took no sample after stopping


def test_hardware_reads_chip_ram_os_and_power():
    answers = {"machdep.cpu.brand_string": "Apple M3 Max\n", "hw.memsize": str(64 * 2**30),
               "-productVersion": "26.5.2\n",
               "batt": "Now drawing from 'AC Power'\n"}  # fmt: skip
    hw = B.hardware(run=lambda cmd: answers[cmd[-1]])
    assert hw == HW
    answers["batt"] = "Now drawing from 'Battery Power'\n"
    assert B.hardware(run=lambda cmd: answers[cmd[-1]])["power"] == "battery"


# ---- BEN-03: llama-bench ----


def test_llama_bench_command_and_parsing(monkeypatch):
    monkeypatch.setattr(llama, "tool", lambda name, cache=None: Path("/v") / name)
    cmd = B.llama_bench_cmd(Path("/m.gguf"))
    want = ["/v/llama-bench", "-m", "/m.gguf", "-p", "512", "-n", "128", "-ngl", "999", "-r"]
    assert cmd == [*want, "5", "-o", "json"]
    pp = {"n_prompt": 512, "n_gen": 0, "avg_ts": 1.5}
    tg = {"n_prompt": 0, "n_gen": 128, "avg_ts": 2.5, "stddev_ts": 0.3}
    out, sd = B.parse_llama_bench(json.dumps([pp, tg]))
    assert out == {"pp512": 1.5, "tg128": 2.5} and sd == {"pp512": 0.0, "tg128": 0.3}
    with pytest.raises(B.BenchError, match="lacks"):
        B.parse_llama_bench(json.dumps([{"n_prompt": 512, "n_gen": 0, "avg_ts": 1.5}]))


# ---- BEN-04: variants ----


def _rec(quant, tg, gen=50.0):  # gen: the per-request p50
    return LaptopBench(**HW, llama_cpp_commit="b1", quant=quant, model_file_mb=1.0, n_prompts=50,
                       repetitions=3, prompt_tps_p50=1.0, gen_tps_p50=gen, gen_tps_p95=gen,
                       ttft_ms_p50=1, answer_latency_ms_p50=1, answer_latency_ms_p95=1,
                       peak_rss_mb=1.0, llama_bench={"pp512": 1.0, "tg128": tg})  # fmt: skip


def test_four_variants_and_the_base_ft_speed_gap(tmp_path):
    assert B.VARIANTS == (("base", "Q4_K_M"), ("ft", "Q4_K_M"), ("base", "Q8_0"), ("ft", "Q8_0"))
    res = {"base-Q4_K_M": _rec("Q4_K_M", 100.0, gen=10.0), "ft-Q4_K_M": _rec("Q4_K_M", 95.0),
           "base-Q8_0": _rec("Q8_0", 80.0), "ft-Q8_0": _rec("Q8_0", 80.0)}  # fmt: skip
    # from tg128, not the per-request rate (which moves with answer length: 10 vs 50 here)
    assert B.per_token_gap(res) == {"Q4_K_M": pytest.approx(0.05), "Q8_0": 0.0}
    assert B.per_token_gap(res, "request") == {"Q4_K_M": pytest.approx(0.8), "Q8_0": 0.0}
    path = B.write(
        tmp_path, res, {k: {"answer_tokens_p50": 1} for k in res}, "2026-11-17T00:00:00Z"
    )
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert {k: LaptopBench.model_validate(v) for k, v in doc["variants"].items()} == res
    assert doc["protocol"] == B.PROTOCOL and doc["measured_at"] == "2026-11-17T00:00:00Z"
    assert set(doc["details"]) == set(res)


# ---- BEN-05: guards ----


def test_refuses_while_a_serve_port_is_busy_or_on_battery():
    with pytest.raises(B.BenchError, match=r"\[8081\]"):
        B.guard(HW, busy=lambda p: p == 8081)
    with pytest.raises(B.BenchError, match="battery"):
        B.guard({**HW, "power": "battery"}, busy=lambda p: False)
    B.guard(HW, busy=lambda p: False)


def test_cli_refuses_before_starting_anything(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(B, "hardware", lambda: HW)
    monkeypatch.setattr(llama, "port_busy", lambda p, host=llama.HOST: p == 8080)
    monkeypatch.setattr(B, "bench_variant", lambda *a, **k: pytest.fail("must not start"))
    assert cli.main(["bench", "--root", str(tmp_path)]) == cli.EXIT_ERROR
    assert "8080" in capsys.readouterr().out


def test_llama_bench_alternates_variants_over_rounds_and_takes_medians(tmp_path, monkeypatch):
    monkeypatch.setattr(llama, "tool", lambda name, cache=None: Path("/v") / name)
    order, speeds = [], iter([100, 90, 120, 95, 110, 85])  # base, ft, base, ft, base, ft

    def run(cmd):
        order.append(Path(cmd[cmd.index("-m") + 1]).name)
        tg = next(speeds)
        return json.dumps([{"n_prompt": 512, "n_gen": 0, "avg_ts": 10.0 * tg},
                           {"n_prompt": 0, "n_gen": 128, "avg_ts": float(tg)}])  # fmt: skip

    med, rounds = B.llama_bench_rounds(tmp_path, [("base", "Q4_K_M"), ("ft", "Q4_K_M")], run=run)
    assert order == ["base-Q4_K_M.gguf", "ft-Q4_K_M.gguf"] * 3  # interleaved, not one then other
    assert med["base-Q4_K_M"]["tg128"] == 110 and med["ft-Q4_K_M"]["tg128"] == 90
    assert [r["tg128"] for r in rounds["ft-Q4_K_M"]] == [90, 95, 85]


def test_cli_runs_the_llama_bench_rounds_before_any_workload(tmp_path, monkeypatch):
    from ggh.train.devrun import DEV_EVAL

    (tmp_path / DEV_EVAL).parent.mkdir(parents=True)
    rows = "".join(json.dumps({"prompt": f"p{i}"}) + "\n" for i in range(60))
    (tmp_path / DEV_EVAL).write_text(rows, encoding="utf-8")
    calls = []
    monkeypatch.setattr(B, "hardware", lambda: HW)
    monkeypatch.setattr(llama, "port_busy", lambda p, host=llama.HOST: False)
    lb = {f"{m}-{q}": {"pp512": 1.0, "tg128": 1.0} for m, q in B.VARIANTS}

    def rounds(root, variants):
        calls.append("rounds")
        return lb, {k: [] for k in lb}

    def variant(root, model, quant, prompts, hw, lb):
        calls.append(f"{model}-{quant}")
        return _rec(quant, 1.0), {}

    monkeypatch.setattr(B, "llama_bench_rounds", rounds)
    monkeypatch.setattr(B, "bench_variant", variant)
    assert cli.main(["bench", "--root", str(tmp_path)]) == 0
    assert calls == ["rounds", *(f"{m}-{q}" for m, q in B.VARIANTS)]
