"""V2.1: `ggh parity --model ft2` and `ggh bench --model ft2` write v2's evidence under
docs/measurements/v2 and leave v1's files alone. Needs no artifacts/, torch or network."""

import gzip
import json
from pathlib import Path

import pytest

from ggh import cli
from ggh.export import parity as P
from ggh.schemas import LaptopBench
from ggh.serve import bench as B
from ggh.serve import llama

HW = {"chip": "Apple M3 Max", "ram_gb": 64, "os": "macOS 26.5.2", "power": "ac"}


def ans(label):
    return json.dumps({"label": label, "answer": "Short.", "citations": [1]})


def test_v1_parity_paths_are_unchanged_and_v2_writes_beside_its_records():
    assert P.paths() == P.paths("ft") == (P.REPORT, P.GENERATIONS, P.SAMPLE)
    assert P.paths("ft2") == (Path("docs/measurements/v2/parity.json"),
                              Path("docs/measurements/v2/parity_generations.jsonl.gz"),
                              Path("docs/measurements/v2/parity_sample.jsonl"))  # fmt: skip
    assert P.q4_name("ft") == P.Q4_NAME and P.q4_name("ft2") == "gemma-e2b-ft2-q4"
    with pytest.raises(ValueError):
        P.paths("base")


def test_ft2_run_writes_only_the_v2_files(tmp_path):
    rows = [{"pmid": i, "label": "yes", "prompt": f"p{i}"} for i in range(30)]
    rep = P.run(tmp_path, rows, model="ft2", bf16_gen=lambda ps: [ans("yes") for _ in ps],
                q4_gen=lambda ps: [ans("yes") for _ in ps])  # fmt: skip
    report, gens, sample = (tmp_path / p for p in P.paths("ft2"))
    assert P.passes(rep) and report.is_file() and sample.is_file()
    assert len(gzip.decompress(gens.read_bytes()).decode().splitlines()) == 30
    assert not (tmp_path / P.REPORT).exists() and not (tmp_path / P.GENERATIONS).exists()


def test_cli_parity_ft2_uses_the_v2_merge_and_the_given_port(tmp_path, monkeypatch):
    from ggh.train.devrun import DEV_EVAL

    (tmp_path / DEV_EVAL).parent.mkdir(parents=True)
    (tmp_path / DEV_EVAL).write_text(json.dumps({"pmid": 1, "label": "yes", "prompt": "p"}) + "\n",
                                     encoding="utf-8")  # fmt: skip
    seen = {}

    def fake_run(root, rows, *, bf16_gen, q4_gen, model):
        seen["model"] = model
        return P.build_report([ans("yes")], [ans("yes")])

    monkeypatch.setattr(P, "run", fake_run)
    assert cli.main(["parity", "--model", "ft2", "--port", "8084", "--root", str(tmp_path)]) == 0
    assert seen == {"model": "ft2"}
    with pytest.raises(SystemExit):  # argparse: parity runs on ft or ft2 only
        cli.main(["parity", "--model", "base", "--root", str(tmp_path)])


def test_bench_v2_variants_pair_each_v2_gguf_with_v1s_and_write_their_own_file():
    assert B.V2_OUT == Path("docs/measurements/v2/laptop_bench.json") != B.OUT
    assert set(B.V2_VARIANTS) == {(m, q) for m in ("ft", "ft2") for q in ("Q4_K_M", "Q8_0")}


def test_guard_ignores_only_the_ports_named_idle():
    B.guard(HW, busy=lambda p: p == 8083, allow=[8083])
    with pytest.raises(B.BenchError, match=r"\[8081\]"):
        B.guard(HW, busy=lambda p: p in (8081, 8083), allow=[8083])


def _rec(quant, tps):
    return LaptopBench(chip="Apple M3 Max", ram_gb=64, os="macOS 26.5.2", power="ac",
                       llama_cpp_commit=llama.TAG, quant=quant, model_file_mb=1.0, n_prompts=50,
                       repetitions=3, prompt_tps_p50=1.0, gen_tps_p50=tps, gen_tps_p95=tps,
                       ttft_ms_p50=1, answer_latency_ms_p50=2, answer_latency_ms_p95=2,
                       peak_rss_mb=2.0, llama_bench={"pp512": 2.0, "tg128": tps})  # fmt: skip


def test_cli_bench_ft2_writes_v2_file_with_conditions_and_leaves_v1(tmp_path, monkeypatch):
    from ggh.train.devrun import DEV_EVAL

    (tmp_path / DEV_EVAL).parent.mkdir(parents=True)
    rows = "".join(json.dumps({"prompt": f"p{i}"}) + "\n" for i in range(60))
    (tmp_path / DEV_EVAL).write_text(rows, encoding="utf-8")
    monkeypatch.setattr(B, "hardware", lambda: HW)
    monkeypatch.setattr(llama, "port_busy", lambda p, host=llama.HOST: p == 8083)
    lb = {f"{m}-{q}": {"pp512": 2.0, "tg128": 1.0} for m, q in B.V2_VARIANTS}
    ran = []
    monkeypatch.setattr(B, "llama_bench_rounds", lambda root, v: (lb, {k: [] for k in lb}))

    def variant(root, model, quant, prompts, hw, lb):
        ran.append(f"{model}-{quant}")
        return _rec(quant, 1.0), {}

    monkeypatch.setattr(B, "bench_variant", variant)
    assert cli.main(["bench", "--model", "ft2", "--root", str(tmp_path)]) == cli.EXIT_ERROR
    args = ["bench", "--model", "ft2", "--allow-idle-port", "8083", "--root", str(tmp_path)]
    assert cli.main(args) == 0
    assert ran == [f"{m}-{q}" for m, q in B.V2_VARIANTS]
    doc = json.loads((tmp_path / B.V2_OUT).read_text(encoding="utf-8"))
    assert doc["conditions"] == {"idle_servers_listening": [8083]}
    assert set(doc["variants"]) == set(lb) and not (tmp_path / B.OUT).exists()
    assert B.per_token_gap({k: _rec(k.split("-", 1)[1], 1.0) for k in lb}, pair=("ft", "ft2")) == {
        "Q4_K_M": 0.0,
        "Q8_0": 0.0,
    }
