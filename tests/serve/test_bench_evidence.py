"""BEN-01 to BEN-04 on the committed `laptop_bench.json`: four valid records under the §8
protocol, the hardware stamp, llama-bench numbers, and the base/fine-tuned per-token gap."""

import json
from pathlib import Path

from ggh.schemas import LaptopBench
from ggh.serve import bench as B
from ggh.serve import llama

ROOT = Path(__file__).resolve().parents[2]
DATA = json.loads((ROOT / B.OUT).read_text(encoding="utf-8"))
RES = {k: LaptopBench.model_validate(v) for k, v in DATA["variants"].items()}
MANIFESTS = {m: json.loads((ROOT / f"docs/measurements/gguf_{m}_manifest.json")
                           .read_text(encoding="utf-8")) for m in ("base", "ft")}  # fmt: skip
RUN_STATE = (ROOT / "specs/X3.2-bench/RUN-STATE.md").read_text(encoding="utf-8")


def test_four_variants_ran_the_protocol():
    assert set(RES) == {f"{m}-{q}" for m, q in B.VARIANTS}
    for k, r in RES.items():
        assert r.n_prompts == 50 and r.repetitions == 3 and r.quant == k.split("-", 1)[1]
        assert (
            r.gen_tps_p95 >= r.gen_tps_p50 > 0
            and r.answer_latency_ms_p95 >= r.answer_latency_ms_p50
        )
        assert 0 < r.ttft_ms_p50 < r.answer_latency_ms_p50


def test_hardware_power_and_tool_are_stamped():
    for r in RES.values():
        assert r.chip.startswith("Apple") and r.ram_gb > 0 and r.os.startswith("macOS ")
        assert r.power == "ac" and r.llama_cpp_commit == llama.TAG


def test_file_sizes_are_the_exported_files_and_memory_exceeds_them():
    for k, r in RES.items():
        model, quant = k.split("-", 1)
        size = MANIFESTS[model]["files"][f"{model}-{quant}.gguf"]["bytes"]
        assert r.model_file_mb == round(size / 2**20, 1)
        assert r.peak_rss_mb > r.model_file_mb  # the server maps the whole file


def test_llama_bench_pp512_and_tg128_are_captured():
    for r in RES.values():
        assert set(r.llama_bench) == {"pp512", "tg128"}
        assert r.llama_bench["pp512"] > r.llama_bench["tg128"] > 0


def test_base_and_ft_speed_agree_within_ten_percent_on_both_measures():
    for on in ("tg128", "request"):
        gaps = B.per_token_gap(RES, on)
        assert set(gaps) == {"Q4_K_M", "Q8_0"} and all(g <= 0.10 for g in gaps.values()), on
        for q, g in gaps.items():
            assert f"{100 * g:.1f} % ({q}" in RUN_STATE  # the recorded numbers are these


def test_llama_bench_values_are_medians_of_three_alternating_rounds():
    import statistics

    for k, r in RES.items():
        rounds = DATA["details"][k]["llama_bench_rounds"]
        assert len(rounds) == 3
        for t in ("pp512", "tg128"):
            assert r.llama_bench[t] == statistics.median(x[t] for x in rounds)


def test_token_counts_and_protocol_are_recorded():
    assert DATA["protocol"] == B.PROTOCOL and DATA["measured_at"].endswith("Z")
    for k in RES:
        d = DATA["details"][k]
        assert d["prompt_tokens_p50"] > 0 and d["answer_tokens_p50"] > 0 and d["rss_samples"] > 100
    ft, base = DATA["details"]["ft-Q4_K_M"], DATA["details"]["base-Q4_K_M"]
    assert ft["prompt_tokens_p50"] == base["prompt_tokens_p50"]  # same prompts, same template
