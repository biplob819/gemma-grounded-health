"""BEN-01 to BEN-05: the laptop benchmark under the fixed protocol of Tech Spec §8 (D-19).

Each variant gets its own `llama-server` with `ggh serve`'s arguments. 50 seeded `dev_eval`
prompts go through the chat endpoint with EvalKit's answer schema, at most 256 new tokens,
greedy, prompt cache off; 3 warm-ups are discarded, then the 50 run 3 times. The server's own
`timings` give the token rates and time to first token; wall time gives per-answer latency.
Each metric is the median across repetitions of that repetition's p50 (or p95). A thread
samples the server's resident memory every 50 ms. `llama-bench` adds pp512 and tg128.
"""

from __future__ import annotations

import json
import platform
import random
import statistics
import subprocess
import threading
import time
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from evalkit.pipeline.answer import answer_schema

from ggh.schemas import LaptopBench
from ggh.serve import llama

N_PROMPTS = 50
WARMUP = 3
REPETITIONS = 3
MAX_TOKENS = 256
SEED = 7
PORT = 8095  # not a `ggh serve` port
SAMPLE_S = 0.05
LB_REPS = 5  # llama-bench repetitions per call (its default)
LB_ROUNDS = 3  # alternating rounds over all variants, before any workload runs
LABELS = ("yes", "no", "maybe")
OUT = Path("docs") / "measurements" / "laptop_bench.json"
VARIANTS = (("base", "Q4_K_M"), ("ft", "Q4_K_M"), ("base", "Q8_0"), ("ft", "Q8_0"))
# V2.1: v2's GGUFs, each beside v1's measured in the same session; v1's file is left alone
V2_OUT = Path("docs") / "measurements" / "v2" / "laptop_bench.json"
V2_VARIANTS = (("ft", "Q4_K_M"), ("ft2", "Q4_K_M"), ("ft", "Q8_0"), ("ft2", "Q8_0"))
# V3.8 (RBR-05): v3b's two shipped GGUFs, Q4_K_M (registered) and Q8_0 (recommended, OWNER D-75)
V3B_OUT = Path("docs") / "measurements" / "v3b" / "laptop_bench.json"
V3B_VARIANTS = (("ft3b", "Q4_K_M"), ("ft3b", "Q8_0"))
# `ggh bench --model`: the variants measured together and the file they are written to
RUNS: dict[str, tuple[tuple[tuple[str, str], ...], Path]] = {
    "ft": (VARIANTS, OUT),
    "ft2": (V2_VARIANTS, V2_OUT),
    "ft3b": (V3B_VARIANTS, V3B_OUT),
}
METRICS = ("prompt_tps", "gen_tps", "ttft_ms", "latency_ms")

Post = Callable[[str, dict[str, Any]], dict[str, Any]]
Run = Callable[[Sequence[str]], str]


class BenchError(RuntimeError):
    pass


def select_prompts(rows: Sequence[dict[str, Any]], n: int = N_PROMPTS, seed: int = SEED
                   ) -> list[str]:  # fmt: skip
    """`n` prompts drawn with a fixed seed, in draw order."""
    if len(rows) < n + WARMUP:
        raise BenchError(f"need at least {n + WARMUP} dev_eval rows, got {len(rows)}")
    picked = random.Random(seed).sample(list(rows), n + WARMUP)
    return [r["prompt"] for r in picked]


def request_body(prompt: str) -> dict[str, Any]:
    return {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "cache_prompt": False,  # every prompt processed fresh, every repetition
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": dict(answer_schema(LABELS)),
                "strict": True,
            },
        },
    }


def measure(resp: dict[str, Any], wall_ms: float) -> dict[str, float]:
    """One request's metrics from llama-server's `timings` and the wall clock."""
    t = resp["timings"]
    if t["predicted_n"] < 1 or t["prompt_n"] < 1:
        raise BenchError(f"server timings report no work: {t}")
    per_token_ms = t["predicted_ms"] / t["predicted_n"]
    return {
        "prompt_tps": float(t["prompt_per_second"]),
        "gen_tps": float(t["predicted_per_second"]),
        "ttft_ms": float(t["prompt_ms"] + per_token_ms),
        "latency_ms": float(wall_ms),
        "prompt_tokens": float(t["prompt_n"]),
        "answer_tokens": float(t["predicted_n"]),
    }


def pct(xs: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (numpy's default), q in [0, 100]."""
    if not xs:
        raise BenchError("no values")
    s = sorted(xs)
    k = (len(s) - 1) * q / 100
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def aggregate(reps: Sequence[Sequence[dict[str, float]]]) -> dict[str, float]:
    """Median across repetitions of each repetition's p50 and p95."""
    if len(reps) != REPETITIONS or any(not r for r in reps):
        raise BenchError(f"need {REPETITIONS} non-empty repetitions")
    out = {}
    for m in METRICS:
        for q in (50, 95):
            out[f"{m}_p{q}"] = statistics.median(pct([x[m] for x in rep], q) for rep in reps)
    return out


def ps_rss_mb(pid: int) -> float | None:  # pragma: no cover - reads the live process
    r = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True)
    text = r.stdout.strip()
    return int(text) / 1024 if text else None


class RssSampler:
    """Peak resident memory of one process, sampled every `interval` seconds in a thread."""

    def __init__(self, pid: int, interval: float = SAMPLE_S,
                 read: Callable[[int], float | None] = ps_rss_mb) -> None:  # fmt: skip
        self.pid, self.interval, self.read = pid, interval, read
        self.peak = 0.0
        self.samples = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            v = self.read(self.pid)
            if v is not None:
                self.peak = max(self.peak, v)
                self.samples += 1
            self._stop.wait(self.interval)

    def __enter__(self) -> RssSampler:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()


def _sh(cmd: Sequence[str]) -> str:  # pragma: no cover - host commands
    return subprocess.run(list(cmd), capture_output=True, text=True, check=True).stdout


def hardware(run: Run = _sh) -> dict[str, Any]:
    chip = run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip()
    ram = int(run(["sysctl", "-n", "hw.memsize"]).strip()) // 2**30
    os_ = f"macOS {run(['sw_vers', '-productVersion']).strip()}"
    batt = run(["pmset", "-g", "batt"])
    power = "ac" if "AC Power" in batt else "battery"
    return {"chip": chip, "ram_gb": ram, "os": os_, "power": power}


def guard(hw: dict[str, Any], busy: Callable[[int], bool] | None = None,
          allow: Sequence[int] = ()) -> None:  # fmt: skip
    """BEN-05: never alongside an E3.2 generation run, never on battery. `allow` names ports
    whose server the caller has checked is idle (recorded in the output's `conditions`)."""
    busy = busy or llama.port_busy
    used = [p for p in sorted(set(llama.PORTS.values())) if p not in allow and busy(p)]
    if used:
        raise BenchError(
            f"a llama-server is listening on {used} (an evaluation run?); stop it first"
        )
    if hw["power"] != "ac":
        raise BenchError("on battery power; §8 needs mains power")


def parse_llama_bench(text: str) -> tuple[dict[str, float], dict[str, float]]:
    """pp512 and tg128 tokens per second from `llama-bench -o json`, and their spread."""
    out, sd = {}, {}
    for e in json.loads(text):
        key = ("pp512" if e.get("n_prompt") == 512 and e.get("n_gen") == 0 else
               "tg128" if e.get("n_gen") == 128 and e.get("n_prompt") == 0 else None)  # fmt: skip
        if key:
            out[key], sd[key] = float(e["avg_ts"]), float(e.get("stddev_ts", 0.0))
    if set(out) != {"pp512", "tg128"}:
        raise BenchError(f"llama-bench output lacks pp512 or tg128: got {sorted(out)}")
    return out, sd


def llama_bench_cmd(gguf: Path, cache: Path | None = None) -> list[str]:
    return [str(llama.tool("llama-bench", cache)), "-m", str(gguf), "-p", "512", "-n", "128",
            "-ngl", "999", "-r", str(LB_REPS), "-o", "json"]  # fmt: skip


def http_post(url: str, body: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover - network
    req = urllib.request.Request(
        url, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


Reps = list[list[dict[str, float]]]


def run_workload(
    prompts: Sequence[str],
    post: Post,
    base_url: str,
    clock: Callable[[], float] = time.perf_counter,
) -> Reps:
    """3 warm-ups discarded, then REPETITIONS passes over the rest."""
    url = f"{base_url}/v1/chat/completions"
    for p in prompts[:WARMUP]:
        post(url, request_body(p))
    reps = []
    for _ in range(REPETITIONS):
        rows = []
        for p in prompts[WARMUP:]:
            t0 = clock()
            resp = post(url, request_body(p))
            rows.append(measure(resp, (clock() - t0) * 1000))
        reps.append(rows)
    return reps


def record(hw: dict[str, Any], quant: str, gguf_bytes: int, n: int, agg: dict[str, float],
           peak_mb: float, lb: dict[str, float]) -> LaptopBench:  # fmt: skip
    return LaptopBench(
        chip=hw["chip"], ram_gb=hw["ram_gb"], os=hw["os"], power=hw["power"],
        llama_cpp_commit=llama.TAG, quant=quant, model_file_mb=round(gguf_bytes / 2**20, 1),
        n_prompts=n, repetitions=REPETITIONS,
        prompt_tps_p50=round(agg["prompt_tps_p50"], 1),
        gen_tps_p50=round(agg["gen_tps_p50"], 1), gen_tps_p95=round(agg["gen_tps_p95"], 1),
        ttft_ms_p50=round(agg["ttft_ms_p50"]),
        answer_latency_ms_p50=round(agg["latency_ms_p50"]),
        answer_latency_ms_p95=round(agg["latency_ms_p95"]),
        peak_rss_mb=round(peak_mb, 1), llama_bench=lb,
    )  # fmt: skip


Details = dict[str, Any]


def bench_variant(
    root: Path,
    model: str,
    quant: str,
    prompts: Sequence[str],
    hw: dict[str, Any],
    *,
    post: Post = http_post,
    start: Callable[..., Any] = llama.start,
    sampler: Callable[[int], RssSampler] = RssSampler,
    run: Run = _sh,
    lb: dict[str, float] | None = None,
) -> tuple[LaptopBench, Details]:
    """The `LaptopBench` record and the details behind it. `lb` is this variant's llama-bench
    result from `llama_bench_rounds`; without it, one llama-bench call is made here."""
    gguf = llama.model_path(root, model, quant)
    proc = start(root, model, quant, PORT, log=root / llama.GGUF_DIR / f"bench-{model}-{quant}.log")
    try:
        with sampler(proc.pid) as rss:
            reps = run_workload(prompts, post, f"http://{llama.HOST}:{PORT}")
    finally:
        proc.terminate()
        proc.wait(timeout=30)
    sd: dict[str, float] = {}
    if lb is None:
        lb, sd = parse_llama_bench(run(llama_bench_cmd(gguf)))
    n = len(prompts) - WARMUP
    rec = record(hw, quant, gguf.stat().st_size, n, aggregate(reps), rss.peak, lb)
    details_sd = {"llama_bench_stddev": sd} if sd else {}
    allrows = [x for rep in reps for x in rep]
    details = {
        "prompt_tokens_p50": statistics.median(x["prompt_tokens"] for x in allrows),
        "answer_tokens_p50": statistics.median(x["answer_tokens"] for x in allrows),
        "answer_tokens_mean": round(statistics.fmean(x["answer_tokens"] for x in allrows), 1),
        "rss_samples": rss.samples,
        **details_sd,
    }
    return rec, details


def llama_bench_rounds(
    root: Path, variants: Sequence[tuple[str, str]], run: Run = _sh, rounds: int = LB_ROUNDS
) -> tuple[dict[str, dict[str, float]], Details]:
    """llama-bench on every variant in turn, `rounds` times, so slow drift in the machine's
    state (heat, background work) falls on all variants alike; each value is the median across
    rounds. Returns the medians and every round's numbers."""
    per: dict[str, list[dict[str, float]]] = {f"{m}-{q}": [] for m, q in variants}
    for _ in range(rounds):
        for m, q in variants:
            out, _sd = parse_llama_bench(run(llama_bench_cmd(llama.model_path(root, m, q))))
            per[f"{m}-{q}"].append(out)
    med = {k: {t: statistics.median(r[t] for r in rs) for t in ("pp512", "tg128")}
           for k, rs in per.items()}  # fmt: skip
    return med, {k: rs for k, rs in per.items()}


def per_token_gap(results: dict[str, LaptopBench], on: str = "tg128",
                  pair: tuple[str, str] = ("base", "ft")) -> dict[str, float]:  # fmt: skip
    """Relative gap in generation speed between base and ft, per quant (BEN-04): on llama-bench
    tg128 (a fixed 128-token generation) or on the workload's per-request p50 (`gen_tps_p50`,
    which moves with answer length and grammar overhead). BEN-04 needs both within 10 %."""

    def speed(r: LaptopBench) -> float:
        return r.llama_bench["tg128"] if on == "tg128" else r.gen_tps_p50

    out = {}
    for q in sorted({r.quant for r in results.values()}):
        b, f = speed(results[f"{pair[0]}-{q}"]), speed(results[f"{pair[1]}-{q}"])
        out[q] = abs(b - f) / max(b, f)
    return out


PROTOCOL = {
    "section": "Tech Spec §8",
    "prompts": f"{N_PROMPTS} dev_eval prompts drawn with seed {SEED}, {WARMUP} more as warm-ups",
    "repetitions": REPETITIONS,
    "request": "chat completions with EvalKit's answer schema (response_format json_schema)",
    "max_tokens": MAX_TOKENS,
    "deviations": [
        "context 8192 (ggh serve's --ctx-size), not llama-server's default: §8 asks for both",
        "cache_prompt false, so every prompt is processed fresh (EvalKit's request leaves the "
        "server default, prompt caching on)",
        "max_tokens 256 per §8 (EvalKit sends max_completion_tokens 2048) and temperature 0 "
        "(the server is already greedy: --temp 0 --top-k 1)",
        "output laptop_bench.json with a measured_at stamp, not §8's laptop-bench-<date>.json",
    ],
    "llama_bench": f"-p 512 -n 128 -ngl 999 -r {LB_REPS}, run on every variant in turn for "
    f"{LB_ROUNDS} rounds before the workloads; each value is the median across rounds",
    "metrics": {
        "ttft_ms": "server prompt_ms plus one generated token's mean time",
        "gen_tps": "server predicted_per_second per request; its p95 is the fast tail",
        "latency_ms": "wall time around the request",
        "aggregate": "median across repetitions of each repetition's p50 (or p95)",
        "peak_rss_mb": "ps rss of the llama-server process every 50 ms from /health to shutdown; "
        "on Apple Silicon it may leave out Metal private buffers such as the KV cache",
    },
}


def write(root: Path, results: dict[str, LaptopBench], details: dict[str, dict[str, Any]],
          measured_at: str, out: Path = OUT, conditions: dict[str, Any] | None = None
          ) -> Path:  # fmt: skip
    data = {k: v.model_dump(mode="json") for k, v in results.items()}
    doc = {"measured_at": measured_at, "host_platform": platform.machine(),
           "protocol": PROTOCOL, "variants": data, "details": details}  # fmt: skip
    if conditions:
        doc["conditions"] = conditions
    path = root / out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return path
