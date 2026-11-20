"""EVK-03, EVK-04, EVK-08: live checks against a running `ggh serve`, recorded to
`docs/measurements/serving_checks.json` (Tech Spec §6, D-6, E4, E9).

- Template parity: each prompt goes through llama-server's `/apply-template` and `/tokenize`
  and must give exactly the token ids of the pinned tokenizer's `apply_chat_template`.
- Structured output: each prompt is sent through EvalKit's own OpenAI-compatible adapter with
  its answer schema, and the reply must pass EvalKit's `parse_answer`. The same prompt without
  the schema is measured and reported separately; it is not gated.
- Determinism: the constrained run is repeated and every reply must be byte-identical.

The prompts are the 20 dev smoke rows (EVK-07), so every check sees what the smoke run sees.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from evalkit.adapters import CompletionRequest, RawCompletion
from evalkit.errors import SchemaError
from evalkit.pipeline.answer import answer_schema, parse_answer

from ggh.evals.tasks import LABELS, SMOKE_TASK, prompts
from ggh.serve import llama

OUT = Path("docs") / "measurements" / "serving_checks.json"


def out_path(model: str) -> Path:
    """The base keeps E3.1's file; the fine-tune gets its own (X3.1)."""
    return OUT if model == "base" else OUT.with_name(f"serving_checks_{model}.json")


MAX_OUTPUT_TOKENS = 2048  # configs/evalkit.yaml
N = 20

Post = Callable[[str, dict[str, Any]], dict[str, Any]]
Complete = Callable[[CompletionRequest], RawCompletion]
HFTokens = Callable[[str], list[int]]


def http_post(url: str, body: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover - network
    req = urllib.request.Request(
        url, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())


def messages(prompt: str) -> list[dict[str, str]]:
    """What EvalKit's adapter sends: one user turn, no system message."""
    return [{"role": "user", "content": prompt}]


def server_tokens(base: str, prompt: str, post: Post) -> list[int]:
    text = post(f"{base}/apply-template", {"messages": messages(prompt)})["prompt"]
    out = post(f"{base}/tokenize", {"content": text, "add_special": True, "parse_special": True})
    return [int(t) for t in out["tokens"]]


def ids_hash(ids: Sequence[int]) -> str:
    return hashlib.sha256(json.dumps(list(ids)).encode()).hexdigest()


def first_mismatch(a: Sequence[int], b: Sequence[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def parity(
    base: str, prompts: Sequence[tuple[str, str]], post: Post, hf: HFTokens
) -> list[dict[str, Any]]:
    rows = []
    for cid, prompt in prompts:
        s, h = server_tokens(base, prompt, post), hf(prompt)
        rows.append(
            {
                "id": cid,
                "n_tokens": len(h),
                "server_sha256": ids_hash(s),
                "hf_sha256": ids_hash(h),
                "first_mismatch": first_mismatch(s, h),
                "equal": s == h,
            }
        )
    return rows


def request(prompt: str, model: str, *, constrained: bool) -> CompletionRequest:
    return CompletionRequest(
        model=model,
        model_id=model,
        prompt=prompt,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        json_schema=answer_schema(LABELS) if constrained else None,
    )


def valid(text: str) -> tuple[bool, str | None]:
    try:
        parse_answer(text, LABELS)
    except SchemaError as exc:
        return False, str(exc)
    return True, None


def structured(
    prompts: Sequence[tuple[str, str]], model: str, complete: Complete
) -> list[dict[str, Any]]:
    rows = []
    for cid, prompt in prompts:
        row: dict[str, Any] = {"id": cid}
        for mode in ("constrained", "unconstrained"):
            raw = complete(request(prompt, model, constrained=mode == "constrained"))
            ok, why = valid(raw.raw_text)
            row[mode] = {
                "valid": ok,
                "error": why,
                "raw_text": raw.raw_text,
                "latency_ms": raw.latency_ms,
                "tokens_out": raw.tokens_out,
            }
        rows.append(row)
    return rows


def determinism(
    prompts: Sequence[tuple[str, str]], model: str, complete: Complete, first: dict[str, str]
) -> list[dict[str, Any]]:
    """Each prompt again, constrained; `first` holds the replies of the structured run."""
    rows = []
    for cid, prompt in prompts:
        again = complete(request(prompt, model, constrained=True)).raw_text
        rows.append(
            {
                "id": cid,
                "sha256_first": hashlib.sha256(first[cid].encode()).hexdigest(),
                "sha256_second": hashlib.sha256(again.encode()).hexdigest(),
                "identical": again == first[cid],
            }
        )
    return rows


def summary(report: dict[str, Any]) -> dict[str, Any]:
    st = report["structured"]
    return {
        "parity_equal": sum(r["equal"] for r in report["parity"]),
        "constrained_valid": sum(r["constrained"]["valid"] for r in st),
        "unconstrained_valid": sum(r["unconstrained"]["valid"] for r in st),
        "identical_reruns": sum(r["identical"] for r in report["determinism"]),
        "n": len(st),
    }


def passed(s: dict[str, Any]) -> bool:
    """EVK-03, EVK-04 and EVK-08 bars; unconstrained validity is reported, never gated."""
    n = s["n"]
    return (
        n == N
        and s["parity_equal"] == n
        and s["constrained_valid"] == n
        and (s["identical_reruns"] == n)
    )


def run(
    root: Path,
    model: str,
    quant: str,
    *,
    post: Post = http_post,
    complete: Complete,
    hf: HFTokens,
) -> dict[str, Any]:
    port = llama.PORTS[model]
    base = f"http://{llama.HOST}:{port}"
    cases = prompts(root, SMOKE_TASK)
    name = f"gemma-e2b-{model}-q4" if quant == "Q4_K_M" else f"gemma-e2b-{model}-{quant.lower()}"
    report: dict[str, Any] = {
        "llama_tag": llama.TAG,
        "gguf": llama.model_path(root, model, quant).name,
        "gguf_sha256": llama.sha256_file(llama.model_path(root, model, quant)),
        "server_args": llama.server_args(Path("llama-server"), Path("model.gguf"), port)[1:],
        "parity": parity(base, cases, post, hf),
    }
    report["structured"] = structured(cases, name, complete)
    first = {r["id"]: r["constrained"]["raw_text"] for r in report["structured"]}
    report["determinism"] = determinism(cases, name, complete, first)
    report["summary"] = summary(report)
    report["passed"] = passed(report["summary"])
    return report


def evalkit_complete(base_url: str) -> Complete:  # pragma: no cover - network
    """EvalKit's own adapter for an `endpoints` model, called synchronously."""
    from evalkit.adapters.openai import OpenAICompatibleAdapter

    def call(req: CompletionRequest) -> RawCompletion:
        # a fresh adapter per call: its async client belongs to the loop asyncio.run makes
        return asyncio.run(OpenAICompatibleAdapter(base_url=base_url).complete(req))

    return call


def hf_tokenizer() -> HFTokens:  # pragma: no cover - needs the nli group and the model cache
    from transformers import AutoTokenizer  # pyright: ignore[reportMissingImports]

    from ggh.export.gguf import BASE_ID, BASE_REVISION

    tok = AutoTokenizer.from_pretrained(BASE_ID, revision=BASE_REVISION)

    def ids(prompt: str) -> list[int]:
        out = tok.apply_chat_template(messages(prompt), tokenize=True, add_generation_prompt=True)
        return list(out["input_ids"] if isinstance(out, dict) or hasattr(out, "keys") else out)

    return ids


# ---- throughput (EVK-09, §11) ----

THROUGHPUT = Path("docs") / "measurements" / "serving_throughput.json"
IN_FLIGHT = (1, 4, 20)  # serial; EvalKit's actual default pool; everything at once

AsyncComplete = Callable[[CompletionRequest], Any]


async def _timed(prompts: Sequence[tuple[str, str]], model: str, complete: AsyncComplete,
                 in_flight: int) -> dict[str, float]:  # fmt: skip
    import time

    sem = asyncio.Semaphore(in_flight)

    async def one(prompt: str) -> RawCompletion:
        async with sem:
            return await complete(request(prompt, model, constrained=True))

    t0 = time.perf_counter()
    done = await asyncio.gather(*(one(p) for _, p in prompts))
    wall = time.perf_counter() - t0
    lat = [r.latency_ms / 1000 for r in done]
    return {
        "in_flight": in_flight,
        "wall_s": round(wall, 2),
        "seconds_per_answer": round(wall / len(prompts), 3),
        "latency_mean_s": round(sum(lat) / len(lat), 3),
        "latency_max_s": round(max(lat), 3),
    }


def throughput(
    root: Path, model: str, quant: str, new_client: Callable[[], AsyncComplete], *, host: str = ""
) -> dict[str, Any]:
    """Wall-clock seconds per answer through EvalKit's adapter at each in-flight level. With
    one server slot, throughput is the same at every level; latency grows with the queue."""
    prompts_ = prompts(root, SMOKE_TASK)
    name = f"gemma-e2b-{model}-q4" if quant == "Q4_K_M" else f"gemma-e2b-{model}-{quant.lower()}"
    # a fresh client per level: an async client belongs to the loop asyncio.run makes
    runs = [asyncio.run(_timed(prompts_, name, new_client(), k)) for k in IN_FLIGHT]
    return {
        "what": (
            f"EvalKit's OpenAICompatibleAdapter against `ggh serve --model {model}` ({quant}, "
            f"llama.cpp {llama.TAG}, one slot), the {len(prompts_)} dev smoke prompts, "
            "constrained JSON, warm prompt cache"
        ),
        "host": host,
        "runs": runs,
        "seconds_per_answer": runs[0]["seconds_per_answer"],
    }


def evalkit_async_client(base_url: str) -> Callable[[], AsyncComplete]:  # pragma: no cover
    from evalkit.adapters.openai import OpenAICompatibleAdapter

    def new() -> AsyncComplete:
        adapter: list[Any] = []

        async def call(req: CompletionRequest) -> RawCompletion:
            if not adapter:  # created inside the running loop
                adapter.append(OpenAICompatibleAdapter(base_url=base_url))
            return await adapter[0].complete(req)

        return call

    return new
