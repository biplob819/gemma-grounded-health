"""RUN-04: an OpenAI-compatible chat endpoint over the merged bf16 fine-tune in Transformers,
so EvalKit can run it like the GGUFs (Tech Spec §7.2 parity run).

`POST /v1/chat/completions` renders the messages with the model's chat template (thinking off,
as in training and serving), decodes greedily, and honours a `json_schema` response format
with xgrammar, as llama-server does. One request at a time. `GET /health` and
`GET /v1/models` answer as llama-server's do. Bound to 127.0.0.1 only.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

HOST = "127.0.0.1"
PORT = 8082
MODEL_ID = "gemma-e2b-grounded-bf16"
# the EvalKit model id each version is served under (V2.2: the judged parity config names v2's)
MODEL_IDS = {"ft": MODEL_ID, "ft2": "gemma-e2b-grounded-v2-bf16"}
DEFAULT_MAX = 2048


@dataclass(frozen=True)
class Reply:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str  # "stop" or "length"


Generate = Callable[[list[dict[str, str]], int, "dict[str, Any] | None"], Reply]


class BadRequest(ValueError):
    pass


def parse_request(body: dict[str, Any]) -> tuple[list[dict[str, str]], int, dict[str, Any] | None]:
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not msgs:
        raise BadRequest("messages must be a non-empty list")
    for m in msgs:
        if not isinstance(m, dict) or not isinstance(m.get("content"), str) or "role" not in m:
            raise BadRequest("each message needs a role and string content")
    max_tokens = body.get("max_completion_tokens", body.get("max_tokens", DEFAULT_MAX))
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
        raise BadRequest("max tokens must be a positive integer")
    schema = None
    rf = body.get("response_format")
    if rf is not None:
        if not isinstance(rf, dict) or not isinstance(rf.get("json_schema", {}), dict):
            raise BadRequest("response_format must be an object")
        if rf.get("type") != "json_schema" or "schema" not in rf.get("json_schema", {}):
            raise BadRequest("only response_format type json_schema is supported")
        schema = rf["json_schema"]["schema"]
    t = body.get("temperature", 0)
    if t not in (0, 0.0, None):
        raise BadRequest("this server is greedy only: temperature must be 0 or absent")
    return [{"role": m["role"], "content": m["content"]} for m in msgs], max_tokens, schema


def completion(reply: Reply, model: str = MODEL_ID) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": reply.finish_reason,
                "message": {"role": "assistant", "content": reply.text},
            }
        ],
        "usage": {
            "prompt_tokens": reply.prompt_tokens,
            "completion_tokens": reply.completion_tokens,
            "total_tokens": reply.prompt_tokens + reply.completion_tokens,
        },
    }


def make_handler(generate: Generate, model: str = MODEL_ID) -> type[BaseHTTPRequestHandler]:
    lock = threading.Lock()  # one generation at a time, like llama-server's single slot

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def _send(self, code: int, obj: dict[str, Any]) -> None:
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send(200, {"status": "ok"})
            elif self.path == "/v1/models":
                self._send(200, {"object": "list", "data": [{"id": model, "object": "model"}]})
            else:
                self._send(404, {"error": {"message": "not found"}})

        def do_POST(self) -> None:
            if self.path != "/v1/chat/completions":
                self._send(404, {"error": {"message": "not found"}})
                return
            try:
                n = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(n) or b"{}")
                msgs, max_tokens, schema = parse_request(body)
            except (BadRequest, ValueError) as exc:
                self._send(400, {"error": {"message": str(exc), "type": "invalid_request_error"}})
                return
            with lock:
                reply = generate(msgs, max_tokens, schema)
            self._send(200, completion(reply, model))

    return Handler


def serve(generate: Generate, port: int = PORT, model: str = MODEL_ID) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((HOST, port), make_handler(generate, model))


def grammar_source(schema: dict[str, Any]) -> str:
    """The schema text the grammar is compiled from, in the request's own key order: xgrammar
    emits object properties in that order, so sorting the keys would put `answer` before
    `label` (an order the model never saw in training)."""
    return json.dumps(schema)


def transformers_generate(merged: Any) -> Generate:  # pragma: no cover - 10 GB model on MPS
    """Greedy generation with the merged bf16 model; xgrammar when a schema is given."""
    import torch  # pyright: ignore[reportMissingImports]
    import xgrammar as xgr  # pyright: ignore[reportMissingImports]

    from ggh.export.parity import load_bf16
    from ggh.train.devrun import vocab_size
    from ggh.train.generate import stop_ids

    model, tok = load_bf16(merged)
    info = xgr.TokenizerInfo.from_huggingface(tok, vocab_size=vocab_size(model))
    compiler = xgr.GrammarCompiler(info)
    compiled: dict[str, Any] = {}

    def gen(msgs: list[dict[str, str]], max_tokens: int, schema: dict[str, Any] | None) -> Reply:
        out = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
        ids = list(out["input_ids"] if hasattr(out, "keys") else out)
        x = torch.tensor([ids], device=model.device)
        kwargs: dict[str, Any] = {"max_new_tokens": max_tokens, "do_sample": False,
                                  "eos_token_id": stop_ids(tok),
                                  "pad_token_id": tok.pad_token_id or 0}  # fmt: skip
        if schema is not None:
            key = grammar_source(schema)  # key order matters (L-26), so it is part of the key
            if key not in compiled:  # any whitespace, as llama-server; property order kept
                compiled[key] = compiler.compile_json_schema(grammar_source(schema))
            kwargs["logits_processor"] = [xgr.contrib.hf.LogitsProcessor(compiled[key])]
        with torch.no_grad():
            seq = model.generate(x, attention_mask=torch.ones_like(x), **kwargs)[0, x.shape[1] :]
        stops = set(stop_ids(tok))
        new = [int(t) for t in seq]
        ended = bool(new) and new[-1] in stops
        body = new[:-1] if ended else new
        text = tok.decode(body, skip_special_tokens=True).strip()
        return Reply(text, len(ids), len(new), "stop" if ended else "length")

    return gen
