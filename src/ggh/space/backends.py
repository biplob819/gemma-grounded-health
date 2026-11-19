"""Generation backends for the demo, each a `Generate` (prompt -> raw reply).

- `endpoint`: an OpenAI-compatible chat endpoint (`ggh serve`'s llama-server), greedy, with
  EvalKit's answer schema as the response format. This is the `ggh space --local` fallback (E7).
- `transformers_pair`: both merged bf16 models in Transformers on the Space, moved to the GPU
  inside one `@spaces.GPU(duration=60)` call (Tech Spec §9). It needs the `space` dependency
  group and is never imported by tests.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable
from typing import Any

from evalkit.pipeline.answer import answer_schema

from ggh.space.engine import LABELS, MAX_NEW_TOKENS, Generate

Post = Callable[[str, bytes], bytes]


def _post(url: str, body: bytes) -> bytes:  # pragma: no cover - network (localhost)
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def chat_body(prompt: str, model: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": MAX_NEW_TOKENS,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "answer", "schema": dict(answer_schema(LABELS))},
        },
    }


def endpoint(base_url: str, model: str = "ggh", post: Post = _post) -> Generate:
    url = base_url.rstrip("/") + "/v1/chat/completions"

    def generate(prompt: str) -> str:
        reply = json.loads(post(url, json.dumps(chat_body(prompt, model)).encode("utf-8")))
        return str(reply["choices"][0]["message"]["content"])

    return generate


def transformers_pair(base_id: str, ft_id: str) -> dict[str, Generate]:  # pragma: no cover
    """Both models load at startup in bf16 on CPU and move to the GPU per request."""
    import spaces  # pyright: ignore[reportMissingImports]
    import torch  # pyright: ignore[reportMissingImports]
    from transformers import (  # pyright: ignore[reportMissingImports]
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    tok = AutoTokenizer.from_pretrained(ft_id)
    models: dict[str, Any] = {  # transformers' stubs mistype .to and .generate
        "base": AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16),
        "ft": AutoModelForCausalLM.from_pretrained(ft_id, torch_dtype=torch.bfloat16),
    }

    def one(name: str, prompt: str) -> str:
        model = models[name].to("cuda")
        ids = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")
        out = model.generate(ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        return tok.decode(out[0][ids.shape[-1] :], skip_special_tokens=True)

    @spaces.GPU(duration=60)
    def both(prompt: str) -> dict[str, str]:  # one GPU allocation answers both models
        return {name: one(name, prompt) for name in models}

    pending: dict[str, dict[str, str]] = {}  # this request's replies only, popped as read

    def generator(name: str) -> Generate:
        def generate(prompt: str) -> str:
            if prompt not in pending:
                pending.clear()
                pending[prompt] = both(prompt)
            replies = pending[prompt]
            text = replies.pop(name)
            if not replies:
                del pending[prompt]
            return text

        return generate

    return {name: generator(name) for name in models}
