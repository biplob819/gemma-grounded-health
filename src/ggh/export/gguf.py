"""EVK-01 (shared with X3.1): HF safetensors → F16 GGUF → Q4_K_M and Q8_0 with the pinned
llama.cpp tools, plus a manifest of hashes (Tech Spec §6).

Base and fine-tuned go through the same function, so they use the same commit and the same
steps. The runner is injectable: tests check the exact commands without running them.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ggh.serve import llama

BASE_ID = "google/gemma-4-E2B-it"
BASE_REVISION = "4adf15748de9e1a0097b6274cee9c15836c41c37"  # Apache-2.0, not gated
QUANTS = llama.QUANTS
MANIFEST_COPY = Path("docs") / "measurements"

Runner = Callable[[Sequence[str], dict[str, str]], None]


def _run(cmd: Sequence[str], env: dict[str, str]) -> None:  # pragma: no cover - heavy
    subprocess.run(list(cmd), check=True, env={**os.environ, **env})


def commands(
    hf_dir: Path, name: str, out_dir: Path, src: Path, quantize: Path, python: str
) -> list[tuple[list[str], dict[str, str]]]:
    """The convert step and one quantize step per quant, in order."""
    f16 = out_dir / f"{name}-f16.gguf"
    steps: list[tuple[list[str], dict[str, str]]] = [
        (
            [
                python,
                str(src / "convert_hf_to_gguf.py"),
                str(hf_dir),
                "--outtype",
                "f16",
                "--outfile",
                str(f16),
            ],
            {"PYTHONPATH": str(src / "gguf-py")},
        )
    ]
    for q in QUANTS:
        steps.append(([str(quantize), str(f16), str(out_dir / f"{name}-{q}.gguf"), q], {}))
    return steps


def export(
    root: Path,
    hf_dir: Path,
    name: str,
    *,
    source_id: str,
    source_revision: str,
    run: Runner = _run,
    cache: Path | None = None,
    python: str = sys.executable,
    now: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.UTC),
) -> dict[str, Any]:
    """Build `<name>-f16.gguf` and each quant under artifacts/gguf; write the manifest there
    and a copy under docs/measurements. Every tool comes from the verified release."""
    if not (hf_dir / "config.json").is_file():
        raise llama.LlamaError(f"{hf_dir} is not a Hugging Face model folder (no config.json)")
    out_dir = root / llama.GGUF_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    src = llama.source(cache)
    quantize = llama.tool("llama-quantize", cache)
    for cmd, env in commands(hf_dir, name, out_dir, src, quantize, python):
        run(cmd, env)
    outputs = [f"{name}-f16.gguf", *(f"{name}-{q}.gguf" for q in QUANTS)]
    files = {}
    for f in outputs:
        path = out_dir / f
        if not path.is_file():
            raise llama.LlamaError(f"export step did not write {path}")
        files[f] = {"sha256": llama.sha256_file(path), "bytes": path.stat().st_size}
    manifest = {
        "name": name,
        "source": {"id": source_id, "revision": source_revision},
        "llama_cpp": {
            "tag": llama.TAG,
            "release_sha256": llama.BIN_SHA256,
            "source_sha256": llama.SRC_SHA256,
        },
        "steps": ["convert_hf_to_gguf.py --outtype f16", *(f"llama-quantize {q}" for q in QUANTS)],
        "files": files,
        "built_at": now().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (out_dir / f"{name}.manifest.json").write_text(text, encoding="utf-8")
    copy = root / MANIFEST_COPY / f"gguf_{name}_manifest.json"
    copy.parent.mkdir(parents=True, exist_ok=True)
    copy.write_text(text, encoding="utf-8")
    return manifest


def hf_snapshot(model_id: str, revision: str) -> Path:  # pragma: no cover - needs the cache
    from huggingface_hub import snapshot_download  # pyright: ignore[reportMissingImports]

    return Path(snapshot_download(model_id, revision=revision))
