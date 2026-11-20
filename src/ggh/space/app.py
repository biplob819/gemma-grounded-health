"""Wiring for the demo: the local app (`ggh space --local`, E7, OWNER D-63: the demo that ships)
and the unused Space entry (`space/app.py` calls `hf_space`) share `make_space`; only the
backends differ. The local app serves v2's Q4_K_M GGUF through `ggh serve ft2`'s llama-server,
the exact file the benchmark scored."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ggh.space import backends, samples
from ggh.space.engine import Generate
from ggh.space.passkey import SECRET_ENV, Gate, from_env
from ggh.space.service import GroundingCheck, Space
from ggh.space.ui import SERVED_MODEL

HF_BASE_ENV = "GGH_SPACE_BASE_MODEL"  # Hub id of the base model (default below)
HF_FT_ENV = "GGH_SPACE_FT_MODEL"  # Hub id of the released fine-tune (R4.2)
DEFAULT_BASE = "google/gemma-4-E2B-it"
LOCAL_FT_MODEL = SERVED_MODEL  # v2's Q4_K_M GGUF, the file the benchmark scored
LOCAL_BASE_MODEL = "gemma-e2b-base-q4"


def make_space(
    env: Mapping[str, str],
    generators: Mapping[str, Generate],
    samples_path: Path,
    grounding: GroundingCheck | None = None,
) -> Space:
    return Space(Gate(from_env(env)), generators, samples.load(samples_path), grounding=grounding)


def sample_labels(space_samples: list[samples.Sample]) -> dict[str, str]:
    return {s.id: s.question for s in space_samples}


def local_generators(ft_url: str, base_url: str | None = None) -> dict[str, Generate]:
    """v2's GGUF server; base's too only when asked (live answers are then side by side)."""
    gens = {"ft": backends.endpoint(ft_url, model=LOCAL_FT_MODEL)}
    if base_url:
        gens = {"base": backends.endpoint(base_url, model=LOCAL_BASE_MODEL), **gens}
    return gens


def passkey_env(env: Mapping[str, str], env_file: Path | None = None) -> dict[str, str]:
    """The passcode hash from the environment, else from `env_file` (KEY=value lines, e.g. the
    owner's .env). Only `SECRET_ENV` is read from the file; nothing is printed."""
    if env.get(SECRET_ENV, "").strip() or env_file is None:
        return {SECRET_ENV: env.get(SECRET_ENV, "")}
    for line in env_file.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.strip().removeprefix("export ").partition("=")
        if sep and key.strip() == SECRET_ENV:
            return {SECRET_ENV: value.strip().strip("'\"")}
    return {SECRET_ENV: ""}


def launch(
    space: Space, samples_path: Path, *, port: int, summary: str = ""
) -> None:  # pragma: no cover - gradio
    from ggh.space import ui

    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # no calls home from the demo

    demo: Any = ui.build(space, sample_labels(samples.load(samples_path)), summary=summary)
    demo.launch(server_name="127.0.0.1", server_port=port, **ui.launch_kwargs())  # local only


def hf_space(path: Path) -> Any:  # pragma: no cover - runs on the Space
    from ggh.space import ui

    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # no calls home from the demo

    env = os.environ
    ft_id = env.get(HF_FT_ENV, "")
    if not ft_id:
        raise RuntimeError(f"set {HF_FT_ENV} to the released model's Hub id")
    gens = backends.transformers_pair(env.get(HF_BASE_ENV, DEFAULT_BASE), ft_id)
    space = make_space(env, gens, path)
    return ui.build(space, sample_labels(samples.load(path)), local=False)
