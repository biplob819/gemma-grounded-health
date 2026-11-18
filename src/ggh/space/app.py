"""Wiring for the demo: the Space (`space/app.py` calls `hf_space`) and the local fallback
(`ggh space --local`, E7) share `make_space`; only the backends differ."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ggh.space import backends, samples
from ggh.space.engine import Generate
from ggh.space.passkey import Gate, from_env
from ggh.space.service import GroundingCheck, Space

HF_BASE_ENV = "GGH_SPACE_BASE_MODEL"  # Hub id of the base model (default below)
HF_FT_ENV = "GGH_SPACE_FT_MODEL"  # Hub id of the released fine-tune (R4.2)
DEFAULT_BASE = "google/gemma-4-E2B-it"


def make_space(
    env: Mapping[str, str],
    generators: Mapping[str, Generate],
    samples_path: Path,
    grounding: GroundingCheck | None = None,
) -> Space:
    return Space(Gate(from_env(env)), generators, samples.load(samples_path), grounding=grounding)


def sample_labels(space_samples: list[samples.Sample]) -> dict[str, str]:
    return {s.id: s.question for s in space_samples}


def local_generators(base_url: str, ft_url: str) -> dict[str, Generate]:
    return {"base": backends.endpoint(base_url), "ft": backends.endpoint(ft_url)}


def launch(space: Space, samples_path: Path, *, port: int) -> None:  # pragma: no cover - gradio
    from ggh.space import ui

    demo: Any = ui.build(space, sample_labels(samples.load(samples_path)))
    demo.launch(server_name="127.0.0.1", server_port=port)  # local only: never shared


def hf_space(path: Path) -> Any:  # pragma: no cover - runs on the Space
    from ggh.space import ui

    env = os.environ
    ft_id = env.get(HF_FT_ENV, "")
    if not ft_id:
        raise RuntimeError(f"set {HF_FT_ENV} to the released model's Hub id")
    gens = backends.transformers_pair(env.get(HF_BASE_ENV, DEFAULT_BASE), ft_id)
    space = make_space(env, gens, path)
    return ui.build(space, sample_labels(samples.load(path)))
