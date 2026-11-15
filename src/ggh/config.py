"""Config loaders. Errors name the field and the file, never a bare pydantic dump."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from ggh.schemas import RecipeConfig


class ConfigError(ValueError):
    """A config file that is missing, not YAML, or fails validation."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that refuses duplicate keys instead of silently keeping the last one."""


def _unique_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode) -> dict:
    seen = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node)
        if key in seen:
            raise yaml.constructor.ConstructorError(
                None, None, f"duplicate key {key!r}", key_node.start_mark
            )
        seen.add(key)
    return loader.construct_mapping(node)


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _unique_mapping,  # type: ignore[arg-type]
)


def load_recipe(path: Path) -> RecipeConfig:
    try:
        raw = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except OSError as exc:
        raise ConfigError(f"cannot read {path}: {exc.strerror or exc}") from exc
    except (yaml.YAMLError, TypeError, UnicodeDecodeError, RecursionError) as exc:
        # TypeError: an unhashable key such as `? [a]`; RecursionError: absurd nesting
        raise ConfigError(f"{path} is not valid YAML: {type(exc).__name__}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must hold a mapping of recipe settings")
    try:
        return RecipeConfig.model_validate(raw)
    except ValidationError as exc:
        lines = []
        for err in exc.errors():
            field = ".".join(str(p) for p in err["loc"]) or "(recipe)"
            lines.append(f"{field}: {err['msg']}")
        raise ConfigError(f"{path}: " + "; ".join(lines)) from exc
