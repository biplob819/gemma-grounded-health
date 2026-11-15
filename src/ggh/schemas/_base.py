"""Shared base model and strict field types for every ggh schema (Tech Spec §3).

Models are frozen with `extra="forbid"` and finite floats (EvalKit L-8, L-9). Sequence fields
are tuples; mapping fields validate into `FrozenDict`, so an object's JSON cannot change after
it is built.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Annotated, Any, NoReturn, TypeVar

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StringConstraints

SCHEMA_VERSION = "1.0"


class FrozenDict(dict):
    """A dict that refuses mutation and can be hashed."""

    def _blocked(self, *_: Any, **__: Any) -> NoReturn:
        raise TypeError("ggh schema objects are immutable")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = _blocked  # type: ignore[assignment]
    __ior__ = _blocked  # type: ignore[assignment]

    def __hash__(self) -> int:  # type: ignore[override]
        return hash(frozenset(self.items()))

    def __iter__(self) -> Iterator[Any]:
        return super().__iter__()


def _freeze_mapping(value: Mapping[Any, Any]) -> FrozenDict:
    return FrozenDict(value)


V = TypeVar("V")

# Strict scalars: no bool-as-int, no "5"-as-5, no NaN or infinity.
NonNegInt = Annotated[int, Field(ge=0, strict=True)]
PosInt = Annotated[int, Field(ge=1, strict=True)]
# strict floats: an int is fine, a bool or a numeric string is not
NonNegFloat = Annotated[float, Field(ge=0.0, allow_inf_nan=False, strict=True)]
Prob = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False, strict=True)]
NonBlank = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
GitSha = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
FrozenMap = Annotated[dict[str, V], AfterValidator(_freeze_mapping)]


class Model(BaseModel):
    """Frozen, strict base: unknown fields are errors, objects are immutable, floats finite."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
