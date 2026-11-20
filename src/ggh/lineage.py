"""Stage contract and lineage manifests (Tech Spec §2.1, SCH-04).

Every pipeline stage writes `manifest.json` next to its outputs. The manifest's key is a
sha256 over the stage name, its code version, the hashes of its inputs, its params and its
seed, so a rerun with the same key can be skipped and `ggh lineage verify` can walk any
artifact back through its parents.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

from pydantic import AfterValidator, AwareDatetime, model_validator

from ggh.schemas._base import FrozenDict, FrozenMap, Model, NonBlank, NonNegInt, Sha256

MANIFEST = "manifest.json"


class LineageError(ValueError):
    """A manifest that cannot be written, read or trusted."""


def deep_freeze(value: Any) -> Any:
    """Lists to tuples and dicts to FrozenDicts, recursively; sets and non-finite floats refused."""
    if isinstance(value, Mapping):
        return FrozenDict({str(k): deep_freeze(v) for k, v in value.items()})
    if isinstance(value, list | tuple):
        return tuple(deep_freeze(v) for v in value)
    if isinstance(value, set | frozenset):
        raise ValueError("sets are not JSON values; use a list")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("NaN and infinity are not allowed")
    return value


JsonValue = Annotated[Any, AfterValidator(deep_freeze)]

# One spelling per file, so inputs and outputs compare as strings: a relative POSIX path with
# no "", ".", ".." segments, no backslash and no leading or trailing slash. Inputs from outside
# the repo (the dataset cache, the Hub) are named by a URI instead: hf://…, cache://….
_URI = re.compile(r"(?:hf|cache)://\S+")  # only sources outside the repo; never file:// etc.


def canonical_rel(path: str) -> str:
    """Return path if it is a canonical relative POSIX path, else raise ValueError."""
    segments = path.split("/")
    if (
        not path
        or "\\" in path
        or "\x00" in path
        or any(seg in ("", ".", "..") for seg in segments)
    ):
        raise ValueError(f"{path!r} is not a canonical relative path (like artifacts/data/x.jsonl)")
    return path


def _paths(value: Mapping[str, str], *, uris: bool) -> FrozenDict:
    for key in value:
        if not (uris and _URI.fullmatch(key)):
            canonical_rel(key)
    return FrozenDict(value)


SKIP_DIRS = {".git", "node_modules", "__pycache__"}  # never lineage; also any .venv* folder


def _canonical(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def stage_key(
    stage: str, code_version: str, inputs: Mapping[str, str], params: Mapping[str, Any], seed: int
) -> str:
    """sha256 of the canonical JSON of (stage, code_version, inputs, params, seed).

    Inputs enter as sorted (path, hash) pairs: swapping the contents of two inputs (train and
    val, say) changes the key (D-32). Dict key order never matters."""
    payload = {
        "stage": stage,
        "code_version": code_version,
        "inputs": sorted(inputs.items()),
        "params": params,
        "seed": seed,
    }
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


# Committed outputs a later build overwrote, whose original bytes are kept under another name.
# v2.x rewrote the targets drop report; v1's copy (test_v1_lineage_copy.py) is the one v1's
# lineage records.
KEPT_COPIES: dict[str, str] = {
    "docs/measurements/targets_drop_report.json": "docs/measurements/targets_drop_report_v1.json",
}


def committed_output(root: Path, rel: str, digest: str) -> Path | None:
    """The file holding committed output `rel` with `digest`: the path itself, else its kept
    copy when a later build overwrote it; None when neither matches."""
    for cand in (rel, KEPT_COPIES.get(rel)):
        if cand and (root / cand).is_file() and sha256_file(root / cand) == digest:
            return root / cand
    return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:  # binary: no encoding applies
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class StageManifest(Model):
    stage: NonBlank
    code_version: NonBlank
    key: Sha256
    inputs: Annotated[dict[str, Sha256], AfterValidator(lambda v: _paths(v, uris=True))]
    outputs: Annotated[dict[str, Sha256], AfterValidator(lambda v: _paths(v, uris=False))]
    params: FrozenMap[JsonValue]
    seed: NonNegInt
    parents: tuple[Sha256, ...]
    started_at: AwareDatetime
    finished_at: AwareDatetime
    host: NonBlank

    @model_validator(mode="after")
    def _key_and_times(self) -> StageManifest:
        expected = stage_key(self.stage, self.code_version, self.inputs, self.params, self.seed)
        if self.key != expected:
            raise ValueError("key does not match stage, code version, inputs, params and seed")
        if self.finished_at < self.started_at:
            raise ValueError("finished_at is before started_at")
        if self.key in self.parents:
            raise ValueError("a stage cannot be its own parent")
        return self


def write_manifest(directory: Path, manifest: StageManifest) -> Path:
    """Write directory/manifest.json atomically: a temp file in the same folder, then rename."""
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / MANIFEST
    tmp = directory / f".{MANIFEST}.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            fh.write(manifest.model_dump_json(indent=2) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()
    return target


def read_manifest(path: Path) -> StageManifest:
    try:
        return StageManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise LineageError(f"cannot read manifest {path}: {exc}") from exc


# Downloaded Kaggle kernel outputs (`kaggle kernels output` into artifacts/kaggle/<kernel>/):
# a mirror of the remote run's working tree, never a local stage. Remote stages enter local
# lineage as manifests reconstructed under artifacts/ (X3.1 known issue), not from here.
MIRROR_DIRS = ("artifacts/kaggle",)


def _is_other_checkout(directory: Path) -> bool:
    """A git worktree or clone nested under the root (`.claude/worktrees/<agent>/`, the repo
    clone inside a Kaggle mirror): its own `.git` file or folder marks it. Its manifests are a
    copy of another checkout's lineage, so they are neither this root's parents nor duplicates."""
    return (directory / ".git").exists()


def _skipped(root: Path, directory: Path) -> bool:
    name = directory.name
    if name in SKIP_DIRS or name.startswith(".venv"):
        return True
    rel = directory.relative_to(root).as_posix()
    if any(rel == m or rel.startswith(f"{m}/") for m in MIRROR_DIRS):
        return True
    return _is_other_checkout(directory)


def manifest_paths(root: Path) -> list[Path]:
    """Every `manifest.json` under root, pruning .git, .venv*, node_modules, __pycache__, other
    git checkouts nested inside root, and Kaggle output mirrors. Any other folder, dot-folders
    included, still counts, so a stray copy of a manifest is still reported as a duplicate."""
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        dirnames[:] = sorted(d for d in dirnames if not _skipped(root, here / d))
        if MANIFEST in filenames:
            found.append(here / MANIFEST)
    return sorted(found)


def _index(root: Path) -> dict[str, list[Path]]:
    """Every readable manifest under root (see `manifest_paths` for what is pruned), by key.
    Unreadable or unrelated `manifest.json` files are simply not indexed: they only matter if
    the chain needs them."""
    index: dict[str, list[Path]] = {}
    for path in manifest_paths(root):
        try:
            m = read_manifest(path)
        except LineageError:
            continue
        index.setdefault(m.key, []).append(path)
    return index


def _inside(root: Path, rel: str) -> Path | None:
    """root/rel when no component of it, file or directory, is a symlink (rel is already a
    canonical relative path, so it cannot climb out of root by itself)."""
    current = root
    for part in Path(rel).parts:
        current = current / part
        if current.is_symlink():
            return None
    return current


def verify(manifest_path: Path, root: Path) -> list[str]:
    """Problems in the chain from manifest_path back to its roots; an empty list means intact.

    Only manifests on the chain are judged. For each one: every output is a relative path
    inside root, not a symlink, present, with its recorded hash; every parent key belongs to
    exactly one manifest under root; every input that a parent declares as an output carries
    that parent's hash (so a parent edited and re-hashed after the child ran is caught); and
    the chain has no cycle."""
    index = _index(root)
    try:
        start = read_manifest(manifest_path)
    except LineageError as exc:
        return [str(exc)]
    problems: list[str] = []
    seen: set[str] = set()
    stack: list[tuple[StageManifest, tuple[str, ...]]] = [(start, (start.key,))]
    while stack:
        m, path = stack.pop()
        if m.key in seen:
            continue
        seen.add(m.key)
        produced: dict[str, str] = {}
        for rel, digest in m.outputs.items():
            f = _inside(root, rel)
            if f is None:
                problems.append(f"{m.stage}: output {rel} is outside {root} or through a symlink")
                continue
            try:
                ok = f.is_file() and sha256_file(f) == digest
                missing = not f.is_file()
            except OSError as exc:
                problems.append(f"{m.stage}: cannot read output {rel}: {exc.strerror or exc}")
                continue
            if missing:
                problems.append(f"{m.stage}: output {rel} is missing")
            elif not ok:
                problems.append(f"{m.stage}: output {rel} no longer matches its recorded hash")
        for parent in m.parents:
            if parent in path:
                problems.append(f"{m.stage}: lineage cycle through {parent[:12]}")
                continue
            found = index.get(parent, [])
            if not found:
                problems.append(f"{m.stage}: parent {parent[:12]} has no manifest under {root}")
                continue
            if len(found) > 1:
                where = ", ".join(str(p) for p in found)
                problems.append(f"{m.stage}: parent {parent[:12]} has several manifests: {where}")
                continue
            pm = read_manifest(found[0])
            for rel, digest in m.inputs.items():
                if rel in pm.outputs and pm.outputs[rel] != digest:
                    problems.append(
                        f"{m.stage}: input {rel} does not match what {pm.stage} recorded"
                    )
            produced.update(pm.outputs)
            stack.append((pm, (*path, parent)))
        # inputs in the repo that no parent produced are sources: they must still be the bytes
        # this stage read (URIs name things outside the repo, which the manifest cannot check)
        for rel, digest in m.inputs.items():
            if "://" in rel or rel in produced:
                continue
            f = _inside(root, rel)
            if f is None or not f.is_file():
                problems.append(f"{m.stage}: source input {rel} is missing or a symlink")
                continue
            try:
                changed = sha256_file(f) != digest
            except OSError as exc:
                problems.append(f"{m.stage}: cannot read source input {rel}: {exc.strerror or exc}")
                continue
            if changed:
                problems.append(f"{m.stage}: source input {rel} changed since {m.stage} ran")
    return problems
