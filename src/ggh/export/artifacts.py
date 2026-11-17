"""EXP-05: stage manifests for train, export and parity, and an `ExportArtifact` record for
every export output (Tech Spec §2.1).

Training ran on Kaggle; its manifest is written here from the frozen record (D-10) with the
targets stage as parent, so `ggh lineage verify` can walk from the parity report back to the
dataset. Each artifact's `parent_key` is the key of the stage manifest that produced it.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal

from ggh.export import parity
from ggh.export.gguf import BASE_ID, BASE_REVISION
from ggh.export.merge import ADAPTER_DIR, MERGED_DIR
from ggh.lineage import StageManifest, read_manifest, sha256_file, stage_key, write_manifest
from ggh.schemas import ExportArtifact, TrainRun
from ggh.serve import llama
from ggh.train.devrun import DEV_EVAL, TRAIN_RUN
from ggh.train.select import ADAPTER_FILES

CODE_VERSION = "1"
TARGETS_MANIFEST = Path("artifacts") / "targets" / "manifest.json"
DATA_MANIFEST = Path("artifacts") / "data" / "manifest.json"
STAGE_DIRS = {
    "train": Path("artifacts") / "train",
    "export_base": Path("artifacts") / "export" / "base",
    "export_ft": Path("artifacts") / "export" / "ft",
    "parity": Path("artifacts") / "parity",
}
RECORDS = Path("docs") / "measurements" / "export_artifacts.json"
LINEAGE_COPY = Path("docs") / "measurements" / "lineage"  # committed copies (artifacts/ is not)
MERGED_FILES = ("model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json",
                "chat_template.jinja")  # fmt: skip
BASE_WEIGHTS = "model.safetensors"
HOST_KAGGLE = "kaggle"

Clock = Callable[[], dt.datetime]


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def hashes(root: Path, rels: list[Path]) -> dict[str, str]:
    out = {}
    for rel in rels:
        if not (root / rel).is_file():
            raise FileNotFoundError(f"{rel} is missing")
        out[rel.as_posix()] = sha256_file(root / rel)
    return out


def _manifest(stage: str, inputs: Mapping[str, str], outputs: Mapping[str, str],
              params: Mapping[str, Any], parents: tuple[str, ...], started: dt.datetime,
              finished: dt.datetime, host: str) -> StageManifest:  # fmt: skip
    return StageManifest(
        stage=stage,
        code_version=CODE_VERSION,
        key=stage_key(stage, CODE_VERSION, inputs, params, 7),
        inputs=dict(inputs),
        outputs=dict(outputs),
        params=dict(params),
        seed=7,
        parents=parents,
        started_at=started,
        finished_at=finished,
        host=host,
    )


def train_manifest(root: Path, now: Clock = _now) -> StageManifest:
    """The Kaggle run, recorded locally from `TrainRun` (D-10): its inputs are the targets
    stage's outputs (replayed byte-identically on Kaggle) and the recipe."""
    run = TrainRun.model_validate_json((root / TRAIN_RUN).read_text(encoding="utf-8"))
    targets = read_manifest(root / TARGETS_MANIFEST)
    rels = [Path("artifacts/targets/train.jsonl"), Path("artifacts/targets/val.jsonl"),
            Path("configs/recipe.yaml")]  # fmt: skip
    inputs = hashes(root, rels)
    for rel in rels[:2]:
        if targets.outputs.get(rel.as_posix()) != inputs[rel.as_posix()]:
            raise ValueError(f"{rel} differs from what the targets stage recorded")
    outputs = hashes(root, [ADAPTER_DIR / f for f in ADAPTER_FILES] + [TRAIN_RUN])
    params = {"kaggle_kernel": f"ggh-t22-train v{run.kaggle_kernel_version}",
              "repo_commit": run.repo_commit, "selected_step": run.selected_step,
              "base": f"{BASE_ID}@{BASE_REVISION}"}  # fmt: skip
    t = now()
    return _manifest("train", inputs, outputs, params, (targets.key,), t, t, HOST_KAGGLE)


def _gguf_rels(model: str) -> list[Path]:
    return [llama.GGUF_DIR / f"{model}-f16.gguf",
            *(llama.GGUF_DIR / f"{model}-{q}.gguf" for q in llama.QUANTS)]  # fmt: skip


def _gguf_params() -> dict[str, Any]:
    return {"llama_cpp": llama.TAG, "release_sha256": llama.BIN_SHA256,
            "steps": ["convert_hf_to_gguf.py --outtype f16",
                      *(f"llama-quantize {q}" for q in llama.QUANTS)]}  # fmt: skip


def base_uri() -> str:
    return f"hf://{BASE_ID}@{BASE_REVISION}/{BASE_WEIGHTS}"


def export_base_manifest(root: Path, base_dir: Path, now: Clock = _now) -> StageManifest:
    inputs = {base_uri(): sha256_file(base_dir / BASE_WEIGHTS)}
    t = now()
    return _manifest("export", inputs, hashes(root, _gguf_rels("base")), _gguf_params(), (),
                     t, t, "local")  # fmt: skip


def export_ft_manifest(root: Path, base_dir: Path, train: StageManifest, now: Clock = _now
                       ) -> StageManifest:  # fmt: skip
    inputs = {base_uri(): sha256_file(base_dir / BASE_WEIGHTS),
              **hashes(root, [ADAPTER_DIR / f for f in ADAPTER_FILES])}  # fmt: skip
    outputs = hashes(root, [MERGED_DIR / f for f in MERGED_FILES] + _gguf_rels("ft"))
    params = {**_gguf_params(), "merge": "bf16 on CPU, peft merge_and_unload"}
    t = now()
    return _manifest("export", inputs, outputs, params, (train.key,), t, t, "local")


def parity_manifest(root: Path, export_ft: StageManifest, now: Clock = _now) -> StageManifest:
    data = read_manifest(root / DATA_MANIFEST)
    inputs = hashes(root, [MERGED_DIR / "model.safetensors", llama.GGUF_DIR / "ft-Q4_K_M.gguf",
                           DEV_EVAL])  # fmt: skip
    outputs = hashes(root, [parity.REPORT, parity.GENERATIONS, parity.SAMPLE])
    params = {"agreement_bar": parity.AGREEMENT_BAR, "schema_gap": parity.SCHEMA_GAP}
    t = now()
    return _manifest("parity", inputs, outputs, params, (export_ft.key, data.key), t, t, "local")


def records(root: Path, train: StageManifest, base: StageManifest, ft: StageManifest
            ) -> list[ExportArtifact]:  # fmt: skip
    out = [ExportArtifact(kind="adapter", model="ft", path=ADAPTER_DIR.as_posix(),
                          sha256=train.outputs[(ADAPTER_DIR / "adapter_model.safetensors")
                                               .as_posix()],
                          parent_key=train.key),
           ExportArtifact(kind="merged_bf16", model="ft",
                          path=(MERGED_DIR / "model.safetensors").as_posix(),
                          sha256=ft.outputs[(MERGED_DIR / "model.safetensors").as_posix()],
                          parent_key=ft.key)]  # fmt: skip
    pairs: tuple[tuple[Literal["base", "ft"], StageManifest], ...] = (("base", base), ("ft", ft))
    for model, m in pairs:
        for rel in _gguf_rels(model):
            quant = rel.stem.split("-", 1)[1].upper()
            out.append(ExportArtifact(kind="gguf", model=model, quant=quant, path=rel.as_posix(),
                                      sha256=m.outputs[rel.as_posix()], llama_cpp_commit=llama.TAG,
                                      parent_key=m.key))  # fmt: skip
    return out


def _write(root: Path, name: str, m: StageManifest) -> None:
    path = write_manifest(root / STAGE_DIRS[name], m)
    copy = root / LINEAGE_COPY / f"{name}.json"
    copy.parent.mkdir(parents=True, exist_ok=True)
    copy.write_bytes(path.read_bytes())


def write_all(root: Path, base_dir: Path, now: Clock = _now) -> list[ExportArtifact]:
    train = train_manifest(root, now)
    _write(root, "train", train)
    base = export_base_manifest(root, base_dir, now)
    _write(root, "export_base", base)
    ft = export_ft_manifest(root, base_dir, train, now)
    _write(root, "export_ft", ft)
    _write(root, "parity", parity_manifest(root, ft, now))
    for name, rel in (("data", DATA_MANIFEST), ("targets", TARGETS_MANIFEST)):  # the chain's start
        m = read_manifest(root / rel).model_copy(update={"host": "local"})  # no machine name
        (root / LINEAGE_COPY / f"{name}.json").write_text(m.model_dump_json(indent=2) + "\n",
                                                          encoding="utf-8")  # fmt: skip
    recs = records(root, train, base, ft)
    text = json.dumps([r.model_dump(mode="json") for r in recs], indent=2) + "\n"
    (root / RECORDS).write_text(text, encoding="utf-8")
    return recs
