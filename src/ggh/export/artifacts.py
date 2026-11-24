"""EXP-05: stage manifests for train, export and parity, and an `ExportArtifact` record for
every export output (Tech Spec §2.1).

Training ran on Kaggle; its manifest is written here from the frozen record (D-10) with the
targets stage as parent, so `ggh lineage verify` can walk from the parity report back to the
dataset. Each artifact's `parent_key` is the key of the stage manifest that produced it.

v2 (`ft2`, V2.1) gets the same records from its own files (`write_v2`, `ggh export-records
--model ft2`): its train, export and parity manifests, committed as `train_v2`, `export_ft2`
and `parity_v2` beside v1's, with the v2 targets stage committed as `targets_v2`. Every hash is
of the file on disk; the GGUF and adapter hashes must also equal the ones V2.1 recorded
(docs/measurements/v2/gguf_ft2_manifest.json, merge_record.json). The base GGUFs are v1's,
unchanged, so v2's base records come from v1's export_base manifest once their files re-hash.

v3b (`ft3b`, V3.8, `write_v3b`, `ggh export-records --model ft3b`) is a GRPO run from v2's frozen
adapter: its train stage (`train_v3b`) reads v2's adapter (train_v2's outputs), the v2 targets'
training prompts, its GRPO recipe and the reward and checker its plan pins, with train_v2 and
the targets stage as parents; `export_ft3b` and `parity_v3b` follow v2's, the parity stage also
recording the exploratory Q8_0 check (OWNER D-73). Every record names the frozen v3b adapter.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal

import yaml

from ggh.export import parity
from ggh.export.gguf import BASE_ID, BASE_REVISION
from ggh.export.merge import ADAPTER_DIR, MERGED_DIR, SELECTION
from ggh.lineage import StageManifest, read_manifest, sha256_file, stage_key, write_manifest
from ggh.schemas import ExportArtifact, RLRun, TrainRun
from ggh.serve import llama
from ggh.train.devrun import DEV_EVAL, TRAIN_RUN
from ggh.train.run import config_hash
from ggh.train.select import ADAPTER_FILES, adapter_hash
from ggh.versions import version

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
# v2 (`ft2`): its own stage folders, committed copies and records, never overwriting v1's
V2_STAGE_DIRS = {
    "train_v2": Path("artifacts") / "train_v2",
    "export_ft2": Path("artifacts") / "export" / "ft2",
    "parity_v2": Path("artifacts") / "parity_v2",
}
RECORDS_V2 = Path("docs") / "measurements" / "export_artifacts_v2.json"
GGUF_RECORD_V2 = version("ft2").measurements / "gguf_ft2_manifest.json"
MERGE_RECORD_V2 = version("ft2").measurements / "merge_record.json"
# v3b (`ft3b`): its own stage folders, committed copies and records
V3B_STAGE_DIRS = {
    "train_v3b": Path("artifacts") / "train_v3b",
    "export_ft3b": Path("artifacts") / "export" / "ft3b",
    "parity_v3b": Path("artifacts") / "parity_v3b",
}
RECORDS_V3B = Path("docs") / "measurements" / "export_artifacts_v3b.json"
V3B = version("ft3b")
RL_RUN_V3B = V3B.measurements / "rl_run.json"
GGUF_RECORD_V3B = V3B.measurements / "gguf_ft3b_manifest.json"
MERGE_RECORD_V3B = V3B.measurements / "merge_record.json"
PLAN_V3B = Path("configs") / "eval_plan_v3b.yaml"
PINNED_V3B = ("reward", "checker")  # the plan's pinned files the train stage reads
PARITY_Q8_V3B = (V3B.measurements / "parity_q8_0.json",
                 V3B.measurements / "parity_q8_0_generations.jsonl.gz")  # fmt: skip
KERNELS = {"ft": "ggh-t22-train", "ft2": "ggh-v21-train"}  # the Kaggle kernel per version
LEDGER = Path("docs") / "measurements" / "gpu-ledger.jsonl"
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


def train_manifest(root: Path, model: str = "ft") -> StageManifest:
    """The Kaggle run, recorded locally from `TrainRun` (D-10): its inputs are the targets
    stage's outputs (replayed byte-identically on Kaggle) and the recipe. `model` is ft (v1)
    or ft2 (v2, its run, selection, adapter and recipe from `ggh.versions`)."""
    v = version(model)
    train_run = TRAIN_RUN if model == "ft" else v.train_run
    run = TrainRun.model_validate_json((root / train_run).read_text(encoding="utf-8"))
    targets = read_manifest(root / TARGETS_MANIFEST)
    rels = [Path("artifacts/targets/train.jsonl"), Path("artifacts/targets/val.jsonl"),
            v.recipe]  # fmt: skip
    inputs = hashes(root, rels)
    for rel in rels[:2]:
        if targets.outputs.get(rel.as_posix()) != inputs[rel.as_posix()]:
            raise ValueError(f"{rel} differs from what the targets stage recorded")
    if config_hash(root / rels[2]) != run.config_hash:
        raise ValueError(f"{v.recipe.as_posix()} is not the recipe the run recorded")
    adapter, selection = (ADAPTER_DIR, SELECTION) if model == "ft" else (v.adapter_dir, v.selection)
    frozen = json.loads((root / selection).read_text(encoding="utf-8"))["adapter_sha256"]
    if adapter_hash(root / adapter) != frozen:
        raise ValueError(f"{adapter} is not the frozen adapter")
    outputs = hashes(root, [adapter / f for f in ADAPTER_FILES] + [train_run])
    kernel = f"{KERNELS[model]} v{run.kaggle_kernel_version}"
    params = {"kaggle_kernel": kernel, "repo_commit": run.repo_commit,
              "selected_step": run.selected_step, "base": f"{BASE_ID}@{BASE_REVISION}"}  # fmt: skip
    started, finished = session_times(root, kernel)
    return _manifest("train", inputs, outputs, params, (targets.key,), started, finished,
                     HOST_KAGGLE)  # fmt: skip


def session_times(root: Path, kernel: str) -> tuple[dt.datetime, dt.datetime]:
    """An approximate span of the Kaggle session from its GPU ledger entry: `at` is when the
    entry was written (soon after the session ended) and `gpu_hours` is rounded up, so
    `started_at` and `finished_at` are estimates, not the run's own clock."""
    rows = [json.loads(x) for x in (root / LEDGER).read_text(encoding="utf-8").splitlines() if x]
    (row,) = [r for r in rows if r.get("kernel") == kernel and r.get("kind") == "train"]
    end = dt.datetime.fromisoformat(row["at"])
    return end - dt.timedelta(hours=float(row["gpu_hours"])), end


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


def _dirs(model: str) -> tuple[Path, Path]:
    """The adapter and merged folders of ft (v1's module constants) or ft2."""
    if model == "ft":
        return ADAPTER_DIR, MERGED_DIR
    return version(model).adapter_dir, version(model).merged_dir


def export_ft_manifest(root: Path, base_dir: Path, train: StageManifest, now: Clock = _now,
                       model: str = "ft", adapter_sha256: str | None = None
                       ) -> StageManifest:  # fmt: skip
    adapter, merged = _dirs(model)
    inputs = {base_uri(): sha256_file(base_dir / BASE_WEIGHTS),
              **hashes(root, [adapter / f for f in ADAPTER_FILES])}  # fmt: skip
    outputs = hashes(root, [merged / f for f in MERGED_FILES] + _gguf_rels(model))
    params = {**_gguf_params(), "merge": "bf16 on CPU, peft merge_and_unload"}
    if adapter_sha256 is not None:  # v3b on: the export names the frozen adapter it merged
        params["adapter_sha256"] = adapter_sha256
    t = now()
    return _manifest("export", inputs, outputs, params, (train.key,), t, t, "local")


def parity_manifest(root: Path, export_ft: StageManifest, now: Clock = _now, model: str = "ft",
                    extra_inputs: tuple[Path, ...] = (), extra_outputs: tuple[Path, ...] = ()
                    ) -> StageManifest:  # fmt: skip
    data = read_manifest(root / DATA_MANIFEST)
    merged = _dirs(model)[1]
    inputs = hashes(root, [merged / "model.safetensors", llama.GGUF_DIR / f"{model}-Q4_K_M.gguf",
                           DEV_EVAL, *extra_inputs])  # fmt: skip
    outputs = hashes(root, [*parity.paths(model), *extra_outputs])
    params = {"agreement_bar": parity.AGREEMENT_BAR, "schema_gap": parity.SCHEMA_GAP}
    t = now()
    return _manifest("parity", inputs, outputs, params, (export_ft.key, data.key), t, t, "local")


def records(root: Path, train: StageManifest, base: StageManifest, ft: StageManifest,
            model: str = "ft") -> list[ExportArtifact]:  # fmt: skip
    """`model` names the fine-tune's files (ft or ft2); a record's own `model` stays "ft" (the
    fine-tune, against "base"), and its path tells v1 from v2."""
    adapter, merged = _dirs(model)
    out = [ExportArtifact(kind="adapter", model="ft", path=adapter.as_posix(),
                          sha256=train.outputs[(adapter / "adapter_model.safetensors")
                                               .as_posix()],
                          parent_key=train.key),
           ExportArtifact(kind="merged_bf16", model="ft",
                          path=(merged / "model.safetensors").as_posix(),
                          sha256=ft.outputs[(merged / "model.safetensors").as_posix()],
                          parent_key=ft.key)]  # fmt: skip
    pairs: tuple[tuple[Literal["base", "ft"], str, StageManifest], ...] = (
        ("base", "base", base), ("ft", model, ft))  # fmt: skip
    for kind, name, m in pairs:
        for rel in _gguf_rels(name):
            quant = rel.stem.split("-", 1)[1].upper()
            out.append(ExportArtifact(kind="gguf", model=kind, quant=quant, path=rel.as_posix(),
                                      sha256=m.outputs[rel.as_posix()], llama_cpp_commit=llama.TAG,
                                      parent_key=m.key))  # fmt: skip
    return out


def _write(root: Path, name: str, m: StageManifest) -> None:
    path = write_manifest(root / {**STAGE_DIRS, **V2_STAGE_DIRS, **V3B_STAGE_DIRS}[name], m)
    copy = root / LINEAGE_COPY / f"{name}.json"
    copy.parent.mkdir(parents=True, exist_ok=True)
    copy.write_bytes(path.read_bytes())


def write_all(root: Path, base_dir: Path, now: Clock = _now) -> list[ExportArtifact]:
    train = train_manifest(root)
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


def check_v2_records(root: Path, train: StageManifest, ft: StageManifest) -> None:
    """The hashes just taken must be the ones V2.1 recorded when it built v2: the GGUF
    manifest's files and source adapter, and the merge record's adapter."""
    v = version("ft2")
    sel = json.loads((root / v.selection).read_text(encoding="utf-8"))
    frozen = sel["adapter_sha256"]
    gg = json.loads((root / GGUF_RECORD_V2).read_text(encoding="utf-8"))
    merge = json.loads((root / MERGE_RECORD_V2).read_text(encoding="utf-8"))
    if gg["source"]["revision"] != frozen or merge["adapter_sha256"] != frozen:
        raise ValueError(f"{GGUF_RECORD_V2} or {MERGE_RECORD_V2} names another adapter")
    for rel in _gguf_rels("ft2"):
        if gg["files"][rel.name]["sha256"] != ft.outputs[rel.as_posix()]:
            raise ValueError(f"{rel} differs from what {GGUF_RECORD_V2} recorded")
    if gg["llama_cpp"]["tag"] != llama.TAG or gg["llama_cpp"]["release_sha256"] != llama.BIN_SHA256:
        raise ValueError(f"{GGUF_RECORD_V2} was built with another llama.cpp")
    if train.params["selected_step"] != sel["selected_step"]:
        raise ValueError(f"{v.selection} selected another step than the run")


def write_v2(root: Path, base_dir: Path, now: Clock = _now) -> list[ExportArtifact]:
    """v2's train, export and parity manifests, its committed lineage copies (with the v2
    targets stage as `targets_v2`) and `export_artifacts_v2.json`."""
    train = train_manifest(root, "ft2")
    base = read_manifest(root / STAGE_DIRS["export_base"] / "manifest.json")
    if hashes(root, _gguf_rels("base")) != base.outputs:
        raise ValueError("the base GGUFs differ from v1's export_base manifest")
    ft = export_ft_manifest(root, base_dir, train, now, "ft2")
    check_v2_records(root, train, ft)
    _write(root, "train_v2", train)
    _write(root, "export_ft2", ft)
    _write(root, "parity_v2", parity_manifest(root, ft, now, "ft2"))
    targets = read_manifest(root / TARGETS_MANIFEST).model_copy(update={"host": "local"})
    (root / LINEAGE_COPY / "targets_v2.json").write_text(targets.model_dump_json(indent=2) + "\n",
                                                         encoding="utf-8")  # fmt: skip
    recs = records(root, train, base, ft, "ft2")
    text = json.dumps([r.model_dump(mode="json") for r in recs], indent=2) + "\n"
    (root / RECORDS_V2).write_text(text, encoding="utf-8")
    return recs


def rl_train_manifest(root: Path) -> StageManifest:
    """v3b's GRPO run on Kaggle, recorded locally from `RLRun` (D-10). Its inputs: v2's adapter
    (train_v2's outputs, the adapter the run started from), the v2 targets' training prompts,
    the GRPO recipe and the reward and checker files the v3b plan pins by sha256."""
    run = RLRun.model_validate_json((root / RL_RUN_V3B).read_text(encoding="utf-8"))
    v2 = version("ft2")
    train_v2 = read_manifest(root / V2_STAGE_DIRS["train_v2"] / "manifest.json")
    targets = read_manifest(root / TARGETS_MANIFEST)
    if targets.key not in train_v2.parents:
        raise ValueError(f"{TARGETS_MANIFEST} is not the targets stage v2 trained on")
    plan = yaml.safe_load((root / PLAN_V3B).read_text(encoding="utf-8"))
    pinned = [(Path(plan["pinned"][k]["file"]), plan["pinned"][k]["sha256"]) for k in PINNED_V3B]
    prompts = Path("artifacts/targets/train.jsonl")
    v2_adapter = [v2.adapter_dir / f for f in ADAPTER_FILES]
    rels = [prompts, V3B.recipe, *v2_adapter, *(f for f, _ in pinned)]
    inputs = hashes(root, rels)
    if targets.outputs.get(prompts.as_posix()) != inputs[prompts.as_posix()]:
        raise ValueError(f"{prompts} differs from what the targets stage recorded")
    for rel in v2_adapter:
        if train_v2.outputs.get(rel.as_posix()) != inputs[rel.as_posix()]:
            raise ValueError(f"{rel} differs from what train_v2 recorded")
    if adapter_hash(root / v2.adapter_dir) != run.init_adapter_sha256:
        raise ValueError(f"{v2.adapter_dir} is not the adapter the GRPO run started from")
    v2_sel = json.loads((root / v2.selection).read_text(encoding="utf-8"))["adapter_sha256"]
    if v2_sel != run.init_adapter_sha256:
        raise ValueError("the GRPO run did not start from v2's frozen adapter")
    if inputs[V3B.recipe.as_posix()] != run.config_hash:
        raise ValueError(f"{V3B.recipe.as_posix()} is not the recipe the run recorded")
    for rel, want in pinned:
        if inputs[rel.as_posix()] != want:
            raise ValueError(f"{rel} is not the file {PLAN_V3B} pins")
    frozen = json.loads((root / V3B.selection).read_text(encoding="utf-8"))["adapter_sha256"]
    if adapter_hash(root / V3B.adapter_dir) != frozen or frozen != run.adapter_sha256:
        raise ValueError(f"{V3B.adapter_dir} is not the frozen v3b adapter")
    outputs = hashes(root, [V3B.adapter_dir / f for f in ADAPTER_FILES] + [RL_RUN_V3B])
    kernel = run.kaggle_kernel_version  # "ggh-v3b-grpo v1": the kernel and version, as logged
    params = {"kaggle_kernel": kernel, "repo_commit": run.repo_commit,
              "selected_step": run.selected_step, "base": f"{BASE_ID}@{BASE_REVISION}",
              "method": "grpo", "reward": run.reward,
              "init_adapter_sha256": run.init_adapter_sha256,
              "adapter_sha256": run.adapter_sha256}  # fmt: skip
    started, finished = session_times(root, kernel)
    return _manifest("train", inputs, outputs, params, (train_v2.key, targets.key), started,
                     finished, HOST_KAGGLE)  # fmt: skip


def check_v3b_records(root: Path, train: StageManifest, ft: StageManifest) -> None:
    """The hashes just taken must be the ones V3.6 recorded when it built v3b, and every record
    must name the frozen adapter: the selection, the run, the GGUF manifest and the merge."""
    sel = json.loads((root / V3B.selection).read_text(encoding="utf-8"))
    run = json.loads((root / RL_RUN_V3B).read_text(encoding="utf-8"))
    gg = json.loads((root / GGUF_RECORD_V3B).read_text(encoding="utf-8"))
    merge = json.loads((root / MERGE_RECORD_V3B).read_text(encoding="utf-8"))
    frozen = sel["adapter_sha256"]
    named = {
        RL_RUN_V3B.as_posix(): run["adapter_sha256"],
        GGUF_RECORD_V3B.as_posix(): gg["source"]["revision"],
        MERGE_RECORD_V3B.as_posix(): merge["adapter_sha256"],
        "train_v3b": train.params["adapter_sha256"],
        "export_ft3b": ft.params["adapter_sha256"],
    }
    bad = sorted(k for k, v in named.items() if v != frozen)
    if bad:
        raise ValueError(f"{', '.join(bad)} name another adapter than {frozen[:12]}…")
    for rel in _gguf_rels("ft3b"):
        if gg["files"][rel.name]["sha256"] != ft.outputs[rel.as_posix()]:
            raise ValueError(f"{rel} differs from what {GGUF_RECORD_V3B} recorded")
    if gg["llama_cpp"]["tag"] != llama.TAG or gg["llama_cpp"]["release_sha256"] != llama.BIN_SHA256:
        raise ValueError(f"{GGUF_RECORD_V3B} was built with another llama.cpp")
    if train.params["selected_step"] != sel["selected_step"]:
        raise ValueError(f"{V3B.selection} selected another step than the run")


def write_v3b(root: Path, base_dir: Path, now: Clock = _now) -> list[ExportArtifact]:
    """v3b's train, export and parity manifests, their committed lineage copies and
    `export_artifacts_v3b.json`. The base GGUFs are v1's, unchanged."""
    train = rl_train_manifest(root)
    base = read_manifest(root / STAGE_DIRS["export_base"] / "manifest.json")
    if hashes(root, _gguf_rels("base")) != base.outputs:
        raise ValueError("the base GGUFs differ from v1's export_base manifest")
    frozen = train.params["adapter_sha256"]
    ft = export_ft_manifest(root, base_dir, train, now, "ft3b", adapter_sha256=frozen)
    check_v3b_records(root, train, ft)
    q8 = (llama.GGUF_DIR / "ft3b-Q8_0.gguf",)
    par = parity_manifest(root, ft, now, "ft3b", extra_inputs=q8, extra_outputs=PARITY_Q8_V3B)
    _write(root, "train_v3b", train)
    _write(root, "export_ft3b", ft)
    _write(root, "parity_v3b", par)
    recs = records(root, train, base, ft, "ft3b")
    text = json.dumps([r.model_dump(mode="json") for r in recs], indent=2) + "\n"
    (root / RECORDS_V3B).write_text(text, encoding="utf-8")
    return recs
