"""EXP-01 (X3.1): merge the frozen LoRA adapter into the bf16 base on CPU (Tech Spec §6).

Merging into a 4-bit base loses precision silently, so a quantized base is refused. The
merged folder gets the base checkpoint's tokenizer, chat template and processor files, so
`convert_hf_to_gguf.py` and the Space read exactly what training read.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from ggh.train.select import ADAPTER_FILES, adapter_hash

CARRIED = (  # files from the base snapshot the merged folder needs besides the weights
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "processor_config.json",
    "generation_config.json",
)
MERGED_DIR = Path("artifacts") / "merged" / "ft-bf16"
SELECTION = Path("docs") / "measurements" / "train_selection.json"
ADAPTER_DIR = Path("artifacts") / "adapter"


def frozen_hash(root: Path) -> str:
    sel = json.loads((root / SELECTION).read_text(encoding="utf-8"))
    if not sel.get("adapter_sha256"):
        raise MergeError(f"{SELECTION} records no frozen adapter hash")
    return str(sel["adapter_sha256"])


class MergeError(ValueError):
    pass


def check_unquantized(model: Any) -> None:
    """EXP-01: never merge into a 4-bit (or 8-bit) base."""
    cfg = getattr(model, "config", None)
    if getattr(cfg, "quantization_config", None) is not None or getattr(
        model, "is_quantized", False
    ):
        raise MergeError("the base is quantized; merge only into the bf16 base on CPU")
    for p in model.parameters():
        if str(p.dtype) not in ("torch.bfloat16", "torch.float32"):
            raise MergeError(f"the base holds {p.dtype} weights; merge only into bf16")


def check_adapter(adapter: Path, expected_sha256: str) -> None:
    missing = [f for f in ADAPTER_FILES if not (adapter / f).is_file()]
    if missing:
        raise MergeError(f"{adapter} is missing {missing}")
    got = adapter_hash(adapter)
    if got != expected_sha256:
        raise MergeError(f"adapter hash {got[:12]}… is not the frozen {expected_sha256[:12]}…")


def lora_count(model: Any) -> int:
    """LoRA-wrapped modules: each holds a `lora_A` submodule."""
    return sum(1 for _, m in model.named_modules() if hasattr(m, "lora_A"))


WEIGHTS = "model.safetensors"


def restore_dropped(base_weights: Path, merged_weights: Path, trained: set[str]) -> list[str]:
    """Put back base tensors that Transformers loads but does not keep (Gemma 4's K/V
    projections in its KV-shared layers). The pinned llama.cpp converts them for the base, so
    the fine-tune must carry them too for its GGUF to be built the same way. None of them may
    be a tensor the adapter trained: those are the merged values, never the base's."""
    from safetensors import safe_open  # pyright: ignore[reportMissingImports]
    from safetensors.torch import save_file  # pyright: ignore[reportMissingImports]

    with safe_open(str(base_weights), "pt") as b, safe_open(str(merged_weights), "pt") as m:
        missing = sorted(set(b.keys()) - set(m.keys()))
        extra = sorted(set(m.keys()) - set(b.keys()))
        if extra:
            raise MergeError(f"the merged checkpoint has tensors the base lacks: {extra[:3]}")
        clash = [k for k in missing if k.removesuffix(".weight") in trained]
        if clash:
            raise MergeError(f"trained tensors are missing from the merge: {clash[:3]}")
        if not missing:
            return []
        state = {k: m.get_tensor(k) for k in m.keys()}  # noqa: SIM118 - a safetensors handle
        state.update({k: b.get_tensor(k) for k in missing})
        meta = m.metadata() or {"format": "pt"}
    save_file(state, str(merged_weights), metadata=meta)
    return missing


def trained_modules(peft_model: Any) -> set[str]:
    """Base module names the adapter wraps, in checkpoint spelling (`model.language_model…`)."""
    out = set()
    for name, mod in peft_model.named_modules():
        if hasattr(mod, "lora_A"):
            out.add(name.removeprefix("base_model.model."))
    return out


def carry_files(base_dir: Path, out: Path) -> list[str]:
    copied = []
    for name in CARRIED:
        src = base_dir / name
        if src.is_file():
            shutil.copy2(src, out / name)
            copied.append(name)
    if "chat_template.jinja" not in copied and "tokenizer_config.json" not in copied:
        raise MergeError(f"{base_dir} has no chat template or tokenizer config to carry")
    return copied


def merge(
    base_dir: Path, adapter: Path, out: Path, *, expected_sha256: str, load_base: Any = None
) -> dict[str, Any]:  # pragma: no cover - loads the 10 GB base; tested with a tiny model
    """Load the base in bf16 on CPU, apply the adapter, merge, save safetensors."""
    import torch  # pyright: ignore[reportMissingImports]
    from peft import PeftModel  # pyright: ignore[reportMissingImports]

    check_adapter(adapter, expected_sha256)

    def _load(path: Path) -> Any:
        from transformers import AutoModelForImageTextToText  # pyright: ignore[reportMissingImports]

        return AutoModelForImageTextToText.from_pretrained(
            path, dtype=torch.bfloat16, device_map={"": "cpu"}
        )

    load_base = load_base or _load
    base = load_base(base_dir)
    check_unquantized(base)
    peft_model = PeftModel.from_pretrained(base, str(adapter))
    n_lora = lora_count(peft_model)
    trained = trained_modules(peft_model)
    if n_lora == 0:
        raise MergeError("the adapter attached no LoRA layer to the base; nothing to merge")
    merged = peft_model.merge_and_unload()
    left = [n for n, _ in merged.named_parameters() if "lora_" in n]
    if left:
        raise MergeError(f"LoRA weights survived the merge: {left[:3]}")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    merged.save_pretrained(out, safe_serialization=True)
    carried = carry_files(base_dir, out)
    restored = restore_dropped(base_dir / WEIGHTS, out / WEIGHTS, trained)
    record = {"adapter_sha256": expected_sha256, "lora_layers_merged": n_lora,
              "dtype": "bfloat16", "device": "cpu", "carried": carried,
              "restored_from_base": restored}  # fmt: skip
    (out / "ggh_merge.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record
