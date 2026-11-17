"""EXP-01: the adapter is hash-checked and merged into a bf16 base; quantized bases are refused;
base-only tensors the merge drops are put back (and trained ones never are)."""

import json
from types import SimpleNamespace

import pytest

from ggh.export import merge as M
from ggh.train.select import adapter_hash


def test_quantized_or_low_precision_bases_are_refused():
    torch = pytest.importorskip("torch")

    class Fake:
        def __init__(self, dtype, qcfg=None):
            self.config = SimpleNamespace(quantization_config=qcfg)
            self._p = [torch.zeros(2, dtype=dtype)]

        def parameters(self):
            return iter(self._p)

    M.check_unquantized(Fake(torch.bfloat16))
    M.check_unquantized(Fake(torch.float32))
    with pytest.raises(M.MergeError, match="quantized"):
        M.check_unquantized(Fake(torch.bfloat16, qcfg={"load_in_4bit": True}))
    with pytest.raises(M.MergeError, match=r"uint8|float16"):
        M.check_unquantized(Fake(torch.uint8))
    with pytest.raises(M.MergeError, match="float16"):
        M.check_unquantized(Fake(torch.float16))


def test_a_changed_or_incomplete_adapter_is_refused(tmp_path):
    a = tmp_path / "adapter"
    a.mkdir()
    (a / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(M.MergeError, match="missing"):
        M.check_adapter(a, "0" * 64)
    (a / "adapter_model.safetensors").write_bytes(b"weights")
    good = adapter_hash(a)
    M.check_adapter(a, good)
    (a / "adapter_model.safetensors").write_bytes(b"weightz")
    with pytest.raises(M.MergeError, match="not the frozen"):
        M.check_adapter(a, good)


def test_the_frozen_hash_comes_from_the_committed_selection(tmp_path):
    p = tmp_path / M.SELECTION
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"adapter_sha256": "ab" * 32}), encoding="utf-8")
    assert M.frozen_hash(tmp_path) == "ab" * 32
    p.write_text(json.dumps({"adapter_sha256": None}), encoding="utf-8")
    with pytest.raises(M.MergeError, match="no frozen"):
        M.frozen_hash(tmp_path)


def _write(path, tensors):
    from safetensors.torch import save_file  # pyright: ignore[reportMissingImports]

    save_file(tensors, str(path), metadata={"format": "pt"})


def test_restore_puts_back_only_untrained_base_tensors(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from safetensors import safe_open  # pyright: ignore[reportMissingImports]

    base, merged = tmp_path / "base.safetensors", tmp_path / "merged.safetensors"
    kv = "model.layers.3.self_attn.k_proj.weight"
    _write(base, {"model.layers.0.q.weight": torch.zeros(2), kv: torch.full((2,), 7.0)})
    _write(merged, {"model.layers.0.q.weight": torch.ones(2)})
    assert M.restore_dropped(base, merged, {"model.layers.0.q"}) == [kv]
    with safe_open(str(merged), "pt") as f:
        assert set(f.keys()) == {"model.layers.0.q.weight", kv}
        assert f.get_tensor("model.layers.0.q.weight").tolist() == [1.0, 1.0]  # merged kept
        assert f.get_tensor(kv).tolist() == [7.0, 7.0]  # the base's, untouched
    assert M.restore_dropped(base, merged, set()) == []  # nothing left to restore


def test_restore_refuses_a_trained_tensor_missing_or_an_unknown_one(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    base, merged = tmp_path / "b.safetensors", tmp_path / "m.safetensors"
    _write(base, {"a.weight": torch.zeros(1), "t.weight": torch.zeros(1)})
    _write(merged, {"a.weight": torch.zeros(1)})
    with pytest.raises(M.MergeError, match="trained tensors"):
        M.restore_dropped(base, merged, {"t"})
    _write(
        merged, {"a.weight": torch.zeros(1), "t.weight": torch.zeros(1), "x.weight": torch.zeros(1)}
    )
    with pytest.raises(M.MergeError, match="lacks"):
        M.restore_dropped(base, merged, set())


def test_carry_needs_a_chat_template_or_tokenizer_config(tmp_path):
    base, out = tmp_path / "base", tmp_path / "out"
    base.mkdir()
    out.mkdir()
    with pytest.raises(M.MergeError, match="no chat template"):
        M.carry_files(base, out)
    (base / "chat_template.jinja").write_text("{{ x }}", encoding="utf-8")
    (base / "tokenizer.json").write_text("{}", encoding="utf-8")
    assert M.carry_files(base, out) == ["tokenizer.json", "chat_template.jinja"]
    assert (out / "chat_template.jinja").read_text(encoding="utf-8") == "{{ x }}"


@pytest.fixture(scope="module")
def tiny_base(tmp_path_factory):
    """A tiny Gemma 4 text model saved in bf16 with the pinned tokenizer and template."""
    torch = pytest.importorskip("torch")
    tr = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    try:
        tok = tr.AutoTokenizer.from_pretrained(
            "google/gemma-4-E2B-it",
            revision="4adf15748de9e1a0097b6274cee9c15836c41c37",
            local_files_only=True,
        )
    except OSError:
        pytest.skip("the pinned Gemma 4 tokenizer is not in the local Hugging Face cache")
    cfg = tr.Gemma4TextConfig(
        vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=32, global_head_dim=32,
        hidden_size_per_layer_input=8, vocab_size_per_layer_input=len(tok),
        layer_types=["sliding_attention", "full_attention"], num_kv_shared_layers=0,
        sliding_window=64, max_position_embeddings=2048,
    )  # fmt: skip
    torch.manual_seed(0)
    base = tmp_path_factory.mktemp("base")
    tr.Gemma4ForCausalLM(cfg).to(torch.bfloat16).save_pretrained(base)
    tok.save_pretrained(base)
    return base, tr, torch


def _adapter(tiny_base, tmp_path):
    base, tr, torch = tiny_base
    from peft import LoraConfig, get_peft_model  # pyright: ignore[reportMissingImports]

    model = tr.Gemma4ForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
    lc = LoraConfig(
        r=4, lora_alpha=8, target_modules=["q_proj", "down_proj"], task_type="CAUSAL_LM"
    )
    peft = get_peft_model(model, lc)
    torch.manual_seed(1)
    for n, p in peft.named_parameters():
        if "lora_B" in n:
            p.data.normal_()  # a non-zero update, so the merge is visible
    out = tmp_path / "adapter"
    peft.save_pretrained(out)
    return out


def test_tiny_gemma4_merge_changes_exactly_the_trained_weights(tiny_base, tmp_path):
    base, tr, torch = tiny_base
    adapter = _adapter(tiny_base, tmp_path)
    digest = adapter_hash(adapter)
    out = tmp_path / "merged"

    def load(p):
        return tr.Gemma4ForCausalLM.from_pretrained(p, dtype=torch.bfloat16)

    rec = M.merge(base, adapter, out, expected_sha256=digest, load_base=load)
    assert rec["lora_layers_merged"] == 4 and rec["dtype"] == "bfloat16"  # 2 layers x 2 modules
    assert (
        json.loads((out / "ggh_merge.json").read_text(encoding="utf-8"))["adapter_sha256"] == digest
    )
    assert (out / "tokenizer.json").is_file() and (out / "model.safetensors").is_file()
    from safetensors import safe_open  # pyright: ignore[reportMissingImports]

    with (
        safe_open(str(base / "model.safetensors"), "pt") as b,
        safe_open(str(out / "model.safetensors"), "pt") as m,
    ):
        bk, mk = set(b.keys()), set(m.keys())  # safetensors handles are not iterable
        assert bk == mk and not any("lora" in k for k in mk)
        changed = {k for k in bk if not torch.equal(b.get_tensor(k), m.get_tensor(k))}
    assert changed and all(k.endswith(("q_proj.weight", "down_proj.weight")) for k in changed)
    assert len(changed) == 4


def test_tiny_merge_refuses_a_changed_adapter_before_loading_anything(tiny_base, tmp_path):
    base, _, _ = tiny_base
    adapter = _adapter(tiny_base, tmp_path)
    called = []
    with pytest.raises(M.MergeError, match="not the frozen"):
        M.merge(base, adapter, tmp_path / "m", expected_sha256="0" * 64,
                load_base=lambda p: called.append(p))  # fmt: skip
    assert not called and not (tmp_path / "m").exists()
