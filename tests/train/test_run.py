import math
from pathlib import Path

import pytest
import yaml

from ggh.config import load_recipe
from ggh.train import run as R

ROOT = Path(__file__).resolve().parents[2]


def test_lora_targets_the_language_model_projections_only():
    names = [
        "model.language_model.layers.0.self_attn.q_proj",
        "model.language_model.layers.0.mlp.down_proj",
        "model.language_model.layers.0.self_attn.q_norm",
        "model.vision_tower.encoder.layers.0.self_attn.q_proj",
        "model.audio_tower.layers.0.self_attn.q_proj",
        "model.embed_vision.embedding_projection",
    ]
    assert R.lora_targets(names, ("q_proj", "down_proj")) == names[:2]
    with pytest.raises(R.RecipeError, match="no module matches"):
        R.lora_targets(names, ("w_in",))


def test_an_unpinned_base_is_refused(tmp_path):
    raw = yaml.safe_load((ROOT / "configs/recipe.yaml").read_text(encoding="utf-8"))
    raw["base_revision"] = "unpinned"
    p = tmp_path / "r.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(R.RecipeError, match="unpinned"):
        R.check_recipe(load_recipe(p))
    R.check_recipe(load_recipe(ROOT / "configs/recipe.yaml"))


def test_stats_record_losses_and_count_non_finite_values():
    s = R.Stats()
    s.log(10, {"loss": 2.0, "grad_norm": 1.5})
    s.log(20, {"loss": float("nan"), "grad_norm": float("inf")})
    s.log(250, {"eval_loss": 1.2})
    assert s.losses == [(10, 2.0), (20, s.losses[1][1])] and math.isnan(s.losses[1][1])
    assert s.eval_losses == [(250, 1.2)] and s.non_finite == 2
    s.tokens, s.seconds = 3000, 2.0
    assert s.tokens_per_s == 1500 and R.Stats().tokens_per_s == 0


def test_full_run_hours_from_measured_throughput():
    assert R.full_run_hours(7_200_000, 1, 1000) == 2.0
    assert R.full_run_hours(7_200_000, 2, 2000) == 2.0
    with pytest.raises(ValueError):
        R.full_run_hours(1, 1, 0)


def test_training_args_follow_the_recipe():
    pytest.importorskip("transformers")
    cfg = load_recipe(ROOT / "configs/recipe.yaml")
    a = R.training_args(cfg, Path("/tmp/x"), max_steps=300, cuda=False)
    assert (a.per_device_train_batch_size, a.gradient_accumulation_steps) == (4, 4)
    assert a.learning_rate == cfg.learning_rate and a.seed == a.data_seed == 7
    assert a.max_steps == 300 and a.eval_steps == a.save_steps == cfg.eval_every_steps
    assert a.save_total_limit == 2 and a.load_best_model_at_end
    assert a.metric_for_best_model == "eval_loss" and not a.greater_is_better
    assert (
        a.warmup_steps == cfg.warmup_ratio
        and getattr(a.lr_scheduler_type, "value", a.lr_scheduler_type) == "cosine"
    )
    assert not a.fp16 and a.use_cpu
    g = R.training_args(cfg, Path("/tmp/x"), max_steps=None, cuda=True)
    assert (
        g.fp16
        and not g.bf16
        and g.max_steps == -1
        and getattr(g.optim, "value", g.optim) == "paged_adamw_8bit"
    )


# ---- RCP-03: a tiny Gemma 4 architecture model, trained on CPU ----


@pytest.fixture(scope="module")
def tiny():
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

    def make():
        torch.manual_seed(0)
        return tr.Gemma4ForCausalLM(cfg)

    return tok, make


def tiny_recipe(tmp_path):
    raw = yaml.safe_load((ROOT / "configs/recipe.yaml").read_text(encoding="utf-8"))
    raw.update(tool="trl", per_device_batch=2, grad_accum=1, learning_rate=0.01,
               warmup_ratio=0.0, eval_every_steps=4, max_seq_len=1024)  # fmt: skip
    p = tmp_path / "recipe.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return p


def train_tiny(tiny, tmp_path, name):
    from peft import LoraConfig, get_peft_model

    tok, make = tiny
    model = make()
    base = {n: p.detach().clone() for n, p in model.named_parameters()}

    def load(cfg):
        return R.Loaded(R.attach_lora(model, cfg, get_peft_model, LoraConfig), tok, "trl")

    rep = R.run(tiny_recipe(tmp_path), ROOT, tmp_path / name, max_steps=8, limit=16,
                load=load, log_every=1)  # fmt: skip
    return rep, model, base


@pytest.mark.recipe
@pytest.mark.skipif(not (ROOT / R.TARGETS / "train.jsonl").exists(), reason="targets are local")
def test_tiny_gemma4_trains_on_cpu_loss_falls_and_only_lora_weights_move(tiny, tmp_path):
    rep, model, base = train_tiny(tiny, tmp_path, "a")
    losses = [v for _, v in rep["losses"]]
    assert rep["steps"] == 8 and len(losses) == 8 and rep["non_finite"] == 0
    assert sum(losses[-3:]) / 3 < sum(losses[:3]) / 3  # it learns
    assert rep["tokens_seen"] > 0 and rep["tokens_per_s"] > 0 and rep["backend"] == "trl"
    assert rep["eval_losses"] and (tmp_path / "a" / "report.json").exists()
    moved_lora = moved_base = 0
    for n, p in model.named_parameters():
        clean = n.replace("base_model.model.", "").replace(".base_layer", "")
        if "lora_" in n:
            moved_lora += not bool((p.detach() == p.detach()).all() and p.abs().sum() == 0)
        elif clean in base:
            moved_base += not bool((p.detach() == base[clean]).all())
    assert moved_lora > 0 and moved_base == 0


@pytest.mark.recipe
@pytest.mark.skipif(not (ROOT / R.TARGETS / "train.jsonl").exists(), reason="targets are local")
def test_tiny_training_is_reproducible_with_the_same_seed(tiny, tmp_path):
    a, _, _ = train_tiny(tiny, tmp_path, "a")
    b, _, _ = train_tiny(tiny, tmp_path, "b")
    assert a["losses"] == b["losses"] and a["eval_losses"] == b["eval_losses"]
