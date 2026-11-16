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
    assert a.num_train_epochs == cfg.epochs
    assert a.save_total_limit == 2 and a.load_best_model_at_end
    assert a.metric_for_best_model == "eval_loss" and not a.greater_is_better
    assert (
        a.warmup_steps == cfg.warmup_ratio
        and getattr(a.lr_scheduler_type, "value", a.lr_scheduler_type) == "cosine"
    )
    assert not a.fp16 and a.use_cpu and a.logging_nan_inf_filter is False
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
    pytest.importorskip("torch")
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

    def make():  # unseeded: run() must seed the base and LoRA initialisation itself
        return tr.Gemma4ForCausalLM(cfg)

    return tok, make


def tiny_recipe(tmp_path):
    raw = yaml.safe_load((ROOT / "configs/recipe.yaml").read_text(encoding="utf-8"))
    raw.update(tool="trl", per_device_batch=2, grad_accum=1, learning_rate=0.01,
               warmup_ratio=0.0, eval_every_steps=4, max_seq_len=1024)  # fmt: skip
    p = tmp_path / "recipe.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return p


def synthetic_targets(root, n=16):
    """A small training set built by the real N1.3 code, so the test needs no local artifacts.
    Val holds half as many examples as train, so a mix-up of the two shows in the counts."""
    from ggh.targets import make as targets
    from tests.targets.test_targets import row

    for split, start, count in (("train", 0, n), ("val", 100, n // 2)):
        exs = []
        for i in range(start, start + count):
            label = "yes" if i % 3 else "no"
            t = targets.AnswerTarget(f"Outcome beta{i} improved markedly.", (3,), 1.0, 0.9)
            exs.append(targets.answer_example(row(i, label), t, split))
        path = root / R.TARGETS / f"{split}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(e.model_dump_json() + "\n" for e in exs), encoding="utf-8")
    return root


def train_tiny(tiny, tmp_path, name):
    from peft import LoraConfig, get_peft_model  # pyright: ignore[reportMissingImports]

    tok, make = tiny
    held = {}

    def load(cfg):  # built after run() seeds, as a real loader is
        model = make()
        held["base"] = {n: p.detach().clone() for n, p in model.named_parameters()}
        peft = R.attach_lora(model, cfg, get_peft_model, LoraConfig)
        held["lora0"] = {n: p.detach().clone() for n, p in peft.named_parameters() if "lora_" in n}
        held["model"] = peft
        return R.Loaded(peft, tok, "trl")

    root = synthetic_targets(tmp_path / f"repo-{name}")
    rep = R.run(tiny_recipe(tmp_path), root, tmp_path / name, max_steps=8, limit=16,
                load=load, log_every=1)  # fmt: skip
    return rep, held


@pytest.mark.recipe
def test_tiny_gemma4_trains_on_cpu_loss_falls_and_only_lora_weights_move(tiny, tmp_path):
    rep, held = train_tiny(tiny, tmp_path, "a")
    model, base = held["model"], held["base"]
    losses = [v for _, v in rep["losses"]]
    assert rep["steps"] == 8 and len(losses) == 8 and rep["non_finite"] == 0
    assert sum(losses[-3:]) / 3 < sum(losses[:3]) / 3  # it learns
    assert rep["tokens_seen"] > 0 and rep["tokens_per_s"] > 0 and rep["backend"] == "trl"
    assert rep["eval_losses"] and (tmp_path / "a" / "report.json").exists()
    moved_lora = moved_base = 0
    for n, p in model.named_parameters():
        clean = n.replace("base_model.model.", "").replace(".base_layer", "")
        if "lora_" in n:
            moved_lora += not bool((p.detach() == held["lora0"][n]).all())
        elif clean in base:
            moved_base += not bool((p.detach() == base[clean]).all())
    assert moved_lora > 0 and moved_base == 0


@pytest.mark.recipe
def test_tiny_training_is_reproducible_with_the_same_seed(tiny, tmp_path):
    a, _ = train_tiny(tiny, tmp_path, "a")
    b, _ = train_tiny(tiny, tmp_path, "b")
    assert a["losses"] == b["losses"] and a["eval_losses"] == b["eval_losses"]


@pytest.mark.recipe
def test_tiny_run_counts_every_non_padding_token_once_and_reports_precision(tiny, tmp_path):
    rep, _ = train_tiny(tiny, tmp_path, "a")
    # 8 steps x 2 sequences = the 16 training examples, each seen once
    assert rep["tokens_seen"] == rep["train_tokens"] and rep["train_examples"] == 16
    tok, _ = tiny
    from ggh.train import format as fmt

    exs = R.load_examples(tmp_path / "repo-a", "train")
    assert rep["train_tokens"] == fmt.build(tok, exs, 1024).tokens
    assert rep["non_finite_steps"] == 0 and rep["skipped_steps"] == 0
    assert rep["precision"]["amp"] is None and set(rep["precision"]["trainable"]) == {"float32"}
    assert rep["base_loaded"] is None  # an injected loader declares no source


def test_the_loaded_checkpoint_must_be_the_pinned_one():
    cfg = load_recipe(ROOT / "configs/recipe.yaml")
    R.check_source(cfg, (cfg.base_model, cfg.base_revision))
    for bad in (("unsloth/gemma-4-E2B-it-unsloth-bnb-4bit", None),
                (cfg.base_model, "0" * 40), (cfg.base_model, None)):  # fmt: skip
        with pytest.raises(R.RecipeError, match="recipe pins"):
            R.check_source(cfg, bad)


def test_loaded_source_reads_the_config_transformers_recorded():
    class Cfg:
        _name_or_path = "google/gemma-4-E2B-it"
        _commit_hash = "abc"

    class Base:
        config = Cfg()

    class Peft:
        def get_base_model(self):
            return Base()

    assert R.loaded_source(Peft()) == ("google/gemma-4-E2B-it", "abc")
    assert R.loaded_source(Base()) == ("google/gemma-4-E2B-it", "abc")


def test_run_picks_the_loader_named_by_tool_and_refuses_a_wrong_checkpoint(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    called = []

    def fake(name, source=None):
        def load(cfg):
            called.append((name, cfg.tool))
            raise RuntimeError("stop after loading")  # nothing else is needed here

        return load

    monkeypatch.setattr(R, "LOADERS", {"trl": fake("trl"), "unsloth": fake("unsloth")})
    for tool, want in ((None, "unsloth"), ("trl", "trl")):
        with pytest.raises(RuntimeError, match="stop after loading"):
            R.run(ROOT / "configs/recipe.yaml", tmp_path, tmp_path / "o", tool=tool)
        assert called[-1] == (want, want)

    def mirror(cfg):
        return R.Loaded(None, None, "unsloth", ("unsloth/gemma-4-E2B-it-unsloth-bnb-4bit", None))

    with pytest.raises(R.RecipeError, match="recipe pins"):
        R.run(ROOT / "configs/recipe.yaml", tmp_path, tmp_path / "o", load=mirror)


# ---- round 2: the numeric guards and the loaders' arguments ----


def test_count_skipped_reads_the_accelerators_flag_and_fails_loudly_on_a_rename():
    from types import SimpleNamespace

    s = R.Stats()
    R.count_skipped(
        s, SimpleNamespace(accelerator=SimpleNamespace(optimizer_step_was_skipped=True))
    )
    R.count_skipped(
        s, SimpleNamespace(accelerator=SimpleNamespace(optimizer_step_was_skipped=False))
    )
    assert s.skipped_steps == 1
    for unwired in (None, SimpleNamespace(accelerator=None)):
        with pytest.raises(RuntimeError, match="not wired"):
            R.count_skipped(s, unwired)
    with pytest.raises(AttributeError):
        R.count_skipped(s, SimpleNamespace(accelerator=SimpleNamespace()))


def test_non_finite_flags_nan_and_inf_losses():
    torch = pytest.importorskip("torch")
    assert R.non_finite(torch.tensor(float("nan"))) and R.non_finite(torch.tensor(float("inf")))
    assert not R.non_finite(torch.tensor(0.5))


@pytest.mark.recipe
def test_a_nan_loss_on_one_step_is_counted(tiny, tmp_path):
    from peft import LoraConfig, get_peft_model  # pyright: ignore[reportMissingImports]

    tok, make = tiny
    calls = {"n": 0}

    def load(cfg):
        peft = R.attach_lora(make(), cfg, get_peft_model, LoraConfig)

        def poison(module, args, kwargs, output):
            calls["n"] += 1
            if module.training and calls["n"] == 3 and output.loss is not None:
                # a NaN value with finite gradients: later steps stay finite, so the count
                # must be exactly one (every step checked, not only the last)
                output.loss = output.loss + float("nan") * output.loss.detach()
            return output

        peft.register_forward_hook(poison, with_kwargs=True)
        return R.Loaded(peft, tok, "trl")

    root = synthetic_targets(tmp_path / "repo")
    rep = R.run(tiny_recipe(tmp_path), root, tmp_path / "o", max_steps=4, limit=16,
                load=load, log_every=1)  # fmt: skip
    assert rep["non_finite_steps"] == 1


@pytest.mark.recipe
def test_seeding_happens_inside_run_even_after_the_global_rng_moves(tiny, tmp_path):
    import torch  # pyright: ignore[reportMissingImports]

    a, _ = train_tiny(tiny, tmp_path, "a")
    torch.manual_seed(12345)
    torch.rand(1000)  # move the global RNG between the runs
    b, _ = train_tiny(tiny, tmp_path, "b")
    assert a["losses"] == b["losses"]


@pytest.mark.recipe
def test_a_partial_epoch_reports_seen_tokens_apart_from_the_dataset_total(tiny, tmp_path):
    from peft import LoraConfig, get_peft_model  # pyright: ignore[reportMissingImports]

    tok, make = tiny

    def load(cfg):
        return R.Loaded(R.attach_lora(make(), cfg, get_peft_model, LoraConfig), tok, "trl")

    root = synthetic_targets(tmp_path / "repo")
    rep = R.run(tiny_recipe(tmp_path), root, tmp_path / "o", max_steps=3, limit=16,
                load=load, log_every=1)  # fmt: skip
    assert 0 < rep["tokens_seen"] < rep["train_tokens"]  # 6 of 16 examples seen
    prec = rep["precision"]
    assert sum(prec["trainable"].values()) == rep["trainable_params"] and prec["frozen"]


def fake_module(name, **attrs):
    import types

    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    return mod


class FakeHF:
    def __init__(self, name, commit):
        from types import SimpleNamespace

        self.config = SimpleNamespace(_name_or_path=name, _commit_hash=commit)

    def get_base_model(self):
        return self

    def named_modules(self):
        return iter([])


def test_load_unsloth_passes_the_exact_pin_the_targets_and_no_vision(monkeypatch):
    pytest.importorskip("torch")
    seen = {}
    cfg = load_recipe(ROOT / "configs/recipe.yaml")

    class FastModel:
        @staticmethod
        def from_pretrained(**kw):
            seen["load"] = kw
            return FakeHF(cfg.base_model, cfg.base_revision), "tok"

        @staticmethod
        def get_peft_model(model, **kw):
            seen["peft"] = kw
            return model

    monkeypatch.setitem(
        __import__("sys").modules, "unsloth", fake_module("unsloth", FastModel=FastModel)
    )
    loaded = R.load_unsloth(cfg)
    assert seen["load"]["use_exact_model_name"] is True
    assert seen["load"]["model_name"] == cfg.base_model
    assert seen["load"]["revision"] == cfg.base_revision
    assert seen["peft"]["target_modules"] == list(cfg.target_modules)
    assert seen["load"]["load_in_4bit"] is cfg.load_in_4bit is True
    assert str(seen["load"]["dtype"]) == f"torch.{cfg.compute_dtype}"
    assert seen["load"]["max_seq_length"] == cfg.max_seq_len
    assert seen["load"]["full_finetuning"] is False
    pk = seen["peft"]
    assert pk["finetune_vision_layers"] is False and pk["r"] == cfg.lora_r
    assert pk["finetune_language_layers"] and pk["finetune_attention_modules"]
    assert pk["finetune_mlp_modules"] is True
    assert (pk["lora_alpha"], pk["lora_dropout"], pk["random_state"]) == (
        cfg.lora_alpha, cfg.lora_dropout, cfg.seed)  # fmt: skip
    assert loaded.source == (cfg.base_model, cfg.base_revision) and loaded.backend == "unsloth"


def test_load_trl_pins_the_revision_for_model_and_tokenizer(monkeypatch):
    pytest.importorskip("torch")
    import sys

    seen = {}
    cfg = load_recipe(ROOT / "configs/recipe.yaml")

    class Auto:
        @staticmethod
        def from_pretrained(name, **kw):
            seen.setdefault("calls", []).append((name, kw.get("revision")))
            return FakeHF(name, kw.get("revision"))

    def get_peft_model(model, lc):
        seen["lora"] = lc
        return model

    monkeypatch.setitem(sys.modules, "transformers", fake_module(
        "transformers", AutoModelForCausalLM=Auto, AutoTokenizer=Auto,
        BitsAndBytesConfig=lambda **kw: kw))  # fmt: skip
    monkeypatch.setitem(sys.modules, "peft", fake_module(
        "peft", LoraConfig=lambda **kw: kw, get_peft_model=get_peft_model))  # fmt: skip
    monkeypatch.setattr(R, "lora_targets", lambda names, wanted: ["x.q_proj"])
    loaded = R.load_trl(cfg)
    assert seen["calls"] == [(cfg.base_model, cfg.base_revision)] * 2  # tokenizer, then model
    lora = seen["lora"]
    assert (lora["r"], lora["lora_alpha"], lora["lora_dropout"]) == (
        cfg.lora_r, cfg.lora_alpha, cfg.lora_dropout)  # fmt: skip
    assert lora["task_type"] == "CAUSAL_LM" and loaded.source == (cfg.base_model, cfg.base_revision)


@pytest.mark.recipe
def test_eval_runs_on_val_and_every_skipped_step_is_counted(tiny, tmp_path, monkeypatch):
    import accelerate  # pyright: ignore[reportMissingImports]

    # force the loss scaler's flag on: the wiring from the trainer to the counter is what runs
    monkeypatch.setattr(accelerate.Accelerator, "optimizer_step_was_skipped",
                        property(lambda self: True))  # fmt: skip
    rep, _ = train_tiny(tiny, tmp_path, "a")
    assert rep["skipped_steps"] == rep["steps"] == 8
    assert rep["val_examples"] == rep["eval_rows"] == 8 and rep["train_examples"] == 16
    assert rep["epochs"] == load_recipe(tiny_recipe(tmp_path)).epochs
