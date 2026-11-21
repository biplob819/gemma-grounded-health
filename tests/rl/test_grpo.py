"""RLR-03..RLR-06: the v3 GRPO recipe, the logged reward components, the trainer arguments, dev
scoring and the Kaggle notebook (no GPU needed)."""

import json
from pathlib import Path

import pytest

from ggh.config import ConfigError, load_recipe
from ggh.rl import grpo as G
from ggh.rl import reward as R
from ggh.rl.config import load_grpo
from ggh.rl.notebook import render
from tests.rl.test_reward import CONTEXT, GOOD, ans

ROOT = Path(__file__).resolve().parents[2]
RECIPE = ROOT / "configs/recipe_v3_grpo.yaml"


def test_the_recipe_continues_v2_exactly():
    g, base = load_grpo(RECIPE, ROOT)
    assert base == load_recipe(ROOT / "configs/recipe_v2.yaml")
    train_run = json.loads((ROOT / "docs/measurements/v2/train_selection.json").read_text("utf-8"))
    assert g.init_adapter_sha256 == train_run["adapter_sha256"]
    assert g.init_repo == "biplob123/gemma-4-e2b-grounded-health" and g.init_subfolder == "adapter"
    assert g.prompts == "targets_train"
    assert (g.num_generations, g.per_device_batch, g.grad_accum) == (8, 8, 4)
    assert g.prompts_per_step() == 4 and g.beta == 0.0 and g.learning_rate == 5e-6


def test_the_recipe_rejects_a_group_that_does_not_fit_the_batch(tmp_path):
    bad = RECIPE.read_text("utf-8").replace("per_device_batch: 8 ", "per_device_batch: 6 ")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/recipe_v2.yaml").write_text(
        (ROOT / "configs/recipe_v2.yaml").read_text("utf-8"), encoding="utf-8"
    )
    path = tmp_path / "r.yaml"
    path.write_text(bad, encoding="utf-8")
    with pytest.raises(ConfigError, match="multiple of num_generations"):
        load_grpo(path, tmp_path)
    path.write_text(RECIPE.read_text("utf-8") + "extra: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_grpo(path, tmp_path)


def test_the_components_sum_to_the_reward():
    texts = [
        ans("no", GOOD, [6]),
        ans("yes", GOOD, [2]),
        ans("maybe", "Unclear.", []),
        "not json",
        ans("no", "No.", [1, 6]),
    ]
    golds, ctx = ["no"] * len(texts), [CONTEXT] * len(texts)
    parts = [f(texts, golds, ctx) for f in (G.r_parse, G.r_label, G.r_grounded, G.r_aim)]
    total = [sum(p[i] for p in parts) for i in range(len(texts))]
    assert total == R.grpo_reward(texts, golds, ctx)
    assert G.probe_maybe(texts) == [0.0, 0.0, 1.0, 0.0, 0.0]
    assert G.probe_words(texts)[2:4] == [1.0, 0.0]
    assert len(G.REWARD_FUNCS) == len(G.REWARD_WEIGHTS) and G.REWARD_WEIGHTS[4:] == (0.0, 0.0)


def test_grpo_args_follow_the_recipe(tmp_path):
    g, base = load_grpo(RECIPE, ROOT)
    a = G.grpo_args(g, base, tmp_path, max_steps=None)
    assert a["max_completion_length"] < base.max_seq_len and "max_prompt_length" not in a
    assert a["warmup_steps"] == 0.05 and "warmup_ratio" not in a
    assert (a["loss_type"], a["scale_rewards"], a["epsilon"], a["num_iterations"]) == (
        "dapo", "group", 0.2, 1,
    )  # fmt: skip
    assert a["max_steps"] == -1 and a["num_train_epochs"] == 1.0 and a["beta"] == 0.0
    assert a["fp16"] and not a["bf16"] and a["save_only_model"]
    assert a["reward_weights"] == list(G.REWARD_WEIGHTS) and a["seed"] == 7
    assert G.grpo_args(g, base, tmp_path, max_steps=40)["max_steps"] == 40


# Field names read from trl v1.15.0 trl/trainer/grpo_config.py and transformers v5.17.0
# training_args.py (the Kaggle pins); `warmup_ratio` and `max_prompt_length` are in neither.
TRL_115_GRPO_FIELDS = {
    "learning_rate", "remove_unused_columns", "num_generations", "max_completion_length",
    "temperature", "top_p", "beta", "num_iterations", "epsilon", "reward_weights",
    "scale_rewards", "loss_type", "mask_truncated_completions",
}  # fmt: skip
TRANSFORMERS_517_FIELDS = {
    "output_dir", "lr_scheduler_type", "warmup_steps", "per_device_train_batch_size",
    "gradient_accumulation_steps", "num_train_epochs", "max_steps", "save_strategy", "save_steps",
    "save_only_model", "logging_steps", "seed", "fp16", "bf16", "report_to",
}  # fmt: skip


def test_every_trainer_argument_is_a_field_of_the_pinned_versions(tmp_path):
    g, base = load_grpo(RECIPE, ROOT)
    keys = set(G.grpo_args(g, base, tmp_path, max_steps=None))
    assert keys <= TRL_115_GRPO_FIELDS | TRANSFORMERS_517_FIELDS, keys - (
        TRL_115_GRPO_FIELDS | TRANSFORMERS_517_FIELDS
    )


def test_check_init_refuses_an_adapter_that_is_not_v2s(tmp_path):
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(G.InitError, match="hashes to"):
        G.check_init(tmp_path, "0" * 64)


def test_prompt_rows_carry_the_gold_label_and_sentences():
    from ggh.schemas import TrainExample

    e = TrainExample.model_validate(
        {
            "id": "pqa-a-1", "pmid": "1", "kind": "answer", "split": "train", "question": "Q?",
            "context": list(CONTEXT), "prompt": "P", "source_label": "no",
            "target": {"label": "no", "answer": GOOD, "citations": [6]},
            "rule_ids": ["TGT-R1"], "support": 0.5, "nli_entail": 0.9,
        }
    )  # fmt: skip
    rows = G.prompt_rows([e])
    assert rows == [{"id": "pqa-a-1", "prompt": [{"role": "user", "content": "P"}],
                     "gold_label": "no", "context": list(CONTEXT)}]  # fmt: skip
    assert G.label_mix(rows) == {"no": 1}


def test_reward_trace_keeps_reward_rows_and_components():
    hist = [
        {"step": 1, "loss": 0.1},
        {"step": 1, "reward": 1.2, "reward_std": 0.3, "rewards/r_label/mean": 0.8,
         "rewards/r_label/std": 0.1, "completions/mean_length": 50.0},
    ]  # fmt: skip
    assert G.reward_trace(hist) == [
        {"step": 1, "reward": 1.2, "reward_std": 0.3, "completions/mean_length": 50.0,
         "r_label": 0.8}
    ]  # fmt: skip


def test_dev_summary_reports_reward_and_hack_signals():
    rows = [{"label": "no", "sentences": list(CONTEXT)}] * 4
    texts = [ans("no", GOOD, [6]), ans("no", GOOD, [2]), ans("maybe", "Unclear here.", []), "x"]
    s = G.dev_summary(texts, rows)
    assert s["n"] == 4 and s["mean_reward"] == round((1.5 + 0.5 + 0.0 - 1.0) / 4, 4)
    assert (s["label_accuracy"], s["grounded"], s["aim_cited"], s["unparsed"]) == (
        0.5, 0.25, 0.25, 0.25,
    )  # fmt: skip
    assert s["label_mix"] == {"maybe": 1, "no": 2} and s["max_words"] == 13
    with pytest.raises(ValueError):
        G.dev_summary(texts[:2], rows)


def test_the_notebook_pins_the_commit_and_runs_trial_or_main():
    sha = "a" * 40
    trial = json.dumps(render(sha, 40))
    main = json.dumps(render(sha, None))
    assert f"git checkout -q {sha}" in trial and "requirements-kaggle-grpo.txt" in trial
    assert "--max-steps 40" in trial and "--n 200" in trial
    assert "--max-steps" not in main and "--n 500" in main
    assert "ggh targets --replay" in main and "HF_TOKEN" not in main
    with pytest.raises(ValueError):
        render("abc", 40)


def test_the_grpo_requirements_pin_trl_inside_unsloths_range():
    req = (ROOT / "notebooks/requirements-kaggle-grpo.txt").read_text("utf-8")
    assert "-r requirements-kaggle-unsloth.txt" in req and "trl==1.15.0" in req
