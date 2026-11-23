"""RLB-03, RLB-04: v3b's trainer wrapper, dev check and Kaggle notebooks."""

import json
import sys
from pathlib import Path

import pytest

from ggh.rl import grpo as G
from ggh.rl import grpo_b as GB
from ggh.rl import reward_b as RB
from ggh.rl.config import load_grpo
from ggh.rl.notebook import render
from tests.rl.test_grpo import TRANSFORMERS_517_FIELDS, TRL_115_GRPO_FIELDS
from tests.rl.test_reward import CONTEXT, GOOD, ans
from tests.rl.test_reward_b import FakeChecker

ROOT = Path(__file__).resolve().parents[2]
RECIPE_B = ROOT / "configs/recipe_v3b_grpo.yaml"


@pytest.fixture
def fake():
    f = FakeChecker()
    GB.set_checker(f)
    yield f
    GB.set_checker(None)


def _batch():
    comps = [[{"role": "assistant", "content": t}] for t in (
        ans("no", GOOD, [6]), ans("yes", GOOD, [6]), "junk", ans("no", GOOD, [2]),
        ans("no", CONTEXT[6], [7]),
    )]  # fmt: skip
    return comps, ["no"] * 5, [list(CONTEXT)] * 5


def test_the_components_sum_to_the_v3b_reward_with_one_checker_call(fake):
    comps, golds, ctxs = _batch()
    parts = [f(comps, gold_label=golds, context=ctxs) for f in GB.REWARD_FUNCS]
    total = [sum(w * p[i] for w, p in zip(GB.REWARD_WEIGHTS, parts, strict=True))
             for i in range(5)]  # fmt: skip
    want = [s.reward for s in RB.score_batch([c[0]["content"] for c in comps], golds, ctxs, fake)]
    assert total == want == [1.25, 0.0, -1.0, -0.25, 1.25]
    assert len(fake.calls) == 2 + 2  # once in the reward functions (cached), once in `want`
    assert parts[-1] == [0.0, 0.0, 0.0, 0.0, 1.0]  # the copy probe
    assert len(GB.REWARD_FUNCS) == len(GB.REWARD_WEIGHTS) and GB.REWARD_WEIGHTS[4:] == (0, 0, 0)


def test_the_batch_cache_is_keyed_by_context_too(fake):
    text = [[{"role": "assistant", "content": ans("no", GOOD, [6])}]]
    other = list(CONTEXT)
    other[5] = "Insulin levels rose after glucose in every subject."
    first = GB.r_grounded(text, gold_label=["no"], context=[list(CONTEXT)])
    second = GB.r_grounded(text, gold_label=["no"], context=[other])
    assert first == [1.0] and second == [0.0]


def test_v3b_trainer_arguments_are_v3s_with_its_own_reward_weights(tmp_path):
    g, base = load_grpo(RECIPE_B, ROOT)
    a = GB.grpo_args(g, base, tmp_path, max_steps=300)
    v3 = G.grpo_args(g, base, tmp_path, max_steps=300)
    assert {k: v for k, v in a.items() if k != "reward_weights"} == {
        k: v for k, v in v3.items() if k != "reward_weights"}  # fmt: skip
    assert a["reward_weights"] == list(GB.REWARD_WEIGHTS) and a["max_steps"] == 300
    assert a["save_steps"] == 25 and set(a) <= TRL_115_GRPO_FIELDS | TRANSFORMERS_517_FIELDS


def test_the_v3b_recipe_is_v3s_but_for_checkpoint_spacing():
    g3, _ = load_grpo(ROOT / "configs/recipe_v3_grpo.yaml", ROOT)
    gb, _ = load_grpo(RECIPE_B, ROOT)
    diff = {k for k in type(g3).model_fields if getattr(g3, k) != getattr(gb, k)}
    assert diff == {"save_every_steps"} and gb.save_every_steps == 25


def test_dev_rows_leave_out_the_checkers_tuning_cases(tmp_path):
    (tmp_path / "artifacts/data").mkdir(parents=True)
    (tmp_path / "docs/measurements/v3").mkdir(parents=True)
    (tmp_path / "docs/measurements/v3/checker_tuning_ids.json").write_text(
        json.dumps(["2", "4"]), encoding="utf-8")  # fmt: skip
    lines = [json.dumps({"pmid": str(i)}) for i in range(6)]
    (tmp_path / "artifacts/data/dev_eval.jsonl").write_text("\n".join(lines) + "\n", "utf-8")
    assert [r["pmid"] for r in GB.dev_rows(tmp_path, 3)] == ["0", "1", "3"]
    with pytest.raises(ValueError, match="outside"):
        GB.dev_rows(tmp_path, 5)
    real = ROOT / "artifacts/data/dev_eval.jsonl"
    if real.is_file():  # built by `ggh data`, not in CI
        rows = GB.dev_rows(ROOT, 300)
        assert len(rows) == 300 and not {r["pmid"] for r in rows} & GB.tuning_ids(ROOT)


def test_dev_summary_has_every_selection_guard_key(fake):
    rows = [{"label": "no", "sentences": list(CONTEXT), "pmid": str(i)} for i in range(3)]
    texts = [ans("no", GOOD, [6]), ans("no", GOOD, [2]), "junk"]
    s = GB.dev_summary(texts, rows, fake)
    for key in ("mean_reward", "unparsed", "maybe_share", "yes_share", "no_accuracy",
                "mean_words", "label_accuracy", "aim_cited"):  # fmt: skip
        assert key in s, key
    assert s["n"] == 3 and s["unparsed"] == round(1 / 3, 4) and s["copied"] == 0.0
    assert "copied_close" in s and "v3_reward" in s  # an independent lexical signal beside it
    assert s["mean_reward"] == round((1.25 - 0.25 - 1.0) / 3, 4)
    with pytest.raises(ValueError):
        GB.dev_summary(texts[:2], rows, fake)


def test_each_v3b_adapter_is_scored_in_its_own_process(tmp_path):
    from ggh.cli import build_parser

    cmd = GB.adapter_cmd(RECIPE_B, tmp_path / "checkpoint-25", "checkpoint-25", ROOT,
                         tmp_path / "dev", 300)  # fmt: skip
    assert cmd[:4] == [sys.executable, "-m", "ggh.cli", "rl-dev-one-b"]
    args = build_parser().parse_args(cmd[3:])
    assert args.func.__name__ == "_cmd_rl_dev_one_b" and args.n == 300
    for name in ("grpo-b", "rl-dev-b"):
        parsed = build_parser().parse_args([name, "--out", "x"] + (["--run", "r"] if name ==
                                                                  "rl-dev-b" else []))  # fmt: skip
        assert parsed.config == "configs/recipe_v3b_grpo.yaml"


def _text(nb):
    return "".join("".join(c["source"]) for c in nb["cells"])


def test_the_v3b_notebooks_run_the_checker_reward_on_held_out_dev():
    sha = "d" * 40
    trial = _text(render(sha, 40, variant="v3b"))
    train = _text(render(sha, None, part="train", variant="v3b"))
    score = _text(render(sha, None, part="score", variant="v3b"))
    assert "ggh grpo-b --config configs/recipe_v3b_grpo.yaml" in trial and "--max-steps 40" in trial
    assert "ggh rl-dev-b" in trial and "--n 200" in trial
    assert "--max-steps 300" in train and "rl-dev" not in train
    assert "ggh rl-dev-b" in score and "--n 300" in score and "grpo-b" not in score
    assert "HF_TOKEN" not in trial + train + score


def test_v3b_kernels_are_rendered_at_one_commit_with_their_metadata():
    from ggh.train.notebook import dumps

    commits = set()
    for folder, kwargs, slug, src in (
        ("trial", {"max_steps": 40}, "ggh-v3b-grpo-trial", []),
        ("main/train", {"max_steps": None, "part": "train"}, "ggh-v3b-grpo", []),
        ("main/score", {"max_steps": None, "part": "score"}, "ggh-v3b-grpo-score",
         ["biplob819/ggh-v3b-grpo"]),
    ):  # fmt: skip
        d = ROOT / "notebooks/v3b" / folder
        nb = (d / "train_kaggle.ipynb").read_text("utf-8")
        commit = nb.split("git checkout -q ")[1][:40]
        commits.add(commit)
        assert nb == dumps(render(commit, variant="v3b", **kwargs))
        meta = json.loads((d / "kernel-metadata.json").read_text("utf-8"))
        assert meta["id"] == f"biplob819/{slug}" and meta["is_private"] is True
        assert meta["machine_shape"] == "NvidiaTeslaT4" and meta["kernel_sources"] == src
    assert len(commits) == 1
