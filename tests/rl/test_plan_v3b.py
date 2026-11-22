"""RLB-05: the v3b pre-registration pins the checker reward and recipe by hash, keeps selection off
the checker's tuning cases, and is hashed before any v3b training."""

import hashlib
import json
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "configs/eval_plan_v3b.yaml"


def _plan() -> dict:
    return yaml.safe_load(PLAN.read_text(encoding="utf-8"))


def test_the_plan_is_final_and_its_hash_is_committed():
    plan = _plan()
    assert plan["status"] == "final" and plan["node"] == "V3.3" and plan["version"] == "v3b"
    want = (ROOT / "docs/measurements/eval_plan_v3b.sha256").read_text(encoding="utf-8").strip()
    assert want == hashlib.sha256(PLAN.read_bytes()).hexdigest()


def test_everything_that_shapes_training_is_pinned():
    pinned = _plan()["pinned"]
    assert set(pinned) == {"reward", "checker", "trainer", "trainer_v3_parts", "reward_v3_parts",
                           "rules", "recipe_schema", "recipe", "packages",
                           "checker_tuning_ids"}  # fmt: skip
    for name, pin in pinned.items():
        got = hashlib.sha256((ROOT / pin["file"]).read_bytes()).hexdigest()
        assert got == pin["sha256"], name


def test_the_primary_test_guardrails_and_stretch_match_v3s():
    b = _plan()
    v3 = yaml.safe_load((ROOT / "configs/eval_plan_v3.yaml").read_text(encoding="utf-8"))
    assert b["primary"] == [["gemma-e2b-grounded-v3b-q4", "gemma-e2b-grounded-v2-q4"]]
    for key in ("primary_metrics", "guardrails", "stretch", "judges", "extractor", "n_cases",
                "seed", "quantization", "evalkit", "task_file", "bootstrap"):  # fmt: skip
        assert b[key] == v3[key], key
    assert "excludes zero" in b["rule"] and "Holm" in b["rule"] and "tie" in b["rule"]
    assert "neither is chosen after the fact" in b["rule"]


def test_selection_is_on_held_out_dev_with_hack_guards():
    plan = _plan()
    sel, held = plan["checkpoint_selection"], plan["held_out_dev"]
    assert "300" in sel and "checker_tuning_ids.json" in held
    guards = ("parse failures", '"maybe" share', '"yes" share', 'gold-"no" rows',
              "answer words", "label accuracy", "aim-citation share", "copy share")  # fmt: skip
    for guard in guards:
        assert guard in sel, guard
    assert "minus 0.10" in sel and "115 seconds" in plan["trial"]
    assert plan["main_run"].startswith("300 GRPO steps")
    assert [r["name"] for r in plan["runs"]] == ["main_v3b"] and plan["runs"][0]["n"] == 1000
    assert "owner" in plan["spend"] and "max_cost_usd" in plan["spend"]


def test_the_plan_row_is_recorded_before_the_v3b_notebooks():
    rows = [json.loads(x) for x in
            (ROOT / "docs/measurements/preregistrations.jsonl").read_text("utf-8").splitlines()
            if x.strip()]  # fmt: skip
    b = [r for r in rows if r["plan"] == "configs/eval_plan_v3b.yaml"]
    assert len(b) == 1 and b[0]["sha256"] == hashlib.sha256(PLAN.read_bytes()).hexdigest()
    nb = (ROOT / "notebooks/v3b/trial/train_kaggle.ipynb").read_text("utf-8")
    commit = nb.split("git checkout -q ")[1][:40]
    git = subprocess.run(["git", "merge-base", "--is-ancestor", b[0]["commit"], commit],
                         cwd=ROOT, check=False)  # fmt: skip
    if git.returncode in (0, 1):  # 128: a shallow clone without the history
        assert git.returncode == 0
