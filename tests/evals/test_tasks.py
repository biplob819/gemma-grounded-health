import json
from pathlib import Path

import pytest
from evalkit.run import load_cases, load_task
from evalkit.tasks import builtin_task, render_prompt

from ggh.data import build as data
from ggh.evals import tasks

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("task_rel", "split"), [(tasks.SMOKE_TASK, "dev_eval"), (tasks.OFFSCOPE_TASK, "offscope_eval")]
)
def test_local_tasks_load_in_evalkit_and_render_like_builtin_pubmedqa(task_rel, split):
    task, base = load_task(str(ROOT / task_rel))
    cases = load_cases(task, base)
    builtin = builtin_task("pubmedqa")
    assert len(cases) == 20 and len({c.id for c in cases}) == 20
    assert task.split == split
    for field in ("labels", "prompt_template", "output_schema", "scorers", "thresholds"):
        assert getattr(task, field) == getattr(builtin, field), field
    for c in cases:
        assert render_prompt(task, c) == render_prompt(builtin, c)
        assert c.reference_label in builtin.labels


def test_task_files_are_exactly_what_ggh_tasks_generates():
    assert (ROOT / tasks.SMOKE_TASK).read_text(encoding="utf-8") == tasks.task_yaml(
        "ggh_dev_smoke", "../../data/dev_smoke.jsonl", "dev_eval"
    )
    assert (ROOT / tasks.OFFSCOPE_TASK).read_text(encoding="utf-8") == tasks.task_yaml(
        "ggh_offscope", "../../data/offscope_task.jsonl", "offscope_eval"
    )


def test_offscope_task_holds_the_frozen_offscope_set():
    frozen = {r["id"]: r for r in data.read_jsonl(ROOT / data.OFFSCOPE_PATH)}
    for cid, prompt in tasks.prompts(ROOT, tasks.OFFSCOPE_TASK):
        assert prompt == frozen[cid]["prompt"]  # the prompt N1.2 froze, rendered by EvalKit
    assert tasks.offscope_cases(ROOT)[0]["label"] == "maybe"


def test_smoke_selection_is_seeded_and_never_from_train(tmp_path):
    rows = [
        {"pmid": str(p), "question": f"Q{p}?", "sentences": ["A.", "B."], "label": "yes",
         "prompt": f"P{p}"}
        for p in range(100, 160)
    ]  # fmt: skip
    path = tmp_path / tasks.DEV_EVAL
    path.parent.mkdir(parents=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    a = tasks.dev_smoke_cases(tmp_path)
    b = tasks.dev_smoke_cases(tmp_path)
    assert [c["id"] for c in a] == [c["id"] for c in b] and len(a) == 20
    assert [c["id"] for c in a] != [f"dev-{p}" for p in range(100, 120)]  # not the first 20
    reordered = tmp_path / "r"
    (reordered / tasks.DEV_EVAL).parent.mkdir(parents=True)
    (reordered / tasks.DEV_EVAL).write_text(
        "".join(json.dumps(r) + "\n" for r in reversed(rows)), encoding="utf-8"
    )
    assert [c["id"] for c in tasks.dev_smoke_cases(reordered)] == [c["id"] for c in a]


@pytest.mark.data
@pytest.mark.skipif(not (ROOT / tasks.DEV_EVAL).exists(), reason="dev_eval is built locally")
def test_committed_smoke_data_regenerates_from_dev_eval():
    for rel, text in tasks.files(ROOT).items():
        assert (ROOT / rel).read_text(encoding="utf-8") == text, rel
    dev = {f"dev-{r['pmid']}" for r in data.read_jsonl(ROOT / tasks.DEV_EVAL)}
    assert {c for c, _ in tasks.prompts(ROOT, tasks.SMOKE_TASK)} <= dev


def test_committed_smoke_cases_are_the_seeded_first_twenty_with_their_labels():
    rows = data.read_jsonl(ROOT / tasks.SMOKE_DATA)
    pmids = [r["id"].removeprefix("dev-") for r in rows]
    assert pmids == sorted(pmids, key=tasks._rank)  # the seeded order, not file order
    assert all(set(r) == {"id", "question", "context", "label"} for r in rows)  # no prompt
    # the labels agree with the cases EvalKit loaded for the committed dev smoke run
    run_cases = {
        c["id"]: c["reference_label"]
        for c in data.read_jsonl(ROOT / "docs" / "results" / "dev_smoke_base" / "cases.jsonl")
    }
    assert {r["id"]: r["label"] for r in rows} == run_cases
    assert len({r["label"] for r in rows}) == 2  # PQA-A dev rows: yes and no
