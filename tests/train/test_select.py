import json
from pathlib import Path

import pytest

from ggh.schemas import DevMetrics
from ggh.train import select as sel


def metrics(schema=1.0, has=1.0, cite=1.0):
    return DevMetrics(n=500, schema_valid=schema, schema_valid_unconstrained=0.5,
                      citation_valid=cite, has_citation=has, label_accuracy={"yes": 0.8},
                      mean_words=40.0, over_80_words=0.0)  # fmt: skip


def cand(step, loss):
    return sel.Candidate(step, loss, Path(f"/ck/{step}"))


def test_candidates_are_tried_by_validation_loss_and_the_first_passing_is_chosen():
    cands = [cand(250, 0.80), cand(500, 0.70), cand(527, 0.71)]
    seen = []

    def m(c):
        seen.append(c.step)
        return metrics(schema=0.97) if c.step == 500 else metrics()

    chosen, tried = sel.select(cands, m)
    assert seen == [500, 527] and chosen == cands[2]  # best loss first; 250 never generated
    assert [t.passed for t in tried] == [False, True]
    assert tried[0].failed == ("schema_valid 0.970 < 0.98",)


def test_ties_go_to_the_earlier_step_and_no_pass_selects_nothing():
    cands = [cand(527, 0.7), cand(500, 0.7)]
    assert [c.step for c in sel.ordered(cands)] == [500, 527]
    chosen, tried = sel.select(cands, lambda c: metrics(has=0.5, cite=0.97))
    assert chosen is None and len(tried) == 2
    assert tried[0].failed == ("has_citation 0.500 < 0.98", "citation_valid 0.970 < 0.98")


def test_each_bar_is_inclusive_at_098():
    for k in ("schema", "has", "cite"):
        assert sel.select([cand(1, 0.1)], lambda c, k=k: metrics(**{k: 0.98}))[0] is not None
        assert sel.select([cand(1, 0.1)], lambda c, k=k: metrics(**{k: 0.979}))[0] is None


def test_checkpoints_read_the_eval_loss_saved_at_their_step(tmp_path):
    for step, loss in ((250, 0.8), (527, 0.7)):
        d = tmp_path / f"checkpoint-{step}"
        d.mkdir()
        hist = [{"step": 10, "loss": 2.0}, {"step": step, "eval_loss": loss + 1},
                {"step": step, "eval_loss": loss}]  # fmt: skip
        (d / "trainer_state.json").write_text(json.dumps({"log_history": hist}), encoding="utf-8")
    got = sel.checkpoints(tmp_path)
    assert [(c.step, c.eval_loss) for c in got] == [(250, 0.8), (527, 0.7)]
    bad = tmp_path / "checkpoint-9"
    bad.mkdir()
    (bad / "trainer_state.json").write_text('{"log_history": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="no eval loss"):
        sel.checkpoints(tmp_path)
    with pytest.raises(ValueError, match="no checkpoints"):
        sel.checkpoints(tmp_path / "missing")


def test_adapter_hash_covers_names_and_bytes(tmp_path):
    a = tmp_path / "a"
    a.mkdir()
    (a / "adapter_config.json").write_text("{}", encoding="utf-8")
    (a / "adapter_model.safetensors").write_bytes(b"w")
    h = sel.adapter_hash(a)
    assert h == sel.adapter_hash(a)  # stable
    (a / "adapter_model.safetensors").write_bytes(b"x")
    assert sel.adapter_hash(a) != h
    (a / "adapter_model.safetensors").write_bytes(b"w")
    (a / "adapter_model.safetensors").rename(a / "renamed.safetensors")
    assert sel.adapter_hash(a) != h
    with pytest.raises(ValueError, match="no files"):
        sel.adapter_hash(tmp_path / "empty_dir_that_is_missing")


def test_selection_record_keeps_every_tried_candidate():
    def m(c):
        return metrics(schema=0.5) if c.step == 500 else metrics()

    chosen, tried = sel.select([cand(500, 0.7), cand(527, 0.71)], m)
    rec = sel.selection_record(chosen, tried, "ab" * 32)
    assert rec["selected_step"] == 527 and rec["adapter_sha256"] == "ab" * 32
    assert [t["step"] for t in rec["tried"]] == [500, 527]
    assert DevMetrics.model_validate(rec["tried"][1]["metrics"]).schema_valid == 1.0
