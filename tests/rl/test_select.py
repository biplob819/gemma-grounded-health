"""RLM-01, RLM-03: the v3 plan's trial stop rules and dev checkpoint selection."""

import copy
import json
from pathlib import Path

import pytest

from ggh.rl.select import guards, select_checkpoint, step_of, trial_verdict

ROOT = Path(__file__).resolve().parents[2]
TRIAL = ROOT / "docs/measurements/v3/trial"

V2 = {
    "n": 500, "mean_reward": 1.30, "label_accuracy": 0.91, "no_accuracy": 0.80,
    "yes_share": 0.85, "maybe_share": 0.0, "aim_cited": 0.04, "unparsed": 0.0,
    "mean_words": 14.0,
}  # fmt: skip


def _cand(**kw):
    c = dict(V2)
    c.update(kw)
    return c


def _rl(cands, steps=679, rows=500):
    return {"run_steps": steps, "dev_rows": rows, "adapters": {"v2": V2, **cands}}


def test_the_committed_trial_passes_every_stop_rule():
    rl = json.loads((TRIAL / "rl_dev.json").read_text(encoding="utf-8"))
    v = trial_verdict(rl)
    assert v.passed and all(v.checks.values()) and len(v.checks) == 4
    assert rl["run_steps"] == 40 and rl["dev_rows"] == 200
    assert list(rl["adapters"]) == ["v2", "checkpoint-40", "final"]
    assert v.reported == {"no_accuracy": 0.8095, "v2_no_accuracy": 0.8095}


def test_the_trial_summaries_recompute_from_the_committed_replies():
    from ggh.rl.grpo import dev_summary

    dev = ROOT / "artifacts/data/dev_eval.jsonl"
    rl = json.loads((TRIAL / "rl_dev.json").read_text(encoding="utf-8"))
    for name, summary in rl["adapters"].items():
        assert json.loads((TRIAL / f"dev_{name}.json").read_text(encoding="utf-8")) == summary
        lines = (TRIAL / f"dev_{name}.jsonl").read_text(encoding="utf-8").splitlines()
        replies = [json.loads(x) for x in lines]
        assert len(replies) == 200
        if dev.is_file():  # built by `ggh data`, not in CI
            rows = [json.loads(x) for x in dev.read_text(encoding="utf-8").splitlines()][:200]
            assert [r["pmid"] for r in replies] == [r["pmid"] for r in rows]
            assert dev_summary([r["reply"] for r in replies], rows) == summary


def test_each_trial_stop_rule_can_stop_the_plan():
    rl = json.loads((TRIAL / "rl_dev.json").read_text(encoding="utf-8"))
    for key, value in (("unparsed", 0.03), ("mean_reward", 1.2), ("maybe_share", 0.06),
                       ("yes_share", 0.91)):  # fmt: skip
        bad = copy.deepcopy(rl)
        bad["adapters"]["final"][key] = value
        assert not trial_verdict(bad).passed, key


def test_selection_takes_the_best_reward_that_keeps_every_guard():
    rl = _rl({
        "checkpoint-50": _cand(mean_reward=1.32),
        "checkpoint-100": _cand(mean_reward=1.40, maybe_share=0.08),  # maybe drift: out
        "checkpoint-150": _cand(mean_reward=1.35),
        "final": _cand(mean_reward=1.35),
    })  # fmt: skip
    s = select_checkpoint(rl)
    assert s.selected == "checkpoint-150" and s.beats_v2  # tie with final goes to the earlier
    assert [r["name"] for r in s.table] == ["checkpoint-50", "checkpoint-100", "checkpoint-150",
                                            "final"]  # fmt: skip
    assert not s.table[1]["keeps_guards"]


def test_every_guard_excludes_a_hacking_candidate():
    hacks = {
        "parse_failures_at_most_0.01": {"unparsed": 0.02},
        "maybe_share_within_0.05_of_v2": {"maybe_share": 0.06},
        "yes_share_at_most_v2_plus_0.05": {"yes_share": 0.91},
        "no_accuracy_at_least_v2_minus_0.05": {"no_accuracy": 0.74},
        "mean_words_at_least_0.8_of_v2": {"mean_words": 11.1},
        "label_accuracy_at_least_v2_minus_0.02": {"label_accuracy": 0.88},
        "aim_cited_at_most_v2": {"aim_cited": 0.05},
    }
    assert set(hacks) == set(guards(V2, V2)) and all(guards(V2, V2).values())
    for name, change in hacks.items():
        g = guards(_cand(**change), V2)
        assert not g[name] and sum(not x for x in g.values()) == 1, name


def test_no_candidate_or_none_above_v2_is_reported_not_hidden():
    s = select_checkpoint(_rl({"final": _cand(unparsed=0.05)}))
    assert s.selected is None and not s.beats_v2
    s = select_checkpoint(_rl({"checkpoint-50": _cand(mean_reward=1.29), "final": _cand()}))
    assert s.selected == "final" and not s.beats_v2  # equal to v2 is not above it


def test_selection_is_on_500_dev_rows_and_named_candidates():
    with pytest.raises(ValueError, match="500"):
        select_checkpoint(_rl({"final": _cand()}, rows=200))
    with pytest.raises(ValueError):
        select_checkpoint({"run_steps": 1, "dev_rows": 500, "adapters": {"v2": V2}})
    assert step_of("final", 679) == 679 and step_of("checkpoint-650", 679) == 650
    with pytest.raises(ValueError):
        step_of("v3", 679)


def test_v3b_selects_on_300_held_out_rows_with_a_wider_no_slack():
    rl = _rl({"final": _cand(no_accuracy=0.72)}, rows=300)  # two of 21 gold-"no" rows fewer
    with pytest.raises(ValueError, match="300"):
        select_checkpoint(rl)  # v3's default stays 500 rows
    s = select_checkpoint(rl, rows=300, no_slack=0.10)
    assert s.selected == "final" and "no_accuracy_at_least_v2_minus_0.10" in s.table[0]["guards"]
    assert select_checkpoint(_rl({"final": _cand(no_accuracy=0.72)})).selected is None
