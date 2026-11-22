"""RLM-03: the v3 plan's trial stop rules and its dev checkpoint-selection rule, applied to an
`rl_dev.json` (configs/eval_plan_v3.yaml, `trial` and `checkpoint_selection`).

Every guard compares a candidate with v2's replies scored in the same Kaggle session. Shares are
over all dev rows, as `ggh rl-dev` computes them. Nothing here reads a model or calls an API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

EPS = 1e-9  # the summaries are rounded to 4 places; a guard at its exact bound passes


@dataclass(frozen=True)
class Verdict:
    passed: bool
    checks: dict[str, bool]
    reported: dict[str, Any] = field(default_factory=dict)


def trial_verdict(rl_dev: dict[str, Any]) -> Verdict:
    """The trial's stop rules on its final adapter against v2. The gold-"no" accuracy guard is
    reported, not applied (the trial's 200 rows hold only about 21 gold "no" rows)."""
    a = rl_dev["adapters"]
    v2, f = a["v2"], a["final"]
    checks = {
        "parse_failures_at_most_0.02": f["unparsed"] <= 0.02 + EPS,
        "mean_reward_within_0.05_of_v2": f["mean_reward"] >= v2["mean_reward"] - 0.05 - EPS,
        "maybe_share_within_0.05_of_v2": abs(f["maybe_share"] - v2["maybe_share"]) <= 0.05 + EPS,
        "yes_share_at_most_v2_plus_0.05": f["yes_share"] <= v2["yes_share"] + 0.05 + EPS,
    }
    reported = {"no_accuracy": f["no_accuracy"], "v2_no_accuracy": v2["no_accuracy"]}
    return Verdict(all(checks.values()), checks, reported)


def guards(c: dict[str, Any], v2: dict[str, Any], no_slack: float = 0.05) -> dict[str, bool]:
    """The plan's seven hack guards for one candidate against v2. v3b's plan widens the gold-"no"
    slack to 0.10, because its 300 held-out dev rows hold only 21 gold "no" rows."""
    no_c, no_v2 = c["no_accuracy"], v2["no_accuracy"]
    return {
        "parse_failures_at_most_0.01": c["unparsed"] <= 0.01 + EPS,
        "maybe_share_within_0.05_of_v2": abs(c["maybe_share"] - v2["maybe_share"]) <= 0.05 + EPS,
        "yes_share_at_most_v2_plus_0.05": c["yes_share"] <= v2["yes_share"] + 0.05 + EPS,
        f"no_accuracy_at_least_v2_minus_{no_slack:.2f}": no_c is not None
        and no_v2 is not None
        and no_c >= no_v2 - no_slack - EPS,
        "mean_words_at_least_0.8_of_v2": c["mean_words"] >= 0.8 * v2["mean_words"] - EPS,
        "label_accuracy_at_least_v2_minus_0.02": c["label_accuracy"]
        >= v2["label_accuracy"] - 0.02 - EPS,
        "aim_cited_at_most_v2": c["aim_cited"] <= v2["aim_cited"] + EPS,
    }


def step_of(name: str, final_step: int) -> int:
    if name == "final":
        return final_step
    if name.startswith("checkpoint-"):
        return int(name.split("-", 1)[1])
    raise ValueError(f"{name!r} is not a candidate (checkpoint-N or final)")


@dataclass(frozen=True)
class Selection:
    selected: str | None  # None: no candidate keeps every guard
    beats_v2: bool  # the selected candidate's mean reward is above v2's
    table: list[dict[str, Any]]  # every candidate in step order, with its guards


def select_checkpoint(rl_dev: dict[str, Any], rows: int = 500, no_slack: float = 0.05) -> Selection:
    """The highest mean reward among candidates keeping every guard; ties go to the earlier step
    (the final adapter counts as the run's last step)."""
    a = rl_dev["adapters"]
    if "v2" not in a or len(a) < 2:
        raise ValueError("rl_dev.json needs v2 and at least one candidate")
    if rl_dev.get("dev_rows") != rows:
        raise ValueError(f"selection is on {rows} dev rows, not {rl_dev.get('dev_rows')}")
    v2, last = a["v2"], int(rl_dev["run_steps"])
    table = []
    for name, c in a.items():
        if name == "v2":
            continue
        g = guards(c, v2, no_slack)
        table.append(
            {"name": name, "step": step_of(name, last), "mean_reward": c["mean_reward"],
             "keeps_guards": all(g.values()), "guards": g}
        )  # fmt: skip
    table.sort(key=lambda r: r["step"])
    ok = [r for r in table if r["keeps_guards"]]
    if not ok:
        return Selection(None, False, table)
    best = max(r["mean_reward"] for r in ok)
    pick = next(r for r in ok if r["mean_reward"] >= best - EPS)  # earliest of the tied best
    return Selection(pick["name"], pick["mean_reward"] > v2["mean_reward"] + EPS, table)
