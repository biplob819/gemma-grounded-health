# v3 trial, attempt 2 (kernel `biplob819/ggh-v3-grpo-trial` version 2, repo at 84d5dc4)

The 40 GRPO steps completed (`report.json`, copied from the kernel output). The dev read then
failed: `ggh rl-dev` scored v2 and died loading the second model, because the first was still
held on the T4 (`cuda:0 has 5.51 GB free`). No trial adapter was scored on dev, so this attempt
makes no stop or go decision. It is kept as a training record only (D-68). The trial runs again,
in full, from the fixed commit.

The one dev line printed before the failure, v2 on the first 200 dev prompts:

    v2 {"n": 200, "mean_reward": 1.2975, "label_accuracy": 0.915, "no_accuracy": 0.8095,
        "n_gold_no": 21, "yes_share": 0.85, "maybe_share": 0.0, "grounded_paid": 0.8,
        "citations_pass": 0.88, "aim_cited": 0.035, "unparsed": 0.0,
        "label_mix": {"no": 30, "yes": 170}, "mean_words": 13.87, "max_words": 42}

Timings from the kernel log: about 11 minutes of setup (`ggh data` 7.7 min), 46.97 s per GRPO step
(31 min for 40 steps, peak 13.4 GB), and about 2.5 minutes to load and score one adapter on 200 dev
prompts.
