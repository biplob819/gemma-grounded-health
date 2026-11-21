# V3.1 run state

**Now:** reward, recipe, trainer, dev check, notebook renderer and the v3 plan built and tested locally
**Next:** converge twice, merge, render the trial notebook at the merge commit, hand it to the portfolio session

## Log
- 2026-11-21 — The portfolio-site session relayed a v3 brief (GRPO, programmatic reward, $15 scoring cap, 2–3 days). Confirmed in this session's ask widget (OWNER D-66): build v3 here; the full-size run even if it slips past 2–3 days; prepare the Hugging Face release and ask before publishing; scoring on the OpenAI API, with the dollar cap to be asked again before the scoring run (the owner answered "Use open AI API" rather than approving $15).
- 2026-11-21 — Reward built. A first draft also required each cited sentence to report a result (TGT-R11's detector); it withheld the grounding credit from 459 of v2's 2,423 gold answers, whose cited sentences entail them, so it was dropped before the plan was hashed. Final reward on v2's 2,722 gold targets: 2,670 earn the maximum (1.5), 52 earn 1.0 (33 answers under 6 words, 17 citing a sentence sharing under 2 words, 2 with 4 citations). On v2's 500 Kaggle dev replies: mean 1.346; label accuracy 0.930; grounded 0.876; aim citation 0.044; 410 of 500 at the maximum.
