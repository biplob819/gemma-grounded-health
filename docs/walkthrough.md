# Walkthrough GIF

`docs/assets/walkthrough.gif` is a 25-second, looping tour of ten real screens for the portfolio page and the README. Every screen is rendered from files committed in this repo: no API keys, no network calls, no spend. Numbers are read from the committed JSON when the GIF is built, never typed in. Since R4.5 it shows the shipped release (v2) by default.

Rebuild it (needs Google Chrome or Chromium and ffmpeg):

```sh
uv run python scripts/walkthrough.py                    # writes docs/assets/walkthrough.gif
uv run python scripts/walkthrough.py --html-only /tmp/w # the HTML pages only, no capture
uv run python scripts/walkthrough.py --summary docs/results/main/summary.json  # v1's run
```

Each screen is captured at 1280×800 with headless Chrome, held for 2.5 s, scaled to 960 px wide and encoded with a 128-colour palette (the ffmpeg line is in `FFMPEG_FILTER`). A page that would show a home path, the local username, an email address or anything key-like is refused before capture (the inlined screenshots are left out of that scan, since they are images, not text). The CLI screen runs with no credentials in the environment and with the repo path replaced by `~/gemma-grounded-health`. The audit page is served from a temporary copy of the committed sample, so nothing in the repo can be written while it is captured.

## Shot list

| # | Screen | Source |
| --- | --- | --- |
| 1 | Overview: tagline, then base, v1 and v2 on label accuracy, correct and fully grounded, cited sentences that support a claim and answers citing an unsupporting aim sentence; run id, questions, models, judge, spend, stack | `portfolio.yml`, `docs/results/main_v2/summary.json`, `docs/results/main_v2/run.json` |
| 2 | Results: the v2 run's three models with 95 % intervals, then the six pre-registered comparisons with their verdicts; the two v2-vs-base citation verdicts are marked as depending on the aim rule (OWNER D-62) | `docs/results/main_v2/summary.json`, `docs/results/results_bundle.json`, `docs/measurements/aim_rule_check.json` |
| 3 | What failed in v1: unsupported claims citing sentence 1, aim-citation rate and citation precision, one judged example | `docs/results/main/records.jsonl.gz`, `cases.jsonl.gz`, `docs/results/main_citation/summary.json` |
| 4 | The released model's Hugging Face page | `docs/assets/screenshots/hf_model_page.png` |
| 5 | The local demo answering a sample question, base against v2 | `docs/assets/screenshots/demo_answer.png` |
| 6 | Data flow diagram | `docs/assets/dataflow.svg` |
| 7 | Laptop benchmark: v1 and v2 GGUF variants, tokens/s, first token, answer tokens, latency, memory | `docs/measurements/v2/laptop_bench.json` |
| 8 | Checks: v2's Q4_K_M against bf16 parity (with the dev bar and whether it was met), and the paraphrase contamination probe | `docs/measurements/parity_headline_v2.json`, `docs/measurements/v2/parity.json`, `docs/measurements/probe_report_v2.json` |
| 9 | `ggh doctor` (no keys set) and `ggh audit score` (the pass bar on the committed labels) | live CLI output, `docs/measurements/audit_labels.jsonl` |
| 10 | The hand-audit page from `ggh audit serve` on 127.0.0.1, framed at 80 % | `docs/measurements/audit_sample.jsonl`, `audit_labels.jsonl` |

The architecture diagram (`docs/assets/architecture.svg`) and the `ggh --help` screen left the GIF in R4.5 to make room for the Hugging Face page and the demo; both stay on the portfolio page and in the repo.

## Another run

Pass a run's `summary.json` with `--summary`: screens 1 and 2 then show that run (the newest fine-tuned Gemma in the summary is treated as the fine-tune, and the comparisons come from the bundle's pre-registered set for that run, or from EvalKit's run-wide tests if the bundle does not carry it). Screen 3 stays on v1's main run on purpose, since it tells the v1 failure. Add a display name for a new model id in `DISPLAY` in `scripts/walkthrough.py`.
