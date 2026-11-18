# Walkthrough GIF

`docs/assets/walkthrough.gif` is a 25-second, looping tour of ten real screens for the portfolio page. Every screen is rendered from files committed in this repo: no API keys, no network calls, no spend. Numbers are read from the committed JSON when the GIF is built, never typed in.

Rebuild it (needs Google Chrome or Chromium and ffmpeg):

```sh
uv run python scripts/walkthrough.py                    # writes docs/assets/walkthrough.gif
uv run python scripts/walkthrough.py --html-only /tmp/w # the HTML pages only, no capture
uv run python scripts/walkthrough.py --summary docs/results/<v2 run>/summary.json  # after v2
```

Each screen is captured at 1280×800 with headless Chrome, held for 2.5 s, scaled to 960 px wide and encoded with a 128-colour palette (the ffmpeg line is in `FFMPEG_FILTER`). A page that would show a home path, the local username, an email address or anything key-like is refused before capture. The CLI screens run with no credentials in the environment and with the repo path replaced by `~/gemma-grounded-health`. The audit page is served from a temporary copy of the committed sample, so nothing in the repo can be written while it is captured.

## Shot list

| # | Screen | Source |
| --- | --- | --- |
| 1 | Overview: tagline, label accuracy, correct and fully grounded, claims supported (base → fine-tuned), run id, questions, models, judges, spend, stack, next step | `portfolio.yml`, `docs/results/main/summary.json`, `docs/results/main/run.json` |
| 2 | Data flow diagram | `docs/assets/dataflow.svg` |
| 3 | Architecture diagram | `docs/assets/architecture.svg` |
| 4 | `ggh --help`: every pipeline command (usage line trimmed) | live CLI output |
| 5 | `ggh doctor` (no keys set) and `ggh audit score` (the pass bar on the committed labels) | live CLI output, `docs/measurements/audit_labels.jsonl` |
| 6 | The hand-audit page from `ggh audit serve` on 127.0.0.1, framed at 80 % | `docs/measurements/audit_sample.jsonl`, `audit_labels.jsonl` |
| 7 | Results: four models with 95 % intervals per judge, then fine-tuned minus base with separated / tie verdicts | `docs/results/main/summary.json` |
| 8 | What failed: unsupported claims citing sentence 1, aim-citation rate and citation precision, one judged example | `docs/results/main/records.jsonl.gz`, `cases.jsonl.gz`, `docs/results/main_citation/summary.json` |
| 9 | Laptop benchmark: four GGUF variants, tokens/s, first token, answer tokens, latency, memory | `docs/measurements/laptop_bench.json` |
| 10 | Checks: Q4_K_M against bf16 parity, and the paraphrase contamination probe | `docs/measurements/parity_headline.json`, `parity_report.json`, `probe_report.json` |

## After v2

Pass the v2 run's `summary.json` with `--summary`: screens 1 and 7 then show v2 against the base and the other models (the newest fine-tuned Gemma in the summary is treated as the fine-tune). Screen 8 stays on v1's main run on purpose, since it tells the v1 failure. Add a display name for the v2 model id in `DISPLAY` in `scripts/walkthrough.py` if it is not `gemma-e2b-grounded-v2-q4`.
