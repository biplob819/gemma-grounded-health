# E3.3 calibration prep (OWNER D-24, RES-03)

This file gets the owner's labelling ready. Nothing in it has been labelled or scored yet. It
needs no API calls and costs $0. Three points need an owner ruling before the `calibrated` stamp
can be green (see the open questions at the end).

## What D-24 asks, and where each part lives

| Part | Source | Judges scored | Size | API calls |
|---|---|---|---|---|
| EvalKit's `pubmedqa` calibration | EvalKit's published run `r-e8dac378d2d3` (`~/EvalKit/runs/`: 200 questions; Sonnet, Haiku, Sol and Luna answers; Haiku extracted the claims) | **Claude Opus 5.5 and GPT-6.1 Sol**, from their recorded verdicts | 100 answers = **461 claims** (233 held-out, 228 tuning), plus a 20-answer relabel = **88 claims** | **none** |
| 40 Gemma claims (the Gemma-family check) | GGH's V2.2 run `main_v2` (base, v1 and v2 answers; GPT-6 Luna extracted the claims) | **GPT-6.1 Sol**, its recorded verdicts (`main_v2` has no Opus verdicts, OWNER D-57) | 40 claims: 20 base, 10 v1, 10 v2 | **none** |

Facts checked on 2026-11-19:

- `evalkit calibrate score` compares the owner's labels with verdicts already in the run's
  `records.jsonl`. It makes no model calls. A dry run on a scratch copy of `r-e8dac378d2d3`,
  with keys unset and synthetic labels, scored both judges and flipped that copy's summary to
  `calibrated`. The real run folder was not touched.
- Every verdict in `r-e8dac378d2d3` (3,571 per judge), `main` and `main_v2` uses rubric hash
  `9a3a8246049f6f85`. That is the current `JUDGE_PROMPT_HASH` in both GGH's pinned EvalKit 0.4.0
  and the EvalKit repo's 0.4.1, so a calibration scored now counts for these runs.
- **Opus can be calibrated without Anthropic credit.** EvalKit's run already holds Opus's
  verdicts on every claim, so Opus's held-out kappa costs nothing. Opus can be part of the
  Gemma-family check only if the 40 claims come from a run Opus judged, which means v1's `main`
  run (base and v1 only, Haiku claims). Re-judging `main_v2`'s Luna claims with Opus would need
  Anthropic calls, which are not possible now. The upper bound is under $0.20 for 40 claims:
  `main` cost $53.68 for 12,388 claims, about $0.0043 per claim with everything included.
- EvalKit stamps calibration **per run folder**. Scoring `r-e8dac378d2d3` calibrates that folder
  only. No GGH run folder becomes calibrated by itself (question 1).

## What was built

- `src/ggh/results/family_check.py`: the 40-claim check.
  - `draw` takes a seeded sample (seed 24) and gives half the claims to the base model and
    splits the rest over the fine-tunes. It uses one claim per answer and one answer per
    question, so the owner never sees two models' claims on the same abstract. The order is
    shuffled.
  - `write_sheet` writes `docs/measurements/gemma_family_sheet.json`. This is the private join:
    it names models and claim ids and holds no verdicts. It refuses to redraw a sheet that
    already has labels.
  - `serve` opens **EvalKit's own labelling page** (`evalkit.pipeline.labeller`) on the sheet's
    blind rows.
  - `score` and `write_report` join the labels to the run's recorded verdicts. They write
    `docs/measurements/gemma_family_check.json`, a map from judge to agreement in the shape
    `ResultsBundle.gemma_family_check` reads (the path is already in `configs/results.yaml`).
    They also write `gemma_family_report.json`, which holds kappa with its interval, the
    confusion matrix, coverage and agreement per model.
  - `score` refuses partial labels, labels made for another sheet, a run whose id or claims
    changed since the draw, another rubric, and coverage under 95%.
- `src/ggh/cli_calibrate.py`: `ggh calibrate sample | label | score`.
- `tests/results/test_family_check.py`: 8 tests on the synthetic `main_v2`.

The real sheet is **not committed**, because `main_v2` is not committed yet and the sheet copies
its claim text. The draw is deterministic: run against the main checkout's `main_v2`
(`r-715fc213a8a5`), it gives sheet id `gf4e32c13cc2aa`, 40 distinct questions, 20/10/10. Commit
the sheet, labels, check and report together with `docs/results/main_v2/`.

## Owner steps (laptop)

The page runs on the laptop's browser only (question 4 covers the phone). Keys: **S**
supported, **U** unsupported, **E** extraction error (the claim misstates the answer),
**Backspace** back. Ask yourself: *do the cited sentences alone establish the claim?* Ignore
outside knowledge. Every label saves the moment you press a key, so you can close the tab and
resume any time with the same command.

1. **EvalKit sheet, first pass (about 1.5 to 3 h for 461 claims; EvalKit's guide quotes 1 to 2 h
   for about 300):**
   ```
   cd ~/EvalKit
   uv run evalkit calibrate export runs/r-e8dac378d2d3     # once, takes seconds; prints "461 claims to label"
   uv run evalkit calibrate label runs/r-e8dac378d2d3      # open the printed http://127.0.0.1:PORT/
   ```
   The builder did not run `export`. It writes into EvalKit's `runs/`, and each export draws a
   fresh secret, so whoever exports owns the sheet. Do not look at the judges' tuning or
   held-out agreement while labelling.
2. **Gemma claims (about 10 to 15 min for 40):**
   ```
   cd ~/"Gemma Grounded Health"            # the main checkout, on a branch with this commit
   uv run ggh calibrate sample             # writes docs/measurements/gemma_family_sheet.json (sheet gf4e32c13cc2aa)
   uv run ggh calibrate label              # open the printed http://127.0.0.1:PORT/
   ```
3. **Relabel, blind, ideally on another day (about 20 to 35 min for 88 claims):**
   `uv run evalkit calibrate label runs/r-e8dac378d2d3 --relabel` (in `~/EvalKit`).
4. **Score, once, with no API calls (seconds):**
   ```
   cd ~/EvalKit && uv run evalkit calibrate score runs/r-e8dac378d2d3
   cd ~/"Gemma Grounded Health" && uv run ggh calibrate score
   ```
   The first command prints the held-out kappa for Opus and Sol and writes
   `runs/r-e8dac378d2d3/calibration/calibration.json` (score once per rubric: EvalKit refuses a
   rescore under another rubric). The second prints Sol's agreement and kappa on the 40 claims.

Total owner time is about 2.5 to 4 h. The Tech Spec's 60 to 90 min estimate (§14) was made
before the sheet size was known.

## How the labels become the `calibrated` stamp, today

- `ggh results bundle` stamps `calibrated` only if **every** committed GGH run's `summary.json`
  says `calibrated` (`src/ggh/results/bundle.py`, the `calibrated = all(...)` line in `build`).
  It then requires `gemma_family_check.json` (schema validator in `src/ggh/schemas/release.py`).
- EvalKit writes `calibrated` into a run's summary only when that run folder's own
  `calibration/calibration.json` has held-out agreement for every judge of the run, on the
  run's **task name** and rubric (`calibration_status_of` in `evalkit/pipeline/aggregate.py`).
- The README's calibration words come from `main_v2`'s summary stamp lines
  (`src/ggh/release/readme.py`, `results_block`), not from the bundle.
- So after all the labelling above, the bundle **still says uncalibrated**. EvalKit's score
  marks only `r-e8dac378d2d3`. In addition, GGH's off-scope and probe runs have their own task
  names (`ggh_offscope`, `ggh_probe_original`, `ggh_probe_paraphrase`), which a `pubmedqa`
  calibration can never cover under EvalKit's rule. Question 1 is the fix.

## Cost

Every step above costs **$0**: no extraction, no judging, no generation. The only paid option
anywhere is Opus re-judging `main_v2`'s 40 claims (under $0.20). It is blocked by the Anthropic
credit and is not needed if question 2 is answered either way below.

## Open questions for the owner

1. **How the GGH stamp reads the calibration.** Pick one:
   - (a) Commit a copy of EvalKit's `calibration.json` as
     `docs/measurements/evalkit_pubmedqa_calibration.json`. `bundle.py` then stamps
     `calibrated` when that report is for `pubmedqa`, under each run's rubric hash, with
     held-out agreement for every judge the `pubmedqa` runs used, and the Gemma-family check
     exists. This is a small code change and leaves every committed summary unchanged.
   - (b) Copy `calibration.json` into each GGH `pubmedqa` run folder and re-summarise with
     EvalKit. This is EvalKit-native and fixes the README stamp too, but it rewrites committed
     `summary.json` files (v1's published ones included; the numbers stay the same, the hashes
     change).
   - Either way, decide whether the off-scope and probe runs (other task names) are exempt or
     keep the bundle uncalibrated.
2. **Opus in the Gemma-family check.** Pick one:
   - Sol only, from `main_v2` as built. This is consistent with D-57 for v2, and Opus is
     calibrated on EvalKit's run but not checked on Gemma claims.
   - Also draw 40 claims from v1's `main` (`ggh calibrate sample --run main`, $0, about 10 to
     15 min more labelling). Opus and Sol would then both be checked on Gemma (base and v1)
     claims for v1's published numbers. The sample command supports this, but it would need a
     second sheet path.
3. **A kappa bar.** EvalKit's groundedness calibration certifies on *any* held-out kappa,
   however low. EvalKit's own criterion calibration uses a 95% lower bound of at least 0.4.
   Should GGH's stamp require a bar for Sol (and Opus), and what should be disclosed if Sol
   falls under it?
4. **Phone labelling.** EvalKit's page cannot be used from a phone:
   - It binds to 127.0.0.1 and refuses any other Host header.
   - It has keys only, with no tap buttons.

   Making it work on a phone is an EvalKit change (OWNER D-44): tap buttons, plus an opt-in LAN
   bind with a one-time token. The only phone-friendly route today is the first pass of the
   EvalKit sheet. You can fill `runs/r-e8dac378d2d3/calibration/sheet.csv` (S or U in the
   `label` column) in a spreadsheet app, then run
   `uv run evalkit calibrate import runs/r-e8dac378d2d3 FILE.csv`. Import refuses if the page
   has already saved labels. The relabel and the 40 Gemma claims stay on the laptop page.
   Should the builder add phone support to EvalKit first?
5. **The 20/10/10 split.** Tech Spec §7.5 says 20 base and 20 fine-tuned, so v1 and v2 get 10
   each. 14/13/13 (`--allocation`) would weigh the three models equally. The default follows
   the spec.
