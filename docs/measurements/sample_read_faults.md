# Sample read: the pipeline faults, explained (E3.3, RES-05, D-18)

The blind sample read (OWNER D-61: Claude Opus 5.5 under `configs/sample_read_prompt.md`, 20
records per model of `main` and `main_v2`, 140 in all) marked 12 records as possible pipeline
faults. Tech Spec §7.6 stops publication until each is explained. The builder then read the
records and EvalKit 0.4.0's code (`evalkit/pipeline/judge.py`, `extract.py`, `aggregate.py`) and
measured each pattern over the whole run. **None is a code or alignment bug**: every citation
verdict belongs to the claim and sentence it names (`parse_citations` stamps the claim id from
the call's own input and requires each cited sentence answered once; batch results are matched by
a hash that includes the claim id). They are judge and extractor behaviour. One of them changes a
published conclusion, disclosed under OWNER D-62. The counts below are in
`docs/measurements/aim_rule_check.json`, written by `ggh results aim-check`.

## aim-rule

Records: `main_v2` base pqa-21550158, pqa-15588538, pqa-24695920.

EvalKit's citation rubric (hash `be34ebe577bc96ae`) says a sentence that only states an aim
"does not establish" a claim. GPT-6.1 Sol applies this even when the claim only restates the
aim ("The investigation assessed the effect of PRP gel on ..."), so it marks the aim sentence as
not supporting, while the claim judge (a separate call) cites the same sentence as its evidence.
EvalKit counts a cited aim sentence that supports no claim as an aim citation, so this raises the
aim-citation rate and lowers citation precision of a model that restates aims. That is mostly the
base model. The post-hoc count in `aim_rule_check.json` (aim sentences the claim judge used for a
supported aim- or design-restating claim treated as supporting) moves base, not v1 or v2: the
v2-vs-v1 citation results hold, the v2-vs-base ones reverse (base better on both). OWNER D-62:
the pre-registered numbers stay published, and every place that compares v2 with base on
citations says this.

## part-of-claim

Records: `main_v2` base pqa-27928673, pqa-27615402.

EvalKit gives every claim the answer's full citation list (`ground_on_answer_citations`, EvalKit
D-52), and the rubric counts a sentence that supports "the part of the claim it is cited for".
So a result sentence can be marked supporting for a claim about the study's design or about
previous work. Because a sentence counts as supporting if it supports any claim, and these
sentences support another claim of the same answer, the metrics do not move. Judge behaviour,
no change to a published number.

## extractor-drop

Records: `main` Haiku pqa-17598882, pqa-15141797; `main_v2` base pqa-11713724.

The extractor returned no claims for an answer with factual content. The record then has
coverage 0 and its headline counts as ungrounded. Measured over whole runs, in `main_v2` this
holds 3, 2 and 1 label-correct answers for base, v1 and v2 (under 0.3 points of headline); in
v1's `main` run, 10 for base, 6 for v1, 26 for Haiku and 16 for Luna (Haiku extracted those
claims). Extractor behaviour, disclosed; the published headlines count these answers as
ungrounded, which slightly understates Haiku and Luna in v1's table.

## extractor-split

Records: `main` base pqa-10732884; `main_v2` v1 pqa-18388848.

The extractor dropped a qualifier the answer had, or merged two outcomes into one claim, and the
judge ruled on the altered claim. Extractor noise at the level of single claims; not measurable
automatically, not a pattern.

## label-only

Record: `main` Luna pqa-23870157.

Haiku's extraction in v1's `main` run sometimes made a bare label ("Yes.") a claim. Claude Opus
5.5 marks it supported, GPT-6.1 Sol unsupported ("no specified proposition"). It occurs only for
the comparator models (Luna 70 claims, Haiku 9) and costs Luna's Sol headline about 4.4 points
(44 answers) and Haiku's about 0.5. It does not touch the Gemma models or `main_v2`. Disclosed
in v1's results; the v1-vs-Luna comparison under Sol is affected, the v1-vs-Luna comparison
under Opus is not.

## judge-error

Record: `main_v2` v2 pqa-27131771.

The claim judge marked unsupported a claim its cited sentence plainly states, while the citation
check (a separate call) marked that sentence as supporting it. Measured over `main_v2`, a
supported verdict whose evidence the citation check marks non-supporting for that claim happens
for 2 to 6% of supported claims per model, and the mirror case occurs too. Judge noise, within
what the calibration (Sol held-out kappa 0.598 [0.468, 0.711]) already reports.
