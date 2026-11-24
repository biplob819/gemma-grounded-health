# Sample read: the pipeline faults, explained (E3.3, RES-05, D-18)

The blind sample read (OWNER D-61: Claude Opus 5.5 under `configs/sample_read_prompt.md`, 20
records per model of `main` and `main_v2`, 140 in all) marked 12 records as possible pipeline
faults. Tech Spec §7.6 stops publication until each is explained. The builder then read the
records and EvalKit 0.4.0's code (`evalkit/pipeline/judge.py`, `extract.py`, `aggregate.py`) and
measured each pattern over the whole run. **None is a code or alignment bug**: every citation
verdict belongs to the claim and sentence it names (`parse_citations` stamps the claim id from
the call's own input and requires each cited sentence answered once; batch results are matched by
a hash that includes the claim id). They are judge and extractor behaviour. Two of them bear on a
published conclusion: the aim rule (the v2-vs-base citation results, disclosed under OWNER
D-62) and the label-only claims (v1 beating Luna under Sol would no longer hold). The aim-rule, extractor-drop and label-only
counts are in `docs/measurements/aim_rule_check.json`, written by `ggh results aim-check`; the
label-correct split of the extractor drops and the judge-error rate were computed from the same
records as defined in their sections. The blind batches the reviewers saw, their outputs and the
keys that map them back are in `docs/measurements/blind_reviews/`.

## aim-rule

Records: `main_v2` base pqa-21550158, pqa-15588538, pqa-24695920 (pqa-21550158's note also names
result sentences marked supporting for its scope claim: that half is the part-of-claim behaviour
below).

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

Records: `main_v2` base pqa-27928673, pqa-27615402 (and the second halves of pqa-21550158 and
pqa-15588538 above); `main_v3b` base pqa-23690198 (cached from `main_v2`).

EvalKit gives every claim the answer's full citation list (`ground_on_answer_citations`, EvalKit
D-52), and the rubric counts a sentence that supports "the part of the claim it is cited for".
So a result sentence can be marked supporting for a claim about the study's design or about
previous work. A sentence counts as supporting if it supports any claim, so this moves nothing
when the sentence also supports another claim of the answer (pqa-27615402: sentence 6 also
supports claim 1; pqa-21550158: sentences 11 and 13 also support claims 1 and 2). When it
supports no other claim, the mark is the only reason the sentence counts as supporting, and if it
is the lenient reading it raises that answer's citation precision in that model's favour:
pqa-27928673 (sentence 10, this study's odds ratios, marked supporting for "previous studies
have shown ..." and not for claim 1: 2/2 instead of 1/2) and pqa-15588538 (sentence 5, a method
sentence giving the patient groups, marked supporting only for the design claim: 3/6 instead of
2/6 if that mark is lenient; it is a defensible mark, since the sentence does describe the
design's groups). Both are base answers.

How often this happens over a whole run was not measured: it needs a reading of each mark. As an
upper gauge, the cited sentences counted as supporting that are not in the evidence of a
supported verdict for any claim they are marked supporting are 11.3% (base), 11.0% (v1) and
25.3% (v2) of cited sentences in `main_v2`; many are correct marks for claims rejected on other
grounds. Even if every one of them were wrong, citation precision (supporting cited sentences over
all cited sentences) would be 0.483 for v2 against 0.252 for v1 ((758 − 261)/1030 and
(718 − 219)/1982), so v2-vs-v1 holds; base would be 0.634, so v2-vs-base would favour base, which
the README already allows (OWNER D-62).

## extractor-drop

Records: `main` Haiku pqa-17598882, pqa-15141797; `main_v2` base pqa-11713724; `main_v3b` v3b-Q4_K_M pqa-27989969.

The extractor returned no claims for an answer with factual content. The record then has
coverage 0 and its headline counts as ungrounded. Measured over whole runs, in `main_v2` this
holds 3, 2 and 1 label-correct answers for base, v1 and v2 (at most 0.3 points of headline); in
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
in v1's results. Without these claims Luna's Sol headline would be 0.334 against v1's 0.330,
so the published v1-beats-Luna comparison under Sol (separated in the bundle) would no longer
hold: Luna would be 0.004 ahead on the point estimate, most likely a tie. The v1-vs-Luna
comparison under Opus is unaffected. No published text may claim v1 beats Luna under
Sol.

## judge-error

Records: `main_v2` v2 pqa-27131771; `main_v3b` v3b-Q8_0 pqa-25501465, v2 pqa-25787073 (cached from `main_v2`), base pqa-16249670 and pqa-20629769 (cached).

The claim judge marked unsupported a claim its cited sentence largely states (Sol's rationale: the
sentence reports 6 months only, the claim says "follow-up visits"), while the citation
check (a separate call) marked that sentence as supporting it. The two calls disagree in both
directions. In this record's direction, an unsupported verdict with a cited sentence the citation
check marks supporting for that claim, the loose count is 55% (base), 38% (v1) and 49% (v2) of
unsupported claims in `main_v2`, plausibly much of it the part-of-claim clause above (a sentence
supports part of a claim the claim judge rejects as a whole) rather than judge error. In the other direction, a
supported verdict all of whose evidence sentences are marked non-supporting for that claim, it is
4.7% (base), 5.9% (v1) and 2.3% (v2) of supported claims. How many of either are true claim-judge
errors, as here, was not measured. The headline uses the claim judge's verdicts, whose agreement
with the reference labeller is what the calibration reports (Sol held-out kappa 0.598 [0.468,
0.711]).

## presentation

v2's off-scope answers reached the B8 reviewers with their citations appended (" [n]") for 19 of
its 20 answers; no other model's answers had citations appended, though all have citations in
their records (GPT-6 Luna's own answer text cites inline as "[n]" in at least 6 of its 20). No
model name was shown, but v2's answers were easier to tell apart in the blind batches
(`docs/measurements/blind_reviews/b8_batch*.json`). All 20 were labelled no advice.

## role-label

Records: `main_v3b` v2 pqa-23690198 and pqa-22012962, base pqa-21952349 (all three cached from
`main_v2`).

The citation check labelled an opening background or hypothesis sentence with the role
"conclusion" (in pqa-23690198 the same sentence is labelled "background" in another claim's
check). Every one of these sentences is also marked non-supporting, and EvalKit counts a cited
sentence as an aim citation only when it supports no claim and its role is "aim", so a
background sentence labelled "conclusion" rather than "background" moves no published metric:
neither label is "aim". Judge behaviour, disclosed; not a code bug.

## v3b-sample

The V3.8 blind sample read of `main_v3b` (20 records per model: base, v2, v3b-Q4_K_M and
v3b-Q8_0, 80 in all; `docs/measurements/sample_read_main_v3b.jsonl`) marked 9 records as
possible pipeline faults. Six are base and v2 records, which `main_v3b` takes unchanged from
`main_v2`'s cache, so they are faults of that run's judging, read again here. They fall under
the patterns above: judge-error (v3b-Q8_0 pqa-25501465; v2 pqa-25787073; base pqa-16249670 and
pqa-20629769: the claim judge and the citation check disagree on a cited sentence), part-of-claim
(base pqa-23690198: a sentence marked supporting for an efficacy claim it does not state),
role-label (three records) and extractor-drop (v3b-Q4_K_M pqa-27989969). Measured over the whole
run, the extractor returned no claims for 4, 2, 3 and 1 yes/no answers of base, v2, v3b-Q4_K_M
and v3b-Q8_0, of which 2, 1, 2 and 1 had the right label (at most 0.2 points of headline each).
None is a code or alignment bug.
