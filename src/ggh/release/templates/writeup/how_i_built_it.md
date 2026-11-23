# How I built Gemma Grounded Health

${header}

> **Research only, not medical advice.** The model summarises what a study abstract says. It is not guidance for anyone's diagnosis or treatment.

Every number in it is filled in by `ggh writeup` from the [results bundle](../results/results_bundle.json) or a committed evidence file, never typed (D-20).

## The problem

A health feature needs a model that answers only from its source and shows where each claim came from; small open models answer from memory.

The model gets a research question and one study's abstract, split into numbered sentences. It answers as JSON: a yes, no or maybe label, a short answer, and the numbers of the sentences that support it. The test set is PubMedQA's expert-labelled set, all ${n} questions, kept out of training (a leakage check removes any overlap) and scored in full (OWNER D-17). The model is Gemma 4 E2B, tuned with LoRA; its ${quant} file runs offline on a laptop.

The score that matters is "correct and fully grounded": the label is right and every claim is supported by the sentences the answer cites. I call it the headline.

## The harness

I set the goals, wrote the PRD and made the product calls. Claude Code wrote the code, under the same lean Kova harness that built EvalKit (D-11):

- **A build graph** of ${nodes} nodes; a node starts only when the nodes it depends on are green.
- **A capability catalog** of ${catalog_rows} rows, each with proof cells naming the tests that prove it.
- **Converge gates**: ${gates} automated checks per node, from lint and tests to the secret scan and the records.
- **A skeptical verifier**: a fresh agent that tries to break each node. After the harness node took many rounds, rounds were capped (OWNER D-33); what is left goes into the known-issues file.

Across ${verified_nodes} nodes the verifiers ran ${verifier_rounds} rounds; the harness node alone took ${max_rounds}. Any node that touches real data or a real model also reads real outputs before it can turn green (D-18).

## Data and targets

Training data is PubMedQA's large artificial set; a leakage check against the expert set dropped ${leak_drops} rows. Prompts are rendered by EvalKit's own PubMedQA template, so the benchmark measures grounding, not a new prompt format (D-2).

The targets were the hard part. The abstracts come without their conclusion, and the raw conclusions often say things the remaining sentences never say, including advice. So each target is the author's conclusion, filtered sentence by sentence for support in the cited sentences, with recommendations removed (D-3). The artificial set has only yes and no labels, so ${v1_insufficient} of v1's ${v1_train} training examples teach "maybe" by removing the evidence (OWNER D-15). Before training, I read ${v1_audit_n} examples by hand against a pass bar, and the audit ${v1_audit}.

## Training on Kaggle

Training ran on one free Kaggle ${gpu} with Unsloth, from a thin notebook that pins the repo commit and checks the data's hash. v1 trained for ${v1_steps} steps in ${v1_gpu_hours} GPU hours, peaking at ${v1_vram} GB. Checkpoints are chosen on the dev split only; the expert set never chooses anything (D-4). On ${v1_dev_n} dev prompts v1 had schema validity ${v1_schema} and citation validity ${v1_cites}.

## Export and the laptop

The adapter is merged and exported to GGUF with a pinned llama.cpp build. The benchmark scores the shipped ${quant} file, served by llama-server, for base and fine-tune alike (D-5).

Serving turned up a surprise: with a JSON schema, Gemma 4's chat grammar still allowed a hidden thinking channel, and the fine-tune opened it, which changed its labels. llama-server now runs with the reasoning budget at zero (D-41). Ollama showed the same thing: Gemma 4 thinks first unless run with `--think=false`, and the first check of the published file, thinking on, did not return valid JSON. The README and the model card give the command with thinking off.

On an ${chip} (${ram} GB) the v2 file (${file_mb} MB) generates ${gen_tps} tokens a second and answers in ${latency} ms (medians), peaking at ${rss} MB. v1's file generated ${v1_gen_tps} tokens a second in the same session at about the same llama-bench speed (${v1_tg128} against ${tg128}), a gap logged as a known issue; the v2 bench ran on a busy machine, and a quiet re-run is still to do.

## Evaluation with EvalKit

Every benchmark number comes from EvalKit, my open-source grounded-answer benchmark. One model splits each answer into claims; a judge rules on each claim using only the sentences the answer cites. Every metric gets a bootstrap interval, comparisons are paired and Holm-corrected, and a gap is "separated" only when its Holm-corrected interval excludes zero. The comparisons are hashed before each run.

v1's main run compared ${v1_models} models (base, v1, Claude Haiku 5.5 and GPT-6 Luna), judged by ${v1_judges}, for ${v1_spend}. v1's and v2's paid runs cost ${total_spend} in all. Against the frontier models, v1 was level with Claude Haiku 5.5 under Claude Opus 5.5 and ahead of it under GPT-6.1 Sol, and level with GPT-6 Luna under Claude Opus 5.5 (the Sol comparison with Luna is not claimed, see below).

## v1: the failure

v1 learned the label: accuracy rose from ${base_acc} to ${v1_acc} (${v1_acc_verdict}). But on the headline it fell below its own base: ${v1_head_opus} against ${base_head_opus} under Claude Opus 5.5 (${d_v1base_opus}) and ${v1_head_sol} against ${base_head_sol} under GPT-6.1 Sol (${d_v1base_sol}).

The judges' reasons showed why: v1 often cited the abstract's first sentence, which usually states the aim, not a result. A per-sentence citation check I added to EvalKit (OWNER D-44) measured it under GPT-6.1 Sol on v1's run: ${v1_aim} of v1's answers cite an aim sentence that does not support the claim (base ${base_aim_v1run}), and ${v1_cp} of the sentences it cites support a claim (base ${base_cp_v1run}).

The cause was the training data. Citation targets were picked by word overlap with the author's conclusion, and an aim sentence ("Does X improve Y?") shares most of its words with the conclusion ("X improves Y"). The hand audit judged each cited set as a whole, so it passed; the model learned to lead with the aim. The dev checks passed too, because a valid citation only meant the sentence existed.

## v2: the retrain

I chose to fix it, labelling v2 as built after v1's test results were seen (OWNER D-43). v2's targets cite only sentences that each support the answer on their own, ranked by an entailment model instead of word overlap (OWNER D-48). The new targets went through ${v2_audits} per-sentence audits (the first mine, the later ones by a blind Claude reviewer, OWNER D-47) and ${v2_audit_result}; v2 was trained anyway, with the miss disclosed (OWNER D-50), on ${v2_train} examples for ${v2_steps} steps in ${v2_gpu_hours} GPU hours, and checked on dev against bars fixed beforehand (D-52).

A second pre-registered run followed on the same ${n} questions. Base and v1 answers were re-scored from EvalKit's cache, so every comparison shares one extractor and one judge. Across ${family} pre-registered tests, Holm-corrected:

- **Against v1, the citations are fixed:** citation precision ${d_v2v1_cp}; aim-citation rate ${d_v2v1_aim}.
- **The headline did not move:** ${v2_head_v2run} against v1's ${v1_head_v2run} (${d_v2v1_head}).
- **Base is still better on the headline:** ${base_head_v2run} against v2's ${v2_head_v2run} (${d_v2base_head}).

On average per answer, only ${v2_grounded} of v2's claims are judged supported: many still go beyond the sentence they cite.

## The caveats

- **One judge, one provider (OWNER D-57).** The Anthropic credit ran out during the v2 dev check, so the v2 benchmark ran on OpenAI only: ${v2_extractor} extracted claims and ${v2_judge} judged them. Claude Opus 5.5 never judged v2 on the test set, and v1's published numbers are not compared with v2's.
- **Calibrated against a model, not a person (OWNER D-59, D-60).** I am not a domain expert, so Claude Opus 5.5 made the calibration labels, blind. GPT-6.1 Sol agrees with that reference labeller at kappa ${sol_kappa} on ${cal_n} held-out claims; the lower bound clears ${cal_bar}, so Sol is stamped ${sol_status}, worded as agreement with a model labeller. Opus agreeing with its own labels (kappa ${opus_kappa}) is never stamped.
- **Model reviews, not mine (OWNER D-61).** The off-scope advice check (B8) and the sample reads (B5) are blind Claude Opus 5.5 reviews under committed prompts. They found advice in ${b8_v1} of v1's ${b8_n} off-scope answers and ${b8_v2} of v2's. The sample read flagged ${faults} of ${read} records as possible pipeline faults; each was traced, and none was a code bug.
- **The aim rule (OWNER D-62).** The citation check marks an aim sentence as unsupporting even when the claim only restates the aim, which mostly hits base. The pre-registered numbers stand (aim-citation rate: base ${base_aim_v2run}, v2 ${v2_aim_v2run}). A post-hoc count that treats those sentences as supporting gives base ${base_aim_corrected} and citation precision ${base_cp_corrected}, against v2's ${v2_aim_corrected} and ${v2_cp_corrected}, so base is better on ${aim_worse}. v2 is therefore not claimed to beat base on citations; v2 against v1 holds under either count.
- **A local demo (OWNER D-63).** A ZeroGPU Space would have cost money, so the demo is the same app on my laptop against the benchmarked ${quant} file, behind a passcode, recorded as screenshots and a GIF.
- **Dev-check and parity misses (OWNER D-55).** On ${dev_n} dev questions v2 passed both citation bars (citation precision ${dev_cp}, bar ${bar_cp}; aim-citation rate ${dev_aim}, bar ${bar_aim}) but its headline was ${dev_head} against v1's ${dev_v1_head}, and its label accuracy ${dev_acc} missed the ${bar_acc} bar. The ${quant} file agrees with the merged weights on the label for ${parity} of ${parity_n} dev prompts, short of its bar (Q8_0: ${q8_parity}). These misses were disclosed, not tuned away.

## What the verifiers caught

These findings would have shipped something untrue:

- The first training pilot had silently loaded a mirror of the base model, not the pinned checkpoint (T2.1).
- Paid runs were missing from the spend ledger (E3.2).
- The v2 dev record recorded the last target audit as passed (V2.1).
- A "matches base" claim and wrong explanations on the pipeline-faults page (E3.3).
- The README said the demo serves the merged weights, not the GGUF (R4.5).

A verifier also found that Luna's bare one-word answers became claims Sol marked unsupported, so v1's Sol win over Luna is not claimed.

## What I took away

A citation that exists is not a citation that supports. Audit the unit the model will copy. Pre-register, then live with the result: v2 fixed the citations but not the headline. And say who did the work: model labels and model reviews are named as such wherever they appear.

## Links

The release is tagged [${tag}](${release_url}); the EvalKit contribution was ${evalkit_pr_state} as [a pull request](${evalkit_pr}) with a [report page](${evalkit_page}).

${links}
