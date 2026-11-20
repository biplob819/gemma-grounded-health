# Gemma Grounded Health

${header}

> **Research only, not medical advice.** The model summarises what a study abstract says. It is not guidance for anyone's diagnosis or treatment.

A small open model, Gemma 4 E2B with LoRA, fine-tuned to answer a health research question only from the abstract it is given: a yes, no or maybe label, a short answer and the numbers of the sentences it relied on. Its ${quant} GGUF runs offline on a laptop. Every model is scored with EvalKit on all ${n} expert-labelled PubMedQA questions.

![Base, v1 and v2 on the expert-labelled PubMedQA questions: correct and grounded, citation precision and aim-citation rate, each with its interval](../assets/headline_v2.svg)

## The result in one paragraph

The first fine-tune (v1) learned the label, raising accuracy from ${base_acc} to ${v1_acc}, but cited the study's aim instead of its results, so it fell below its own base model on "correct and fully grounded" (separated under GPT-6.1 Sol, a tie under Claude Opus 5.5). The retrain (v2) fixed the citations against v1 (citation precision ${d_v2v1_cp}; aim-citation rate ${d_v2v1_aim}). It did not move the headline against v1 (${d_v2v1_head}), and the base model is still better on it (${d_v2base_head}).

## Results

From the pre-registered v2 run, scored by ${v2_extractor} (claims) and ${v2_judge} (judge):

${table}

## What failed, and what changed

- **v1's citations pointed at the aim.** Under ${v2_judge}, ${v1_aim_v2run} of v1's answers cite an aim sentence that does not support the claim; base: ${base_aim_v2run}. Training targets had picked citations by word overlap with the author's conclusion, and the study's aim shares most of its words with the conclusion.
- **v2 cites what supports the answer.** Its targets cite only sentences that each support the answer on their own (OWNER D-43, D-48); its aim-citation rate is ${v2_aim_v2run}.
- **The headline did not follow.** Per answer, on average ${v2_grounded} of v2's claims are judged supported; many still go beyond the sentence they cite.

![The same answer from v1 and v2: v1 cites the study's aim sentence, v2 cites the result sentence](../assets/screenshots/v1_vs_v2_citation.png)

## Offline on a laptop

On an ${chip} the ${quant} file (${file_mb} MB) generates ${gen_tps} tokens a second and answers in ${latency} ms (medians), peaking at ${rss} MB of memory. Run it with Ollama, thinking off:

```bash
ollama run --think=false hf.co/${gguf_repo}:${quant}
```

The demo is a local app against the same benchmarked file, behind a passcode (OWNER D-63):

![The local demo: base and v2 answers side by side with the cited sentences highlighted, then a live answer refused with a wrong passcode](../assets/demo.gif)

## Caveats

${caveats}

## Read more

- [How I built it](how-i-built-it.md): the harness, the data, training, the v1 failure, the v2 retrain and every caveat.
- [Decisions that shaped the result](decisions-summary.md) and the [one-page PRD summary](prd-one-page.md).
- [Video script](video-script.md).

${links}
