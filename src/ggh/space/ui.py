"""SPC-02..04: the Gradio page over `Space` (Tech Spec §9). Gradio is imported only here, and
only when the page is built, so the logic stays testable without it.

Default view: the six samples (no passcode). Secondary: paste a question and an abstract, enter
the passcode, optionally a visitor's own Anthropic or OpenAI key for the grounding check. The
key field is a password box whose value is passed to one request and never kept in state.

The local app (OWNER D-63) says on screen that the benchmark numbers are for the Q4_K_M GGUF and
that it serves exactly that GGUF (D-14), and shows the shipped release's results from the
committed results bundle (`results_summary`), never typed.

No `from __future__ import annotations` here: Gradio finds the `gr.Request` parameter of a
handler by its evaluated annotation, and a string annotation hides it.
"""

from html import escape
from typing import Any

from ggh.release.card import outcome_table, scoring
from ggh.release.links import md
from ggh.release.readme import disclosures
from ggh.release.sources import Sources
from ggh.space.engine import BANNER, card_html, highlight
from ggh.space.service import Result, Space

SERVED_MODEL = "gemma-e2b-grounded-v2-q4"  # EvalKit's name for v2's Q4_K_M GGUF
TITLES = {"base": "Base Gemma 4 E2B (Q4_K_M)", "ft": "Fine-tuned v2 (Q4_K_M GGUF)"}

CSS = """
.banner {background:#fff4d6;color:#4a3500;padding:8px 12px;border-radius:6px;font-weight:600}
.cited {background:#d9f2e3} .err {color:#a40000} .notice {font-size:0.85em;opacity:0.8}
.card {border:1px solid #8884;border-radius:8px;padding:8px 12px}
"""


def served_note(model: str = SERVED_MODEL) -> str:
    return (
        "The benchmark numbers are for the Q4_K_M GGUF, and this demo serves exactly that GGUF "
        f"(`{model}`, through llama-server on this laptop)."
    )


def footer(local: bool = True, model: str = SERVED_MODEL) -> str:
    links = f"{md('benchmark')} · {md('hf_model')} · {md('hf_gguf')} · {md('repo')}"
    if local:  # OWNER D-63: the demo that ships
        return f"{served_note(model)} {links}. {BANNER}"
    return (
        "Runs on Hugging Face ZeroGPU, which has a free daily quota per visitor: if it runs out, "
        "the samples still work. The demo runs the merged bf16 weights; the benchmark scores the "
        f"Q4_K_M GGUF ({md('parity')}). {links}. {BANNER}"
    )


def results_summary(src: Sources, served_model: str = SERVED_MODEL) -> str:
    """The shipped release's benchmark results as Markdown, from the committed bundle (D-20).
    Refuses a bundle whose shipped model is not the one this demo serves."""
    if src.ft_model != served_model:
        raise ValueError(
            f"the bundle's shipped model is {src.ft_model}, but the demo serves {served_model}"
        )
    main, b = src.main(), src.bundle
    ref = src.ref(main)
    n = max((m.n for m in main.models), default=0)
    caveats = "; ".join(disclosures(src))
    lines = [
        f"### {b.shipped.version} results (from the committed results bundle)",
        "",
        f"{b.shipped.version} on {n} expert-labelled PubMedQA questions (EvalKit run "
        f"`{main.run_id}`, {scoring(ref)}, calibration: {b.shipped.calibration_status}). "
        f"{served_note(served_model)} 95% intervals in square brackets.",
        "",
        outcome_table(src, ref, bold=served_model),
    ]
    if caveats:
        lines += ["", f"Caveats: {caveats}."]
    return "\n".join(lines) + "\n"


def launch_kwargs() -> dict[str, Any]:  # pragma: no cover - gradio
    import gradio as gr  # pyright: ignore[reportMissingImports]

    return {"css": CSS} if int(gr.__version__.split(".")[0]) >= 6 else {}


def render(result: Result) -> tuple[str, str, str, str]:
    """(message, abstract with the fine-tune's citations marked, base card, ft card) as HTML."""
    base, ft = result.cards.get("base"), result.cards.get("ft")
    msg = f"<p>{escape(result.message)}</p>" if result.message else ""
    for name, text in result.grounding.items():
        msg += f"<p><b>{escape(name)}</b> grounding check: {escape(text)}</p>"
    shown = ft or base
    return (
        msg,
        highlight(result.sentences, shown) if shown else "",
        card_html(TITLES["base"], base) if base else "",
        card_html(TITLES["ft"], ft) if ft else "",
    )


def build(
    space: Space, samples_labels: dict[str, str], *, summary: str = "", local: bool = True
) -> Any:  # pragma: no cover - gradio
    import gradio as gr  # pyright: ignore[reportMissingImports]

    # Gradio 6 takes the page CSS in launch() (`launch_kwargs`); Gradio 5 in Blocks
    old = int(gr.__version__.split(".")[0]) < 6
    style: dict[str, Any] = {"css": CSS} if old else {}
    with gr.Blocks(title="Gemma Grounded Health", **style) as demo:
        gr.HTML(f"<div class='banner'>{BANNER}</div>")
        gr.Markdown("# Gemma Grounded Health: base against fine-tuned, from the abstract only")
        if local:
            gr.Markdown(served_note())
        with gr.Tab("Samples"):
            first = next(iter(samples_labels), None)
            pick = gr.Dropdown(
                choices=[(v, k) for k, v in samples_labels.items()],
                value=first,
                label="Sample question (precomputed, no passcode)",
            )
        with gr.Tab("Your abstract"):
            q = gr.Textbox(label="Question", max_lines=3)
            a = gr.Textbox(label="Abstract", lines=10)
            code = gr.Textbox(label="Passcode", type="password")
            key = gr.Textbox(
                label="Your Anthropic or OpenAI key (optional, grounding check, "
                "used for this request only)",
                type="password",
            )
            check = gr.Checkbox(label="Run the grounding check")
            go = gr.Button("Answer")
        msg = gr.HTML()
        abstract = gr.HTML()
        with gr.Row():
            base_card, ft_card = gr.HTML(), gr.HTML()
        if summary:
            with gr.Accordion("Benchmark results", open=False):
                gr.Markdown(summary)
        gr.Markdown(footer(local))

        def on_pick(sample_id: str) -> tuple[str, str, str, str]:
            return render(space.sample(sample_id))

        def on_go(
            qv: str, av: str, cv: str, kv: str, ck: bool, request: gr.Request
        ) -> tuple[Any, ...]:
            out = render(space.live(qv, av, cv, request.session_hash or "anon", kv or None, ck))
            return (*out, "", "")  # clear the passcode and key boxes after every request

        pick.change(on_pick, pick, [msg, abstract, base_card, ft_card], api_name="sample")
        # SPC-03: the first sample is shown on arrival, not only after a pick
        demo.load(on_pick, pick, [msg, abstract, base_card, ft_card])
        go.click(
            on_go,
            [q, a, code, key, check],
            [msg, abstract, base_card, ft_card, code, key],
            api_name="answer",
        )
    return demo
