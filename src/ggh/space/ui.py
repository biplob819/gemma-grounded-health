"""SPC-02..04: the Gradio page over `Space` (Tech Spec §9). Gradio is imported only here, and
only when the page is built, so the logic stays testable without it.

Default view: the six samples (no passcode). Secondary: paste a question and an abstract, enter
the passcode, optionally a visitor's own Anthropic or OpenAI key for the grounding check. The
key field is a password box whose value is passed to one request and never kept in state.
"""

from __future__ import annotations

from html import escape
from typing import Any

from ggh.release.links import md
from ggh.space.engine import BANNER, card_html, highlight
from ggh.space.service import Result, Space

CSS = """
.banner {background:#fff4d6;color:#4a3500;padding:8px 12px;border-radius:6px;font-weight:600}
.cited {background:#d9f2e3} .err {color:#a40000} .notice {font-size:0.85em;opacity:0.8}
.card {border:1px solid #8884;border-radius:8px;padding:8px 12px}
"""


def footer() -> str:
    return (
        "Runs on Hugging Face ZeroGPU, which has a free daily quota per visitor: if it runs out, "
        "the samples still work. The demo runs the merged bf16 weights; the benchmark scores the "
        f"Q4_K_M GGUF ({md('parity')}). {md('benchmark')} · {md('hf_model')} · "
        f"{md('hf_gguf')} · {md('repo')}. {BANNER}"
    )


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
        card_html("Base Gemma 4 E2B", base) if base else "",
        card_html("Fine-tuned (grounded)", ft) if ft else "",
    )


def build(space: Space, samples_labels: dict[str, str]) -> Any:  # pragma: no cover - gradio
    import gradio as gr  # pyright: ignore[reportMissingImports]

    with gr.Blocks(css=CSS, title="Gemma Grounded Health") as demo:
        gr.HTML(f"<div class='banner'>{BANNER}</div>")
        gr.Markdown("# Gemma Grounded Health: base against fine-tuned, from the abstract only")
        with gr.Tab("Samples"):
            pick = gr.Dropdown(
                choices=[(v, k) for k, v in samples_labels.items()],
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
        gr.Markdown(footer())

        def on_pick(sample_id: str) -> tuple[str, str, str, str]:
            return render(space.sample(sample_id))

        def on_go(
            qv: str, av: str, cv: str, kv: str, ck: bool, request: gr.Request
        ) -> tuple[Any, ...]:
            out = render(space.live(qv, av, cv, request.session_hash or "anon", kv or None, ck))
            return (*out, "", "")  # clear the passcode and key boxes after every request

        pick.change(on_pick, pick, [msg, abstract, base_card, ft_card])
        go.click(on_go, [q, a, code, key, check], [msg, abstract, base_card, ft_card, code, key])
    return demo
