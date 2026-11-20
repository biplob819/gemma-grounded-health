"""OSS-06, SPC-04: every link the README, the Space footer and the release notes need (OWNER
D-37), in one place. A link that does not exist yet is `None` and renders as a visible
placeholder; R4.2 to R4.5 fill them in as each thing is published. The README link test fails
on any placeholder once the release is tagged.
"""

from __future__ import annotations

REPO = "https://github.com/biplob819/gemma-grounded-health"
PENDING = "(link pending)"

# name -> (label, url or None until published)
LINKS: dict[str, tuple[str, str | None]] = {
    "repo": ("GitHub repository", REPO),
    "release": ("GitHub release v1.0.0", None),  # R4.5
    "hf_model": (
        "Hugging Face model",
        "https://huggingface.co/biplob123/gemma-4-e2b-grounded-health",
    ),
    "hf_gguf": (
        "Hugging Face GGUF",
        "https://huggingface.co/biplob123/gemma-4-e2b-grounded-health-GGUF",
    ),
    "space": ("Demo (local, recorded; OWNER D-63)", "docs/assets/demo.gif"),
    "evalkit_report": ("EvalKit report page", None),  # R4.4
    "evalkit_pr": ("EvalKit upstream pull request", None),  # R4.4
    "parity": (
        "Parity (Q4_K_M against the merged weights)",
        f"{REPO}/blob/main/docs/measurements/v2/parity.json",
    ),
    "benchmark": ("Benchmark results", f"{REPO}/tree/main/docs/results"),
    "data_card": ("DATA_CARD.md", "DATA_CARD.md"),
    "model_card": ("MODEL_CARD.md", "MODEL_CARD.md"),
    "prd": ("PRD", "Gemma%20Grounded%20Health%20PRD.md"),
    "tech_spec": ("Tech Spec", "Gemma%20Grounded%20Health%20Tech%20Spec.md"),
    "bundle": ("Results bundle", "docs/results/results_bundle.json"),
    "citation": ("CITATION.cff", "CITATION.cff"),
    "writeup": ("Write-up", None),  # P5.1
    "case_study": ("Case study: the v1 grounding failure", "docs/case-study.md"),
}


def md(name: str) -> str:
    label, url = LINKS[name]
    return f"[{label}]({url})" if url else f"{label} {PENDING}"


def pending() -> list[str]:
    return [name for name, (_, url) in LINKS.items() if url is None]
