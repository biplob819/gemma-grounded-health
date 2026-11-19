"""Hugging Face Space entry point (R4.3, Tech Spec §9). All logic lives in `ggh.space`; the
Space repo holds this file, requirements.txt and samples.json.

Space secrets: GGH_SPACE_PASSKEY (the salted hash from `ggh passkey set`, never the passcode)
and GGH_SPACE_FT_MODEL (the released model's Hub id).
"""

from pathlib import Path

from ggh.space.app import hf_space

demo = hf_space(Path(__file__).resolve().parent / "samples.json")

if __name__ == "__main__":
    demo.launch()
