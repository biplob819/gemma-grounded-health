"""Record real terminal commands and render them as a looping GIF (no vhs/asciinema needed).

    uv run python scripts/termgif.py record steps.json cast.json
    uv run python scripts/termgif.py render cast.json out.gif --title "..."

`steps.json` is a list of {"show": "<command as displayed>", "run": "<shell command>"} or
{"comment": "<text>"}. Each command really runs; its output is captured with timestamps so
the GIF replays what happened. Long silent waits are shortened to `--max-gap` seconds and the
title says so, so time is never presented as faster than it was without a note.
"""

from __future__ import annotations

import json
import os
import selectors
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT = "/System/Library/Fonts/Menlo.ttc"
BG, FG, DIM, PROMPT, TITLE_BG = (
    (24, 26, 33),
    (220, 223, 228),
    (128, 136, 150),
    (110, 200, 140),
    (40, 43, 54),
)


def record(steps_path: str, cast_path: str) -> None:
    steps = json.loads(Path(steps_path).read_text(encoding="utf-8"))
    events: list[dict] = []
    for step in steps:
        if "comment" in step:
            events.append({"kind": "comment", "text": step["comment"]})
            continue
        events.append({"kind": "cmd", "text": step["show"]})
        t0 = time.monotonic()
        proc = subprocess.Popen(
            step["run"], shell=True, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            cwd=step.get("cwd"), env={**os.environ, "NO_COLOR": "1", "TERM": "dumb"},
        )  # fmt: skip
        assert proc.stdout is not None
        out = proc.stdout
        sel = selectors.DefaultSelector()
        sel.register(out, selectors.EVENT_READ)
        while True:
            if not sel.select(timeout=0.5):
                if proc.poll() is not None:
                    break
                continue
            chunk = os.read(out.fileno(), 4096)
            if not chunk:
                break
            events.append({"kind": "out", "t": time.monotonic() - t0,
                           "text": chunk.decode("utf-8", "replace")})  # fmt: skip
        proc.wait()
        events.append({"kind": "exit", "t": time.monotonic() - t0, "code": proc.returncode})
        print(
            f"{step['show'][:60]}: exit {proc.returncode} in {time.monotonic() - t0:.1f}s",
            flush=True,
        )
    Path(cast_path).write_text(json.dumps(events, indent=1), encoding="utf-8")


def _clean(text: str) -> str:
    out, i = [], 0
    while i < len(text):  # drop ANSI escapes and spinner carriage returns
        if text[i] == "\x1b":
            j = i + 1
            if j < len(text) and text[j] == "[":
                j += 1
                while j < len(text) and not (0x40 <= ord(text[j]) <= 0x7E):
                    j += 1
            i = j + 1
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


def render(cast_path: str, gif_path: str, title: str, cols: int, rows: int, max_gap: float) -> None:
    events = json.loads(Path(cast_path).read_text(encoding="utf-8"))
    font = ImageFont.truetype(FONT, 15)
    cw = int(font.getbbox("M")[2])
    lh, pad, th = 20, 14, 30
    w, h = cols * cw + 2 * pad, rows * lh + 2 * pad + th
    lines: list[tuple[str, tuple]] = [("", FG)]
    frames: list[tuple[Image.Image, int]] = []

    def put(text: str, color=FG) -> None:
        for ch in _clean(text):
            if ch == "\n":
                lines.append(("", color))
            elif ch == "\r":
                lines[-1] = ("", color)
            else:
                cur, _ = lines[-1]
                if len(cur) >= cols:
                    lines.append((ch, color))
                else:
                    lines[-1] = (cur + ch, color)

    def snap(ms: int) -> None:
        img = Image.new("RGB", (w, h), BG)
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, w, th], fill=TITLE_BG)
        for k, c in enumerate([(237, 106, 94), (245, 191, 79), (98, 197, 84)]):
            d.ellipse([12 + k * 20, 9, 24 + k * 20, 21], fill=c)
        d.text((80, 7), title, font=font, fill=DIM)
        for r, (text, color) in enumerate(lines[-rows:]):
            if text.startswith("$ "):
                d.text((pad, th + pad + r * lh), "$", font=font, fill=PROMPT)
                d.text((pad + 2 * cw, th + pad + r * lh), text[2:], font=font, fill=FG)
            else:
                d.text((pad, th + pad + r * lh), text, font=font, fill=color)
        if frames and ms < 40:  # merge tiny steps into the previous frame
            frames[-1] = (img, frames[-1][1] + ms)
        else:
            frames.append((img, max(ms, 20)))

    last_t = 0.0
    for ev in events:
        if ev["kind"] == "comment":
            put(ev["text"] + "\n", DIM)
            snap(900)
        elif ev["kind"] == "cmd":
            put("$ ")
            snap(400)
            for i, ch in enumerate(ev["text"]):
                put(ch)
                if i % 3 == 2 or i == len(ev["text"]) - 1:
                    snap(45)
            put("\n")
            snap(250)
            last_t = 0.0
        elif ev["kind"] == "out":
            gap = min(ev["t"] - last_t, max_gap)
            if frames:
                frames[-1] = (frames[-1][0], frames[-1][1] + int(gap * 1000))
            put(ev["text"])
            snap(30)
            last_t = ev["t"]
        elif ev["kind"] == "exit":
            gap = min(ev["t"] - last_t, max_gap)
            frames[-1] = (frames[-1][0], frames[-1][1] + int(gap * 1000))
    frames[-1] = (frames[-1][0], 4000)
    imgs = [f[0].convert("P", palette=Image.Palette.ADAPTIVE, colors=32) for f in frames]
    imgs[0].save(gif_path, save_all=True, append_images=imgs[1:], loop=0, optimize=True,
                 duration=[f[1] for f in frames], disposal=1)  # fmt: skip
    total = sum(f[1] for f in frames) / 1000
    print(
        f"{gif_path}: {len(frames)} frames, {total:.1f}s, {Path(gif_path).stat().st_size:,} bytes"
    )


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "record":
        record(sys.argv[2], sys.argv[3])
    else:
        import argparse

        ap = argparse.ArgumentParser()
        ap.add_argument("cast")
        ap.add_argument("gif")
        ap.add_argument("--title", default="Terminal")
        ap.add_argument("--cols", type=int, default=100)
        ap.add_argument("--rows", type=int, default=28)
        ap.add_argument("--max-gap", type=float, default=2.0)
        a = ap.parse_args(sys.argv[2:])
        render(a.cast, a.gif, a.title, a.cols, a.rows, a.max_gap)
