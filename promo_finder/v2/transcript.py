"""Render a clip exactly as a viewer would experience it: play-time seconds, what is said and shown."""
from __future__ import annotations

from ..heuristics import beat_intensity, beats_in


def render(promo: dict, eps: dict, with_seconds: bool = True) -> str:
    out, off = [], 0.0
    for i, s in enumerate(promo["segments"]):
        if i:
            out.append("[CUT]")
        ep = eps[s["ep"]]
        for b in beats_in(ep, s["start"], s["end"]):
            t = f"[{off + b.start - s['start']:4.1f}s] " if with_seconds else ""
            if b.kind == "dialogue":
                out.append(f'{t}{b.speaker}: "{b.text}"')
            elif b.kind == "action" or (b.kind == "camera" and beat_intensity(b) >= 5):
                out.append(f"{t}(on screen: {b.text[:160]})")
            elif b.kind == "sfx":
                out.append(f"{t}(sound: {b.text[:100]})")
        off += s["end"] - s["start"]
    out.append(f"[END {off:.0f}s]")
    return "\n".join(out)
