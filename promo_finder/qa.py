"""Input quality check, run BEFORE spending any LLM calls.

A promo finder is only as good as its scripts. This flags packs where:
  - the same lines repeat across most episodes (template / boilerplate / recap footage)
  - every episode has identical timing (a generator template, not a cut)
  - beats are too coarse to place a 3-second hook (one beat lasts 6-13 s)
  - "dialogue" that is really narration (a character saying a plot summary about themselves)
"""
from __future__ import annotations

import re
from collections import Counter

from .parser import Episode


def norm(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()


def inspect(eps: list[Episode]) -> dict:
    n = len(eps)
    beats = [b for e in eps for b in e.beats]
    dl = [b for b in beats if b.kind == "dialogue"]
    # in how many episodes does each line/description appear?
    per_text = Counter()
    for e in eps:
        for t in {norm(b.text) for b in e.beats if b.kind in ("dialogue", "camera", "action") and len(b.text) > 15}:
            per_text[t] += 1
    boiler = {t for t, c in per_text.items() if n >= 4 and c >= 0.5 * n}
    rep_dialogue = sum(norm(b.text) in boiler for b in dl) / max(1, len(dl))
    rep_all = sum(norm(b.text) in boiler for b in beats if b.kind != "sfx") / max(1, len(beats))
    timing = Counter(tuple(round(b.start, 1) for b in e.beats) for e in eps)
    same_timing = timing.most_common(1)[0][1] / n if n else 0
    mean_beat = sum(b.end - b.start for b in beats) / max(1, len(beats))
    narr = 0
    for b in dl:
        first = b.speaker.split()[0].title() if b.speaker else ""
        if first and len(first) > 2 and re.search(rf"\b{first}\b", b.text):
            narr += 1
    speakers = {b.speaker for b in dl}
    warn = []
    if rep_dialogue >= 0.4:
        warn.append(f"{rep_dialogue:.0%} of dialogue lines repeat word-for-word in at least half the episodes (templated or recap content)")
    if same_timing >= 0.5 and n >= 4:
        warn.append(f"{same_timing:.0%} of episodes have identical beat timing")
    if mean_beat > 5:
        warn.append(f"beats are coarse (mean {mean_beat:.1f}s), so a 3-second hook cannot be placed precisely")
    if narr and narr / max(1, len(dl)) > 0.03:
        warn.append(f"{narr} dialogue line(s) look like narration (the speaker names themselves), probably plot summary text")
    if len(speakers) <= 2:
        warn.append(f"only {len(speakers)} speaking character(s)")
    verdict = "templated" if (rep_dialogue >= 0.4 or same_timing >= 0.8) else ("warn" if warn else "ok")
    return {"verdict": verdict, "episodes": n, "beats": len(beats), "beats_per_episode": round(len(beats) / max(1, n), 1),
            "mean_beat_seconds": round(mean_beat, 1), "speakers": len(speakers),
            "repeated_dialogue_share": round(rep_dialogue, 3), "repeated_beat_share": round(rep_all, 3),
            "identical_timing_share": round(same_timing, 3), "narration_as_dialogue": narr,
            "boilerplate_texts": len(boiler), "warnings": warn}
