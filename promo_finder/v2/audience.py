"""Step 4: a synthetic audience panel predicts WHEN each viewer swipes away.

Platforms judge promos by retention (3-second hold, completion) and click-through, not by an
editor's taste. Argyle et al. (2023) show language models conditioned on specific personas can
reproduce the response patterns of matching human subgroups ("silicon sampling"). Here five US
viewer personas each watch every finalist second by second; we build a predicted retention curve
per clip. This is a cheap pre-test to order the shortlist, NOT a replacement for real A/B data -
the panel's verdicts should be recalibrated against real ad metrics once they exist.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .transcript import render

PERSONAS = {
    "maya": "Maya, 19, college student in Texas. Scrolls TikTok 3 hours a day, loves enemies-to-lovers and "
            "revenge stories, swipes in under 2 seconds if nothing is happening.",
    "denise": "Denise, 41, nurse and mom in Ohio. Watches ReelShort/DramaBox on breaks, loves secret-billionaire "
              "and hidden-identity romance, gives up if she cannot tell who is who.",
    "jake": "Jake, 27, warehouse supervisor in Florida. Not into romance at all; stops only for fights, "
            "humiliation, or something genuinely shocking.",
    "linda": "Linda, 58, retired teacher in Georgia, watches Facebook Reels. Loves period dramas and justice for "
             "the underdog; needs the setup to be clear.",
    "priya": "Priya, 33, product manager in California, K-drama fan. Rewards clever twists, rolls her eyes at "
             "cheesy melodrama and swipes on cliches.",
}

SYSTEM = """You simulate ONE specific real viewer scrolling a short-video feed. Stay in character: their tastes,
patience and attention span. You see ad clips for a drama app as transcripts with play-time seconds.
For each clip decide honestly the second at which THIS person swipes away (most ads lose most viewers in
the first 3 seconds), or null if they watch to the end, and whether they tap to watch the show.
Be as impatient and picky as real feeds make people: most ads are swiped, a clip has to EARN the full
watch, and tapping through to an app is rare. Do not be polite to the clip.
Return ONLY JSON."""


def _prompt(persona: str, clips: list[tuple[str, str]]) -> str:
    body = "\n\n".join(f"### CLIP {cid}\n{txt}" for cid, txt in clips)
    return f"""YOU ARE: {persona}

{body}

Answer for every clip:
{{"clips": [{{"id": "<clip id>", "swipe_at": <seconds or null>, "tap_to_watch": true/false,
  "inner_voice": "what you think while watching, max 15 words"}}]}}"""


def run(llm, promos: list[dict], eps: dict, raw_dir: Path, log=print, errors=None):
    clips = [(p["id"], render(p, eps)) for p in promos]
    resp = {p["id"]: {} for p in promos}
    for name, persona in PERSONAS.items():
        try:
            data = llm.complete_json(SYSTEM, _prompt(persona, clips))
        except Exception as e:
            (errors if errors is not None else []).append(f"audience {name}: {str(e)[:200]}")
            log(f"  audience {name} FAILED: {str(e)[:160]}")
            continue
        (raw_dir / f"audience_{name}.json").write_text(json.dumps(data, indent=1, ensure_ascii=False))
        items = data.get("clips", []) if isinstance(data, dict) else data
        for c in items if isinstance(items, list) else []:
            m = re.search(r"[A-Z]+\d+", str(c.get("id", "")))
            if m and m.group(0) in resp:
                resp[m.group(0)][name] = c
        log(f"  audience {name}: {sum(1 for r in resp.values() if name in r)}/{len(promos)} clips")

    out = {}
    for p in promos:
        r = resp[p["id"]]
        dur = p["duration"]
        if not r:
            continue
        swipes = []
        for c in r.values():
            s = c.get("swipe_at")
            try:
                swipes.append(None if s in (None, "null", "") else max(0.0, min(dur, float(s))))
            except (TypeError, ValueError):
                swipes.append(None)
        n = len(swipes)
        watching = lambda t: sum(1 for s in swipes if s is None or s > t) / n
        curve = [round(watching(t), 3) for t in range(0, int(dur) + 1)]
        out[p["id"]] = {
            "panel": n,
            "hold_3s": round(watching(3.0), 3),
            "completion": round(sum(1 for s in swipes if s is None) / n, 3),
            "tap_rate": round(sum(1 for c in r.values() if c.get("tap_to_watch") in (True, "true", "yes")) / n, 3),
            "curve": curve,
            "voices": {k: {"swipe_at": v.get("swipe_at"), "tap": v.get("tap_to_watch"), "says": v.get("inner_voice", "")}
                       for k, v in r.items()},
        }
    return out
