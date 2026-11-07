"""Prompts. Kept in one file so an editor can tune them without touching code."""
from __future__ import annotations

from .parser import Episode

PROMO_RULES = """WHAT A PROMO IS
A 20-60 second clip cut from the show itself, posted on TikTok / Reels / Shorts to pull in NEW viewers
(think ReelShort / DramaBox ads). It is usually one continuous stretch, but may join 2-3 moments.
A good promo:
1. HOOK - something in the FIRST 3 SECONDS that stops a scrolling thumb: a slap, a reveal, a shocking or
   humiliating line, a threat, a dramatic image. NOT an establishing shot, NOT a calm line.
2. CLIFFHANGER - it CUTS at an unresolved moment (a question hangs, a threat lands, a door bursts open,
   a secret is about to come out). Never end after the tension is resolved.
3. STANDS ALONE - a stranger who has never seen the show understands who is wronged, by whom, and what is at
   stake. Avoid clips that depend on earlier episodes, open mid-thought ("And then...", "That's why..."),
   or need you to know many side characters.
Prefer strong emotional premises: betrayal, humiliation of a good heroine, a secret identity, a protector
stepping in, a villain about to win. Cutting just BEFORE the payoff is better than showing it."""

SYSTEM_FIND = f"""You are a senior social-video editor for a vertical microdrama app. You pick promo clips from
screenplays. You are strict: you would rather return one great promo than three average ones.

{PROMO_RULES}

HOW TO ANSWER
- The script is given as beats: [BEAT_ID START-END] TYPE: text. Times are MM:SS.ss inside the episode.
- Refer to cut points ONLY by beat id. A segment runs from the START of `start_beat` to the END of `end_beat`.
  Do not write timestamps; code computes them from the ids.
- Never start or end a segment in the middle of a spoken line. A segment must not end on a CAMERA beat that
  merely continues past the last line unless that image is the cliffhanger.
- Total duration across segments: 20-60 seconds. Aim for 25-45.
- `hook` = exactly what the viewer sees/hears in the first 3 seconds (quote the line if it is dialogue).
- `cliffhanger` = the unresolved moment the clip cuts on.
- `why` = ONE line on why it works for a cold viewer.
- Scores 1-10, be honest and spread them out; a 9-10 is a promo you would bet ad money on.
Return ONLY JSON."""

FIND_SCHEMA = """{
  "promos": [
    {
      "title": "short internal name",
      "segments": [{"start_beat": "e01b012", "end_beat": "e01b040"}],
      "hook": "...",
      "cliffhanger": "...",
      "why": "...",
      "scores": {"hook": 1-10, "cliffhanger": 1-10, "standalone": 1-10},
      "risks": "what could make this promo fail (spoilers, confusing, weak open) or empty"
    }
  ]
}"""


COMPACT = False  # set by the pipeline: drop low-information camera beats (token diet)


def _beats_block(ep: Episode) -> str:
    from .heuristics import beat_intensity
    lines = []
    for b in ep.beats:
        if b.kind == "camera" and COMPACT and beat_intensity(b) < 5:
            continue  # plain camera moves carry little promo signal; ~40% of beats, ~half the tokens
        if b.kind == "camera":
            txt = b.text if len(b.text) <= 120 else b.text[:117] + "..."
            lines.append(f"[{b.id} {_t(b.start)}-{_t(b.end)}] CAMERA: {txt}")
        else:
            lines.append(b.line())
    return f"""EPISODE {ep.num} (duration {_t(ep.duration)})
LOGLINE (for your understanding only - the viewer does NOT know this): {ep.logline}
CONTINUITY: {ep.continuity}

SCRIPT BEATS:
{chr(10).join(lines)}"""


def episode_prompt(ep: Episode, show_title: str, max_promos: int = 3) -> str:
    return f"""SHOW: {show_title}
{_beats_block(ep)}

TASK: Find up to {max_promos} distinct, non-overlapping promos in THIS episode (0 is allowed if nothing is
strong enough). Multi-segment promos may only use beats from this episode here.

JSON schema:
{FIND_SCHEMA}"""


def batch_prompt(eps: list[Episode], show_title: str, max_promos: int = 3) -> str:
    """Several episodes in one call - for free tiers with a daily request cap."""
    body = "\n\n".join("=" * 20 + "\n" + _beats_block(e) for e in eps)
    nums = ", ".join(str(e.num) for e in eps)
    return f"""SHOW: {show_title}
You get {len(eps)} episodes ({nums}). Treat EACH episode separately.

{body}

TASK: For EACH episode, find up to {max_promos} distinct, non-overlapping promos inside that episode (0 is allowed
if nothing is strong enough). Every segment must use beats from one episode only. Add "episode": <number> to
each promo.

JSON schema:
{FIND_SCHEMA}"""


SYSTEM_CROSS = f"""You are a senior social-video editor for a vertical microdrama app.

{PROMO_RULES}

Here you build promos that JOIN moments from DIFFERENT episodes: e.g. open on the most shocking line of a
later episode, then show the setup that explains it, or show the humiliation in one episode and cut on the
cliffhanger of a later one. Each joined promo must still read as one coherent mini-story to a stranger.
Use ONLY beat ids that appear in the moment list. Segments run from start of start_beat to end of end_beat
and must stay inside one listed moment each. Total 20-60 seconds. Return ONLY JSON."""


def cross_prompt(show_title: str, digest: str, max_promos: int = 6) -> str:
    return f"""SHOW: {show_title}

AVAILABLE MOMENTS (story order). Each moment lists its beats; you may trim a moment by choosing a later
start_beat or an earlier end_beat inside it.
{digest}

TASK: Propose up to {max_promos} promos that join 2-3 moments from DIFFERENT episodes.

JSON schema:
{FIND_SCHEMA}"""


SYSTEM_COLD = """You are a typical TikTok viewer who has NEVER seen this show. You are shown the transcript of a
short ad clip, exactly as it plays. Judge only what is on screen. Be blunt. Return ONLY JSON."""


def cold_prompt(items: list[tuple[str, str]]) -> str:
    blocks = "\n\n".join(f"### CLIP {cid}\n{txt}" for cid, txt in items)
    return f"""{blocks}

For EACH clip answer:
{{"clips": [{{"id": "...", "what_is_happening": "one sentence, your understanding",
  "confused_by": "what you did not understand, or empty",
  "stop_scrolling": 1-10, "need_to_know_next": 1-10, "understood": 1-10}}]}}"""


def _t(sec: float) -> str:
    m = int(sec // 60)
    return f"{m:02d}:{sec - m * 60:05.2f}"
