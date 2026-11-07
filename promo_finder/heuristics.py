"""Rule-based scoring. No AI. Used for (a) an offline baseline that runs
without any API key and (b) an independent second opinion that the
ranker and the QC report compare against the LLM's picks.

All scores are 0-10.
"""
from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field

from .parser import Beat, Episode

# --- lexicons --------------------------------------------------------------
# Physical / visual shocks that stop a scroll.
SHOCK_ACTS = r"slap|strike|strikes|struck|punch|kick|grab|seiz|shove|throw|hurl|smash|shatter|explo|blast|blood|bleed|wound|knife|blade|sword|dagger|pistol|gun|crop|whip|choke|strangl|collaps|faint|fall|drag|chain|shackle|burn|fire|flame|ablaze|scream|shriek|sob|tear[s ]|kneel|one knee|kiss|rip|tear[s]? (?:apart|open)|burst|bang|crash|slam|poison|arsenic|dead|corpse|noose|gallows|hot iron|brand|briar|thorn|mud|spit|gasp|arrest|handcuff"
SHOCK_WORDS = r"marry|married|wedding|husband|wife|divorce|betray|liar|lie[sd]?\b|kill|murder|die|dead|death|hang|execut|witch|poison|traitor|treason|duke|duchess|heir|bastard|pregnan|baby|son\b|mother|disown|inherit|never|how dare|kneel|beg|nothing|worthless|plain|ugly|cheat|steal|thief|stole|prison|dungeon|burn|blood|secret|truth|who are you|you are"
INTENSE_TONES = r"furious|rage|enraged|outrag|shock|stunn|horrif|gasp|scream|shout|yell|cry|crying|sob|desperat|beg|plead|menac|threat|cruel|sadist|venom|vicious|gloat|mock|sneer|cold|icy|lethal|deadly|commanding|explosive|devastat|broken|choking|seeth|livid|hiss|roar|defiant"
CLIFF_SFX = r"cliffhanger|ominous|cut to black|cuts to black|abrupt|sudden|silence|gasp|drone|sting|boom|bang|slam|crash|burst|thunder"
RESOLVED = r"warm|tender|gentle|peace|safe|smil|laugh|embrace|relief|soft|fading out|fade-out|fades out|hope|calm|content|sleep|rest\b|serene"
WEAK_OPENERS = r"^(\.\.\.|\u2026|and\b|but\b|so\b|then\b|that\b|this\b|it\b|he\b|she\b|they\b|him\b|her\b|them\b|yes\b|well\b|also\b|because\b|which\b)"
PREMISE = r"wedding|marry|married|husband|wife|sister|mother|father|family|duke|duchess|soldier|witch|poison|altar|bride|groom|engag|heir|title"


def _rx(p):
    return re.compile(p, re.I)


SHOCK_ACTS_RE, SHOCK_WORDS_RE, INTENSE_RE = _rx(SHOCK_ACTS), _rx(SHOCK_WORDS), _rx(INTENSE_TONES)
CLIFF_SFX_RE, RESOLVED_RE, WEAK_RE, PREMISE_RE = _rx(CLIFF_SFX), _rx(RESOLVED), _rx(WEAK_OPENERS), _rx(PREMISE)


_INT_CACHE: dict[tuple, float] = {}


def beat_intensity(b: Beat) -> float:
    """0-10: how arresting a single beat is (memoised - search calls this a lot)."""
    key = (b.id, b.text)
    if key not in _INT_CACHE:
        _INT_CACHE[key] = _beat_intensity(b)
    return _INT_CACHE[key]


def _beat_intensity(b: Beat) -> float:
    s = 0.0
    if b.kind == "dialogue":
        txt = b.text
        s += 1.0
        s += 2.0 * min(2, len(SHOCK_WORDS_RE.findall(txt)))
        s += 2.0 * min(2, len(INTENSE_RE.findall(b.tone)))
        if "!" in txt:
            s += 1.5
        if "?" in txt:
            s += 0.5
        if len(txt.split()) <= 7:
            s += 1.0  # short punchy lines land in 3s
    elif b.kind in ("action", "camera"):
        s += 2.5 * min(2, len(SHOCK_ACTS_RE.findall(b.text)))
        s += 1.0 * min(2, len(INTENSE_RE.findall(b.text)))
    elif b.kind == "sfx":
        s += 2.0 * min(2, len(SHOCK_ACTS_RE.findall(b.text)) + len(CLIFF_SFX_RE.findall(b.text)))
    return min(10.0, s)


# --- cut points ------------------------------------------------------------
CONT = ("...", "\u2026", "-", "\u2014")


def _continuations(ep: Episode) -> list[tuple[float, float]]:
    """Sentences split across two dialogue beats ("I said..." / "...let her go").
    Returns the (start, end) span of each split sentence; no cut may fall strictly inside it."""
    dialog = [b for b in ep.beats if b.kind == "dialogue"]
    spans = []
    for a, b in zip(dialog, dialog[1:]):
        if b.text.lstrip().startswith(CONT) and b.start - a.end < 6.0:
            spans.append((a.start, b.end))
    return spans


def cut_points(ep: Episode) -> tuple[list[float], list[float]]:
    """Times where we can start / end a clip without slicing a spoken line or sentence."""
    cached = ep.__dict__.get("_cuts")
    if cached:
        return cached
    dialog = [b for b in ep.beats if b.kind == "dialogue"]
    spans = _continuations(ep)

    def inside_line(t: float) -> bool:
        return any(d.start + 0.05 < t < d.end - 0.05 for d in dialog)

    def inside_sentence(t: float) -> bool:
        return any(s + 0.05 < t < e - 0.05 for s, e in spans)

    def ok_start(b):
        return b.kind != "sfx" and not inside_line(b.start) and not inside_sentence(b.start) \
            and not (b.kind == "dialogue" and b.text.lstrip().startswith(CONT))

    starts = sorted({round(b.start, 2) for b in ep.beats if ok_start(b)})
    ends = sorted({round(b.end, 2) for b in ep.beats if not inside_line(b.end) and not inside_sentence(b.end)})
    ep.__dict__["_cuts"] = (starts, ends)
    return starts, ends


def beats_in(ep: Episode, s: float, e: float) -> list[Beat]:
    # beats are sorted by start; bisect for speed (called millions of times in search)
    starts = ep.__dict__.setdefault("_starts", [b.start for b in ep.beats])
    i = bisect.bisect_left(starts, s - 0.05)
    out = []
    for b in ep.beats[i:]:
        if b.start > e + 0.05:
            break
        if b.end <= e + 0.05:
            out.append(b)
    return out


def _resolve_word(b: Beat) -> bool:
    return bool(RESOLVED_RE.search(b.text + " " + b.tone))


@dataclass
class Scored:
    hook: float
    cliff: float
    intensity: float
    standalone: float
    total: float
    notes: list[str] = field(default_factory=list)


def score_window(ep: Episode, segs: list[tuple[float, float]]) -> Scored:
    """Score a promo made of one or more (start, end) segments of `ep`."""
    notes = []
    all_beats = [b for s, e in segs for b in beats_in(ep, s, e)]
    if not all_beats:
        return Scored(0, 0, 0, 0, 0, ["empty"])

    # Hook: what plays in the first 3 seconds of the first segment.
    s0, e0 = segs[0]
    first = [b for b in beats_in(ep, s0, e0) if b.start < s0 + 3.0]
    hook = max([beat_intensity(b) for b in first] or [0.0])
    first_lines = [b for b in first if b.kind == "dialogue"]
    if not first_lines:
        hook *= 0.75
        notes.append("no dialogue in first 3s")

    # Cliffhanger: the last ~5 seconds of the last segment.
    sl, el = segs[-1]
    last = [b for b in beats_in(ep, sl, el) if b.end > el - 5.0]
    cliff = 0.0
    last_line = next((b for b in reversed(last) if b.kind == "dialogue"), None)
    if last_line:
        t = last_line.text.rstrip()
        if t.endswith("?"):
            cliff += 3
        if t.endswith(("...", "…", "-", "—")):
            cliff += 3
        if t.endswith("!"):
            cliff += 1.5
        cliff += 1.5 * min(2, len(SHOCK_WORDS_RE.findall(t)))
        cliff += 1.0 * min(1, len(INTENSE_RE.findall(last_line.tone)))
    for b in last:
        if b.kind in ("sfx", "camera", "action") and CLIFF_SFX_RE.search(b.text):
            cliff += 2.5
            break
    if el >= ep.duration - 6:
        cliff += 3.0  # writers put the episode cliffhanger at the very end
        notes.append("ends on episode cliffhanger")
    if sum(_resolve_word(b) for b in last) >= 2:
        cliff -= 3.0
        notes.append("ending reads as resolved/warm")
    cliff = max(0.0, min(10.0, cliff))

    intensity = sum(beat_intensity(b) for b in all_beats) / len(all_beats)
    intensity = min(10.0, intensity * 1.6)

    # Standalone: could a cold viewer follow it?
    standalone = 6.0
    if first_lines and WEAK_RE.search(first_lines[0].text):
        standalone -= 2.5
        notes.append("opens mid-thought: '" + first_lines[0].text[:30] + "'")
    speakers = {b.speaker for b in all_beats if b.kind == "dialogue"}
    if len(speakers) > 4:
        standalone -= 1.5 * (len(speakers) - 4)
        notes.append(f"{len(speakers)} speakers")
    text_blob = " ".join(b.text for b in all_beats)
    if PREMISE_RE.search(text_blob):
        standalone += 2.0
    n_lines = sum(1 for b in all_beats if b.kind == "dialogue")
    dur = sum(e - s for s, e in segs)
    if n_lines < dur / 8:
        standalone -= 2.0
        notes.append("little dialogue")
    standalone = max(0.0, min(10.0, standalone))

    total = 0.35 * hook + 0.35 * cliff + 0.15 * intensity + 0.15 * standalone
    if 25 <= dur <= 45:
        total += 0.3
    return Scored(round(hook, 2), round(cliff, 2), round(intensity, 2), round(standalone, 2), round(total, 3), notes)


# --- candidate search ------------------------------------------------------
def overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    return inter / max(1e-6, min(a[1] - a[0], b[1] - b[0]))


_BW_CACHE: dict = {}


def best_windows(ep: Episode, k: int = 3, min_d: float = 20, max_d: float = 60) -> list[dict]:
    """Exhaustive search over clean cut points, then non-max suppression.
    Memoised on the episode's content: several stages ask for the same search."""
    key = (ep.num, len(ep.beats), hash(tuple((b.start, b.end, b.text) for b in ep.beats)), k, min_d, max_d)
    if key not in _BW_CACHE:
        _BW_CACHE[key] = _best_windows(ep, k, min_d, max_d)
    return _BW_CACHE[key]


def _best_windows(ep: Episode, k: int = 3, min_d: float = 20, max_d: float = 60) -> list[dict]:
    starts, ends = cut_points(ep)
    cands = []
    for s in starts:
        for e in ends:
            d = e - s
            if min_d <= d <= max_d:
                sc = score_window(ep, [(s, e)])
                cands.append(((s, e), sc))
    cands.sort(key=lambda x: -x[1].total)
    picked: list[tuple[tuple[float, float], Scored]] = []
    for win, sc in cands:
        if all(overlap(win, p[0]) < 0.35 for p in picked):
            picked.append((win, sc))
        if len(picked) >= k:
            break
    return [{"segments": [w], "scores": sc} for w, sc in picked]


def hook_beat(ep: Episode, exclude: tuple[float, float]) -> Beat | None:
    """Most arresting short dialogue beat outside a window (for cold opens)."""
    pool = [b for b in ep.beats if b.kind == "dialogue" and b.end - b.start <= 4.0
            and not (exclude[0] - 0.1 <= b.start <= exclude[1])]
    pool.sort(key=lambda b: -beat_intensity(b))
    return pool[0] if pool else None
