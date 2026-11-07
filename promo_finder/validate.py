"""Deterministic checks and fixes for every promo, whoever proposed it.

The LLM picks moments; code owns the timestamps. Every promo goes through
here, and every fix is recorded as a flag so a human can see what changed.
"""
from __future__ import annotations

import re

from .heuristics import CONT, beat_intensity, beats_in, cut_points, score_window
from .parser import Beat, Episode, fmt

MIN_D, MAX_D, HOOK_WINDOW = 20.0, 60.0, 3.0


class Index:
    def __init__(self, episodes: list[Episode]):
        self.eps = {e.num: e for e in episodes}
        self.beats: dict[str, Beat] = {b.id: b for e in episodes for b in e.beats}
        self._cuts = {e.num: cut_points(e) for e in episodes}

    def cuts(self, ep: int):
        return self._cuts[ep]


def _inside_line(ep: Episode, t: float) -> Beat | None:
    for b in ep.beats:
        if b.kind == "dialogue" and b.start + 0.05 < t < b.end - 0.05:
            return b
    return None


def _beat(idx: Index, raw, flags: list[str]) -> Beat | None:
    """Look up a beat id, tolerating small model slips: '12b031', 'E12B31', one past the last beat."""
    t = str(raw or "").strip()
    if t in idx.beats:
        return idx.beats[t]
    m = re.search(r"e?p?(\d+)\s*b\s*(\d+)", t, re.I)
    if not m:
        return None
    ep, k = int(m.group(1)), int(m.group(2))
    if ep not in idx.eps:
        return None
    beats = idx.eps[ep].beats
    if k < len(beats):
        flags.append(f"fixed: beat id '{t}' read as {beats[k].id}")
        return beats[k]
    if k <= len(beats) + 2:
        flags.append(f"fixed: beat id '{t}' is past the episode end; used last beat")
        return beats[-1]
    return None


def resolve(idx: Index, raw_segments: list[dict], flags: list[str]) -> list[dict] | None:
    segs = []
    for rs in raw_segments:
        if not isinstance(rs, dict):
            flags.append(f"REJECT: bad segment {rs!r}")
            return None
        a, z = _beat(idx, rs.get("start_beat"), flags), _beat(idx, rs.get("end_beat"), flags)
        if not a or not z:
            flags.append(f"REJECT: unknown beat id {rs}")
            return None
        if a.ep != z.ep:
            flags.append("REJECT: a segment spans two episodes")
            return None
        if z.end < a.start:
            a, z = z, a
            flags.append("fixed: start/end beats were reversed")
        segs.append({"ep": a.ep, "start": a.start, "end": z.end, "start_beat": a.id, "end_beat": z.id})
    return segs


def _snap(idx: Index, seg: dict, flags: list[str]):
    ep = idx.eps[seg["ep"]]
    # a line that continues a sentence ("...to leave") is not a clean start: back up one line
    dialog = [b for b in ep.beats if b.kind == "dialogue"]
    for a, b in zip(dialog, dialog[1:]):
        if abs(b.start - seg["start"]) < 0.3 and b.text.lstrip().startswith(CONT) and b.start - a.end < 2.5:
            seg["start"] = a.start
            flags.append(f"fixed: start moved back to the beginning of the sentence '{a.text[:25]}'")
            break
    b = _inside_line(ep, seg["start"])
    if b:
        seg["start"] = b.start
        flags.append(f"fixed: start moved to beginning of line '{b.text[:25]}'")
    b = _inside_line(ep, seg["end"])
    if b:
        seg["end"] = b.end
        flags.append(f"fixed: end moved to end of line '{b.text[:25]}'")


def _trim_dead_air(idx: Index, seg: dict, flags: list[str], is_first: bool, is_last: bool):
    ep = idx.eps[seg["ep"]]
    inside = beats_in(ep, seg["start"], seg["end"])
    content = [b for b in inside if b.kind != "camera"]
    if not content:
        return
    if is_first:
        first = min(b.start for b in content)
        if first - seg["start"] > HOOK_WINDOW:
            new = max(seg["start"], first - 0.5)
            flags.append(f"fixed: cut {new - seg['start']:.1f}s of silent lead-in so the hook lands in 3s")
            seg["start"] = round(new, 2)
    last_end = max(b.end for b in content)
    # Keep a closing image if it is the cliffhanger (episode end / cut to black); trim long dead tails.
    if seg["end"] - last_end > 4.0 and not (is_last and seg["end"] >= ep.duration - 0.5):
        new = last_end + 1.5
        if not _inside_line(ep, new):
            flags.append(f"fixed: trimmed {seg['end'] - new:.1f}s of dead air at the end")
            seg["end"] = round(new, 2)


def total(segs):
    return sum(s["end"] - s["start"] for s in segs)


def _fit_length(idx: Index, segs: list[dict], flags: list[str]) -> bool:
    d = total(segs)
    if MIN_D <= d <= MAX_D:
        return True
    if len(segs) != 1:
        # multi-segment: trim/extend the longest segment's start, keep the final cut
        flags.append(f"multi-segment duration {d:.1f}s outside 20-60s")
    seg = max(segs, key=lambda s: s["end"] - s["start"]) if len(segs) > 1 else segs[0]
    ep = idx.eps[seg["ep"]]
    starts, ends = idx.cuts(seg["ep"])
    other = d - (seg["end"] - seg["start"])
    if d > MAX_D:
        if d > 180:
            flags.append(f"REJECT: {d:.1f}s is far over 60s")
            return False
        # Keep the chosen hook (opening) AND the chosen cliffhanger (ending); drop the middle.
        # Trimming either end would throw away exactly what the picker chose the clip for.
        budget = MAX_D - 2 - other
        head_ends = [e for e in ends if seg["start"] + 8 <= e <= seg["start"] + min(22, budget - 15)]
        best = None
        for he in head_ends:
            for ts in starts:
                if ts <= he + 3:
                    continue
                tot = (he - seg["start"]) + (seg["end"] - ts)
                if 25 <= tot <= budget and seg["end"] - ts >= 15:
                    # keep the most intense material (the reveal lines), then a punchy line at the jump
                    kept = beats_in(ep, seg["start"], he) + beats_in(ep, ts, seg["end"])
                    val = (sum(beat_intensity(b) for b in kept if b.kind != "camera")
                           + 0.5 * score_window(ep, [(ts, seg["end"])]).hook)
                    if best is None or val > best[0]:
                        best = (val, he, ts)
        if not best:
            flags.append(f"REJECT: could not cut {d:.1f}s down to 60s at clean cuts")
            return False
        _, he, ts = best
        tail = dict(seg, start=ts)
        flags.append(f"fixed: {d:.0f}s too long - kept opening {fmt(seg['start'])}-{fmt(he)} and ending "
                     f"{fmt(ts)}-{fmt(seg['end'])}, cut the middle (jump cut)")
        seg["end"] = he
        segs.insert(segs.index(seg) + 1, tail)
        return True
    # too short: extend the opening backwards first (keeps the chosen cliffhanger)
    opts = [s for s in starts if s <= seg["start"] and MIN_D <= seg["end"] - s + other <= 40]
    if opts:
        best = max(opts, key=lambda s: score_window(ep, [(s, seg["end"])]).hook)
        flags.append(f"fixed: extended opening back to {fmt(best)} to reach 20s")
        seg["start"] = best
        return True
    opts = [e for e in ends if e >= seg["end"] and MIN_D <= e - seg["start"] + other <= 40]
    if opts:
        best = max(opts, key=lambda e: score_window(ep, [(seg["start"], e)]).cliff)
        flags.append(f"fixed: extended ending to {fmt(best)} to reach 20s (cliffhanger changed)")
        seg["end"] = best
        return True
    flags.append(f"REJECT: {d:.1f}s is under 20s and could not be extended")
    return False


def _words(t: str) -> set[str]:
    return {w for w in re.findall(r"[a-z']+", t.lower()) if len(w) > 2}


def _check_hook_text(idx: Index, promo: dict, segs: list[dict], flags: list[str]):
    """The stated hook must actually play in the first 3 seconds. Catches pickers (or our own
    fixes) that describe a hook which is not where the clip starts."""
    hook = str(promo.get("hook", ""))
    hw = _words(hook)
    if not hw or not segs:
        return
    s0 = segs[0]
    first = [b for b in beats_in(idx.eps[s0["ep"]], s0["start"], s0["end"]) if b.start < s0["start"] + HOOK_WINDOW]
    fw = set().union(*[_words(b.text + " " + b.speaker) for b in first]) if first else set()
    if len(hw & fw) / len(hw) < 0.4:
        # Is the hook a line further into the first segment? Then open the clip on it.
        ep = idx.eps[s0["ep"]]
        later = [b for b in beats_in(ep, s0["start"], s0["end"]) if b.kind != "camera"
                 and len(hw & _words(b.text + " " + b.speaker)) / len(hw) >= 0.5]
        starts, _ = idx.cuts(s0["ep"])
        if later:
            b = later[0]
            new = max([t for t in starts if t <= b.start + 0.05] or [b.start])
            if total(segs) - (new - s0["start"]) >= MIN_D:
                flags.append(f"fixed: opened the clip on its stated hook (moved start {fmt(s0['start'])} -> {fmt(new)})")
                s0["start"] = new
                return
        inside = [b for s in segs for b in beats_in(idx.eps[s["ep"]], s["start"], s["end"])]
        aw = set().union(*[_words(b.text) for b in inside]) if inside else set()
        where = "plays later in the clip" if len(hw & aw) / len(hw) >= 0.4 else "is not in the clip at all"
        flags.append(f"warn: stated hook {where}, not in the first 3s")


def check(idx: Index, promo: dict) -> dict:
    """Validate + fix in place. Sets promo['status'] to ok / fixed / rejected."""
    flags: list[str] = promo.setdefault("flags", [])
    segs = promo.get("segments")
    if segs and "start_beat" in segs[0] and "start" not in segs[0]:
        segs = resolve(idx, segs, flags)
    if not segs:
        promo["status"] = "rejected"
        return promo
    for i, s in enumerate(segs):
        _snap(idx, s, flags)
        _trim_dead_air(idx, s, flags, i == 0, i == len(segs) - 1)
    ok = _fit_length(idx, segs, flags)
    _check_hook_text(idx, promo, segs, flags)
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            a, b = segs[i], segs[j]
            if a["ep"] == b["ep"] and min(a["end"], b["end"]) - max(a["start"], b["start"]) > 0.5:
                flags.append("warn: segments overlap (same footage twice)")
    # re-derive beat ids after fixes so outputs stay consistent
    for s in segs:
        ep = idx.eps[s["ep"]]
        inside = beats_in(ep, s["start"], s["end"])
        if inside:
            s["start_beat"], s["end_beat"] = inside[0].id, inside[-1].id
        s["start"], s["end"] = round(s["start"], 2), round(s["end"], 2)
    promo["segments"] = segs
    promo["duration"] = round(total(segs), 2)
    if not ok or any(f.startswith("REJECT") for f in flags):
        promo["status"] = "rejected"
    else:
        promo["status"] = "fixed" if any(f.startswith("fixed") for f in flags) else "ok"
    return promo
