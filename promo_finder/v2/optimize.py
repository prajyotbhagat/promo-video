"""Step 2: search every clean clip and score it from the beat annotations. No LLM here.

score = 0.30*Hook + 0.30*Gap + 0.15*Suspense + 0.10*TurningPoint - 0.20*ContextDebt - 0.20*Resolution

Hook        surprise + arousal of what plays in the first 3s; bonus if a question opens there.
Gap         information gap left open at the cut (Loewenstein 1994): importance of questions opened
            inside the clip and NOT answered inside it; recent ones weigh more (cut right after the
            question lands).
Suspense    open-question load and arousal across the clip (sustained uncertainty, Wilmot & Keller 2020).
TurningPt   clip contains an opportunity / change of plans / point of no return / setback
            (TRIPOD types, Papalampidi et al. 2019) - trailers are built from these (Papalampidi 2021).
ContextDebt how much a stranger needs earlier episodes: labelled context need, plus answers to
            questions the clip never set up.
Resolution  the clip answers its own question at the end, ends on a triumph, or contains the climax.

Clip shapes searched: every single window between clean cut points, and "flash-forward" cold opens
(a shocking beat from later in the episode played first, then the setup that leads up to it).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..heuristics import beats_in, cut_points, overlap
from ..parser import Episode, fmt
from .annotate import Ann, _keep

W = {"hook": 0.30, "gap": 0.30, "suspense": 0.15, "tp": 0.10, "debt": -0.20, "resolution": -0.20}
SETUP_TPS = {"OPP", "CHG", "PNR", "SET"}


@dataclass
class Clip:
    ep: int
    segments: list[tuple[float, float]]
    shape: str = "window"              # window | flash-forward
    feats: dict = field(default_factory=dict)
    score: float = 0.0
    open_qs: list[str] = field(default_factory=list)
    hook_beat: str = ""
    end_beat: str = ""

    @property
    def duration(self):
        return sum(e - s for s, e in self.segments)


class Scorer:
    def __init__(self, ep: Episode, ann: dict[str, Ann], questions: dict[str, dict], weights=None):
        self.ep, self.ann, self.q = ep, ann, questions
        self.w = dict(W, **(weights or {}))
        self.opened_at = {}
        for b in ep.beats:
            for qid in ann[b.id].opens:
                self.opened_at.setdefault(qid, b.start)

    def imp(self, qid):
        return float(self.q.get(qid, {}).get("importance", 1.0))

    def features(self, segs: list[tuple[float, float]], teaser: bool = False) -> tuple[dict, list[str], str, str]:
        played = []  # (play_offset, beat)
        off = 0.0
        for s, e in segs:
            for b in beats_in(self.ep, s, e):
                if _keep(b):
                    played.append((off + b.start - s, b))
            off += e - s
        dur = off
        if not played:
            return {k: 0.0 for k in W}, [], "", ""
        A = lambda b: self.ann[b.id]

        # Hook
        first = [(o, b) for o, b in played if o < 3.0]
        hook = max([(A(b).a + A(b).s) / 6 for _, b in first] or [0.0])
        if any(self.imp(q) >= 2 for _, b in first for q in A(b).opens):
            hook += 0.25
        hook_beat = max(first, key=lambda ob: A(ob[1]).a + A(ob[1]).s)[1].id if first else ""

        # Question bookkeeping in play order
        opened, closed, mass_trace = {}, {}, []
        outside_answers = 0
        for o, b in played:
            for q in A(b).closes:
                if q in opened:
                    closed[q] = o
                else:
                    outside_answers += 1     # answer to a question the clip never raised
            for q in A(b).opens:
                opened.setdefault(q, o)
            mass_trace.append(sum(self.imp(q) for q in opened if q not in closed))
        still_open = [q for q in opened if q not in closed]

        gap = sum(self.imp(q) * (1.0 if dur - opened[q] <= 10 else 0.7) for q in still_open) / 5
        if any(dur - opened[q] <= 5 for q in still_open):
            gap += 0.2                       # cut lands right after the question
        if teaser:
            gap += 0.25                      # the flash-forward itself is an unexplained future moment
        gap = min(1.0, gap)

        mean_a = sum(A(b).a for _, b in played) / len(played)
        suspense = 0.5 * min(1.0, (sum(mass_trace) / len(mass_trace)) / 4) + 0.5 * mean_a / 3

        tp = 1.0 if any(A(b).tp in SETUP_TPS for _, b in played) else 0.0

        dl = [b for _, b in played if b.kind == "dialogue"] or [b for _, b in played]
        debt = min(1.0, sum(A(b).c for b in dl) / len(dl) / 2 + 0.15 * outside_answers)

        res = sum(self.imp(q) for q, o in closed.items() if dur - o <= 6) / 3
        if any(A(b).tp == "CLX" and dur - o <= 8 for o, b in played):
            res += 0.5
        last_lines = [b for o, b in played if b.kind == "dialogue" and dur - o <= 6]
        if last_lines and A(last_lines[-1]).v >= 1.5:
            res += 0.3                       # ends on a triumph, not a question
        res = min(1.0, res)

        f = {"hook": round(min(1.0, hook), 3), "gap": round(gap, 3), "suspense": round(suspense, 3),
             "tp": tp, "debt": round(debt, 3), "resolution": round(res, 3)}
        return f, still_open, hook_beat, played[-1][1].id

    def score(self, f: dict) -> float:
        return round(sum(self.w[k] * f[k] for k in W), 4)


def search(ep: Episode, ann, questions, k: int = 3, weights=None, min_d=20.0, max_d=60.0) -> list[Clip]:
    sc = Scorer(ep, ann, questions, weights)
    starts, ends = cut_points(ep)
    clips: list[Clip] = []
    for s in starts:
        for e in ends:
            if min_d <= e - s <= max_d:
                f, oq, hb, eb = sc.features([(s, e)])
                clips.append(Clip(ep.num, [(s, e)], "window", f, sc.score(f), oq, hb, eb))
    clips.sort(key=lambda c: -c.score)

    # Flash-forward cold opens: a high-shock beat from AFTER the body plays first.
    shocks = [b for b in ep.beats if _keep(b) and b.kind in ("dialogue", "action")
              and ann[b.id].a + ann[b.id].s >= 4 and b.end - b.start <= 5.0]
    ff = []
    for body in clips[:40]:
        bs, be = body.segments[0]
        for t in shocks:
            if t.start < be + 2:
                continue
            ts = [x for x in starts if x <= t.start + 0.05]
            te = [x for x in ends if x >= t.end - 0.05]
            if not ts or not te:
                continue
            seg_t = (ts[-1], te[0])
            if seg_t[1] - seg_t[0] > 6:
                continue
            segs = [seg_t, (bs, be)]
            d = sum(e - s for s, e in segs)
            if not (min_d <= d <= max_d):
                continue
            f, oq, hb, eb = sc.features(segs, teaser=True)
            ff.append(Clip(ep.num, segs, "flash-forward", f, sc.score(f), oq, hb, eb))
    allc = sorted(clips + ff, key=lambda c: -c.score)

    picked: list[Clip] = []
    for c in allc:
        main = max(c.segments, key=lambda x: x[1] - x[0])
        if all(overlap(main, max(p.segments, key=lambda x: x[1] - x[0])) < 0.35 for p in picked):
            picked.append(c)
        if len(picked) >= k:
            break
    return picked


def describe(c: Clip, ep: Episode, ann, questions) -> dict:
    """Human-readable hook / cliffhanger / why, generated from the signals (no LLM)."""
    beats = {b.id: b for b in ep.beats}
    hb = beats.get(c.hook_beat)
    hook = (f'{hb.speaker}: "{hb.text}"' if hb and hb.kind == "dialogue" else (hb.text if hb else ""))
    eb = beats.get(c.end_beat)
    last_line = None
    s, e = c.segments[-1]
    for b in reversed(beats_in(ep, s, e)):
        if b.kind == "dialogue":
            last_line = b
            break
    open_txt = [questions[q]["text"] for q in sorted(c.open_qs, key=lambda q: -float(questions.get(q, {}).get("importance", 1)))
                if q in questions][:2]
    cliff = (f'Cuts on {last_line.speaker}: "{last_line.text}"' if last_line else (eb.text if eb else ""))
    if open_txt:
        cliff += " - leaves open: " + " / ".join(open_txt)
    f = c.feats
    bits = []
    if c.shape == "flash-forward":
        bits.append("flash-forward cold open (shows a shock from later, then how we got there)")
    bits.append(f"hook {f['hook']:.2f}")
    bits.append(f"{len(c.open_qs)} question(s) unanswered at the cut (gap {f['gap']:.2f})")
    if f["tp"]:
        bits.append("contains a turning point")
    if f["debt"] > 0.35:
        bits.append(f"needs some backstory (debt {f['debt']:.2f})")
    if f["resolution"] > 0.3:
        bits.append(f"partly resolves itself (resolution {f['resolution']:.2f})")
    return {"hook": hook, "cliffhanger": cliff, "why": "; ".join(bits) + ".", "open_questions": open_txt}


def to_promo(c: Clip, ep: Episode, ann, questions, pid: str) -> dict:
    d = describe(c, ep, ann, questions)
    return {"id": pid, "source": f"v2-{c.shape}", "title": f"Ep{ep.num} {c.shape}",
            "segments": [{"ep": ep.num, "start": s, "end": e, "start_ts": fmt(s), "end_ts": fmt(e)} for s, e in c.segments],
            "duration": round(c.duration, 2), "signals": c.feats, "signal_score": c.score, "flags": [], **d}
