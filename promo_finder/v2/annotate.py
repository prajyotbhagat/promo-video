"""Step 1: annotate each beat with narrative signals (one LLM pass per show).

Output per beat:
  A  arousal 0-3            (how activating: calm talk 0 ... slap/scream/explosion 3)
  V  valence -2..2          (for the sympathetic lead)
  S  surprise 0-3           (unexpected given only what came before - backward-looking surprise,
                             Wilmot & Keller 2020)
  C  context need 0-2       (0 a stranger gets it, 2 meaningless without earlier episodes)
  TP turning point          (TRIPOD types, Papalampidi et al. 2019: OPP opportunity, CHG change of
                             plans, PNR point of no return, SET major setback, CLX climax/resolution)
  OPEN / CLOSE              questions the beat raises / answers in a viewer's mind
                            (information gaps, Loewenstein 1994)
Questions carry an importance 1-3.

The format is pipe-separated text, not JSON: ~40% fewer output tokens and a broken line only
loses that line, not the whole reply.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

from ..heuristics import beat_intensity
from ..parser import Beat, Episode, fmt

TPS = {"OPP", "CHG", "PNR", "SET", "CLX"}

SYSTEM = """You are a story analyst producing labels for a computational model of curiosity and suspense
in short vertical dramas. Label every listed beat. Be consistent across episodes. Use the full scales.

FIELDS (per beat)
A  arousal 0-3: 0 calm/neutral, 1 tense talk, 2 heated confrontation / strong emotion, 3 explosive
   (slap, attack, scream, shocking reveal, explosion).
V  valence -2..2 for the heroine/sympathetic side: -2 humiliated or in danger, +2 triumph.
S  surprise 0-3: how unexpected this beat is given ONLY the beats before it in this episode.
C  context need 0-2 for a stranger who has never seen the show: 0 understandable on its own,
   1 earlier events help, 2 meaningless without earlier episodes (unexplained names, callbacks).
TP turning point inside this episode's own mini-arc, or -: OPP opportunity, CHG change of plans,
   PNR point of no return, SET major setback, CLX climax/resolution. Use at most 3 per episode.
OPEN ids of questions this beat makes the viewer ask (e.g. "Why is he holding her hand?",
   "Will she be executed?"). CLOSE ids of questions this beat answers. Use - if none.
   Put each question on the EXACT beat that triggers it, and its CLOSE on the exact beat that answers
   it (often later, sometimes never in this episode). Typically 4-8 questions per episode. Include the
   big ones (Will she be saved? Who is he really?) and the immediate ones (What is in the box?).

OUTPUT FORMAT (plain text, nothing else):
## EPISODE <n>
<beat_id>|A|V|S|C|TP|OPEN|CLOSE
...
QUESTIONS
<qid>|importance 1-3|question text
...
Question ids must look like e<ep>q<k>, e.g. e01q1, e01q2 (two-digit episode number)."""


@dataclass
class Ann:
    a: float = 0.0
    v: float = 0.0
    s: float = 0.0
    c: float = 0.0
    tp: str = ""
    opens: list[str] = field(default_factory=list)
    closes: list[str] = field(default_factory=list)
    labelled: bool = False


def _keep(b: Beat) -> bool:
    return b.kind != "camera" or beat_intensity(b) >= 5


def episode_block(ep: Episode) -> str:
    lines = [f"## EPISODE {ep.num} - logline (for you only): {ep.logline}"]
    for b in ep.beats:
        if _keep(b):
            lines.append(b.line()[:220])
    return "\n".join(lines)


def prompt(eps: list[Episode]) -> str:
    body = "\n\n".join(episode_block(e) for e in eps)
    return f"""Label every beat below (camera-only beats are omitted).

{body}

Now output the labels for ALL {sum(sum(_keep(b) for b in e.beats) for e in eps)} beats in the exact format."""


def _num(x, lo, hi):
    try:
        return max(lo, min(hi, float(x)))
    except (TypeError, ValueError):
        return 0.0


def _ids(x: str) -> list[str]:
    return [t.strip().lower() for t in re.split(r"[,; ]+", x or "") if re.match(r"e\d+q\d+", t.strip().lower())]


def parse(text: str, ann: dict[str, Ann], questions: dict[str, dict]):
    in_q = False
    for raw in text.splitlines():
        line = raw.strip().rstrip(",").strip().strip('"').strip("`").strip()
        if not line:
            continue
        if line.upper().startswith("QUESTIONS"):
            in_q = True
            continue
        if line.startswith("##"):
            in_q = False
            continue
        parts = [p.strip() for p in line.split("|")]
        if in_q or re.match(r"^e\d+q\d+\|", line, re.I):
            if len(parts) >= 3 and re.match(r"e\d+q\d+", parts[0].lower()):
                questions[parts[0].lower()] = {"importance": _num(re.sub(r"\D", "", parts[1]) or 1, 1, 3),
                                               "text": "|".join(parts[2:])}
            continue
        if len(parts) >= 6 and parts[0] in ann:
            x = ann[parts[0]]
            x.a, x.v, x.s, x.c = _num(parts[1], 0, 3), _num(parts[2], -2, 2), _num(parts[3], 0, 3), _num(parts[4], 0, 2)
            tp = parts[5].upper()
            x.tp = tp if tp in TPS else ""
            x.opens = _ids(parts[6]) if len(parts) > 6 else []
            x.closes = _ids(parts[7]) if len(parts) > 7 else []
            x.labelled = True


def ep_cached(llm, ep: Episode) -> bool:
    from ..cache import store
    return store().exists("annot", _ep_key(llm, ep))


def _ep_key(llm, ep: Episode) -> str:
    """Content address of one episode's annotation: model + instructions + the episode text itself.
    If none of these change, the episode is never sent to the model again."""
    import hashlib
    return hashlib.sha256(json.dumps([llm.name, SYSTEM, episode_block(ep)]).encode()).hexdigest()[:24]


def annotate(llm, eps: list[Episode], batch: int, raw_dir: Path, log=print, errors=None, stats=None):
    ann = {b.id: Ann() for e in eps for b in e.beats}
    questions: dict[str, dict] = {}
    # 1) per-episode content cache: re-runs and weekly additions only pay for new/changed episodes
    from ..cache import store
    st = store()
    todo = []
    for e in eps:
        hit = st.get("annot", _ep_key(llm, e))
        if hit is not None:
            d = json.loads(hit)
            for bid, x in d["beats"].items():
                if bid in ann:
                    ann[bid] = Ann(**x)
            questions.update(d["questions"])
        else:
            todo.append(e)
    if stats is not None:
        stats.update({"episodes": len(eps), "from_episode_cache": len(eps) - len(todo), "annotated_now": len(todo)})
    if len(todo) < len(eps):
        log(f"  annotation cache: {len(eps) - len(todo)}/{len(eps)} episodes reused, {len(todo)} to annotate")
    groups = [todo[i:i + batch] for i in range(0, len(todo), batch)]
    for g in groups:
        tag = f"ep{g[0].num:03d}-{g[-1].num:03d}"
        try:
            text = llm.complete(SYSTEM, prompt(g), json_mode=False)
        except Exception as e:  # keep going: unlabelled beats fall back to rules
            (errors if errors is not None else []).append(f"annotate {tag}: {str(e)[:200]}")
            log(f"  annotate {tag} FAILED: {str(e)[:160]}")
            continue
        (raw_dir / f"annot_{tag}.txt").write_text(text)
        parse(text, ann, questions)
        want = sum(_keep(b) for e in g for b in e.beats)
        got = sum(ann[b.id].labelled for e in g for b in e.beats if _keep(b))
        if got < 0.6 * want:  # truncated or malformed: retry once, asking for every episode explicitly
            log(f"  annotate {tag}: only {got}/{want} labelled, retrying once")
            try:
                text2 = llm.complete(SYSTEM, prompt(g) + "\nYour reply must cover EVERY episode listed, one line per beat.",
                                     json_mode=False)
                (raw_dir / f"annot_{tag}_retry.txt").write_text(text2)
                parse(text2, ann, questions)
                got = sum(ann[b.id].labelled for e in g for b in e.beats if _keep(b))
            except Exception as e:
                (errors if errors is not None else []).append(f"annotate retry {tag}: {str(e)[:200]}")
        log(f"  annotate {tag}: {got}/{want} beats labelled, {sum(1 for q in questions if int(q[1:3]) in {e.num for e in g})} questions")
        if got >= 0.6 * want:  # only cache good annotations
            for e in g:
                qs = {k: v for k, v in questions.items() if re.match(rf"e0*{e.num}q", k)}
                st.set("annot", _ep_key(llm, e), json.dumps(
                    {"ep": e.num, "beats": {b.id: asdict(ann[b.id]) for b in e.beats if ann[b.id].labelled},
                     "questions": qs}))
    # fallback for beats the model skipped: map rule-based intensity onto arousal
    for e in eps:
        for b in e.beats:
            x = ann[b.id]
            if not x.labelled and _keep(b):
                x.a = round(min(3.0, beat_intensity(b) / 3.3), 1)
    return ann, questions


def save(path: Path, eps, ann, questions):
    with open(path, "w") as f:
        for e in eps:
            for b in e.beats:
                x = ann[b.id]
                if x.labelled:
                    f.write(json.dumps({"beat": b.id, "t": fmt(b.start), "kind": b.kind, "text": b.text[:120], **asdict(x)}) + "\n")
    (path.parent / "questions.json").write_text(json.dumps(questions, indent=1, ensure_ascii=False))
