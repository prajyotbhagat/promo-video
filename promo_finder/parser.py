"""Parse Kuku-style timestamped screenplays into a flat list of beats.

A beat is one CAMERA / ACTION / SOUND FX / dialogue block with its own
start and end time. Everything downstream (heuristics, LLM prompts,
validation) works on beats, and the LLM refers to beats by id - it never
writes timestamps itself.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

TIME_RE = re.compile(r"(\d+):(\d+(?:\.\d+)?)")
BLOCK_RE = re.compile(r"^\*\*(.+?):\*\*\s*$")
# Accept "Start: (00:01.00) | End: (00:04.80)" and "Start: 00:01.00 | End: 00:04.80" (seen in a second show)
RANGE_RE = re.compile(r"Start:\s*\(?\s*(\d+:\d+(?:\.\d+)?)\s*\)?\s*\|\s*End:\s*\(?\s*(\d+:\d+(?:\.\d+)?)\s*\)?")
EP_NUM_RE = re.compile(r"episode-(\d+)", re.I)


def to_sec(ts: str) -> float:
    m = TIME_RE.search(ts)
    if not m:
        raise ValueError(f"bad timestamp {ts!r}")
    return int(m.group(1)) * 60 + float(m.group(2))


def fmt(sec: float) -> str:
    """Seconds -> MM:SS.ss (same format as the scripts)."""
    sec = max(0.0, sec)
    m = int(sec // 60)
    return f"{m:02d}:{sec - m * 60:05.2f}"


@dataclass
class Beat:
    id: str            # e.g. "e03b041" (unique across the show)
    ep: int
    idx: int           # position within episode
    scene: int
    kind: str          # camera | action | sfx | dialogue
    speaker: str       # dialogue only, normalised name
    tone: str          # parenthetical(s) e.g. "Gloating, Smug" / "V.O."
    start: float
    end: float
    text: str

    def line(self) -> str:
        """Compact one-line form used in prompts and reports."""
        t = f"[{self.id} {fmt(self.start)}-{fmt(self.end)}]"
        if self.kind == "dialogue":
            tone = f" ({self.tone})" if self.tone else ""
            return f'{t} {self.speaker}{tone}: {self.text}'
        return f"{t} {self.kind.upper()}: {self.text}"


@dataclass
class Episode:
    num: int
    title: str
    logline: str
    continuity: str
    duration: float
    path: str
    beats: list[Beat] = field(default_factory=list)

    def to_dict(self):
        d = asdict(self)
        d["beats"] = [asdict(b) for b in self.beats]
        return d


def _split_speaker(label: str) -> tuple[str, str]:
    """'JULIANA (V.O.) (Tender, Safe)' -> ('JULIANA', 'V.O., Tender, Safe')."""
    parens = re.findall(r"\(([^)]*)\)", label)
    name = re.sub(r"\([^)]*\)", "", label).strip()
    return name, ", ".join(p.strip() for p in parens)


def parse_episode(path: str | Path) -> Episode:
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    def header(key: str) -> str:
        m = re.search(rf"^{key}:\s*(.*)$", text, re.M | re.I)
        return m.group(1).strip() if m else ""

    title = header("EPISODE NUMBER")
    m = re.search(r"(\d+)", title) or EP_NUM_RE.search(path.name)
    num = int(m.group(1)) if m else 0

    beats: list[Beat] = []
    scene = 0
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        if re.match(r"^SCENE\s+\d+", ln, re.I):
            scene = int(re.findall(r"\d+", ln)[0])
        bm = BLOCK_RE.match(ln)
        if bm and i + 1 < len(lines):
            label = bm.group(1).strip()
            rm = RANGE_RE.search(lines[i + 1])
            if rm:
                # body = following non-empty lines until blank / next block
                body = []
                j = i + 2
                while j < len(lines) and lines[j].strip() and not BLOCK_RE.match(lines[j].strip()):
                    body.append(lines[j].strip())
                    j += 1
                upper = label.upper()
                if upper == "CAMERA":
                    kind, spk, tone = "camera", "", ""
                elif upper == "ACTION":
                    kind, spk, tone = "action", "", ""
                elif upper.startswith("SOUND"):
                    kind, spk, tone = "sfx", "", ""
                else:
                    kind = "dialogue"
                    spk, tone = _split_speaker(label)
                body_txt = " ".join(body).strip().strip('"').strip("“”")
                st, en = to_sec(rm.group(1)), to_sec(rm.group(2))
                if en < st:
                    en = st
                beats.append(Beat("", num, 0, scene, kind, spk, tone, st, en, body_txt))
                i = j
                continue
        i += 1

    beats.sort(key=lambda b: (b.start, b.end))
    for k, b in enumerate(beats):
        b.idx = k
        b.id = f"e{num:02d}b{k:03d}"
    scene_ends = [to_sec(x) for x in re.findall(r"^Scene end:\s*(\S+)", text, re.M | re.I)]
    duration = max([b.end for b in beats] + scene_ends + [0.0])
    return Episode(num, title, header("EPISODE LOGLINE"), header("CONTINUITY NOTES"),
                   duration, str(path), beats)


def load_show(folder: str | Path) -> list[Episode]:
    folder = Path(folder)
    files = sorted(p for p in folder.rglob("*.txt") if not p.name.startswith("."))
    eps = [parse_episode(p) for p in files]
    eps = [e for e in eps if e.beats]
    eps.sort(key=lambda e: e.num)
    return eps
