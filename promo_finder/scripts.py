"""Write a script for each promo clip, in the same format as the episode scripts.

    python -m promo_finder.scripts outputs/v3 --top 10

For each promo, <run>/promo_scripts/<id>.txt holds:
  - a header: promo id, source episode(s) and timestamps, duration, hook, cliffhanger, open questions
  - one SCENE per continuous piece of footage (a jump cut starts a new scene)
  - every CAMERA / ACTION / SOUND FX / dialogue block of the clip, re-timed so 00:00.00 is the first
    frame of the promo, with the original episode time noted in the scene header

The files use the Kuku screenplay format, so the same parser (and any tool that reads episode
scripts) can read a promo script. A combined promo_scripts.md is written for reading and review.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .heuristics import beats_in
from .parser import Beat, Episode, fmt, load_show


def scene_headings(ep: Episode) -> dict[int, str]:
    text = Path(ep.path).read_text(encoding="utf-8", errors="replace")
    out, cur = {}, None
    for line in text.splitlines():
        m = re.match(r"^\s*SCENE\s+(\d+)", line, re.I)
        if m:
            cur = int(m.group(1))
            continue
        m = re.match(r"^\s*Scene heading:\s*(.+)$", line, re.I)
        if m and cur is not None:
            out[cur] = m.group(1).strip()
    return out


def _label(b: Beat) -> str:
    if b.kind == "camera":
        return "CAMERA"
    if b.kind == "action":
        return "ACTION"
    if b.kind == "sfx":
        return "SOUND FX"
    tone = f" ({b.tone})" if b.tone else ""
    return f"{b.speaker}{tone}"


def promo_script(promo: dict, emap: dict[int, Episode], show: str = "") -> str:
    segs = promo["segments"]
    total = sum(s["end"] - s["start"] for s in segs)
    src = " + ".join(f"Ep{s['ep']} {fmt(s['start'])}-{fmt(s['end'])}" for s in segs)
    hook = str(promo.get("hook", "")).replace("\n", " ")
    cliff = str(promo.get("cliffhanger", "")).replace("\n", " ")
    L = [f"EPISODE NUMBER: Promo {promo['id']}",
         f"EPISODE LOGLINE: {show + ' - ' if show else ''}promo cut from {src}. Hook: {hook}",
         f"CONTINUITY NOTES: Standalone promo ({total:.1f}s). Cliffhanger: {cliff}",
         "", "--- SCRIPT START ---", ""]
    off, n = 0.0, 0
    for si, s in enumerate(segs):
        ep = emap[s["ep"]]
        heads = scene_headings(ep)
        beats = beats_in(ep, s["start"], s["end"])
        # one promo scene per (segment, source scene) run
        groups: list[list[Beat]] = []
        for b in beats:
            if groups and groups[-1][0].scene == b.scene:
                groups[-1].append(b)
            else:
                groups.append([b])
        for gi, g in enumerate(groups):
            n += 1
            g_start = s["start"] if gi == 0 else g[0].start
            g_end = s["end"] if gi == len(groups) - 1 else groups[gi + 1][0].start
            heading = heads.get(g[0].scene, "INT./EXT. - CONTINUOUS")
            cut = "JUMP CUT FROM PREVIOUS SHOT. " if si > 0 and gi == 0 else ""
            L += [f"SCENE {n:02d}",
                  f"Scene heading: {heading}",
                  f"Scene start: {fmt(off + g_start - s['start'])}",
                  f"Scene end: {fmt(off + g_end - s['start'])}",
                  f"Source: {cut}Episode {s['ep']}, {fmt(g_start)}-{fmt(g_end)}",
                  "", "Actions and dialogue:", ""]
            for b in g:
                a, z = off + b.start - s["start"], off + min(b.end, s["end"]) - s["start"]
                body = f'"{b.text}"' if b.kind == "dialogue" else b.text
                L += [f"**{_label(b)}:**", f"Start: ({fmt(a)}) | End: ({fmt(z)})", body, ""]
        off += s["end"] - s["start"]
    L += ["--- SCRIPT END ---", ""]
    return "\n".join(L)


def promo_markdown(promo: dict, emap: dict[int, Episode], rank: int | None = None) -> str:
    segs = promo["segments"]
    total = sum(s["end"] - s["start"] for s in segs)
    head = f"## {'#' + str(rank) + ' - ' if rank else ''}Promo {promo['id']} ({total:.1f}s)"
    L = [head, "",
         "- **Source:** " + " + ".join(f"Ep{s['ep']} {fmt(s['start'])}-{fmt(s['end'])}" for s in segs),
         f"- **Hook (0-3s):** {promo.get('hook', '')}",
         f"- **Cliffhanger:** {promo.get('cliffhanger', '')}", "",
         "| Time | | Line |", "|---|---|---|"]
    off = 0.0
    for si, s in enumerate(segs):
        if si:
            L.append("| | | *— jump cut —* |")
        for b in beats_in(emap[s["ep"]], s["start"], s["end"]):
            t = off + b.start - s["start"]
            who = f"**{b.speaker}**" if b.kind == "dialogue" else f"*{b.kind}*"
            txt = (f'"{b.text}"' if b.kind == "dialogue" else b.text).replace("|", "/")
            L.append(f"| {fmt(t)} | {who} | {txt} |")
        off += s["end"] - s["start"]
    return "\n".join(L) + "\n"


def write_scripts(run_dir, top=10, ids=None, out=None, input_dir=None) -> list[Path]:
    run_dir = Path(run_dir)
    run = json.loads((run_dir / "promos.json").read_text())
    in_dir = Path(input_dir or run["meta"].get("input_dir") or "input")
    emap = {e.num: e for e in load_show(in_dir)}
    subs = [p for p in in_dir.iterdir() if p.is_dir()] if in_dir.is_dir() else []
    show = subs[0].name if len(subs) == 1 else in_dir.name
    by = {p["id"]: p for p in run["promos"]}
    order = ids or [i for i in run.get("post_first", []) if i in by][:top]
    out = Path(out or run_dir / "promo_scripts")
    out.mkdir(parents=True, exist_ok=True)
    files, md = [], [f"# Promo scripts - {show}", ""]
    for r, pid in enumerate(order, 1):
        p = by[pid]
        f = out / f"{pid}.txt"
        f.write_text(promo_script(p, emap, show), encoding="utf-8")
        files.append(f)
        md.append(promo_markdown(p, emap, r))
    (out / "promo_scripts.md").write_text("\n".join(md), encoding="utf-8")
    return files


def main():
    ap = argparse.ArgumentParser(prog="promo_finder.scripts", description="Write a script for each promo clip.")
    ap.add_argument("run", help="run folder with promos.json")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--ids", help="comma-separated promo ids")
    ap.add_argument("-o", "--out")
    ap.add_argument("--input", help="scripts folder if it moved since the run")
    a = ap.parse_args()
    files = write_scripts(a.run, a.top, a.ids.split(",") if a.ids else None, a.out, a.input)
    print(f"wrote {len(files)} promo scripts + promo_scripts.md to {files[0].parent if files else '-'}")


if __name__ == "__main__":
    main()
