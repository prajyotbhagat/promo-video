"""End-to-end pipeline.

  parse scripts (code)
    -> heuristic candidates per episode (code, free)
    -> LLM: pick promos per episode by beat id (1 call / episode)
    -> validate + fix timestamps (code)
    -> LLM: join moments across episodes (1 call / show)
    -> LLM: "cold viewer" test of the shortlist with zero show context
    -> rank (code: blend of LLM, heuristic and cold-viewer scores)
"""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import prompts
from .heuristics import beat_intensity, beats_in, best_windows, overlap, score_window
from .llm import LLM
from .parser import Episode, fmt, load_show
from .validate import Index, check

SERIOUS = ("stated hook", "hook late", "speakers", "resolved", "multi-segment duration", "overlap", "opens mid-thought",
           "cliffhanger changed")


def as_items(data, key):
    """Models sometimes return {"promos": [...]} and sometimes a bare [...] - accept both."""
    if isinstance(data, list):
        out = []
        for x in data:
            if isinstance(x, dict) and isinstance(x.get(key), list):
                out += [y for y in x[key] if isinstance(y, dict)]  # [{"promos": [...]}, ...] per episode
            elif isinstance(x, dict):
                out.append(x)
        return out
    if isinstance(data, dict):
        v = data.get(key)
        if isinstance(v, list):
            return [x for x in v if isinstance(x, dict)]
        if "segments" in data or "id" in data:
            return [data]
    return []


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ----------------------------------------------------------------------------
def heuristic_promos(eps: list[Episode], k: int) -> list[dict]:
    out = []
    for ep in eps:
        for w in best_windows(ep, k):
            s, e = w["segments"][0]
            out.append(_heuristic_text({"source": "heuristic", "title": f"Ep{ep.num} auto-window",
                                        "segments": [{"ep": ep.num, "start": s, "end": e}]}, ep))
    return out


def _heuristic_text(p: dict, ep: Episode) -> dict:
    seg = p["segments"]
    inside = [b for s in seg for b in beats_in(ep, s["start"], s["end"])]
    s0 = seg[0]["start"]
    first = [b for b in inside if b.start < s0 + 3 and b.kind in ("dialogue", "action")]
    first.sort(key=lambda b: -beat_intensity(b))
    lines = [b for b in inside if b.kind == "dialogue"]
    hb = first[0] if first else (lines[0] if lines else inside[0])
    p["hook"] = (f'{hb.speaker}: "{hb.text}"' if hb.kind == "dialogue" else hb.text)
    lb = lines[-1] if lines else inside[-1]
    p["cliffhanger"] = f'{lb.speaker}: "{lb.text}"' if lb.kind == "dialogue" else lb.text
    sc = score_window(ep, [(s["start"], s["end"]) for s in seg])
    bits = []
    bits.append("opens on a high-intensity beat" if sc.hook >= 6 else "opens mid-conflict")
    bits.append("cuts on the episode's own cliffhanger" if "ends on episode cliffhanger" in sc.notes
                else ("cuts on an unanswered line" if sc.cliff >= 5 else "ending is soft"))
    bits.append("premise is clear" if sc.standalone >= 7 else "needs some context")
    p["why"] = "Rule-based pick: " + ", ".join(bits) + "."
    p["risks"] = "; ".join(sc.notes)
    return p


# ----------------------------------------------------------------------------
def llm_episode_promos(llm: LLM, ep: Episode, show: str, k: int, raw_dir: Path) -> list[dict]:
    data = llm.complete_json(prompts.SYSTEM_FIND, prompts.episode_prompt(ep, show, k))
    (raw_dir / f"ep{ep.num:03d}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False))
    out = []
    for p in as_items(data, "promos")[: k + 1]:
        p["source"] = "llm"
        out.append(p)
    return out


def llm_batch_promos(llm: LLM, eps: list[Episode], show: str, k: int, raw_dir: Path) -> list[dict]:
    data = llm.complete_json(prompts.SYSTEM_FIND, prompts.batch_prompt(eps, show, k))
    tag = f"ep{eps[0].num:03d}-{eps[-1].num:03d}"
    (raw_dir / f"{tag}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False))
    out = []
    for p in as_items(data, "promos"):
        p["source"] = "llm"
        out.append(p)
    return out


def cross_digest(idx: Index, promos: list[dict], per_ep: int = 2) -> str:
    by_ep: dict[int, list[dict]] = {}
    for p in promos:
        if p["status"] != "rejected" and len({s["ep"] for s in p["segments"]}) == 1:
            by_ep.setdefault(p["segments"][0]["ep"], []).append(p)
    out = []
    for n in sorted(by_ep):
        ep = idx.eps[n]
        best = sorted(by_ep[n], key=lambda p: -_llm_q(p))[:per_ep]
        out.append(f"\n== EPISODE {n}: {ep.logline}")
        for j, p in enumerate(best):
            for s in p["segments"]:
                out.append(f"-- moment {n}.{j} ({p.get('title', '')})")
                for b in beats_in(ep, s["start"], s["end"]):
                    if b.kind != "camera" or beat_intensity(b) >= 5:
                        out.append("   " + b.line()[:170])
    return "\n".join(out)


def cold_test(llm: LLM, idx: Index, promos: list[dict], raw_dir: Path, batch: int = 8, errors=None):
    def transcript(p):
        parts = []
        for i, s in enumerate(p["segments"]):
            if i:
                parts.append("[CUT]")
            for b in beats_in(idx.eps[s["ep"]], s["start"], s["end"]):
                if b.kind == "dialogue":
                    parts.append(f'{b.speaker}: "{b.text}"')
                elif b.kind == "action" or (b.kind == "camera" and beat_intensity(b) >= 5):
                    parts.append(f"(on screen: {b.text})")
        return "\n".join(parts)

    for i in range(0, len(promos), batch):
        chunk = promos[i:i + batch]
        items = [(p["id"], transcript(p)) for p in chunk]
        try:
            data = llm.complete_json(prompts.SYSTEM_COLD, prompts.cold_prompt(items))
        except Exception as e:  # never let QC kill the run
            log(f"cold test batch failed: {str(e)[:200]}")
            if errors is not None:
                errors.append(f"cold-viewer batch {i // batch}: {str(e)[:200]}")
            continue
        (raw_dir / f"cold_{i // batch:02d}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False))
        res = {}
        for c in as_items(data, "clips"):
            m = re.search(r"[PX]\d+", str(c.get("id", "")))  # model may echo "CLIP P004"
            if m:
                res[m.group(0)] = c
        for p in chunk:
            if p["id"] in res:
                p["cold"] = res[p["id"]]


# ----------------------------------------------------------------------------
def _num(x, default=5.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _llm_q(p):
    s = p.get("scores") or {}
    if not s:
        return 0.0
    return (_num(s.get("hook")) + _num(s.get("cliffhanger")) + _num(s.get("standalone"))) / 30


def _cold_q(p):
    c = p.get("cold")
    if not c:
        return None
    return (_num(c.get("stop_scrolling")) + _num(c.get("need_to_know_next")) + _num(c.get("understood"))) / 30


def score_all(idx: Index, promos: list[dict], mode: str):
    for p in promos:
        segs = p["segments"]
        h = [score_window(idx.eps[s["ep"]], [(s["start"], s["end"])]) for s in segs]
        # hook from first segment, cliffhanger from last, rest averaged
        p["heuristic"] = {"hook": h[0].hook, "cliffhanger": h[-1].cliff,
                          "intensity": round(sum(x.intensity for x in h) / len(h), 2),
                          "standalone": round(sum(x.standalone for x in h) / len(h), 2),
                          "total": round(0.35 * h[0].hook + 0.35 * h[-1].cliff +
                                         0.15 * sum(x.intensity for x in h) / len(h) +
                                         0.15 * sum(x.standalone for x in h) / len(h), 3),
                          "notes": sorted({n for x in h for n in x.notes})}
        if h[0].hook < 4:
            p["flags"].append("warn: hook late/weak by rule check (little happens in first 3s)")
        for n in p["heuristic"]["notes"]:
            if "speakers" in n or "mid-thought" in n or "resolved" in n:
                p["flags"].append("warn: " + n)
    totals = sorted(p["heuristic"]["total"] for p in promos)

    def pct(v):
        return sum(t <= v for t in totals) / max(1, len(totals))

    for p in promos:
        hq = pct(p["heuristic"]["total"])
        pen = 0.04 * sum(any(k in f for k in SERIOUS) for f in p["flags"])
        pen += 0.15 * sum("not in the clip at all" in f for f in p["flags"])
        early = 0.03 if all(s["ep"] <= 10 for s in p["segments"]) else 0.0
        if mode == "heuristic":
            base = p["heuristic"]["total"] / 10
            comp = {"heuristic": round(base, 3)}
        else:
            lq, cq = _llm_q(p), _cold_q(p)
            if cq is None:
                base = 0.65 * lq + 0.35 * hq
            else:
                base = 0.5 * lq + 0.2 * hq + 0.3 * cq
            comp = {"llm": round(lq, 3), "heuristic_pct": round(hq, 3), "cold_viewer": None if cq is None else round(cq, 3)}
        p["rank_score"] = round(base - pen + early, 4)
        p["score_parts"] = {**comp, "flag_penalty": round(pen, 3), "early_episode_bonus": early}


def dedupe(promos: list[dict]) -> list[dict]:
    keep = []
    for p in sorted(promos, key=lambda p: -p["rank_score"]):
        dup = False
        for q in keep:
            for a in p["segments"]:
                for b in q["segments"]:
                    if a["ep"] == b["ep"] and overlap((a["start"], a["end"]), (b["start"], b["end"])) > 0.7 \
                            and len(p["segments"]) == len(q["segments"]):
                        dup = True
        if dup:
            p["status"] = "duplicate"
        else:
            keep.append(p)
    return keep


def post_first(promos: list[dict], n: int = 10, per_ep: int = 2) -> list[dict]:
    chosen, count = [], {}
    for p in promos:
        eps = {s["ep"] for s in p["segments"]}
        if any(count.get(e, 0) >= per_ep for e in eps):
            continue
        chosen.append(p)
        for e in eps:
            count[e] = count.get(e, 0) + 1
        if len(chosen) >= n:
            break
    return chosen


# ----------------------------------------------------------------------------
def run(input_dir: str, out_dir: str, model: str = "heuristic", per_episode: int = 3,
        cross: bool = True, cold_top: int = 16, episodes: list[int] | None = None, workers: int = 4,
        batch: int = 1, compact: bool = False) -> dict:
    prompts.COMPACT = compact
    t0 = time.time()
    out = Path(out_dir)
    raw = out / "raw_llm"
    out.mkdir(parents=True, exist_ok=True)
    eps = load_show(input_dir)
    if episodes:
        eps = [e for e in eps if e.num in set(episodes)]
    if not eps:
        raise SystemExit(f"no episodes parsed from {input_dir}")
    show = Path(input_dir).resolve().name
    sub = [p for p in Path(input_dir).iterdir() if p.is_dir()]
    if len(sub) == 1:
        show = sub[0].name
    idx = Index(eps)
    log(f"parsed {len(eps)} episodes, {len(idx.beats)} beats - show: {show}")

    llm = None
    errors = []
    if model == "heuristic":
        promos = heuristic_promos(eps, per_episode)
    else:
        llm = LLM(model)
        raw.mkdir(exist_ok=True)
        log(f"model: {llm.name}")
        promos = []

        groups = [eps[i:i + batch] for i in range(0, len(eps), batch)]

        def work(group):
            label = f"{group[0].num}" if len(group) == 1 else f"{group[0].num}-{group[-1].num}"
            try:
                if len(group) == 1:
                    return label, llm_episode_promos(llm, group[0], show, per_episode, raw)
                return label, llm_batch_promos(llm, group, show, per_episode, raw)
            except Exception as e:
                errors.append(f"episodes {label}: {str(e)[:300]}")
                log(f"  ep {label} FAILED: {str(e)[:200]}")
                return label, []

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for label, res in ex.map(work, groups):
                promos.extend(res)
                log(f"  ep {label}: {len(res)} promos")

    for i, p in enumerate(promos):
        p["id"] = f"P{i + 1:03d}"
        check(idx, p)

    if llm and cross and len(eps) > 1:
        try:
            digest = cross_digest(idx, promos)
            data = llm.complete_json(prompts.SYSTEM_CROSS, prompts.cross_prompt(show, digest))
            (raw / "cross_episode.json").write_text(json.dumps(data, indent=2, ensure_ascii=False))
            for j, p in enumerate(as_items(data, "promos")):
                p["source"], p["id"] = "llm-cross", f"X{j + 1:02d}"
                check(idx, p)
                promos.append(p)
            log(f"cross-episode: {len(as_items(data, 'promos'))} promos")
        except Exception as e:
            errors.append(f"cross-episode: {e}")
            log(f"cross-episode FAILED: {e}")

    valid = [p for p in promos if p["status"] != "rejected"]
    rejected = [p for p in promos if p["status"] == "rejected"]
    mode = "heuristic" if llm is None else "llm"
    score_all(idx, valid, mode)
    valid = dedupe(valid)

    if llm and cold_top:
        short = sorted(valid, key=lambda p: -p["rank_score"])[:cold_top]
        log(f"cold-viewer test on top {len(short)}")
        cold_test(llm, idx, short, raw, errors=errors)
        score_all(idx, valid, mode)  # re-score with cold-viewer signal (flags are idempotent enough)
        for p in valid:
            p["flags"] = list(dict.fromkeys(p["flags"]))

    valid.sort(key=lambda p: -p["rank_score"])
    for r, p in enumerate(valid, 1):
        p["rank"] = r
        for s in p["segments"]:
            s["start_ts"], s["end_ts"] = fmt(s["start"]), fmt(s["end"])
    top = post_first(valid)

    meta = {
        "show": show, "episodes": len(eps), "beats": len(idx.beats), "model": llm.name if llm else "heuristic",
        "seconds": round(time.time() - t0, 1), "promos_valid": len(valid), "promos_rejected": len(rejected),
        "llm_calls": llm.calls if llm else 0, "cache_hits": llm.cache_hits if llm else 0,
        "usage": llm.usage if llm else {}, "approx_input_chars": llm.in_chars if llm else 0,
        "errors": errors, "batch": batch, "post_first_ids": [p["id"] for p in top],
    }
    from .report import write_all
    write_all(out, idx, valid, rejected, top, meta)
    log(f"done in {meta['seconds']}s -> {out}")
    return meta
