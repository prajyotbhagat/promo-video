"""v3: budget-aware promo finder with rank fusion.

Budget levels (each adds candidates and/or rankers; the final ranking is always RRF fusion):
  B0  rules only                                   0 LLM calls
  B1  + beat annotation -> curiosity optimizer     ~10 small-model calls (cached per episode)
  B2  + LLM picker (second candidate generator)    ~20 small-model calls          <- default, best value
  B3  + pairwise tournament on a shortlist          +4 judge calls
  B4  + persona audience panel on finalists         +5 small-model calls
On our labelled test, B2 matched B4's accuracy (13/14) with ~30% fewer calls and no stronger model.
"""
from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path

from ..heuristics import score_window
from ..llm import LLM
from ..parser import fmt, load_show
from ..pipeline import heuristic_promos
from ..validate import Index, check
from ..v2 import annotate as A
from ..v2 import audience as AU
from ..v2 import tournament as T
from ..v2.optimize import Scorer, search, to_promo
from ..v2.pipeline import seed_promos, shortlist
from .fusion import _main, consensus_ranks, cover, fuse

LEVELS = ["B0", "B1", "B2", "B3", "B4"]


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def rule_score(ep, segs):
    h = [score_window(ep, [x]) for x in segs]
    return 0.35 * h[0].hook + 0.35 * h[-1].cliff + 0.15 * sum(x.intensity for x in h) / len(h) \
        + 0.15 * sum(x.standalone for x in h) / len(h)


def _last_line(p, emap):
    from ..heuristics import beats_in
    s = p["segments"][-1]
    lines = [b for b in beats_in(emap[s["ep"]], s["start"], s["end"]) if b.kind == "dialogue"]
    return lines[-1] if lines else None


def _words(p, emap):
    from ..heuristics import beats_in
    ws = set()
    for s in p["segments"]:
        for b in beats_in(emap[s["ep"]], s["start"], s["end"]):
            if b.kind in ("dialogue", "action"):
                ws |= set(re.findall(r"[a-z']+", b.text.lower()))
    return ws


def moments(ranked, emap=None, sim=0.8):
    """Group clips into moments: same episode and >= 50% overlap, OR (across episodes) nearly the same
    words (Jaccard >= 0.8) - recaps, flashbacks and templated scenes would otherwise fill the top 10
    with the same footage. Keeps the best-ranked clip of each moment."""
    reps = []
    for p in ranked:
        ep, w = _main(p)
        ws = _words(p, emap) if emap else set()
        hit = None
        for r in reps:
            if r["_ep"] == ep and cover(w, r["_w"]) >= 0.5:
                hit = r
                break
            if emap and ws and r["_ws"] and len(ws & r["_ws"]) / len(ws | r["_ws"]) >= sim:
                hit = r
                r["repeats_in_episodes"].add(ep)
                break
        if hit:
            hit["also_found_by"].add(p["source"])
            continue
        reps.append({**p, "_ep": ep, "_w": w, "_ws": ws, "also_found_by": {p["source"]}, "repeats_in_episodes": set()})
    for r in reps:
        r.pop("_ep"), r.pop("_w"), r.pop("_ws")
        r["also_found_by"] = sorted(r["also_found_by"])
        r["repeats_in_episodes"] = sorted(r["repeats_in_episodes"])
    return reps


def run(input_dir, out_dir, model="gemini:gemini-3.5-flash-lite", budget="B2", judge_model=None, batch=5,
        per_episode=3, episodes=None, compact=True):
    t0 = time.time()
    lv = LEVELS.index(budget)
    out = Path(out_dir)
    raw = out / "raw_llm"
    raw.mkdir(parents=True, exist_ok=True)
    eps = load_show(input_dir)
    if episodes:
        eps = [e for e in eps if e.num in set(episodes)]
    emap = {e.num: e for e in eps}
    idx = Index(eps)
    errors, cost = [], {}
    from ..qa import inspect
    qa = inspect(eps)
    log(f"parsed {len(eps)} episodes; budget {budget}; input check: {qa['verdict']}")
    for w_ in qa["warnings"]:
        log(f"  input warning: {w_}")

    # B0: rule-based windows
    pool = []
    for p in heuristic_promos(eps, per_episode):
        p["id"] = f"H{len(pool) + 1:03d}"
        check(idx, p)
        if p["status"] != "rejected":
            pool.append(p)

    llm = LLM(model) if lv >= 1 else None
    ann = questions = None
    if lv >= 1:
        st = {}
        c0 = (llm.calls, dict(llm.usage))
        ann, questions = A.annotate(llm, eps, batch, raw, log, errors, stats=st)
        A.save(out / "annotations.jsonl", eps, ann, questions)
        if st.get("annotated_now") and llm.calls == c0[0]:
            log("  WARNING: every annotation call failed - results below fall back to rules only. First error: "
                + (errors[0][:200] if errors else "?"))
        cost["annotate"] = {"calls": llm.calls - c0[0], **st,
                            "input_tokens": llm.usage["input_tokens"] - c0[1]["input_tokens"],
                            "output_tokens": llm.usage["output_tokens"] - c0[1]["output_tokens"],
                            "coverage": round(sum(ann[b.id].labelled for e in eps for b in e.beats if A._keep(b)) /
                                              max(1, sum(A._keep(b) for e in eps for b in e.beats)), 3),
                            "questions": len(questions)}
        for ep in eps:
            for c in search(ep, ann, questions, per_episode):
                p = to_promo(c, ep, ann, questions, f"C{len(pool) + 1:03d}")
                check(idx, p)
                if p["status"] != "rejected":
                    pool.append(p)
    if lv >= 2:
        from ..pipeline import run as run_v1
        c0 = llm.calls
        meta1 = run_v1(input_dir, str(out / "v1_llm_picks"), model, per_episode, cross=False, cold_top=0,
                       episodes=episodes, workers=2, batch=batch, compact=compact)
        cost["llm_picker"] = {"calls": meta1["llm_calls"], "cache_hits": meta1["cache_hits"],
                              "input_tokens": meta1["usage"].get("input_tokens", 0),
                              "output_tokens": meta1["usage"].get("output_tokens", 0),
                              "input_chars": meta1["approx_input_chars"], "compact": compact}
        pool += seed_promos(out / "v1_llm_picks" / "promos.json", emap, ann, questions, idx)
        del c0

    # free rankers over the whole pool
    R = {"rules": {}, "optimizer": {}, "llm_pick": {}, "tournament": {}, "audience": {}}
    for p in pool:
        ep = emap[p["segments"][0]["ep"]]
        segs = [(s["start"], s["end"]) for s in p["segments"]]
        R["rules"][p["id"]] = rule_score(ep, segs)
        if ann is not None:
            sc = Scorer(ep, ann, questions)
            f, *_ = sc.features(segs, teaser=p.get("source") == "v2-flash-forward")
            p["signals"], p["signal_score"] = f, sc.score(f)
            R["optimizer"][p["id"]] = p["signal_score"]
        if p.get("llm_rank"):
            R["llm_pick"][p["id"]] = -float(p["llm_rank"])

    def fused_now():
        return fuse(pool, {k: v for k, v in R.items() if v}, method="rrf", consensus=True)

    tstats = None
    if lv >= 3:
        judge = LLM(judge_model) if judge_model else llm
        f = fused_now()
        for p in pool:
            p["signal_score"] = p.get("signal_score", 0.0)
        cands = [p for p in pool if p["id"][0] in "CL"]
        short = shortlist(cands, 24, key=lambda p: -f[p["id"]])
        c0 = (judge.calls, llm.calls)
        bt, tstats = T.run(judge, short, emap, raw, log=log, errors=errors, fallback=llm)
        cost["tournament"] = {"judge_calls": judge.calls - c0[0], "fallback_calls": llm.calls - c0[1]}
        for p in short:
            p["tournament"] = bt[p["id"]]
            R["tournament"][p["id"]] = bt[p["id"]]["bt_strength"]
    if lv >= 4:
        f = fused_now()
        finals = moments(sorted(pool, key=lambda p: -f[p["id"]]), emap)[:10]
        finals = [next(p for p in pool if p["id"] == r["id"]) for r in finals]
        c0 = llm.calls
        panel = AU.run(llm, finals, emap, raw, log=log, errors=errors)
        cost["audience"] = {"calls": llm.calls - c0}
        for p in finals:
            if p["id"] in panel:
                p["audience"] = panel[p["id"]]
                a = panel[p["id"]]
                R["audience"][p["id"]] = 0.4 * a["hold_3s"] + 0.3 * a["completion"] + 0.3 * a["tap_rate"]

    # final: RRF fusion with moment consensus
    used = {k: v for k, v in R.items() if v}
    fused = fuse(pool, used, method="rrf", consensus=True)
    per_ranker = {k: consensus_ranks(pool, v) for k, v in used.items()}
    for p in pool:
        p["fused"] = round(fused[p["id"]], 6)
        p["ranker_ranks"] = {k: r[p["id"]] for k, r in per_ranker.items() if p["id"] in r}
        for s in p["segments"]:
            s["start_ts"], s["end_ts"] = fmt(s["start"]), fmt(s["end"])
    ranked = sorted(pool, key=lambda p: -p["fused"])
    for r, p in enumerate(ranked, 1):
        p["rank"] = r
    mom = moments(ranked, emap)
    # Distinctiveness (the TF-IDF idea): footage that repeats in many episodes (recaps, templates)
    # gets extra "votes" from every copy but says nothing specific about the show - keep it out of
    # the post-first list and flag it.
    limit = max(3, 0.25 * len(eps))
    for p in mom:
        p["flags"] = list(p.get("flags", []))
        if len(p.get("repeats_in_episodes", [])) + 1 >= limit:
            p["boilerplate"] = True
            p["flags"].append(f"boilerplate: same footage in {len(p['repeats_in_episodes']) + 1} episodes - not distinctive")
        last = _last_line(p, emap)
        if last is not None:
            first = last.speaker.split()[0].title() if last.speaker else ""
            if first and len(first) > 2 and re.search(rf"\b{first}\b", last.text):
                p["flags"].append("check: final line looks like narration, not dialogue")
    post, cnt = [], {}
    for p in mom:
        if p.get("boilerplate"):
            continue
        e = p["segments"][0]["ep"]
        if cnt.get(e, 0) < 2:
            post.append(p)
            cnt[e] = cnt.get(e, 0) + 1
        if len(post) >= 10:
            break

    meta = {"engine": "v3-fusion", "input_dir": str(Path(input_dir).resolve()), "input_check": qa, "budget": budget, "compact_picker": compact, "model": model if lv else "none", "episodes": len(eps),
            "candidates": len(pool), "moments": len(mom), "rankers": list(used), "cost_by_stage": cost,
            "tournament": tstats, "errors": errors, "seconds": round(time.time() - t0, 1),
            "post_first_ids": [p["id"] for p in post]}
    write(out, ranked, mom, post, meta)
    log(f"done in {meta['seconds']}s -> {out}  (cost: {cost})")
    return meta


def write(out, ranked, mom, post, meta):
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    (out / "promos.json").write_text(json.dumps({"meta": meta, "post_first": [p["id"] for p in post],
                                                 "promos": ranked, "moments": mom}, indent=1, ensure_ascii=False, default=str))
    seg = lambda p: " + ".join(f"Ep{s['ep']} {s['start_ts']}-{s['end_ts']}" for s in p["segments"])
    with open(out / "promos.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["moment_rank", "id", "source", "found_by", "segments", "duration_s", "hook", "cliffhanger",
                    "why", "fused_score", "ranker_ranks"])
        for i, p in enumerate(mom, 1):
            w.writerow([i, p["id"], p["source"], " ".join(p["also_found_by"]), seg(p), p.get("duration"),
                        p.get("hook", ""), p.get("cliffhanger", ""), p.get("why", ""), p["fused"],
                        json.dumps(p["ranker_ranks"])])
    L = [f"# Promo picks (v3 fusion, budget {meta['budget']})", "",
         f"{meta['candidates']} candidate clips -> {meta['moments']} distinct moments. Rankers fused (RRF, k=60): "
         f"{', '.join(meta['rankers'])}. Cost: {json.dumps(meta['cost_by_stage'])}", "",
         f"**Input check: {meta['input_check']['verdict']}**"] + [f"- {w}" for w in meta["input_check"]["warnings"]] + [
         "", "## Post these first", ""]
    for i, p in enumerate(post, 1):
        rr = ", ".join(f"{k} #{v}" for k, v in p["ranker_ranks"].items())
        L += [f"### {i}. {seg(p)} - {p.get('duration')}s (`{p['id']}`)",
              f"- **Hook (0-3s):** {p.get('hook', '')}",
              f"- **Cliffhanger:** {p.get('cliffhanger', '')}",
              f"- **Why:** {p.get('why', '')}",
              f"- **Evidence:** {rr}; found independently by: {', '.join(p['also_found_by'])}"
              + (f"; same footage also in {len(p['repeats_in_episodes'])} other episode(s)" if p.get("repeats_in_episodes") else "")]
        L += [f"- **Check:** {f}" for f in p.get("flags", []) if f.startswith(("check", "boilerplate"))]
        L.append("")
    nb = [p for p in mom if p.get("boilerplate")]
    if nb:
        L += ["## Left out as boilerplate (repeats across many episodes)", ""]
        L += [f"- {seg(p)}: {p['flags'][0]}" for p in nb[:5]]
    (out / "report.md").write_text("\n".join(L) + "\n")
