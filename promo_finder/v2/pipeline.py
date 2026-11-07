"""v2 pipeline: annotate -> optimize -> tournament -> audience -> rank."""
from __future__ import annotations

import json
import time
from pathlib import Path

from ..llm import LLM
from ..parser import fmt, load_show
from ..validate import Index, check
from . import annotate as A
from . import audience as AU
from . import tournament as T
from .optimize import search, to_promo


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def seed_promos(path, emap, ann, questions, idx):
    """Bring in clips picked directly by another engine (e.g. v1's LLM picks) so both candidate
    generators compete in the same blind tournament. Only single-episode clips are scored."""
    from .optimize import Scorer
    data = json.loads(Path(path).read_text())
    out = []
    for q in data.get("promos", []):
        segs = q.get("segments") or []
        eps_ = {x["ep"] for x in segs}
        if len(eps_) != 1 or next(iter(eps_)) not in emap:
            continue
        ep = emap[next(iter(eps_))]
        sc = Scorer(ep, ann, questions)
        f, oq, _, _ = sc.features([(x["start"], x["end"]) for x in segs], teaser=False)
        open_txt = [questions[k]["text"] for k in oq if k in questions][:2]
        out.append({"id": f"L{len(out) + 1:03d}", "source": "v1-llm-pick", "title": q.get("title", ""),
                    "segments": [{"ep": x["ep"], "start": x["start"], "end": x["end"],
                                  "start_ts": fmt(x["start"]), "end_ts": fmt(x["end"])} for x in segs],
                    "duration": round(sum(x["end"] - x["start"] for x in segs), 2),
                    "signals": f, "signal_score": sc.score(f), "flags": list(q.get("flags", [])),
                    "hook": q.get("hook", ""),
                    "cliffhanger": q.get("cliffhanger", "") + (" - leaves open: " + " / ".join(open_txt) if open_txt else ""),
                    "why": q.get("why", ""), "llm_rank": q.get("rank")})
    return out


def shortlist(promos, n, per_ep=2, key=None):
    out, cnt = [], {}
    key = key or (lambda p: -p["signal_score"])
    for p in sorted(promos, key=key):
        e = p["segments"][0]["ep"]
        if cnt.get(e, 0) >= per_ep:
            continue
        cnt[e] = cnt.get(e, 0) + 1
        out.append(p)
        if len(out) >= n:
            break
    return out


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for k, i in enumerate(order):
            r[i] = k
        return r
    if len(xs) < 3:
        return None
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    d = sum((a - b) ** 2 for a, b in zip(rx, ry))
    return round(1 - 6 * d / (n * (n * n - 1)), 3)


def run(input_dir, out_dir, model="gemini:gemini-3.5-flash-lite", judge_model=None, audience_model=None,
        batch=5, per_episode=3, n_short=24, n_final=10, rounds=4, episodes=None, seed=None):
    t0 = time.time()
    out = Path(out_dir)
    raw = out / "raw_llm"
    raw.mkdir(parents=True, exist_ok=True)
    eps = load_show(input_dir)
    if episodes:
        eps = [e for e in eps if e.num in set(episodes)]
    sub = [p for p in Path(input_dir).iterdir() if p.is_dir()]
    show = sub[0].name if len(sub) == 1 else Path(input_dir).resolve().name
    idx = Index(eps)
    emap = {e.num: e for e in eps}
    errors = []
    llm = LLM(model)
    judge = LLM(judge_model) if judge_model else llm
    aud = LLM(audience_model) if audience_model else llm
    log(f"parsed {len(eps)} episodes - annotate:{llm.name} judge:{judge.name} audience:{aud.name}")

    # 1. annotate
    ann, questions = A.annotate(llm, eps, batch, raw, log, errors)
    A.save(out / "annotations.jsonl", eps, ann, questions)

    # 2. optimize
    promos = []
    for ep in eps:
        for c in search(ep, ann, questions, per_episode):
            p = to_promo(c, ep, ann, questions, f"C{len(promos) + 1:03d}")
            check(idx, p)
            if p["status"] != "rejected":
                for s in p["segments"]:
                    s["start_ts"], s["end_ts"] = fmt(s["start"]), fmt(s["end"])
                promos.append(p)
    log(f"optimizer: {len(promos)} candidate promos")

    # 3. tournament on the shortlist
    if seed == "auto":
        # second candidate generator: v1 engine (LLM picks clips directly), batched, no extras
        from ..pipeline import run as run_v1
        log("running v1 LLM picker as second candidate generator")
        run_v1(input_dir, str(out / "v1_llm_picks"), model, per_episode, cross=False, cold_top=0,
               episodes=episodes, workers=2, batch=batch)
        seed = str(out / "v1_llm_picks" / "promos.json")
    if seed:
        seeded = seed_promos(seed, emap, ann, questions, idx)
        log(f"seeded {len(seeded)} clips from {seed}")
        # half the shortlist from each generator: optimizer by signal score, seeds by their own engine's rank
        short = shortlist(promos, n_short // 2) + shortlist(seeded, n_short - n_short // 2,
                                                             key=lambda p: p.get("llm_rank") or 999)
        promos = promos + seeded
    else:
        short = shortlist(promos, n_short)
    bt, tstats = T.run(judge, short, emap, raw, rounds=rounds, log=log, errors=errors, fallback=llm)
    for p in short:
        p["tournament"] = bt[p["id"]]
    tstats["spearman_signal_vs_tournament"] = spearman([p["signal_score"] for p in short],
                                                       [p["tournament"]["bt_strength"] for p in short])
    by_src = {}
    for p in short:
        r = p["tournament"]["record"].split("/")
        w, n = float(r[0]), float(r[1])
        src = "optimizer" if p["source"].startswith("v2") else p["source"]
        a = by_src.setdefault(src, [0.0, 0.0])
        a[0] += w
        a[1] += n
    tstats["win_rate_by_generator"] = {k: round(v[0] / v[1], 3) if v[1] else None for k, v in by_src.items()}
    log(f"tournament: {tstats}")

    # 4. audience panel on finalists
    finals = sorted(short, key=lambda p: -p["tournament"]["bt_strength"])[:n_final]
    panel = AU.run(aud, finals, emap, raw, log=log, errors=errors)
    for p in finals:
        if p["id"] in panel:
            p["audience"] = panel[p["id"]]

    # 5. final ranking
    def final_score(p):
        bt_n = p["tournament"]["bt_norm"]
        a = p.get("audience")
        if not a:
            return 0.5 * bt_n
        return 0.5 * bt_n + 0.2 * a["hold_3s"] + 0.15 * a["completion"] + 0.15 * a["tap_rate"]

    for p in finals:
        p["rank_score"] = round(final_score(p), 4)
        p["stage"] = "finalist"
    finals.sort(key=lambda p: -p["rank_score"])
    rest_short = sorted([p for p in short if p not in finals], key=lambda p: -p["tournament"]["bt_strength"])
    for p in rest_short:
        p["stage"], p["rank_score"] = "shortlist", round(0.5 * p["tournament"]["bt_norm"], 4)
    rest = sorted([p for p in promos if p not in short], key=lambda p: -p["signal_score"])
    for p in rest:
        p["stage"], p["rank_score"] = "candidate", round(0.3 * p["signal_score"], 4)
    ranked = finals + rest_short + rest
    for r, p in enumerate(ranked, 1):
        p["rank"] = r

    usage = {m.name: {"calls": m.calls, "cache_hits": m.cache_hits, **m.usage} for m in {llm, judge, aud}}
    meta = {"engine": "v2-curiosity", "show": show, "episodes": len(eps), "beats": len(idx.beats),
            "model": llm.name, "judge_model": judge.name, "audience_model": aud.name,
            "seconds": round(time.time() - t0, 1), "candidates": len(promos), "shortlist": len(short),
            "finalists": len(finals), "tournament": tstats, "usage": usage, "errors": errors,
            "annotation_coverage": round(sum(a.labelled for a in ann.values()) /
                                         max(1, sum(A._keep(b) for e in eps for b in e.beats)), 3),
            "questions": len(questions), "post_first_ids": [p["id"] for p in finals], "seed": seed,
            "finalists_by_generator": {s_: sum(1 for p in finals if (p["source"].startswith("v2")) == (s_ == "optimizer"))
                                       for s_ in ("optimizer", "v1-llm-pick")}}
    from .report import write_all
    write_all(out, emap, ranked, finals, meta, ann, questions)
    log(f"done in {meta['seconds']}s -> {out}")
    return meta
