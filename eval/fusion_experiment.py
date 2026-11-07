"""Offline experiment: does fusing the rankers we already have beat each one alone?

Pre-registered primary method (chosen before looking at held-out results): RRF, k=60, equal weights,
moment consensus on, all available rankers. Everything else is reported as a secondary variant.
No LLM calls - uses cached outputs of the v1, heuristic and v2-ensemble runs.

usage: python eval/fusion_experiment.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from promo_finder.heuristics import score_window  # noqa: E402
from promo_finder.parser import fmt, load_show  # noqa: E402
from promo_finder.v2.optimize import Scorer  # noqa: E402
from promo_finder.v3.fusion import fuse  # noqa: E402
from ablate import load_ann  # noqa: E402
from evaluate import hits_gold  # noqa: E402

V2 = ROOT / "outputs" / "v2-ensemble"
OUT = ROOT / "outputs" / "v3-fusion"


def load_pool(eps):
    v2 = json.loads((V2 / "promos.json").read_text())["promos"]
    heur = json.loads((ROOT / "outputs" / "heuristic" / "promos.json").read_text())["promos"]
    pool = [p for p in v2 if len({s["ep"] for s in p["segments"]}) == 1]
    for p in heur:
        q = dict(p)
        q["id"] = "H" + p["id"][1:]
        q["source"] = "rules-window"
        pool.append(q)
    return pool


def rankers_for(pool, eps, ann, q):
    rules, opt, llm, tour, aud = {}, {}, {}, {}, {}
    for p in pool:
        ep = eps[p["segments"][0]["ep"]]
        segs = [(s["start"], s["end"]) for s in p["segments"]]
        h = [score_window(ep, [x]) for x in segs]
        rules[p["id"]] = 0.35 * h[0].hook + 0.35 * h[-1].cliff + 0.15 * sum(x.intensity for x in h) / len(h) \
            + 0.15 * sum(x.standalone for x in h) / len(h)
        sc = Scorer(ep, ann, q)
        f, *_ = sc.features(segs, teaser=p.get("source") == "v2-flash-forward")
        opt[p["id"]] = sc.score(f)
        if p.get("source") == "v1-llm-pick" and p.get("llm_rank"):
            llm[p["id"]] = -float(p["llm_rank"])
        if p.get("tournament"):
            tour[p["id"]] = p["tournament"]["bt_strength"]
        a = p.get("audience")
        if a:
            aud[p["id"]] = 0.4 * a["hold_3s"] + 0.3 * a["completion"] + 0.3 * a["tap_rate"]
    return {"rules": rules, "optimizer": opt, "llm_pick": llm, "tournament": tour, "audience": aud}


def write_run(name, pool, fused):
    ranked = sorted(pool, key=lambda p: -fused[p["id"]])
    post, cnt = [], {}
    for r, p in enumerate(ranked, 1):
        p = dict(p, rank=r, fused=round(fused[p["id"]], 6))
        for s in p["segments"]:
            s.setdefault("start_ts", fmt(s["start"]))
            s.setdefault("end_ts", fmt(s["end"]))
        ranked[r - 1] = p
    for p in ranked:
        e = p["segments"][0]["ep"]
        if cnt.get(e, 0) < 2 and len(post) < 10:
            post.append(p["id"])
            cnt[e] = cnt.get(e, 0) + 1
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "promos.json").write_text(json.dumps({"meta": {"model": f"fusion:{name}"}, "post_first": post,
                                               "promos": ranked}, indent=1, ensure_ascii=False))
    return ranked, post


def score(ranked, post, gold):
    out = {}
    for split in ("dev", "heldout"):
        g = [x for x in gold if x["split"] == split]
        eps_ = sorted({x["ep"] for x in g})
        top1 = 0
        for e in eps_:
            best = next((p for p in ranked if p["segments"][0]["ep"] == e), None)
            top1 += bool(best and hits_gold(best, [x for x in g if x["ep"] == e]))
        intop = sum(any(hits_gold(p, [x]) for p in ranked if p["id"] in post) for x in g)
        out[split] = (top1, len(eps_), intop, len(g))
    return out


def main():
    gold = json.loads((ROOT / "eval" / "gold.json").read_text())["moments"]
    eps = {e.num: e for e in load_show(ROOT / "input")}
    ann, q = load_ann(V2, list(eps.values()))
    pool = load_pool(eps)
    R = rankers_for(pool, eps, ann, q)
    print(f"pool: {len(pool)} clips; ranker coverage: " + ", ".join(f"{k}={len(v)}" for k, v in R.items()))

    def sub(*names):
        return {n: R[n] for n in names}

    variants = {
        # single rankers (fused alone = plain ranking, consensus off)
        "rules alone": (sub("rules"), "rrf", False),
        "optimizer alone": (sub("optimizer"), "rrf", False),
        # PRIMARY (pre-registered)
        "RRF all  [primary]": (R, "rrf", True),
        # secondary variants
        "RRF all, no consensus": (R, "rrf", False),
        "RRF free only (rules+optimizer)": (sub("rules", "optimizer"), "rrf", True),
        "RRF no audience": (sub("rules", "optimizer", "llm_pick", "tournament"), "rrf", True),
        "RRF no tournament/audience": (sub("rules", "optimizer", "llm_pick"), "rrf", True),
        "Borda all": (R, "borda", True),
        "z-score mean all": (R, "zmean", True),
    }
    rows = []
    print(f"\n{'variant':36s} {'dev top1':>9s} {'held-out top1':>14s} {'dev in-top10':>13s} {'held-out in-top10':>18s}")
    for name, (rk, method, cons) in variants.items():
        fused = fuse(pool, rk, method=method, consensus=cons)
        slug = name.split("[")[0].strip().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "-").replace(",", "").replace("/", "-")
        ranked, post = write_run(slug, pool, fused)
        s = score(ranked, post, gold)
        rows.append((name, s))
        print(f"{name:36s} {s['dev'][0]}/{s['dev'][1]:<7} {s['heldout'][0]}/{s['heldout'][1]:<12} "
              f"{s['dev'][2]}/{s['dev'][3]:<11} {s['heldout'][2]}/{s['heldout'][3]}")
    (OUT / "results.json").write_text(json.dumps({n: s for n, s in rows}, indent=1))


if __name__ == "__main__" and "--frontier" not in sys.argv:
    main()


def frontier():
    """Cost / quality frontier: each budget level only uses the candidates and rankers it paid for."""
    gold = json.loads((ROOT / "eval" / "gold.json").read_text())["moments"]
    eps = {e.num: e for e in load_show(ROOT / "input")}
    ann, q = load_ann(V2, list(eps.values()))
    full = load_pool(eps)
    R = rankers_for(full, eps, ann, q)
    levels = [
        ("B0 rules only", ("H",), ("rules",), 0, 0),
        ("B1 + annotation -> optimizer", ("H", "C"), ("rules", "optimizer"), 10, 0),
        ("B2 + LLM picker", ("H", "C", "L"), ("rules", "optimizer", "llm_pick"), 20, 0),
        ("B3 + pairwise tournament", ("H", "C", "L"), ("rules", "optimizer", "llm_pick", "tournament"), 20, 4),
        ("B4 + audience panel", ("H", "C", "L"), ("rules", "optimizer", "llm_pick", "tournament", "audience"), 25, 4),
    ]
    print(f"\n{'budget level':32s} {'small-model calls':>18s} {'judge calls':>12s} {'dev':>6s} {'held-out':>9s} {'total':>6s}")
    rows = []
    for name, prefixes, rk_names, small, judge in levels:
        pool = [p for p in full if p["id"][0] in prefixes]
        ids = {p["id"] for p in pool}
        rk = {n: {i: v for i, v in R[n].items() if i in ids} for n in rk_names}
        fused = fuse(pool, rk, method="rrf", consensus=True) if len(rk) > 1 else {i: v for i, v in rk[rk_names[0]].items()}
        slug = "budget_" + name.split()[0]
        ranked, post = write_run(slug, pool, fused)
        s = score(ranked, post, gold)
        tot = s["dev"][0] + s["heldout"][0]
        rows.append({"level": name, "small_calls": small, "judge_calls": judge, "dev": s["dev"][:2], "heldout": s["heldout"][:2]})
        print(f"{name:32s} {small:>18d} {judge:>12d} {s['dev'][0]}/{s['dev'][1]:>4} {s['heldout'][0]}/{s['heldout'][1]:>7} {tot:>4}/14")
    (OUT / "frontier.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__" and "--frontier" in sys.argv:
    frontier()
