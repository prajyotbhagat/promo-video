"""Score runs against hand-labelled gold moments (eval/gold.json).

usage: python eval/evaluate.py outputs/<run>/promos.json [...] [--random]

Metrics
- recall@all : gold moments covered by ANY promo in that episode (easy - random clips score high)
- top1 hit   : in each gold episode, is the tool's best-ranked promo a gold moment? (the useful one)
- in top 10  : gold moments that made the show-level 'post first' list
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sec(t):
    m, s = t.split(":")
    return int(m) * 60 + float(s)


def cover(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    return inter / max(1e-6, min(a[1] - a[0], b[1] - b[0]))


def hits_gold(p, golds):
    return any(s["ep"] == g["ep"] and cover((sec(g["start"]), sec(g["end"])), (s["start"], s["end"])) >= 0.5
               for g in golds for s in p["segments"])


def evaluate(run_path, gold):
    d = json.loads(Path(run_path).read_text())
    promos, top = d["promos"], set(d["post_first"])
    eps = sorted({g["ep"] for g in gold})
    found = sum(any(hits_gold(p, [g]) for p in promos) for g in gold)
    in_top = sum(any(hits_gold(p, [g]) and p["id"] in top for p in promos) for g in gold)
    top1, none, lines = 0, 0, []
    for e in eps:
        ge = [g for g in gold if g["ep"] == e]
        mine = sorted([p for p in promos if p["segments"][0]["ep"] == e and len(p["segments"]) >= 1
                       and all(s["ep"] == e for s in p["segments"])], key=lambda p: p["rank"])
        best = mine[0] if mine else None
        ok = bool(best and hits_gold(best, ge))
        top1 += ok
        none += best is None
        seg = " + ".join(f"{s['start_ts']}-{s['end_ts']}" for s in best["segments"]) if best else "-"
        lines.append(f"  - Ep{e}: best promo {best['id'] if best else '-'} {seg} -> {'GOLD' if ok else 'not gold'}"
                     f"  [{(best or {}).get('hook', '')[:60]}]")
    print(f"\n## {d['meta']['model']}  ({run_path})")
    print(f"- recall@all : {found}/{len(gold)}")
    print(f"- top1 hit   : {top1}/{len(eps)} episodes" + (f"  ({none} had no single-episode promo = counted as miss)" if none else ""))
    print(f"- in top 10  : {in_top}/{len(gold)}")
    print("\n".join(lines))


def random_baseline(gold, trials=2000, seed=0):
    sys.path.insert(0, str(ROOT))
    from promo_finder.parser import load_show
    eps = {e.num: e for e in load_show(ROOT / "input")}
    rnd = random.Random(seed)
    gep = sorted({g["ep"] for g in gold})
    rec3 = top1 = 0
    for _ in range(trials):
        for e in gep:
            ep, ge = eps[e], [g for g in gold if g["ep"] == e]
            clips = []
            for _ in range(3):
                d = rnd.uniform(20, 60)
                s = rnd.uniform(0, max(0.0, ep.duration - d))
                clips.append((s, s + d))
            hit = lambda c, g: cover((sec(g["start"]), sec(g["end"])), c) >= 0.5
            rec3 += sum(any(hit(c, g) for c in clips) for g in ge)
            top1 += any(hit(clips[0], g) for g in ge)
    print("\n## random baseline (3 random 20-60s clips per episode)")
    print(f"- recall@all : {rec3 / trials:.1f}/{len(gold)}")
    print(f"- top1 hit   : {top1 / trials:.1f}/{len(gep)} episodes")


if __name__ == "__main__":
    gold = json.loads((Path(__file__).parent / "gold.json").read_text())["moments"]
    split = next((a.split("=")[1] for a in sys.argv if a.startswith("--split=")), None)
    if split:
        gold = [g for g in gold if g.get("split") == split]
        print(f"(split: {split}, {len(gold)} moments)")
    for r in [a for a in sys.argv[1:] if not a.startswith("--")]:
        evaluate(r, gold)
    if "--random" in sys.argv:
        random_baseline(gold)
