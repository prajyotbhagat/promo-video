"""Ablation: which signal in the v2 optimizer actually matters?

Re-runs the code-only optimizer from saved annotations with one weight switched off at a time and
checks the best clip per gold episode against eval/gold.json. No LLM calls.

usage: python eval/ablate.py outputs/v2-curiosity
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from promo_finder.parser import load_show  # noqa: E402
from promo_finder.v2.annotate import Ann  # noqa: E402
from promo_finder.v2.optimize import W, search  # noqa: E402
from evaluate import cover, sec  # noqa: E402


def load_ann(run_dir, eps):
    ann = {b.id: Ann() for e in eps for b in e.beats}
    for line in open(Path(run_dir) / "annotations.jsonl"):
        d = json.loads(line)
        if d["beat"] not in ann:
            continue
        x = ann[d["beat"]]
        x.a, x.v, x.s, x.c, x.tp, x.opens, x.closes, x.labelled = (d["a"], d["v"], d["s"], d["c"], d["tp"],
                                                                     d["opens"], d["closes"], True)
    q = json.loads((Path(run_dir) / "questions.json").read_text())
    return ann, q


def main(run_dir, split=None):
    gold = json.loads((ROOT / "eval" / "gold.json").read_text())["moments"]
    if split:
        gold = [g for g in gold if g.get("split") == split]
        print(f"split: {split}")
    gep = sorted({g["ep"] for g in gold})
    eps = {e.num: e for e in load_show(ROOT / "input") if e.num in gep}
    ann, q = load_ann(run_dir, list(eps.values()))
    configs = {"full": {}}
    for k in W:
        configs[f"without {k}"] = {k: 0.0}
    configs["gap only"] = {k: 0.0 for k in W if k != "gap"}
    configs["hook only"] = {k: 0.0 for k in W if k != "hook"}
    print(f"{'config':22s} top1  top3   (gold episodes: {len(gep)})")
    for name, w in configs.items():
        t1 = t3 = 0
        for n in gep:
            ge = [g for g in gold if g["ep"] == n]
            clips = search(eps[n], ann, q, k=3, weights=w)
            hit = [any(cover((sec(g["start"]), sec(g["end"])), seg) >= 0.5 for g in ge for seg in c.segments) for c in clips]
            t1 += bool(hit and hit[0])
            t3 += any(hit)
        print(f"{name:22s} {t1}/{len(gep)}   {t3}/{len(gep)}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sp = next((a.split("=")[1] for a in sys.argv if a.startswith("--split=")), None)
    main(args[0] if args else "outputs/v2-curiosity", sp)
