"""Fuse several rankers into one ranking.

Rankers (higher = better):
  rules       rule-based hook/cliffhanger score              free, covers every clip
  optimizer   curiosity/information-gap score (v2)           free once annotated, covers every clip
  llm_pick    the v1 LLM picker's own rank                   only clips it proposed
  tournament  Bradley-Terry strength from pairwise judging   only the 24-clip shortlist
  audience    persona-panel retention + tap                  only the 10 finalists

Fusion methods:
  RRF     Reciprocal Rank Fusion, score = sum_r 1 / (k + rank_r), k = 60 (Cormack, Clarke & Buettcher,
          SIGIR 2009). Uses ranks, not raw scores, so rankers on different scales combine safely,
          and a ranker that has no opinion on a clip simply adds nothing.
  borda   mean normalised rank over the rankers that rank the clip.
  zmean   mean z-score over the rankers that score the clip.

Moment consensus: two clips that cover >= 50% of each other are the same moment. A clip inherits
each ranker's best rank for its moment, so a moment several engines found independently collects
evidence from all of them (a form of model averaging across engines).
"""
from __future__ import annotations

import math


def cover(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    return inter / max(1e-6, min(a[1] - a[0], b[1] - b[0]))


def _main(p):
    s = max(p["segments"], key=lambda x: x["end"] - x["start"])
    return s["ep"], (s["start"], s["end"])


def ranks(scores: dict[str, float]) -> dict[str, int]:
    order = sorted(scores, key=lambda i: -scores[i])
    return {i: r + 1 for r, i in enumerate(order)}


def consensus_ranks(pool: list[dict], scores: dict[str, float], thresh=0.5) -> dict[str, int]:
    """Each clip gets the best rank this ranker gave to any clip of the same moment."""
    rk = ranks(scores)
    main = {p["id"]: _main(p) for p in pool}
    scored = [i for i in rk]
    out = {}
    for p in pool:
        ep, w = main[p["id"]]
        best = None
        for j in scored:
            ej, wj = main[j]
            if ej == ep and cover(w, wj) >= thresh:
                best = rk[j] if best is None else min(best, rk[j])
        if best is not None:
            out[p["id"]] = best
    return out


def fuse(pool: list[dict], rankers: dict[str, dict[str, float]], method="rrf", k=60, consensus=True,
         weights: dict[str, float] | None = None) -> dict[str, float]:
    weights = weights or {}
    R = {name: (consensus_ranks(pool, sc) if consensus else ranks(sc)) for name, sc in rankers.items() if sc}
    fused = {}
    if method == "rrf":
        for p in pool:
            i = p["id"]
            fused[i] = sum(weights.get(n, 1.0) / (k + r[i]) for n, r in R.items() if i in r)
    elif method == "borda":
        for p in pool:
            i = p["id"]
            vals = [1 - (r[i] - 1) / max(1, len(r) - 1) for r in R.values() if i in r]
            fused[i] = sum(vals) / len(vals) if vals else 0.0
    elif method == "zmean":
        Z = {}
        for n, sc in rankers.items():
            if not sc:
                continue
            m = sum(sc.values()) / len(sc)
            sd = math.sqrt(sum((v - m) ** 2 for v in sc.values()) / len(sc)) or 1.0
            Z[n] = {i: (v - m) / sd for i, v in sc.items()}
        for p in pool:
            vals = [z[p["id"]] for z in Z.values() if p["id"] in z]
            fused[p["id"]] = sum(vals) / len(vals) if vals else -9.0
    return fused
