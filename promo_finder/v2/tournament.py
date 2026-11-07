"""Step 3: rank the shortlist by pairwise LLM judgments + Bradley-Terry.

Why pairwise: LLM judges are better at "which of these two" than at absolute 1-10 scores, and
absolute self-scores bunch up at 8-9 (we saw this in v1). Zheng et al. (2023) show strong LLM
judges reach ~80% agreement with humans in pairwise mode, but have position bias - so every pair
is judged twice with the order swapped; a win counts only when both orders agree, else it is a tie.
Bradley-Terry (fitted with Hunter's 2004 MM algorithm) turns wins into one strength per clip.

To fit free-tier request caps, clips are listed once and many pairs refer to them by id.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

from .transcript import render

SYSTEM = """You are the head of user acquisition at a vertical-drama app. You decide which promo clips get
ad budget on TikTok / Reels. You judge clips by one question: which clip would make MORE strangers who have
never seen the show (1) stop scrolling in the first 3 seconds, (2) watch to the end, and (3) tap through to
watch Episode 1 because they NEED to know what happens next? Clarity for a cold viewer matters.
Ignore production quality; judge the transcript. Return ONLY JSON."""


def _prompt(lib: list[tuple[str, str]], pairs: list[tuple[str, str]]) -> str:
    clips = "\n\n".join(f"### {cid}\n{txt}" for cid, txt in lib)
    ps = "\n".join(f"{i + 1}. A={a}  B={b}" for i, (a, b) in enumerate(pairs))
    return f"""CLIP LIBRARY (transcripts as played, with seconds):

{clips}

PAIRS TO JUDGE (A is listed first, B second):
{ps}

For EVERY pair return:
{{"verdicts": [{{"pair": 1, "winner": "A" or "B", "margin": 1-3, "reason": "max 12 words"}}, ...]}}"""


def make_pairs(ids: list[str], rounds: int, seed: int = 7) -> list[tuple[str, str]]:
    rnd = random.Random(seed)
    seen, pairs = set(), []
    for _ in range(rounds):
        for _try in range(50):
            order = ids[:]
            rnd.shuffle(order)
            cand = [(order[i], order[i + 1]) for i in range(0, len(order) - 1, 2)]
            if not any(tuple(sorted(p)) in seen for p in cand):
                break
        for p in cand:
            seen.add(tuple(sorted(p)))
            pairs.append(p)
    return pairs


def bradley_terry(ids, results, iters=200):
    """results: list of (i, j, score_i) with score_i in {1, 0.5, 0}. MM updates (Hunter 2004).
    One virtual tie against an average opponent acts as a prior so unbeaten clips stay finite."""
    p = {i: 1.0 for i in ids}
    wins = {i: 0.5 for i in ids}  # prior tie
    games = {i: [] for i in ids}
    for i, j, s in results:
        wins[i] += s
        wins[j] += 1 - s
        games[i].append(j)
        games[j].append(i)
    for _ in range(iters):
        new = {}
        for i in ids:
            denom = 1.0 / (p[i] + 1.0) + sum(1.0 / (p[i] + p[j]) for j in games[i])
            new[i] = wins[i] / denom
        g = math.exp(sum(math.log(v) for v in new.values()) / len(new))
        p = {i: v / g for i, v in new.items()}
    return p


def run(llm, promos: list[dict], eps: dict, raw_dir: Path, rounds: int = 4, chunk: int = 24, log=print, errors=None,
        fallback=None):
    ids = [p["id"] for p in promos]
    lib = [(p["id"], render(p, eps)) for p in promos]
    pairs = make_pairs(ids, rounds)
    verdicts = {}  # (a,b) ordered as asked -> winner id
    reasons = {}
    for order in ("AB", "BA"):
        asked = pairs if order == "AB" else [(b, a) for a, b in pairs]
        for k in range(0, len(asked), chunk):
            part = asked[k:k + chunk]
            try:
                try:
                    data = llm.complete_json(SYSTEM, _prompt(lib, part))
                except Exception as e:
                    if fallback is None or fallback is llm or "429" not in str(e):
                        raise
                    log(f"  judge quota hit - falling back to {fallback.name} for this batch")
                    (errors if errors is not None else []).append(f"tournament {order}{k}: judge 429, used fallback")
                    data = fallback.complete_json(SYSTEM, _prompt(lib, part))
            except Exception as e:
                (errors if errors is not None else []).append(f"tournament {order}{k}: {str(e)[:200]}")
                log(f"  tournament batch {order}{k} FAILED: {str(e)[:160]}")
                continue
            (raw_dir / f"tournament_{order}_{k:02d}.json").write_text(json.dumps(data, indent=1, ensure_ascii=False))
            vs = data.get("verdicts", data) if isinstance(data, dict) else data
            for v in vs if isinstance(vs, list) else []:
                try:
                    n = int(v.get("pair")) - 1
                    a, b = part[n]
                except (TypeError, ValueError, IndexError, AttributeError):
                    continue
                w = str(v.get("winner", "")).strip().upper()[:1]
                if w in ("A", "B"):
                    verdicts[(a, b)] = a if w == "A" else b
                    reasons[(a, b)] = str(v.get("reason", ""))
        log(f"  tournament order {order}: {sum(1 for k in verdicts if (k in pairs) == (order == 'AB'))} verdicts")

    results, agree, both, first_wins, judged = [], 0, 0, 0, 0
    for a, b in pairs:
        v1, v2 = verdicts.get((a, b)), verdicts.get((b, a))
        for (x, y), v in (((a, b), v1), ((b, a), v2)):
            if v:
                judged += 1
                first_wins += v == x
        if v1 and v2:
            both += 1
            if v1 == v2:
                agree += 1
                results.append((a, b, 1.0 if v1 == a else 0.0))
            else:
                results.append((a, b, 0.5))       # order-dependent verdict = tie
        elif v1 or v2:
            v = v1 or v2
            results.append((a, b, 0.75 if v == a else 0.25))  # half credit, one order only
    strength = bradley_terry(ids, results)
    lo, hi = min(strength.values()), max(strength.values())
    stats = {"pairs": len(pairs), "judgments": judged, "both_orders": both,
             "order_consistency": round(agree / both, 3) if both else None,
             "first_position_win_rate": round(first_wins / judged, 3) if judged else None}
    rec = {}
    for p in promos:
        i = p["id"]
        w = sum(s if a == i else 1 - s for a, b, s in results if i in (a, b))
        n = sum(1 for a, b, s in results if i in (a, b))
        rec[i] = {"bt_strength": round(strength[i], 4),
                  "bt_norm": round((math.log(strength[i]) - math.log(lo)) / ((math.log(hi) - math.log(lo)) or 1), 3),
                  "record": f"{w:g}/{n}",
                  "sample_reasons": [reasons[k] for k in reasons if i == verdicts.get(k)][:2]}
    return rec, stats
