"""v2 outputs: report.md, promos.csv, promos.json, qc.md, review.html (with signal + retention charts)."""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path

from ..heuristics import beats_in
from .annotate import _keep


def seg_str(p):
    return " + ".join(f"Ep{s['ep']} {s['start_ts']}-{s['end_ts']}" for s in p["segments"])


def series(p, emap, ann, questions):
    """(t, arousal, open-question mass) in play time."""
    pts, off, opened, closed = [], 0.0, {}, set()
    imp = lambda q: float(questions.get(q, {}).get("importance", 1))
    for s in p["segments"]:
        for b in beats_in(emap[s["ep"]], s["start"], s["end"]):
            if not _keep(b):
                continue
            x = ann[b.id]
            for q in x.closes:
                if q in opened:
                    closed.add(q)
            for q in x.opens:
                opened.setdefault(q, 1)
            pts.append((off + b.start - s["start"], x.a, sum(imp(q) for q in opened if q not in closed)))
        off += s["end"] - s["start"]
    return pts, off


def svg_signals(pts, dur, w=320, h=70):
    if not pts or dur <= 0:
        return ""
    X = lambda t: 4 + (w - 8) * t / dur
    ya = lambda a: h - 6 - (h - 14) * a / 3
    mmax = max(4.0, max(m for _, _, m in pts))
    ym = lambda m: h - 6 - (h - 14) * m / mmax
    area = " ".join(f"{X(t):.1f},{ym(m):.1f}" for t, _, m in pts)
    area = f"{X(pts[0][0]):.1f},{h - 6} {area} {X(pts[-1][0]):.1f},{h - 6}"
    line = " ".join(f"{X(t):.1f},{ya(a):.1f}" for t, a, _ in pts)
    return (f'<svg width="{w}" height="{h}" class=chart><rect x=0 y=0 width={w} height={h} fill="#fafafa"/>'
            f'<rect x="{X(0):.1f}" y=0 width="{X(3) - X(0):.1f}" height={h} fill="#fff3c4"/>'
            f'<polygon points="{area}" fill="#f4b6b6" opacity=.7><title>open questions</title></polygon>'
            f'<polyline points="{line}" fill=none stroke="#2b5797" stroke-width=1.6><title>arousal</title></polyline>'
            f'<text x=6 y=11 font-size=9 fill="#555">blue = arousal, red = unanswered questions, yellow = first 3s</text></svg>')


def svg_curve(curve, w=320, h=70):
    if not curve:
        return ""
    n = max(1, len(curve) - 1)
    pts = " ".join(f"{4 + (w - 8) * i / n:.1f},{h - 6 - (h - 16) * v:.1f}" for i, v in enumerate(curve))
    return (f'<svg width="{w}" height="{h}" class=chart><rect x=0 y=0 width={w} height={h} fill="#fafafa"/>'
            f'<polyline points="{pts}" fill=none stroke="#1a7f37" stroke-width=2/>'
            f'<text x=6 y=11 font-size=9 fill="#555">predicted share still watching (persona panel)</text></svg>')


def write_all(out: Path, emap, ranked, finals, meta, ann, questions):
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2))
    (out / "promos.json").write_text(json.dumps({"meta": meta, "post_first": [p["id"] for p in finals],
                                                 "promos": ranked}, indent=1, ensure_ascii=False))
    with open(out / "promos.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "id", "stage", "shape", "segments", "duration_s", "hook", "cliffhanger", "why",
                    "signal_score", "tournament_record", "hold_3s", "completion", "tap_rate", "final_score"])
        for p in ranked:
            t, a = p.get("tournament", {}), p.get("audience", {})
            w.writerow([p["rank"], p["id"], p["stage"], p["source"].replace("v2-", ""), seg_str(p), p["duration"],
                        p["hook"], p["cliffhanger"], p["why"], p["signal_score"], t.get("record", ""),
                        a.get("hold_3s", ""), a.get("completion", ""), a.get("tap_rate", ""), p["rank_score"]])

    L = [f"# Promo picks (v2 Curiosity Engine) - {meta['show']}", "",
         f"Annotate `{meta['model']}` | judge `{meta['judge_model']}` | panel `{meta['audience_model']}` | "
         f"{meta['candidates']} candidates -> {meta['shortlist']} shortlisted -> {meta['finalists']} finalists", "",
         "## Post these first", ""]
    for i, p in enumerate(finals, 1):
        t, a = p["tournament"], p.get("audience", {})
        L += [f"### {i}. {seg_str(p)} - {p['duration']}s ({p['source'].replace('v2-', '')}, `{p['id']}`)",
              f"- **Hook (0-3s):** {p['hook']}",
              f"- **Cliffhanger:** {p['cliffhanger']}",
              f"- **Why (signals):** {p['why']}",
              f"- **Tournament:** won {t['record']} head-to-heads (BT strength {t['bt_strength']})"
              + (f" - judge: \"{t['sample_reasons'][0]}\"" if t.get("sample_reasons") else "")]
        if a:
            L.append(f"- **Panel:** 3s hold {a['hold_3s']:.0%}, completion {a['completion']:.0%}, tap {a['tap_rate']:.0%}")
            for k, v in a["voices"].items():
                L.append(f"  - {k}: swipe at {v['swipe_at']}, tap {v['tap']} - \"{v['says']}\"")
        L.append("")
    L += ["## All candidates", "", "| # | id | stage | segments | sec | hook | signal | record |", "|---|---|---|---|---|---|---|---|"]
    for p in ranked:
        L.append(f"| {p['rank']} | {p['id']} | {p['stage']} | {seg_str(p)} | {p['duration']} | "
                 f"{str(p['hook'])[:70].replace('|', '/')} | {p['signal_score']} | {p.get('tournament', {}).get('record', '')} |")
    (out / "report.md").write_text("\n".join(L) + "\n")

    t = meta["tournament"]
    Q = [f"# Quality check (v2) - {meta['show']}", "",
         "## Annotation", "", f"- beats labelled: {meta['annotation_coverage']:.0%} (rest fall back to rule-based arousal)",
         f"- questions tracked: {meta['questions']}", "",
         "## Tournament (pairwise judge)", "",
         f"- pairs: {t['pairs']}, judgments: {t['judgments']}, judged in both orders: {t['both_orders']}",
         f"- order consistency: {t['order_consistency']} (share of pairs with the same winner in both orders; the rest count as ties)",
         f"- first-position win rate: {t['first_position_win_rate']} (0.5 = no position bias)",
         f"- Spearman(signal score, tournament strength): {t['spearman_signal_vs_tournament']} "
         "(how far the code-only optimizer agrees with the judge)", "",
         f"- win rate by candidate generator (blind head-to-heads): {t.get('win_rate_by_generator')}",
         f"- finalists by generator: {meta.get('finalists_by_generator')}", "",
         "## Audience panel", ""]
    for p in finals:
        a = p.get("audience")
        if a:
            Q.append(f"- {p['id']} {seg_str(p)}: hold3 {a['hold_3s']:.0%}, completion {a['completion']:.0%}, tap {a['tap_rate']:.0%}")
    Q += ["", "## Errors", "", "\n".join(f"- {e}" for e in meta["errors"]) or "none", "",
          "## Usage", "", "```", json.dumps(meta["usage"], indent=1), "```"]
    (out / "qc.md").write_text("\n".join(Q) + "\n")

    css = """body{font-family:system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#222}
    .p{border:1px solid #ddd;border-radius:8px;padding:12px 16px;margin:14px 0}.fin{border:2px solid #c47b00}
    .meta{color:#555;font-size:13px}.beat{font-family:ui-monospace,monospace;font-size:12px;white-space:pre-wrap;margin:1px 0}
    .camera{color:#999}.action{color:#245}.sfx{color:#735}.hook{background:#fff3c4}.charts{display:flex;gap:14px;margin:8px 0}
    .q{color:#a33}.v{font-size:12px;color:#444}"""
    H = ["<!doctype html><meta charset=utf-8><title>v2 promo review</title><style>" + css + "</style>",
         f"<h1>Promo review (v2 Curiosity Engine): {html.escape(meta['show'])}</h1>",
         "<p class=meta>Orange = finalists in post-first order. Charts: what the viewer feels across the clip "
         "(annotations) and the panel's predicted retention.</p>"]
    for p in ranked[:40]:
        pts, dur = series(p, emap, ann, questions)
        a = p.get("audience", {})
        H.append(f"<div class='p{' fin' if p['stage'] == 'finalist' else ''}'>")
        H.append(f"<b>#{p['rank']} {html.escape(seg_str(p))}</b> <span class=meta>{p['duration']}s - "
                 f"{p['source']} - signal {p['signal_score']} - record {p.get('tournament', {}).get('record', '-')}"
                 + (f" - hold3 {a['hold_3s']:.0%} / complete {a['completion']:.0%} / tap {a['tap_rate']:.0%}" if a else "")
                 + "</span>")
        H.append(f"<div><b>hook:</b> {html.escape(p['hook'])}</div><div><b>cliffhanger:</b> {html.escape(p['cliffhanger'])}</div>")
        H.append(f"<div class=meta>{html.escape(p['why'])}</div>")
        H.append(f"<div class=charts>{svg_signals(pts, dur)}{svg_curve(a.get('curve', []))}</div>")
        for k, v in a.get("voices", {}).items():
            H.append(f"<div class=v>{k}: swipes at {v['swipe_at']}, tap {v['tap']} - {html.escape(str(v['says']))}</div>")
        H.append("<details><summary>script</summary>")
        for i, s in enumerate(p["segments"]):
            H.append(f"<div class=meta>-- segment {i + 1}: Ep{s['ep']} {s['start_ts']}-{s['end_ts']}</div>")
            for b in beats_in(emap[s["ep"]], s["start"], s["end"]):
                x = ann[b.id]
                tag = ""
                if x.labelled:
                    tag = f"  [A{x.a:g} S{x.s:g} C{x.c:g}{' ' + x.tp if x.tp else ''}]"
                    if x.opens:
                        tag += " OPENS " + "; ".join(questions.get(q, {}).get("text", q) for q in x.opens)
                    if x.closes:
                        tag += " CLOSES " + ",".join(x.closes)
                cls = b.kind + (" hook" if i == 0 and b.start < s["start"] + 3 else "")
                H.append(f"<div class='beat {cls}'>{html.escape(b.line())}<span class=q>{html.escape(tag)}</span></div>")
        H.append("</details></div>")
    (out / "review.html").write_text("\n".join(H))
