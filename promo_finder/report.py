"""Write run outputs: promos.json, promos.csv, report.md, qc.md, review.html, run_meta.json."""
from __future__ import annotations

import csv
import html
import json
from collections import Counter
from pathlib import Path

from .heuristics import beats_in, best_windows, overlap

from .validate import Index


def seg_str(p):
    return " + ".join(f"Ep{s['ep']} {s['start_ts']}-{s['end_ts']}" for s in p["segments"])


def eps_str(p):
    return ",".join(str(e) for e in dict.fromkeys(s["ep"] for s in p["segments"]))


def write_all(out: Path, idx: Index, valid, rejected, top, meta):
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2))
    clean = []
    for p in valid:
        q = {k: v for k, v in p.items()}
        clean.append(q)
    (out / "promos.json").write_text(json.dumps({"meta": meta, "post_first": [p["id"] for p in top],
                                                 "promos": clean, "rejected": rejected}, indent=2, ensure_ascii=False))
    with open(out / "promos.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "id", "post_first", "source", "episodes", "segments", "duration_s", "hook",
                    "cliffhanger", "why", "score", "status", "flags"])
        top_ids = {p["id"] for p in top}
        for p in valid:
            w.writerow([p["rank"], p["id"], "yes" if p["id"] in top_ids else "", p["source"], eps_str(p), seg_str(p),
                        p["duration"], p.get("hook", ""), p.get("cliffhanger", ""), p.get("why", ""),
                        p["rank_score"], p["status"], " | ".join(p["flags"])])
    _report_md(out, valid, top, meta)
    _qc_md(out, idx, valid, rejected, meta)
    _review_html(out, idx, valid, top, meta)


def _report_md(out, valid, top, meta):
    L = [f"# Promo picks - {meta['show']}", "",
         f"Model: `{meta['model']}` | episodes: {meta['episodes']} | valid promos: {meta['promos_valid']} | "
         f"rejected by validator: {meta['promos_rejected']} | runtime: {meta['seconds']}s", "",
         "## Post these first", "",
         "Top of the ranked list, max 2 promos per episode so the first week of posts is not ten cuts of one scene.", ""]
    for i, p in enumerate(top, 1):
        L += [f"### {i}. {p.get('title', p['id'])}  (`{p['id']}`, score {p['rank_score']})",
              f"- **Segments:** {seg_str(p)}  -  **{p['duration']}s**",
              f"- **Hook (0-3s):** {p.get('hook', '')}",
              f"- **Cliffhanger:** {p.get('cliffhanger', '')}",
              f"- **Why it works:** {p.get('why', '')}"]
        if p.get("cold"):
            c = p["cold"]
            L.append(f"- **Cold-viewer test:** \"{c.get('what_is_happening', '')}\" (stop {c.get('stop_scrolling')}/10, "
                     f"need-next {c.get('need_to_know_next')}/10, understood {c.get('understood')}/10)"
                     + (f"; confused by: {c['confused_by']}" if c.get("confused_by") else ""))
        if p["flags"]:
            L.append(f"- **Checks:** {'; '.join(p['flags'])}")
        L.append("")
    L += ["## Full ranked list", "", "| # | id | episodes | segments | sec | hook | cliffhanger | score | status |",
          "|---|---|---|---|---|---|---|---|---|"]
    for p in valid:
        L.append(f"| {p['rank']} | {p['id']} | {eps_str(p)} | {seg_str(p)} | {p['duration']} | "
                 f"{_cell(p.get('hook', ''))} | {_cell(p.get('cliffhanger', ''))} | {p['rank_score']} | {p['status']} |")
    (out / "report.md").write_text("\n".join(L) + "\n")


def _cell(s, n=90):
    s = str(s).replace("|", "/").replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def _qc_md(out, idx: Index, valid, rejected, meta):
    L = [f"# Quality check - {meta['show']} ({meta['model']})", ""]
    st = Counter(p["status"] for p in valid)
    L += ["## Validator", "",
          f"- valid: {len(valid)} (clean: {st.get('ok', 0)}, auto-fixed: {st.get('fixed', 0)}), rejected: {len(rejected)}",
          f"- LLM calls: {meta['llm_calls']} (cache hits {meta['cache_hits']}), tokens: {meta.get('usage')}",
          f"- errors: {meta['errors'] or 'none'}", ""]
    kinds = Counter()
    for p in valid + rejected:
        for f in p["flags"]:
            key = f.split("'")[0].split(" to ")[0].split(" from ")[0][:60]
            kinds[key] += 1
    if kinds:
        L += ["Flag counts (what the validator had to fix or warn about):", ""]
        L += [f"- {n} x {k}" for k, n in kinds.most_common(20) if n]
        L.append("")
    durs = [p["duration"] for p in valid]
    if durs:
        L += [f"Durations: min {min(durs)}s, median {sorted(durs)[len(durs) // 2]}s, max {max(durs)}s", ""]
    if rejected:
        L += ["### Rejected", ""] + [f"- {p.get('id')} {p.get('title', '')}: {'; '.join(f for f in p['flags'] if 'REJECT' in f or 'multi' in f)}" for p in rejected] + [""]

    covered = {s["ep"] for p in valid for s in p["segments"]}
    missing = sorted(set(idx.eps) - covered)
    L += ["## Coverage", "", f"- episodes with at least one promo: {len(covered)}/{len(idx.eps)}",
          f"- episodes with none: {missing or 'none'}", ""]

    if meta["model"] != "heuristic":
        agree, rows = 0, []
        hw = {n: best_windows(e, 3) for n, e in idx.eps.items()}
        for p in valid:
            s = p["segments"][0]
            hit = any(overlap((s["start"], s["end"]), w["segments"][0]) > 0.3 for w in hw[s["ep"]])
            agree += hit
            lq = p.get("score_parts", {}).get("llm", 0)
            rows.append((lq - p["score_parts"].get("heuristic_pct", 0), p))
        L += ["## LLM vs rule-based agreement", "",
              f"- {agree}/{len(valid)} LLM promos overlap one of the rule-based top-3 windows for that episode.",
              "- Biggest disagreements (LLM loves it, rules don't) - check these by hand:", ""]
        for d, p in sorted(rows, key=lambda r: -r[0])[:5]:
            L.append(f"  - {p['id']} {seg_str(p)}: llm {p['score_parts']['llm']}, rules pct {p['score_parts']['heuristic_pct']} - {p.get('hook', '')[:80]}")
        L += ["", "- Rules love it, LLM doesn't:", ""]
        for d, p in sorted(rows, key=lambda r: r[0])[:5]:
            L.append(f"  - {p['id']} {seg_str(p)}: llm {p['score_parts']['llm']}, rules pct {p['score_parts']['heuristic_pct']} - {p.get('hook', '')[:80]}")
        L.append("")
        cold = [p for p in valid if p.get("cold")]
        if cold:
            L += ["## Cold-viewer test (fresh model call, no show context)", "",
                  "| id | understood | stop | need-next | what they think is happening | confused by |", "|---|---|---|---|---|---|"]
            for p in sorted(cold, key=lambda p: float(p["cold"].get("understood") or 0)):
                c = p["cold"]
                L.append(f"| {p['id']} | {c.get('understood')} | {c.get('stop_scrolling')} | {c.get('need_to_know_next')} | "
                         f"{_cell(c.get('what_is_happening', ''), 120)} | {_cell(c.get('confused_by', ''), 100)} |")
            L.append("")
            sc = [(float(p['cold'].get('stop_scrolling') or 0), float(p['scores'].get('hook') or 0)) for p in cold if p.get('scores')]
            if sc:
                over = sum(1 for c, h in sc if h - c >= 2)
                L += [f"- Self-score inflation: in {over}/{len(sc)} clips the picker's hook score is 2+ points above the cold viewer's stop-scrolling score.", ""]
    (out / "qc.md").write_text("\n".join(L) + "\n")


def _review_html(out, idx: Index, valid, top, meta):
    top_ids = {p["id"] for p in top}
    css = """body{font-family:system-ui,sans-serif;max-width:1000px;margin:24px auto;padding:0 16px;color:#222}
    .p{border:1px solid #ddd;border-radius:8px;padding:12px 16px;margin:14px 0}.top{border-color:#c47b00;border-width:2px}
    .meta{color:#555;font-size:13px}.beat{font-family:ui-monospace,monospace;font-size:12px;margin:1px 0;white-space:pre-wrap}
    .dialogue{color:#000}.camera{color:#888}.action{color:#245}.sfx{color:#735}.hook{background:#fff3c4}.cliff{background:#ffd9d9}
    .flag{color:#a33;font-size:12px}details summary{cursor:pointer;color:#036}h3{margin:4px 0}"""
    H = [f"<!doctype html><meta charset=utf-8><title>Promo review - {html.escape(meta['show'])}</title><style>{css}</style>",
         f"<h1>Promo review: {html.escape(meta['show'])}</h1>",
         f"<p class=meta>model {html.escape(meta['model'])} - {meta['promos_valid']} promos - orange border = post first. "
         "Yellow = first 3 seconds (hook). Red = last 4 seconds (cliffhanger).</p>"]
    for p in valid:
        H.append(f"<div class='p{' top' if p['id'] in top_ids else ''}'>")
        H.append(f"<h3>#{p['rank']} {html.escape(str(p.get('title', p['id'])))} <span class=meta>({p['id']}, {p['source']}, score {p['rank_score']})</span></h3>")
        H.append(f"<div class=meta><b>{html.escape(seg_str(p))}</b> - {p['duration']}s - status {p['status']}</div>")
        for k in ("hook", "cliffhanger", "why"):
            H.append(f"<div><b>{k}:</b> {html.escape(str(p.get(k, '')))}</div>")
        if p.get("cold"):
            c = p["cold"]
            H.append(f"<div class=meta>cold viewer: {html.escape(str(c.get('what_is_happening', '')))} "
                     f"(understood {c.get('understood')}, stop {c.get('stop_scrolling')}, next {c.get('need_to_know_next')})</div>")
        for f in p["flags"]:
            H.append(f"<div class=flag>{html.escape(f)}</div>")
        H.append("<details><summary>script</summary>")
        for i, s in enumerate(p["segments"]):
            H.append(f"<div class=meta>-- segment {i + 1}: Ep{s['ep']} {s['start_ts']}-{s['end_ts']}</div>")
            for b in beats_in(idx.eps[s["ep"]], s["start"], s["end"]):
                cls = b.kind
                if i == 0 and b.start < s["start"] + 3:
                    cls += " hook"
                if i == len(p["segments"]) - 1 and b.end > s["end"] - 4:
                    cls += " cliff"
                H.append(f"<div class='beat {cls}'>{html.escape(b.line())}</div>")
        H.append("</details></div>")
    (out / "review.html").write_text("\n".join(H))
