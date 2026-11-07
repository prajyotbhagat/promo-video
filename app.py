"""Promo Finder - editor UI.

    pip install -r requirements.txt
    streamlit run app.py

Load scripts (zip upload, a local folder, or a previous run), check input quality, see the exact
LLM-call plan for a budget level, run, then review / approve promos.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import html
import io
import json
import os
import re
import zipfile
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from promo_finder.heuristics import beats_in
from promo_finder.parser import fmt, load_show
from promo_finder.qa import inspect

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
UPLOADS = ROOT / ".ui_uploads"

LEVELS = {
    "B0": ("Rules only", "No AI, no key, ~15 s. Good first look; 11/14 on our test labels."),
    "B1": ("+ Story annotation", "LLM labels every beat (curiosity signals). ~10 calls per new show."),
    "B2": ("+ LLM picker (recommended)", "Two generators fused with RRF. ~20 calls per new show, 13/14 on our test labels."),
    "B3": ("+ Pairwise judge", "Adds a stronger judge on a 24-clip shortlist. +4 calls."),
    "B4": ("+ Audience panel", "Adds 5 viewer personas predicting swipe-away time. +5 calls."),
}
KEYS = {"gemini": "GEMINI_API_KEY", "hf": "HF_TOKEN", "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
        "groq": "GROQ_API_KEY", "openrouter": "OPENROUTER_API_KEY"}

st.set_page_config(page_title="Promo Finder", page_icon="🎬", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1350px;}
.card {border: 1px solid rgba(128,128,128,.25); border-radius: 12px; padding: 14px 18px; margin: 4px 0 10px 0;}
.rank {font-size: 2.1rem; font-weight: 800; line-height: 1; opacity: .85}
.seg {font-family: ui-monospace, monospace; font-size: .86rem; opacity: .85}
.hook {background: rgba(255, 200, 0, .16); border-left: 4px solid #e0a800; padding: 6px 10px; border-radius: 6px; margin: 4px 0;}
.cliff {background: rgba(230, 60, 60, .12); border-left: 4px solid #d33; padding: 6px 10px; border-radius: 6px; margin: 4px 0;}
.chip {display: inline-block; padding: 1px 9px; margin: 2px 4px 2px 0; border-radius: 999px; font-size: .78rem;
       background: rgba(100, 120, 255, .14); border: 1px solid rgba(100, 120, 255, .35);}
.chip.warn {background: rgba(255, 150, 0, .15); border-color: rgba(255, 150, 0, .5);}
.chip.ok {background: rgba(30, 160, 80, .14); border-color: rgba(30, 160, 80, .45);}
.beat {font-family: ui-monospace, monospace; font-size: .8rem; padding: 1px 6px; border-radius: 4px; margin: 1px 0;}
.beat.camera {opacity: .55} .beat.action {color: #2b6cb0} .beat.sfx {color: #8b5cf6}
.beat.inhook {background: rgba(255, 200, 0, .18)} .beat.incliff {background: rgba(230, 60, 60, .12)}
.small {font-size: .82rem; opacity: .8}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ data helpers
def safe_extract(upload) -> Path:
    """Extract only .txt files from an uploaded zip into .ui_uploads/<hash>/ (no path traversal)."""
    data = upload.getvalue()
    dest = UPLOADS / hashlib.sha256(data).hexdigest()[:16]
    if not dest.exists():
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for info in z.infolist():
                name = info.filename.replace("\\", "/")
                if info.is_dir() or not name.lower().endswith(".txt") or "__MACOSX" in name:
                    continue
                parts = [p for p in name.split("/") if p not in ("", ".", "..")]
                show = parts[-2] if len(parts) > 1 else Path(upload.name).stem
                target = dest / show / parts[-1]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(info))
    return dest


@st.cache_resource(show_spinner=False)
def episodes(path: str, _mtime: float):
    eps = load_show(path)
    return eps, {e.num: e for e in eps}


def folder_mtime(path: Path) -> float:
    return max((p.stat().st_mtime for p in path.rglob("*.txt")), default=0.0)


def list_runs():
    runs = []
    for p in sorted(OUTPUTS.rglob("promos.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        meta = d.get("meta", {})
        if meta.get("engine") == "v3-fusion" and "moments" in d:
            runs.append((p.parent, meta))
    return sorted(runs, key=lambda r: r[0].stat().st_mtime, reverse=True)


@st.cache_data(show_spinner=False)
def load_run(run_dir: str, _mtime: float):
    d = json.loads((Path(run_dir) / "promos.json").read_text())
    ann = {}
    a = Path(run_dir) / "annotations.jsonl"
    if a.exists():
        for line in a.read_text().splitlines():
            x = json.loads(line)
            ann[x["beat"]] = x
    q = {}
    qp = Path(run_dir) / "questions.json"
    if qp.exists():
        q = json.loads(qp.read_text())
    return d, ann, q


def seg_str(p):
    return " + ".join(f"Ep{s['ep']} {s['start_ts']}–{s['end_ts']}" for s in p["segments"])


def show_name(path: Path) -> str:
    subs = [p for p in path.iterdir() if p.is_dir()] if path.is_dir() else []
    return subs[0].name if len(subs) == 1 else path.name


def feedback_path(run_dir: Path) -> Path:
    return run_dir / "editor_feedback.jsonl"


def load_feedback(run_dir: Path) -> dict:
    fb = {}
    p = feedback_path(run_dir)
    if p.exists():
        for line in p.read_text().splitlines():
            x = json.loads(line)
            fb[x["id"]] = x  # last decision wins
    return fb


def save_feedback(run_dir: Path, promo: dict, decision: str, note: str):
    rec = {"id": promo["id"], "decision": decision, "note": note, "at": dt.datetime.now().isoformat(timespec="seconds"),
           "segments": [{k: s[k] for k in ("ep", "start", "end")} for s in promo["segments"]],
           "ranker_ranks": promo.get("ranker_ranks", {}), "source": promo.get("source")}
    with open(feedback_path(run_dir), "a") as f:
        f.write(json.dumps(rec) + "\n")


def clip_series(p, emap, ann, questions):
    """(t, arousal, open-question load) in play time, from the run's annotations."""
    rows, off, opened, closed = [], 0.0, {}, set()
    imp = lambda q: float(questions.get(q, {}).get("importance", 1))
    for s in p["segments"]:
        for b in beats_in(emap[s["ep"]], s["start"], s["end"]):
            x = ann.get(b.id)
            if not x:
                continue
            for qid in x["closes"]:
                if qid in opened:
                    closed.add(qid)
            for qid in x["opens"]:
                opened.setdefault(qid, 1)
            t = off + b.start - s["start"]
            rows.append({"second": round(t, 1), "arousal": x["a"],
                         "open questions": sum(imp(k) for k in opened if k not in closed)})
        off += s["end"] - s["start"]
    return pd.DataFrame(rows)


def script_html(p, emap, ann, questions):
    out, off, seen = [], 0.0, set()
    for i, s in enumerate(p["segments"]):
        if i:
            out.append("<div class='small'>— jump cut —</div>")
        for b in beats_in(emap[s["ep"]], s["start"], s["end"]):
            t = off + b.start - s["start"]
            cls = b.kind
            if i == 0 and b.start < s["start"] + 3:
                cls += " inhook"
            if i == len(p["segments"]) - 1 and b.end > s["end"] - 4:
                cls += " incliff"
            who = f"<b>{html.escape(b.speaker)}</b>: " if b.kind == "dialogue" else f"<i>{b.kind}</i>: "
            tag = ""
            x = ann.get(b.id)
            new = [q for q in (x["opens"] if x else []) if q not in seen]
            seen.update(new)
            if new:
                tag = " <span class='chip warn'>❓ " + html.escape(
                    "; ".join(questions.get(qid, {}).get("text", qid) for qid in new)[:140]) + "</span>"
            out.append(f"<div class='beat {cls}'>[{t:5.1f}s] {who}{html.escape(b.text)}{tag}</div>")
        off += s["end"] - s["start"]
    return "\n".join(out)


def planned_calls(path: Path, model: str, level: str) -> dict | None:
    if level == "B0":
        return {"total_calls": 0}
    try:
        from promo_finder.agent import Agent
        subs = [p for p in path.iterdir() if p.is_dir()]
        show_dir = subs[0] if len(subs) == 1 else path  # the show name is part of the cached prompt
        a = Agent(show_dir.parent, ROOT / ".ui_plans", model, level, log=lambda *_: None)
        return a.plan(show_dir)
    except Exception as e:
        return {"error": str(e)}


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🎬 Promo Finder")
st.sidebar.caption("Finds 20–60 s promo clips in microdrama scripts.")

mode = st.sidebar.radio("Scripts", ["Open a previous run", "Upload a .zip", "Folder on this computer"], index=0)
run_dir = st.session_state.get("run_dir")
input_path = None

if mode == "Open a previous run":
    runs = list_runs()
    if not runs:
        st.sidebar.info("No runs yet. Upload scripts to start.")
    else:
        def _label(r, m):
            d = Path(m.get("input_dir", ""))
            name = show_name(d) if d.exists() else (d.name or r.name)
            when = dt.datetime.fromtimestamp((r / "promos.json").stat().st_mtime).strftime("%d %b %H:%M")
            return f"{name[:34]} · {m.get('budget')} · {when}"
        labels = [_label(r, m) for r, m in runs]
        i = st.sidebar.selectbox("Run", range(len(runs)), format_func=lambda k: labels[k])
        run_dir = runs[i][0]
        st.session_state["run_dir"] = run_dir
elif mode == "Upload a .zip":
    up = st.sidebar.file_uploader("Zip with one folder of episode .txt files", type="zip")
    if up:
        input_path = safe_extract(up)
else:
    p = st.sidebar.text_input("Folder path", value=str(ROOT / "input"))
    if p and Path(p).expanduser().exists():
        input_path = Path(p).expanduser()
    elif p:
        st.sidebar.error("Folder not found")

if input_path is not None:
    eps, _ = episodes(str(input_path), folder_mtime(input_path))
    if not eps:
        st.sidebar.error("No episode scripts found in that folder.")
    else:
        qa = inspect(eps)
        badge = {"ok": "🟢", "warn": "🟡", "templated": "🔴"}[qa["verdict"]]
        st.sidebar.markdown(f"**{show_name(input_path)}**  \n{len(eps)} episodes · input check {badge} `{qa['verdict']}`")
        for w in qa["warnings"]:
            st.sidebar.caption("⚠️ " + w)

        st.sidebar.divider()
        level = st.sidebar.select_slider("Budget level", options=list(LEVELS), value="B2",
                                         format_func=lambda k: k)
        st.sidebar.caption(f"**{LEVELS[level][0]}** — {LEVELS[level][1]}")
        model = judge = None
        if level != "B0":
            model = st.sidebar.text_input("Model", value="gemini:gemini-3.5-flash-lite",
                                          help="provider:model, e.g. hf:Qwen/Qwen2.5-72B-Instruct, openai:gpt-4.1-mini")
            prov = model.split(":", 1)[0]
            env = KEYS.get(prov)
            if env and not os.environ.get(env):
                key = st.sidebar.text_input(f"{env}", type="password",
                                            help="Kept in this app's memory only; never written to disk.")
                if key:
                    os.environ[env] = key
            if level in ("B3", "B4"):
                judge = st.sidebar.text_input("Judge model", value="gemini:gemini-3.5-flash")
            plan = planned_calls(input_path, model, level)
            if plan and "error" not in plan:
                st.sidebar.metric("Planned LLM calls", plan["total_calls"],
                                  help="Exact count from the cache: work already done costs nothing.")
                if plan.get("episodes_to_annotate") is not None:
                    st.sidebar.caption(f"{plan['episodes_to_annotate']} of {plan['episodes']} episodes need annotation; "
                                       f"{plan['picker_calls']} picker calls")
        if st.sidebar.button("▶  Find promos", type="primary", width="stretch"):
            from promo_finder.v3.pipeline import run as run_v3
            stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
            out = OUTPUTS / "ui_runs" / re.sub(r"[^A-Za-z0-9]+", "-", show_name(input_path))[:40] / f"{stamp}-{level}"
            buf = io.StringIO()
            with st.status(f"Running {level} on {len(eps)} episodes…", expanded=True) as status:
                try:
                    with contextlib.redirect_stdout(buf):
                        run_v3(str(input_path), str(out), model or "gemini:gemini-3.5-flash-lite", level, judge)
                    status.update(label="Done", state="complete")
                    st.session_state["run_dir"] = out
                    run_dir = out
                except Exception as e:
                    status.update(label="Run failed", state="error")
                    st.error(str(e))
                st.code(buf.getvalue()[-4000:] or "(no log)", language=None)

# ------------------------------------------------------------------ main
if not run_dir or not (Path(run_dir) / "promos.json").exists():
    st.title("Promo Finder")
    st.markdown("""
Find the best **20–60 second promo clips** in a microdrama, ranked, with the hook, the cliffhanger, and *why*.

1. **Load scripts** in the sidebar: upload a zip, point at a folder, or open a previous run.
2. **Check the input**: the tool flags templated or broken scripts before spending anything.
3. **Pick a budget level** and see the exact number of AI calls it will make (B0 needs no AI at all).
4. **Review**: approve or reject each promo; your decisions are saved to tune the ranking.
""")
    st.stop()

run_dir = Path(run_dir)
data, ann, questions = load_run(str(run_dir), (run_dir / "promos.json").stat().st_mtime)
meta = data["meta"]
in_dir = Path(meta.get("input_dir") or (ROOT / "input"))
if not in_dir.exists():
    st.warning(f"Scripts for this run are not at `{in_dir}`; script views are disabled.")
    emap = {}
else:
    _, emap = episodes(str(in_dir), folder_mtime(in_dir))
promos = {p["id"]: p for p in data["promos"]}
moments = data["moments"]
post = [promos[i] for i in data["post_first"] if i in promos]
qa = meta.get("input_check", {})
fb = load_feedback(run_dir)

st.title(show_name(in_dir) if in_dir.exists() else "Promo run")
calls = sum(int(v.get("calls", 0)) for v in meta.get("cost_by_stage", {}).values())
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Episodes", meta.get("episodes"))
c2.metric("Candidate clips", meta.get("candidates"))
c3.metric("Distinct moments", meta.get("moments"))
c4.metric("LLM calls this run", calls, help="0 when everything came from the cache")
verdict = qa.get("verdict", "-")
c5.metric(f"Input check: {verdict}", {"ok": "🟢", "warn": "🟡", "templated": "🔴"}.get(verdict, "-"),
          help="Checks the scripts before any AI is used: repeated lines, identical timing, coarse beats, narration as dialogue.")
st.caption(f"Budget {meta.get('budget')} · model `{meta.get('model')}` · rankers fused: "
           f"{', '.join(meta.get('rankers', []))} · {meta.get('seconds')} s · `{run_dir.relative_to(ROOT) if run_dir.is_relative_to(ROOT) else run_dir}`")
if qa.get("warnings"):
    with st.expander(f"⚠️ {len(qa['warnings'])} input warning(s)"):
        for w in qa["warnings"]:
            st.markdown(f"- {w}")

tab_top, tab_all, tab_ep, tab_fb, tab_how = st.tabs(
    ["⭐ Post first", "📋 All promos", "🎞️ Episode explorer", f"✅ Editor review ({len(fb)})", "ℹ️ How it works"])

# ---- Post first
with tab_top:
    st.caption("Top 10 distinct moments, at most 2 per episode, repeated footage left out. Yellow = first 3 s (hook), red = last 4 s (cliffhanger).")
    if emap and post:
        def _scripts_zip():
            from promo_finder.scripts import promo_markdown, promo_script
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                for k, pp in enumerate(post, 1):
                    z.writestr(f"promo_scripts/{k:02d}_{pp['id']}.txt", promo_script(pp, emap, show_name(in_dir)))
                z.writestr("promo_scripts/promo_scripts.md",
                           "\n".join(promo_markdown(pp, emap, k) for k, pp in enumerate(post, 1)))
            return buf.getvalue()
        st.download_button(f"📝 Download scripts for all {len(post)} promos (.zip)", _scripts_zip(),
                           "promo_scripts.zip", "application/zip",
                           help="One screenplay per promo in the episode-script format, plus a readable summary")
    for i, p in enumerate(post, 1):
        with st.container(border=True):
            left, mid, right = st.columns([0.7, 5, 2.3])
            left.markdown(f"<div class='rank'>#{i}</div>", unsafe_allow_html=True)
            dec = fb.get(p["id"], {}).get("decision")
            if dec:
                left.markdown("<span class='chip ok'>approved</span>" if dec == "approve"
                              else "<span class='chip warn'>rejected</span>", unsafe_allow_html=True)
            mid.markdown(f"<span class='seg'>{seg_str(p)} · {p.get('duration')} s</span>", unsafe_allow_html=True)
            mid.markdown(f"<div class='hook'><b>Hook</b> · {html.escape(str(p.get('hook', '')))}</div>"
                         f"<div class='cliff'><b>Cliffhanger</b> · {html.escape(str(p.get('cliffhanger', '')))}</div>",
                         unsafe_allow_html=True)
            if p.get("why"):
                mid.caption(p["why"])
            flags = [f for f in p.get("flags", []) if f.startswith(("check", "boilerplate", "warn"))]
            for f in flags[:3]:
                mid.markdown(f"<span class='chip warn'>{html.escape(f[:110])}</span>", unsafe_allow_html=True)
            found = next((m.get("also_found_by", []) for m in moments if m["id"] == p["id"]), [p.get("source")])
            right.markdown("**Found by** " + " ".join(f"<span class='chip'>{html.escape(s)}</span>" for s in found),
                           unsafe_allow_html=True)
            rr = p.get("ranker_ranks", {})
            if rr:
                right.markdown("**Rank per engine**  \n" + " · ".join(f"{k} **#{v}**" for k, v in rr.items()))
            b1, b2, b3 = right.columns([1, 1, 2])
            note = b3.text_input("note", key=f"n{p['id']}", label_visibility="collapsed", placeholder="note")
            if b1.button("👍", key=f"a{p['id']}", help="Approve"):
                save_feedback(run_dir, p, "approve", note)
                st.rerun()
            if b2.button("👎", key=f"r{p['id']}", help="Reject"):
                save_feedback(run_dir, p, "reject", note)
                st.rerun()
            if emap:
                from promo_finder.scripts import promo_script
                right.download_button("📝 Script", promo_script(p, emap, show_name(in_dir)), f"promo_{p['id']}.txt",
                                      key=f"t{p['id']}", width="stretch",
                                      help="This promo as its own screenplay (Kuku format), timed from 00:00 of the promo")
            if emap:
                with st.expander("Script and signals"):
                    c_col = st.container()
                    s_col = st.container()
                    s_col.markdown(script_html(p, emap, ann, questions), unsafe_allow_html=True)
                    if ann:
                        df = clip_series(p, emap, ann, questions)
                        if not df.empty:
                            base = alt.Chart(df).encode(x=alt.X("second:Q", title="seconds into the promo"))
                            area = base.mark_area(opacity=.3, color="#d33", interpolate="step-after").encode(
                                y=alt.Y("open questions:Q", title="unanswered questions",
                                        axis=alt.Axis(titleColor="#d33", orient="right")))
                            line = base.mark_line(color="#2b6cb0", point=True, interpolate="monotone").encode(
                                y=alt.Y("arousal:Q", title="arousal (0-3)", scale=alt.Scale(domain=[0, 3]),
                                        axis=alt.Axis(titleColor="#2b6cb0")),
                                tooltip=["second", "arousal", "open questions"])
                            band = alt.Chart(pd.DataFrame({"x": [0], "x2": [3]})).mark_rect(
                                opacity=.18, color="#e0a800").encode(x="x:Q", x2="x2:Q")
                            c_col.caption("What the viewer feels: arousal (blue, left axis), unanswered questions "
                                          "(red, right axis), hook window (yellow)")
                            c_col.altair_chart(alt.layer(band, area, line).resolve_scale(y="independent")
                                               .properties(height=190), width="stretch")
                    elif not ann:
                        c_col.caption("No annotations in this run (B0). Run B1+ to see curiosity signals.")

# ---- All promos
with tab_all:
    rows = []
    mids = {m["id"] for m in moments}
    for p in data["promos"]:
        rows.append({"rank": p.get("rank"), "id": p["id"], "episode": p["segments"][0]["ep"], "segments": seg_str(p),
                     "seconds": p.get("duration"), "source": p.get("source"), "distinct moment": p["id"] in mids,
                     "fused score": p.get("fused"), "hook": str(p.get("hook", ""))[:120],
                     "cliffhanger": str(p.get("cliffhanger", ""))[:120]})
    df = pd.DataFrame(rows)
    f1, f2, f3 = st.columns([2, 2, 1])
    ep_f = f1.multiselect("Episodes", sorted(df["episode"].unique()))
    src_f = f2.multiselect("Source", sorted(df["source"].dropna().unique()))
    only_m = f3.toggle("Distinct moments only", value=True)
    v = df.copy()
    if ep_f:
        v = v[v["episode"].isin(ep_f)]
    if src_f:
        v = v[v["source"].isin(src_f)]
    if only_m:
        v = v[v["distinct moment"]]
    st.dataframe(v, width="stretch", hide_index=True, height=520,
                 column_config={"fused score": st.column_config.ProgressColumn(
                     "fused score", min_value=0.0, max_value=float(df["fused score"].max() or 1), format="%.4f")})
    d1, d2, d3 = st.columns(3)
    d1.download_button("Download CSV", v.to_csv(index=False), "promos.csv", "text/csv")
    d2.download_button("Download JSON", (run_dir / "promos.json").read_text(), "promos.json", "application/json")
    if (run_dir / "report.md").exists():
        d3.download_button("Download report", (run_dir / "report.md").read_text(), "report.md", "text/markdown")

# ---- Episode explorer
with tab_ep:
    if not emap:
        st.info("Scripts not available for this run.")
    else:
        n = st.selectbox("Episode", sorted(emap), format_func=lambda k: f"Episode {k}")
        ep = emap[n]
        st.caption(ep.logline)
        brow = []
        opened, closed = {}, set()
        for b in ep.beats:
            x = ann.get(b.id)
            if x:
                for qid in x["closes"]:
                    closed.add(qid)
                for qid in x["opens"]:
                    opened.setdefault(qid, 1)
                brow.append({"second": b.start, "arousal": x["a"], "surprise": x["s"],
                             "open questions": len([k for k in opened if k not in closed]),
                             "line": f"{b.speaker}: {b.text}"[:90] if b.kind == "dialogue" else b.text[:90]})
        clips = []
        for p in data["promos"]:
            for s in p["segments"]:
                if s["ep"] == n:
                    clips.append({"start": s["start"], "end": s["end"], "source": p.get("source"),
                                  "rank": p.get("rank"), "id": p["id"],
                                  "top10": p["id"] in data["post_first"]})
        layers, titles = [], []
        if brow:
            bdf = pd.DataFrame(brow)
            base = alt.Chart(bdf).encode(x=alt.X("second:Q", title="episode time (s)", scale=alt.Scale(domain=[0, ep.duration])))
            layers.append(alt.layer(
                base.mark_area(opacity=.25, color="#d33", interpolate="step-after").encode(
                    y=alt.Y("open questions:Q", title="unanswered questions", axis=alt.Axis(titleColor="#d33", orient="right"))),
                base.mark_line(color="#2b6cb0", point=True, interpolate="monotone").encode(
                    y=alt.Y("arousal:Q", title="arousal (0-3)", scale=alt.Scale(domain=[0, 3]), axis=alt.Axis(titleColor="#2b6cb0")),
                    tooltip=["second", "arousal", "surprise", "line"]),
            ).resolve_scale(y="independent").properties(height=210))
            titles.append("Arousal (blue, left axis) and unanswered questions (red, right axis) through the episode")
        if clips:
            cdf = pd.DataFrame(clips)
            layers.append(alt.Chart(cdf).mark_bar(height=10, cornerRadius=3).encode(
                x=alt.X("start:Q", title="episode time (s)", scale=alt.Scale(domain=[0, ep.duration])), x2="end:Q",
                y=alt.Y("source:N", title=""),
                color=alt.condition("datum.top10", alt.value("#e07b00"), alt.value("#9aa5b1")),
                tooltip=["id", "rank", "source", "start", "end"],
            ).properties(height=150))
            titles.append("Candidate clips by engine (orange = in the post-first list); hover for details")
        for t, ch in zip(titles, layers):
            st.caption(t)
            st.altair_chart(ch, width="stretch")
        with st.expander("Full script"):
            st.markdown("\n".join(
                f"<div class='beat {b.kind}'>[{fmt(b.start)}] "
                + (f"<b>{html.escape(b.speaker)}</b>: " if b.kind == "dialogue" else f"<i>{b.kind}</i>: ")
                + html.escape(b.text) + "</div>" for b in ep.beats), unsafe_allow_html=True)

# ---- Editor review
with tab_fb:
    if not fb:
        st.info("No decisions yet. Use 👍 / 👎 on the Post first tab.")
    else:
        fdf = pd.DataFrame([{"id": k, "decision": v["decision"], "note": v.get("note", ""), "at": v["at"],
                             "segments": " + ".join(f"Ep{s['ep']} {fmt(s['start'])}–{fmt(s['end'])}" for s in v["segments"]),
                             "source": v.get("source")} for k, v in fb.items()])
        a, r = (fdf["decision"] == "approve").sum(), (fdf["decision"] == "reject").sum()
        m1, m2, m3 = st.columns(3)
        m1.metric("Approved", int(a))
        m2.metric("Rejected", int(r))
        m3.metric("Accept rate", f"{a / max(1, a + r):.0%}")
        st.dataframe(fdf, width="stretch", hide_index=True)
        st.download_button("Download feedback", feedback_path(run_dir).read_text(), "editor_feedback.jsonl")
        st.caption("Saved in the run folder (editor_feedback.jsonl). These decisions are the labels for learning "
                   "the fusion weights and for growing the test set.")

# ---- How it works
with tab_how:
    st.markdown("""
**Pipeline (v3).** Code parses each script into timestamped beats. Several independent engines each rank
every clean 20–60 s clip:

| Engine | What it looks for | Cost |
|---|---|---|
| Rules | hook in the first 3 s, cliffhanger at the cut, intensity | free |
| Curiosity optimizer | questions the story opens and leaves **unanswered at the cut** (information-gap theory, Loewenstein 1994) | free after annotation |
| LLM picker | a model reads each episode and names the beats to cut | ~10 calls per show |

Their rankings are combined with **Reciprocal Rank Fusion** (Cormack et al., SIGIR 2009). Clips that cover the
same moment pool their votes, and footage repeated across episodes is kept out of the top 10.

**Safety rails.** The model names beat ids, never timestamps. A validator never cuts mid-line or mid-sentence
and keeps every clip 20–60 s. An input check flags templated scripts before anything is spent.

**Cost.** About 20 small-model calls for a new 50-episode show and about 2 for a weekly drop of 5 episodes.
Everything is cached, in files or Redis, so finished work is never paid for twice.
""")
