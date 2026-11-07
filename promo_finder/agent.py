"""Promo agent: runs the v3 pipeline by itself, for every show, every week, within a call budget.

Loop (one "tick"):
  1. SCAN     inbox/<show>/*.txt - hash every episode; compare with the agent's memory (state.json)
  2. PLAN     for each new or changed show, compute EXACTLY which LLM calls are still needed by checking
              the caches (per-episode annotation cache + reply cache). Unchanged work costs 0.
  3. BUDGET   compare the plan with today's remaining call budget (e.g. the free tier's daily cap).
              If it does not fit, process the earliest episodes that do fit and defer the rest.
  4. ACT      run v3 (B2 by default) -> outbox/<show>/<timestamp>/ and outbox/<show>/latest/
  5. REPORT   diff the new top-10 against the previous one, append to outbox/<show>/digest.md,
              optionally POST a short message to a webhook (Slack-compatible) from PROMO_WEBHOOK_URL.
  6. REMEMBER save hashes, calls spent, and anything deferred or failed, so the next tick resumes.

Run it from cron / a scheduler (`--once`) or as a long-running process (`--watch SECONDS`).
A new 50-episode show plans ~20 calls; a weekly drop of 5 episodes plans ~2.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

from . import prompts
from .llm import LLM
from .parser import load_show
from .v2 import annotate as A


def now():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def today():
    return dt.date.today().isoformat()


class Agent:
    def __init__(self, inbox, outbox, model="gemini:gemini-3.5-flash-lite", budget_level="B2",
                 daily_budget=40, batch=5, per_episode=3, webhook=None, log=print, skip_templated=False):
        self.inbox, self.outbox = Path(inbox), Path(outbox)
        self.outbox.mkdir(parents=True, exist_ok=True)
        self.model, self.level, self.daily_budget = model, budget_level, daily_budget
        self.batch, self.k = batch, per_episode
        self.skip_templated = skip_templated
        self.worker = os.environ.get("PROMO_WORKER_ID", f"{os.uname().nodename}:{os.getpid()}")
        self.webhook = webhook or os.environ.get("PROMO_WEBHOOK_URL")
        self.state_path = self.outbox / "agent_state.json"  # local mirror for humans; the truth is in the store
        self.state = {"shows": {}, "usage": {}}
        self._log = log
        self.logfile = self.outbox / "agent.log"

    # -------------------------------------------------------------- utils
    def log(self, msg):
        line = f"{now()} [{getattr(self, 'worker', '-')}] {msg}"
        self._log(line)
        with open(self.logfile, "a") as f:
            f.write(line + "\n")

    @staticmethod
    def _skey(name):
        return hashlib.sha256(name.encode()).hexdigest()[:24]

    def load_show_state(self, name):
        from .cache import store
        raw = store().get("state", self._skey(name))
        self.state["shows"][name] = json.loads(raw) if raw else {"hashes": {}, "runs": []}
        return self.state["shows"][name]

    def save(self, name=None):
        from .cache import store
        for n in ([name] if name else list(self.state["shows"])):
            store().set("state", self._skey(n), json.dumps(self.state["shows"][n]))
        self.state_path.write_text(json.dumps(self.state, indent=1))

    def spent_today(self):
        # with Redis this counter is shared by every agent worker using the same cache
        from .cache import store
        return store().counter(f"calls:{today()}")

    # -------------------------------------------------------------- 1. scan
    def scan(self):
        shows = {}
        for d in sorted(p for p in self.inbox.iterdir() if p.is_dir()):
            files = sorted(d.rglob("*.txt"))
            if files:
                shows[d.name] = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()[:16] for f in files}
        return shows

    def changes(self, name, hashes):
        mem = self.load_show_state(name)
        old = mem.get("hashes", {})
        pending = mem.get("deferred", False)
        new = [f for f in hashes if f not in old]
        changed = [f for f in hashes if f in old and old[f] != hashes[f]]
        return new, changed, pending

    # -------------------------------------------------------------- 2. plan
    def plan(self, show_dir: Path, episodes=None):
        """Exact count of LLM calls the run still needs, by probing the caches. No API calls."""
        eps = load_show(show_dir)
        if episodes is not None:
            eps = [e for e in eps if e.num in set(episodes)]
        llm = LLM(self.model, require_key=False)
        todo = [e for e in eps if not A.ep_cached(llm, e)]
        ann_calls = 0
        for i in range(0, len(todo), self.batch):
            g = todo[i:i + self.batch]
            if not llm.cached(A.SYSTEM, A.prompt(g), json_mode=False):
                ann_calls += 1
        pick_calls = 0
        if self.level >= "B2":
            prompts.COMPACT = True
            for i in range(0, len(eps), self.batch):
                g = eps[i:i + self.batch]
                if not llm.cached(prompts.SYSTEM_FIND, prompts.batch_prompt(g, show_dir.name, self.k)):
                    pick_calls += 1
        extra = {"B3": 4, "B4": 9}.get(self.level, 0)
        return {"episodes": len(eps), "episodes_to_annotate": len(todo), "annotation_calls": ann_calls,
                "picker_calls": pick_calls, "judge_and_panel_calls": extra,
                "total_calls": ann_calls + pick_calls + extra, "ep_nums": [e.num for e in eps]}

    # -------------------------------------------------------------- 3. budget
    def fit_to_budget(self, show_dir, plan):
        remaining = self.daily_budget - self.spent_today()
        if plan["total_calls"] <= remaining:
            return plan, None
        # Process the earliest episodes that fit; batches are aligned from episode 1, so later
        # ticks reuse everything already done.
        nums = plan["ep_nums"]
        for n in range(len(nums) - 1, 0, -1):
            if n % self.batch:
                continue  # keep batch boundaries aligned so later ticks hit the cache
            p = self.plan(show_dir, nums[:n])
            if p["total_calls"] <= remaining:
                return p, (f"needs {plan['total_calls']} calls, {remaining} left today - processing episodes "
                           f"{nums[0]}-{nums[n - 1]} now ({p['total_calls']} calls), the rest next tick")
        return None, f"needs {plan['total_calls']} calls, {remaining} left today - deferred to tomorrow"

    # -------------------------------------------------------------- 4/5. act + report
    def run_show(self, name, hashes, dry_run=False):
        show_dir = self.inbox / name
        from .qa import inspect
        qa = inspect(load_show(show_dir))
        self.log(f"[{name}] input check: {qa['verdict']}" + "".join(f"\n    - {w}" for w in qa["warnings"]))
        if qa["verdict"] == "templated" and self.skip_templated:
            self.log(f"[{name}] skipped: templated input (use without --skip-templated to run anyway)")
            return None
        plan = self.plan(show_dir)
        self.log(f"[{name}] plan: {plan['episodes']} episodes, {plan['episodes_to_annotate']} to annotate -> "
                 f"{plan['annotation_calls']} annotation + {plan['picker_calls']} picker + "
                 f"{plan['judge_and_panel_calls']} judge/panel = {plan['total_calls']} calls "
                 f"(budget left today: {self.daily_budget - self.spent_today()})")
        todo, note = self.fit_to_budget(show_dir, plan)
        if note:
            self.log(f"[{name}] budget: {note}")
        if dry_run:
            return {"plan": plan, "fitted": todo, "note": note}
        mem = self.state["shows"].get(name) or self.load_show_state(name)
        if todo is None:
            mem["deferred"] = True
            self.save(name)
            return None
        from .v3.pipeline import run as run_v3
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        out = self.outbox / name / stamp
        partial = todo["episodes"] < plan["episodes"]
        try:
            meta = run_v3(str(show_dir), str(out), self.model, self.level, batch=self.batch, per_episode=self.k,
                          episodes=todo["ep_nums"] if partial else None)
        except Exception as e:  # never crash the loop; retry next tick
            self.log(f"[{name}] run FAILED: {e}")
            mem["deferred"] = True
            self.save(name)
            return None
        calls = sum(int(v.get("calls", 0)) for v in meta["cost_by_stage"].values())
        from .cache import store
        store().incr(f"calls:{today()}", calls, ttl=3 * 86400)
        self.state["usage"][today()] = self.state["usage"].get(today(), 0) + calls
        failed = bool(meta["errors"])
        prev = self.outbox / name / "latest"
        diff = self.diff(prev / "promos.json", out / "promos.json")
        if prev.exists():
            shutil.rmtree(prev)
        shutil.copytree(out, prev)
        mem["hashes"] = hashes if not (partial or failed) else {**mem["hashes"],
                                                                **{f: h for f, h in hashes.items() if f in mem["hashes"]}}
        mem["deferred"] = partial or failed
        mem["runs"].append({"at": now(), "out": str(out), "calls": calls, "episodes": todo["episodes"],
                            "partial": partial, "errors": len(meta["errors"]), "worker": self.worker})
        self.save(name)
        self.report(name, out, meta, calls, plan, diff, partial)
        return meta

    def diff(self, old_path: Path, new_path: Path):
        def top(p):
            if not p.exists():
                return []
            d = json.loads(p.read_text())
            ids = d.get("post_first", [])
            by = {x["id"]: x for x in d.get("promos", [])}
            return [by[i] for i in ids if i in by]

        def key(p):
            s = max(p["segments"], key=lambda x: x["end"] - x["start"])
            return s["ep"], s["start"], s["end"]

        def same(a, b):
            ea, sa, xa = key(a)
            eb, sb, xb = key(b)
            inter = max(0.0, min(xa, xb) - max(sa, sb))
            return ea == eb and inter / max(1e-6, min(xa - sa, xb - sb)) >= 0.5

        old, new = top(old_path), top(new_path)
        entered = [p for p in new if not any(same(p, q) for q in old)]
        left = [q for q in old if not any(same(q, p) for p in new)]
        return {"first_run": not old, "entered": entered, "left": left, "top": new}

    def report(self, name, out, meta, calls, plan, diff, partial):
        seg = lambda p: " + ".join(f"Ep{s['ep']} {s['start_ts']}-{s['end_ts']}" for s in p["segments"])
        qa = meta.get("input_check", {})
        L = [f"## {now()} - {name}",
             f"- input check: **{qa.get('verdict')}**" + "".join(f"; {w}" for w in qa.get("warnings", [])),
             f"- episodes: {meta['episodes']}{' (partial, rest deferred)' if partial else ''}; "
             f"LLM calls made: {calls} (planned {plan['total_calls']}); errors: {len(meta['errors'])}",
             f"- output: `{out}`"]
        if diff["first_run"]:
            L.append("- first run for this show - top picks:")
            L += [f"  {i}. {seg(p)} - {str(p.get('hook', ''))[:90]}" for i, p in enumerate(diff["top"][:5], 1)]
        elif diff["entered"]:
            L.append(f"- {len(diff['entered'])} new promo(s) entered the top 10:")
            L += [f"  - {seg(p)} - {str(p.get('hook', ''))[:90]}" for p in diff["entered"]]
        else:
            L.append("- top 10 unchanged")
        text = "\n".join(L) + "\n\n"
        with open(self.outbox / name / "digest.md", "a") as f:
            f.write(text)
        self.log(f"[{name}] done: {calls} calls; " + (f"{len(diff['entered'])} new in top 10" if not diff["first_run"] else "first run"))
        if self.webhook:
            try:
                import requests
                requests.post(self.webhook, json={"text": text[:3500]}, timeout=10)
            except Exception as e:
                self.log(f"[{name}] webhook failed: {e}")

    # -------------------------------------------------------------- tick / loop
    def tick(self, dry_run=False, force=False):
        from .cache import store
        st = store()
        self.log(f"tick (cache: {st.kind}, worker {self.worker})")
        shows = self.scan()
        if not shows:
            self.log(f"inbox {self.inbox} is empty")
        for name, hashes in shows.items():
            # One lock per show (Redis SET NX EX): N workers split the shows, never double-process one.
            lock = f"agent:show:{self._skey(name)}"
            if not st.acquire(lock, 6 * 3600):
                self.log(f"[{name}] being processed by another worker - skipped")
                continue
            try:
                new, changed, pending = self.changes(name, hashes)
                if not (new or changed or pending or force):
                    self.log(f"[{name}] no new or changed episodes - 0 calls")
                    continue
                self.log(f"[{name}] {len(new)} new, {len(changed)} changed episode files"
                         + (", resuming deferred work" if pending else ""))
                self.run_show(name, hashes, dry_run=dry_run)
            finally:
                st.release(lock)


def main():
    ap = argparse.ArgumentParser(prog="promo_finder.agent", description="Autonomous promo agent.")
    ap.add_argument("--inbox", default="inbox", help="one sub-folder per show with episode .txt files")
    ap.add_argument("--outbox", default="promo_runs")
    ap.add_argument("-m", "--model", default="gemini:gemini-3.5-flash-lite")
    ap.add_argument("-b", "--budget-level", default="B2", choices=["B0", "B1", "B2", "B3", "B4"])
    ap.add_argument("--daily-budget", type=int, default=40, help="max LLM calls per day across all shows")
    ap.add_argument("--batch", type=int, default=5)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--once", action="store_true", help="one tick then exit (for cron)")
    g.add_argument("--watch", type=int, metavar="SECONDS", help="tick forever every N seconds")
    ap.add_argument("--dry-run", action="store_true", help="plan and print cost only; no API calls")
    ap.add_argument("--force", action="store_true", help="re-run shows even if nothing changed")
    ap.add_argument("--skip-templated", action="store_true", help="do not spend calls on shows the input check flags as templated")
    a = ap.parse_args()
    agent = Agent(a.inbox, a.outbox, a.model, a.budget_level, a.daily_budget, a.batch, skip_templated=a.skip_templated)
    if a.watch:
        agent.log(f"watching {a.inbox} every {a.watch}s (daily budget {a.daily_budget} calls)")
        while True:
            agent.tick(a.dry_run, a.force)
            time.sleep(a.watch)
    else:
        agent.tick(a.dry_run, a.force)


if __name__ == "__main__":
    sys.exit(main())
