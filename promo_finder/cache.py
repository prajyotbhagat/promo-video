"""Cache layer: Redis when PROMO_CACHE_URL is set, local files otherwise.

Namespaces:
  llm    model replies, keyed by sha256(model, temperature, system prompt, user prompt[, mode])
  annot  per-episode annotations, keyed by sha256(model, instructions, episode text)

Why Redis: several agent workers (or a team) share one cache, so no episode is ever paid for twice;
replies can expire (PROMO_CACHE_TTL_DAYS); and the agent's lock and daily call budget become atomic
across machines (SET NX / INCRBY) instead of living in local files.

If Redis is configured but unreachable, the layer logs a warning and falls back to files, so a cache
outage never stops a run (it may only cost extra calls).

  PROMO_CACHE_URL=redis://localhost:6379/0     use Redis
  PROMO_CACHE_TTL_DAYS=90                      expire LLM replies (annotations never expire)
  PROMO_CACHE_DIR=.cache                       file backend location
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PREFIX = "promo"


class FileStore:
    kind = "file"

    def __init__(self, root=None):
        self.root = Path(root or os.environ.get("PROMO_CACHE_DIR", ".cache"))

    def _p(self, ns, key):
        return self.root / ns / f"{key}.json"

    def get(self, ns, key):
        p = self._p(ns, key)
        return p.read_text() if p.exists() else None

    def set(self, ns, key, value, ttl=None):
        p = self._p(ns, key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(value)

    def exists(self, ns, key):
        return self._p(ns, key).exists()

    def delete(self, ns, key):
        self._p(ns, key).unlink(missing_ok=True)

    def keys(self, ns):
        d = self.root / ns
        return [p.stem for p in d.glob("*.json")] if d.exists() else []

    # coordination primitives (single machine)
    def acquire(self, name, ttl):
        import time
        p = self.root / f".{name}.lock"
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists() and time.time() - p.stat().st_mtime < ttl:
            return False
        p.write_text(str(os.getpid()))
        return True

    def release(self, name):
        (self.root / f".{name}.lock").unlink(missing_ok=True)

    def incr(self, name, n, ttl):
        p = self.root / f".{name}.count"
        p.parent.mkdir(parents=True, exist_ok=True)
        v = int(p.read_text()) if p.exists() else 0
        p.write_text(str(v + n))
        return v + n

    def counter(self, name):
        p = self.root / f".{name}.count"
        return int(p.read_text()) if p.exists() else 0


class RedisStore:
    kind = "redis"

    def __init__(self, url):
        import redis
        self.r = redis.Redis.from_url(url, decode_responses=True, socket_timeout=5, socket_connect_timeout=3)
        self.r.ping()
        self.url = url

    def _k(self, ns, key):
        return f"{PREFIX}:{ns}:{key}"

    def get(self, ns, key):
        return self.r.get(self._k(ns, key))

    def set(self, ns, key, value, ttl=None):
        self.r.set(self._k(ns, key), value, ex=ttl)

    def exists(self, ns, key):
        return bool(self.r.exists(self._k(ns, key)))

    def delete(self, ns, key):
        self.r.delete(self._k(ns, key))

    def keys(self, ns):
        n = len(f"{PREFIX}:{ns}:")
        return [k[n:] for k in self.r.scan_iter(f"{PREFIX}:{ns}:*", count=500)]

    def acquire(self, name, ttl):
        return bool(self.r.set(f"{PREFIX}:lock:{name}", str(os.getpid()), nx=True, ex=int(ttl)))

    def release(self, name):
        self.r.delete(f"{PREFIX}:lock:{name}")

    def incr(self, name, n, ttl):
        k = f"{PREFIX}:count:{name}"
        with self.r.pipeline() as p:
            p.incrby(k, n)
            p.expire(k, int(ttl))
            return int(p.execute()[0])

    def counter(self, name):
        v = self.r.get(f"{PREFIX}:count:{name}")
        return int(v) if v else 0


_STORE = None


def store():
    """Process-wide cache store, chosen from the environment once."""
    global _STORE
    if _STORE is None:
        url = os.environ.get("PROMO_CACHE_URL", "").strip()
        if url.startswith(("redis://", "rediss://", "unix://")):
            try:
                _STORE = RedisStore(url)
            except Exception as e:
                print(f"[cache] Redis at {url} unavailable ({e.__class__.__name__}: {e}); falling back to files",
                      file=sys.stderr)
                _STORE = FileStore()
        else:
            _STORE = FileStore()
    return _STORE


def reset():
    global _STORE
    _STORE = None


def llm_ttl():
    d = os.environ.get("PROMO_CACHE_TTL_DAYS")
    return int(float(d) * 86400) if d else None


def migrate(src_dir=".cache"):
    """Copy an existing file cache into the configured store (e.g. Redis). Idempotent."""
    dst = store()
    src = FileStore(src_dir)
    n = 0
    for ns in ("llm", "annot"):
        for key in src.keys(ns):
            if not dst.exists(ns, key):
                dst.set(ns, key, src.get(ns, key), ttl=llm_ttl() if ns == "llm" else None)
                n += 1
    return n


def stats():
    s = store()
    return {"backend": s.kind, **{ns: len(s.keys(ns)) for ns in ("llm", "annot")}}


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser(prog="promo_finder.cache")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("migrate", help="copy the file cache into the store named by PROMO_CACHE_URL")
    m.add_argument("--from", dest="src", default=".cache")
    sub.add_parser("stats")
    a = ap.parse_args()
    if a.cmd == "migrate":
        print(f"copied {migrate(a.src)} entries into {store().kind}")
    print(json.dumps(stats()))


# ---------------------------------------------------------------- shared rate limit
_local_windows: dict = {}


def rate_limit(model: str, rpm: int | None):
    """Block until a request to `model` fits in its requests-per-minute budget.
    With Redis the window is shared by every worker on every machine (fixed 60 s window, INCR+EXPIRE);
    with files it is per process. rpm=None or 0 means no limit."""
    import time
    if not rpm:
        return
    s = store()
    while True:
        minute = int(time.time() // 60)
        if s.kind == "redis":
            k = f"{PREFIX}:rpm:{model}:{minute}"
            with s.r.pipeline() as p:
                p.incr(k)
                p.expire(k, 120)
                n = int(p.execute()[0])
        else:
            key = (model, minute)
            _local_windows[key] = _local_windows.get(key, 0) + 1
            n = _local_windows[key]
        if n <= rpm:
            return
        time.sleep(60 - time.time() % 60 + 0.05)


def rpm_for(provider: str):
    v = os.environ.get(f"PROMO_RPM_{provider.upper()}") or os.environ.get("PROMO_RPM")
    return int(v) if v else None
