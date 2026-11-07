"""Model-agnostic LLM client. Swap models with one string:

    heuristic                      no LLM at all (offline baseline)
    gemini:gemini-2.5-flash        Google AI Studio free tier  (GEMINI_API_KEY)
    groq:llama-3.3-70b-versatile   Groq free tier              (GROQ_API_KEY)
    openrouter:<model>             OpenRouter                  (OPENROUTER_API_KEY)
    openai:gpt-4.1-mini            OpenAI                      (OPENAI_API_KEY)
    anthropic:<model>              Anthropic Messages API      (ANTHROPIC_API_KEY)
    ollama:llama3.1                local Ollama, no key
    hf:Qwen/Qwen2.5-72B-Instruct   Hugging Face Inference Providers (HF_TOKEN), any hosted open model
    compat:<model>                 any OpenAI-compatible URL   (LLM_BASE_URL, LLM_API_KEY)

Only `requests` is needed. Responses are cached on disk by prompt hash, so
re-running a show costs nothing and is reproducible.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

import requests

from .cache import llm_ttl

PRESETS = {
    # provider: (base_url, key_env, default_model)
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY", "gemini-3.5-flash-lite"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "llama-3.3-70b-versatile"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "meta-llama/llama-3.3-70b-instruct:free"),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-4.1-mini"),
    "ollama": ("http://localhost:11434/v1", "", "llama3.1"),
    # Hugging Face Inference Providers (OpenAI-compatible router over many open models)
    "hf": ("https://router.huggingface.co/v1", "HF_TOKEN", "meta-llama/Llama-3.3-70B-Instruct"),
    "compat": (os.environ.get("LLM_BASE_URL", ""), "LLM_API_KEY", "default"),
    "anthropic": ("https://api.anthropic.com/v1", "ANTHROPIC_API_KEY", "claude-sonnet-5-5"),
}


class LLMError(RuntimeError):
    pass


class LLM:
    def __init__(self, spec: str, cache_dir: str | Path | None = None, temperature: float = 0.3,
                 max_tokens: int = 16000, timeout: int = 180, require_key: bool = True):
        provider, _, model = spec.partition(":")
        provider = provider.strip().lower()
        if provider not in PRESETS:
            raise LLMError(f"unknown provider {provider!r}; choose from {', '.join(PRESETS)}")
        base, key_env, default_model = PRESETS[provider]
        self.provider, self.model = provider, (model or default_model)
        self.base = os.environ.get(f"{provider.upper()}_BASE_URL_OVERRIDE", base).rstrip("/")
        self.key = os.environ.get(key_env, "") if key_env else ""
        if key_env and not self.key and require_key:
            raise LLMError(f"{provider} needs the {key_env} environment variable")
        if provider == "compat" and not self.base:
            raise LLMError("compat provider needs LLM_BASE_URL")
        self.temperature, self.max_tokens, self.timeout = temperature, max_tokens, timeout
        from .cache import FileStore, store
        # cache_dir given explicitly -> private file cache (tests); otherwise the shared store (Redis or files)
        self.store = FileStore(Path(cache_dir).parent) if cache_dir else store()
        self.calls = 0
        self.cache_hits = 0
        self.in_chars = 0
        self.out_chars = 0
        self.usage = {"input_tokens": 0, "output_tokens": 0}

    @property
    def name(self) -> str:
        return f"{self.provider}:{self.model}"

    # ------------------------------------------------------------------
    def cache_key(self, system: str, user: str, json_mode: bool = True) -> str:
        key = [self.name, self.temperature, system, user] + ([] if json_mode else ["text"])
        return hashlib.sha256(json.dumps(key).encode()).hexdigest()[:24]

    def cached(self, system: str, user: str, json_mode: bool = True) -> bool:
        """True if this exact request is already answered in the cache (file or Redis)."""
        return self.store.exists("llm", self.cache_key(system, user, json_mode))

    def complete(self, system: str, user: str, json_mode: bool = True) -> str:
        key = self.cache_key(system, user, json_mode)
        hit = self.store.get("llm", key)
        if hit is not None:
            self.cache_hits += 1
            return json.loads(hit)["text"]
        from .cache import rate_limit, rpm_for
        rate_limit(self.name, rpm_for(self.provider))  # shared across workers when the cache is Redis
        text, usage = self._call_with_retry(system, user, json_mode)
        self.calls += 1
        self.in_chars += len(system) + len(user)
        self.out_chars += len(text)
        for k in self.usage:
            self.usage[k] += int(usage.get(k, 0) or 0)
        self.store.set("llm", key, json.dumps({"model": self.name, "text": text, "usage": usage}), ttl=llm_ttl())
        return text

    def complete_json(self, system: str, user: str) -> dict:
        txt = self.complete(system, user)
        try:
            return parse_json(txt)
        except ValueError:
            self.store.delete("llm", self.cache_key(system, user))  # never cache a broken reply
            # one repair attempt: ask the model to fix its own output
            fixed = self.complete(system, user + "\n\nYour previous reply was not valid JSON:\n" + txt[:6000]
                                  + "\n\nReturn ONLY the corrected JSON object.")
            return parse_json(fixed)

    # ------------------------------------------------------------------
    def _call_with_retry(self, system, user, json_mode=True, tries=6):
        delay = 5.0
        for attempt in range(tries):
            try:
                return self._call(system, user, json_mode)
            except LLMError as e:
                msg = str(e)
                retryable = any(c in msg for c in ("429", "500", "502", "503", "504", "timeout", "overloaded"))
                if not retryable or attempt == tries - 1:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 90)
        raise LLMError("unreachable")

    def _call(self, system, user, json_mode=True):
        try:
            if self.provider == "anthropic":
                r = requests.post(
                    f"{self.base}/messages", timeout=self.timeout,
                    headers={"x-api-key": self.key, "anthropic-version": "2023-06-01",
                             "content-type": "application/json"},
                    json={"model": self.model, "max_tokens": self.max_tokens, "temperature": self.temperature,
                          "system": system, "messages": [{"role": "user", "content": user}]})
                if r.status_code != 200:
                    raise LLMError(f"{r.status_code} {r.text[:400]}")
                d = r.json()
                text = "".join(c.get("text", "") for c in d.get("content", []))
                u = d.get("usage", {})
                return text, {"input_tokens": u.get("input_tokens"), "output_tokens": u.get("output_tokens")}
            headers = {"content-type": "application/json"}
            if self.key:
                headers["authorization"] = f"Bearer {self.key}"
            body = {"model": self.model, "temperature": self.temperature, "max_tokens": self.max_tokens,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
            if json_mode and self.provider in ("openai", "groq", "gemini"):
                body["response_format"] = {"type": "json_object"}
            if self.provider == "gemini" and "2.5" in self.model:
                # 2.5 models think by default and thinking tokens count against max_tokens
                body["reasoning_effort"] = os.environ.get("GEMINI_REASONING", "low")
            r = requests.post(f"{self.base}/chat/completions", headers=headers, json=body, timeout=self.timeout)
            if r.status_code != 200:
                raise LLMError(f"{r.status_code} {r.text[:400]}")
            d = r.json()
            ch = d["choices"][0]
            text = ch["message"].get("content") or ""
            # open reasoning models (DeepSeek-R1, Qwen3, ...) may inline their thinking
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
            if ch.get("finish_reason") == "length":
                raise LLMError("output truncated (finish_reason=length); raise max_tokens")
            u = d.get("usage", {}) or {}
            return text, {"input_tokens": u.get("prompt_tokens"), "output_tokens": u.get("completion_tokens")}
        except requests.Timeout as e:
            raise LLMError(f"timeout {e}")
        except requests.RequestException as e:
            raise LLMError(f"network error {e}")


def parse_json(txt: str):
    t = txt.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    a, b = t.find("{"), t.rfind("}")
    if t.lstrip().startswith("["):
        a, b = t.find("["), t.rfind("]")
    if a != -1 and b > a:
        try:
            return json.loads(t[a:b + 1])
        except json.JSONDecodeError:
            pass
    raise ValueError("no JSON object in model output")
