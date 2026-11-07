"""Minimal OpenAI-compatible chat client (defaults to Gemini) with a persistent cache.

* temperature 0 and a fixed JSON schema for extraction -> repeatable output
* every (model, prompt) pair is cached in SQLite, so JD extraction runs once
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

import httpx

from .config import settings


class LLMError(RuntimeError):
    pass


class _Cache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT)")
        self._lock = threading.Lock()

    def get(self, k: str) -> str | None:
        with self._lock:
            row = self._db.execute("SELECT v FROM cache WHERE k=?", (k,)).fetchone()
        return row[0] if row else None

    def put(self, k: str, v: str) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO cache VALUES (?,?)", (k, v))
            self._db.commit()


class LLMClient:
    def __init__(self, base_url: str | None = None, model: str | None = None, api_key: str | None = None,
                 cache_path: Path | None = None):
        self.base_url = (base_url or settings.llm_base_url).rstrip("/")
        self.model = model or settings.llm_model
        self.api_key = api_key if api_key is not None else settings.llm_api_key
        self.cache = _Cache(cache_path or settings.cache_path)
        self._http = httpx.Client(timeout=settings.llm_timeout)

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def chat(self, messages: list[dict], *, temperature: float = 0.0, json_schema: dict | None = None,
             max_tokens: int = 2048, use_cache: bool = True, retries: int = 4) -> str:
        body: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": temperature,
                                "max_tokens": max_tokens}
        if json_schema is not None:
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "output", "schema": json_schema, "strict": True}}
        key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        if use_cache:
            hit = self.cache.get(key)
            if hit is not None:
                return hit
        if not self.available:
            raise LLMError("No LLM API key configured (set GEMINI_API_KEY or LLM_API_KEY)")
        delay = 2.0
        for attempt in range(retries + 1):
            try:
                r = self._http.post(f"{self.base_url}/chat/completions", json=body,
                                    headers={"Authorization": f"Bearer {self.api_key}"})
            except httpx.HTTPError as e:
                err = str(e)
            else:
                if r.status_code == 200:
                    content = r.json()["choices"][0]["message"].get("content") or ""
                    if use_cache:
                        self.cache.put(key, content)
                    return content
                err = f"HTTP {r.status_code}: {r.text[:300]}"
                if r.status_code not in (429, 500, 502, 503, 504):
                    raise LLMError(err)
            if attempt < retries:
                time.sleep(delay)
                delay *= 2
        raise LLMError(err)

    def json(self, messages: list[dict], schema: dict, **kw) -> dict:
        text = self.chat(messages, json_schema=schema, **kw)
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text[text.find("{"):]
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            s, e2 = text.find("{"), text.rfind("}")
            if s >= 0 and e2 > s:
                return json.loads(text[s:e2 + 1])
            raise LLMError(f"Invalid JSON from LLM: {e}") from e


_client: LLMClient | None = None


def get_llm() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
