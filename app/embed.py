"""Sentence-embedding wrapper (ONNX via fastembed, no torch needed)."""
from __future__ import annotations

import os
import threading
from functools import lru_cache

import numpy as np

from .config import settings

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
_lock = threading.Lock()


class Embedder:
    def __init__(self, model_name: str | None = None, cache_dir: str | None = None):
        from fastembed import TextEmbedding

        self.model_name = model_name or settings.embed_model
        self._model = TextEmbedding(self.model_name, cache_dir=cache_dir or str(settings.model_dir))
        self._memo: dict[str, np.ndarray] = {}

    def encode(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        """L2-normalised embeddings, shape (n, d), float32.  Memoised per text."""
        todo = [t for t in dict.fromkeys(texts) if t not in self._memo]
        if todo:
            with _lock:
                vecs = np.asarray(list(self._model.embed(todo, batch_size=batch_size)), dtype=np.float32)
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
            if len(self._memo) > 200_000:
                self._memo.clear()
            for t, v in zip(todo, vecs):
                self._memo[t] = v
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._memo[t] for t in texts])

    @property
    def dim(self) -> int:
        return 384


@lru_cache(maxsize=4)
def get_embedder(model_name: str | None = None) -> Embedder:
    return Embedder(model_name)
