"""Embedding helpers with a sentence-transformer path and deterministic fallback."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Iterable, List


TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def _normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


def cosine(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class Embedder:
    """Encode text with Dai/Han's model when available, else hashing.

    The fallback is deterministic and dependency-free. It is not a semantic
    substitute for a trained encoder, but it keeps the generated dataset fully
    reproducible in minimal environments.
    """

    def __init__(
        self,
        model_name: str = "multi-qa-MiniLM-L6-cos-v1",
        backend: str = "auto",
        fallback_dim: int = 384,
    ) -> None:
        self.model_name = model_name
        self.backend_requested = backend
        self.fallback_dim = fallback_dim
        self.backend_used = "hashing"
        self._model = None

        if backend in {"auto", "sentence_transformers"}:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                self._model = SentenceTransformer(model_name)
                self.backend_used = "sentence_transformers"
            except Exception:
                if backend == "sentence_transformers":
                    raise

    def encode(self, texts: Iterable[str]) -> List[List[float]]:
        text_list = list(texts)
        if self.backend_used == "sentence_transformers":
            encoded = self._model.encode(text_list)  # type: ignore[union-attr]
            return [_normalize([float(v) for v in row]) for row in encoded]
        return [self._hashing_embedding(text) for text in text_list]

    def _hashing_embedding(self, text: str) -> List[float]:
        vec = [0.0] * self.fallback_dim
        tokens = TOKEN_RE.findall(text.lower())
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.fallback_dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[bucket] += sign
        return _normalize(vec)

