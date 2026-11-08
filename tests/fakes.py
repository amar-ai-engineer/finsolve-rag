"""
Test doubles that let the RBAC tests run offline.

The real embedding model (all-MiniLM-L6-v2) is downloaded from the Hugging
Face Hub, which is not available in every environment (CI sandboxes,
air-gapped machines). Access control does not depend on embedding quality,
only on the metadata filter, so a simple deterministic embedding is enough
to exercise the real ChromaDB filtering code.
"""

import hashlib
import math
import re

from langchain_core.embeddings import Embeddings

_TOKEN = re.compile(r"[a-z0-9]+")


class HashingEmbeddings(Embeddings):
    """Bag-of-words embedding using the hashing trick.

    Each lowercase token is hashed (md5, so results are the same on every
    machine and every run) into one of `dim` buckets with a +1/-1 sign.
    The vector is L2-normalised so Chroma's distance behaves sensibly.
    Texts that share words end up close together, which is all we need.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.md5(token.encode()).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[bucket] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            # Empty or punctuation-only text: return a fixed unit vector
            # instead of all zeros (zero vectors break cosine/L2 maths).
            vec[0] = 1.0
            return vec
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
