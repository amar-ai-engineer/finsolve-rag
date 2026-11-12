"""
Keyword (BM25) search and reciprocal rank fusion for hybrid retrieval.

Hybrid retrieval runs two searches and merges their rankings:

- vector search (embeddings): good at meaning and paraphrases,
- BM25 (keyword statistics): good at exact terms such as "Q1", "Series B"
  or a policy name that the embedding model may blur.

Access control is applied to BOTH searches before they are merged: the BM25
index is only ever built from chunks the role may see, so fusion cannot pull
an unauthorized chunk back in.
"""

import re

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"[a-z0-9]+")

# Standard RRF constant from Cormack et al. (2009). It damps the influence of
# rank 1 vs rank 2, so one list cannot dominate the fused result on its own.
RRF_K = 60


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens. Deliberately simple: no stemming, no stopwords."""
    return _TOKEN.findall(text.lower())


def _allowed_chunks(vectorstore, allowed_departments: list[str]) -> tuple[list[str], list[dict]]:
    """Fetch the text and metadata of every chunk the role may see.

    Uses the store's metadata filter, like the vector search does. If the
    store rejects filters, fetch everything and filter in Python, so the BM25
    index still never contains a forbidden chunk.
    """
    include = ["documents", "metadatas"]
    try:
        data = vectorstore.get(where={"department": {"$in": allowed_departments}}, include=include)
        return data["documents"], data["metadatas"]
    except Exception:
        data = vectorstore.get(include=include)
        pairs = [
            (text, meta) for text, meta in zip(data["documents"], data["metadatas"], strict=True)
            if meta.get("department") in allowed_departments
        ]
        return [p[0] for p in pairs], [p[1] for p in pairs]


def bm25_search(query: str, allowed_departments: list[str], vectorstore, k: int) -> list[Document]:
    """Top-k chunks by BM25 score, searched only among the allowed departments.

    The index is rebuilt on every call. That is fine for this demo corpus
    (about a hundred chunks); a large corpus would need a cached index per
    role or a search engine with BM25 built in.
    """
    query_tokens = tokenize(query)
    texts, metadatas = _allowed_chunks(vectorstore, allowed_departments)
    if not query_tokens or not texts:
        return []

    scores = BM25Okapi([tokenize(t) for t in texts]).get_scores(query_tokens)
    # Sort by score, then by (source, chunk_index) so ties are deterministic.
    ranked = sorted(
        range(len(texts)),
        key=lambda i: (-scores[i], metadatas[i].get("source", ""), metadatas[i].get("chunk_index", 0)),
    )
    # A score of 0 means no query word appears in the chunk: not a keyword match.
    return [Document(page_content=texts[i], metadata=metadatas[i]) for i in ranked[:k] if scores[i] > 0]


def _chunk_key(doc: Document) -> tuple:
    """Identify the same chunk across the two result lists."""
    return (doc.metadata.get("source"), doc.metadata.get("chunk_index"), doc.page_content)


def reciprocal_rank_fusion(ranked_lists: list[list[Document]], rrf_k: int = RRF_K) -> list[Document]:
    """Merge ranked lists: each chunk scores sum(1 / (rrf_k + rank)) over the lists it appears in.

    Only ranks are used, not raw scores, so cosine distances and BM25 scores
    (which are on different scales) never have to be compared directly.
    """
    scores: dict[tuple, float] = {}
    docs: dict[tuple, Document] = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, 1):
            key = _chunk_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            docs.setdefault(key, doc)
    # sorted() is stable: equal scores keep first-seen order (the first list wins ties).
    return [docs[key] for key in sorted(scores, key=lambda key: -scores[key])]
