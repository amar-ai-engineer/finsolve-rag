"""
Tests for hybrid retrieval (BM25 + vector, reciprocal rank fusion).

The leak tests in tests/test_rbac.py already run in both modes. These tests
cover the parts that are specific to hybrid search: the BM25 index only ever
contains allowed chunks, the fusion maths, and the mode switch.
"""

import pytest
from langchain_core.documents import Document

from src.auth import get_allowed_departments
from src.hybrid import RRF_K, bm25_search, reciprocal_rank_fusion, tokenize
from src.rag_chain import get_retrieval_mode, retrieve_with_rbac
from tests.test_rbac import ALL_DEPARTMENTS, CROSS_DEPARTMENT_QUERIES, EXPECTED_ACCESS, INJECTION_QUERIES


def doc(source, chunk_index=0, department="general"):
    return Document(
        page_content=f"{source}#{chunk_index}",
        metadata={"source": source, "chunk_index": chunk_index, "department": department},
    )


def departments_of(docs):
    return {d.metadata["department"] for d in docs}


# ---------- BM25 respects the role filter ----------


def test_tokenize():
    assert tokenize("Q1 2026 budget, Series-B!") == ["q1", "2026", "budget", "series", "b"]


def test_bm25_finds_board_minutes_only_when_allowed(vectorstore):
    """Without the filter BM25 would return the board minutes; with it, it cannot."""
    query = "board meeting minutes"
    executive = bm25_search(query, get_allowed_departments("executive"), vectorstore, k=5)
    assert executive[0].metadata["source"] == "executive_board_minutes.txt"

    employee = bm25_search(query, get_allowed_departments("employee"), vectorstore, k=20)
    assert departments_of(employee) <= {"general"}


@pytest.mark.parametrize("role", sorted(EXPECTED_ACCESS) + ["unknown"])
@pytest.mark.parametrize("query", CROSS_DEPARTMENT_QUERIES + INJECTION_QUERIES)
def test_bm25_never_leaks(vectorstore, role, query):
    docs = bm25_search(query, get_allowed_departments(role), vectorstore, k=50)
    assert departments_of(docs) <= EXPECTED_ACCESS.get(role, {"general"})


def test_bm25_returns_nothing_for_queries_without_matching_words(vectorstore):
    assert bm25_search("", ["general"], vectorstore, k=5) == []
    assert bm25_search("zzzqqq xxyyzz", ["general"], vectorstore, k=5) == []


class NoFilterStore:
    """A store that rejects metadata filters in both search and get.

    Used to check that the hybrid path's Python fallbacks also never leak.
    """

    def __init__(self, departments):
        self.docs = [
            Document(
                page_content=f"policy document number {i}",
                metadata={"department": dept, "source": f"{dept}_{i}.txt", "chunk_index": 0},
            )
            for i, dept in enumerate(departments)
        ]

    def similarity_search(self, query, k=4, filter=None):
        if filter is not None:
            raise ValueError("metadata filters are not supported")
        return self.docs[:k]

    def get(self, where=None, include=None):
        if where is not None:
            raise ValueError("metadata filters are not supported")
        return {"documents": [d.page_content for d in self.docs], "metadatas": [d.metadata for d in self.docs]}


@pytest.mark.parametrize("role", sorted(EXPECTED_ACCESS) + ["unknown"])
def test_hybrid_fallback_never_leaks(role):
    store = NoFilterStore(sorted(ALL_DEPARTMENTS) * 5)
    docs = retrieve_with_rbac("policy document", role, store, k=5, mode="hybrid")
    assert docs, "the fallback should still return allowed chunks"
    assert departments_of(docs) <= EXPECTED_ACCESS.get(role, {"general"})
    assert len(docs) <= 5


# ---------- Reciprocal rank fusion ----------


def test_rrf_rewards_chunks_found_by_both_searches():
    a, b, c = doc("a.txt"), doc("b.txt"), doc("c.txt")
    # b is 2nd in both lists; a and c are 1st in one list only.
    fused = reciprocal_rank_fusion([[a, b], [c, b]])
    assert fused[0].metadata["source"] == "b.txt"
    assert [d.metadata["source"] for d in fused] == ["b.txt", "a.txt", "c.txt"]


def test_rrf_scores_follow_the_formula():
    # A chunk ranked 1st in one list and 3rd in the other beats one ranked 2nd in both
    # only if 1/(k+1) + 1/(k+3) > 2/(k+2). With k=60 that holds, by a small margin.
    x, y, filler = doc("x.txt"), doc("y.txt"), doc("f.txt")
    fused = reciprocal_rank_fusion([[x, y, filler], [filler, y, x]])
    assert 1 / (RRF_K + 1) + 1 / (RRF_K + 3) > 2 / (RRF_K + 2)
    # x and filler have the same score (1st + 3rd); y is 2nd + 2nd.
    assert [d.metadata["source"] for d in fused] == ["x.txt", "f.txt", "y.txt"]


def test_rrf_treats_same_chunk_as_one_result():
    first, second = doc("a.txt", 0), doc("a.txt", 1)
    fused = reciprocal_rank_fusion([[first, second], [doc("a.txt", 0)]])
    assert len(fused) == 2
    assert fused[0].metadata["chunk_index"] == 0


def test_rrf_with_empty_lists():
    assert reciprocal_rank_fusion([[], []]) == []
    only = [doc("a.txt"), doc("b.txt")]
    assert reciprocal_rank_fusion([only, []]) == only


# ---------- Mode switch ----------


def test_default_mode_is_vector(monkeypatch):
    monkeypatch.delenv("FINSOLVE_RETRIEVAL_MODE", raising=False)
    assert get_retrieval_mode() == "vector"


def test_mode_from_environment(monkeypatch):
    monkeypatch.setenv("FINSOLVE_RETRIEVAL_MODE", " Hybrid ")
    assert get_retrieval_mode() == "hybrid"


def test_invalid_mode_fails_loudly(monkeypatch, vectorstore):
    monkeypatch.setenv("FINSOLVE_RETRIEVAL_MODE", "hybird")
    with pytest.raises(ValueError, match="FINSOLVE_RETRIEVAL_MODE"):
        retrieve_with_rbac("budget", "finance", vectorstore)
    with pytest.raises(ValueError, match="unknown retrieval mode"):
        retrieve_with_rbac("budget", "finance", vectorstore, mode="keyword")


def test_hybrid_returns_at_most_k(vectorstore):
    for k in (1, 5, 30):
        assert len(retrieve_with_rbac("annual leave policy", "hr", vectorstore, k=k, mode="hybrid")) <= k
