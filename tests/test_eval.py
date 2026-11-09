"""
Tests for the retrieval evaluation (eval/run_eval.py).

These check that the golden set is well-formed and grounded in the real
documents, and that the metric code computes what it claims to. They run
offline with the fake hashing embeddings; the real-model numbers come from
the Eval workflow in GitHub Actions.
"""

import os

import pytest

from eval.run_eval import evaluate, first_rank, load_golden, summarize
from src.auth import ROLES, get_allowed_departments
from src.document_loader import detect_department
from tests.conftest import DOCS_DIR

GOLDEN = load_golden()
DOC_FILES = set(os.listdir(DOCS_DIR))


def test_golden_set_size_and_unique_ids():
    assert len(GOLDEN) == 30
    assert len({item["id"] for item in GOLDEN}) == len(GOLDEN)


def test_every_role_is_covered():
    assert {item["role"] for item in GOLDEN} == set(ROLES)


@pytest.mark.parametrize("item", [g for g in GOLDEN if g["expected_source"]], ids=lambda g: g["id"])
def test_answerable_question_points_to_a_document_the_role_can_see(item):
    assert item["expected_source"] in DOC_FILES
    assert detect_department(item["expected_source"]) in get_allowed_departments(item["role"])


@pytest.mark.parametrize("item", [g for g in GOLDEN if not g["expected_source"]], ids=lambda g: g["id"])
def test_denied_question_only_has_answers_the_role_cannot_see(item):
    assert item["forbidden_sources"], "say where the answer lives, so a reviewer can check it"
    for source in item["forbidden_sources"]:
        assert source in DOC_FILES
        assert detect_department(source) not in get_allowed_departments(item["role"])


def test_first_rank():
    assert first_rank(["a.txt", "b.txt", "b.txt"], "b.txt") == 2
    assert first_rank(["a.txt"], "b.txt") is None


def test_summary_metrics_on_hand_made_rows():
    rows = [
        {"role": "hr", "expected_source": "x", "rank": 1, "leaked_sources": []},
        {"role": "hr", "expected_source": "x", "rank": 2, "leaked_sources": []},
        {"role": "hr", "expected_source": "x", "rank": None, "leaked_sources": []},
        {"role": "hr", "expected_source": None, "rank": None, "leaked_sources": ["secret.txt"]},
    ]
    s = summarize(rows)["overall"]
    assert s["answerable"] == 3
    assert s["hit@1"] == pytest.approx(1 / 3)
    assert s["hit@5"] == pytest.approx(2 / 3)
    assert s["mrr"] == pytest.approx((1 + 0.5 + 0) / 3)
    assert s["leak_rate"] == pytest.approx(1 / 4)


def test_eval_end_to_end_never_leaks(vectorstore):
    """The full golden set through real Chroma filtering: zero unauthorized chunks."""
    rows = evaluate(GOLDEN, vectorstore)
    assert len(rows) == len(GOLDEN)
    assert summarize(rows)["overall"]["leak_rate"] == 0
