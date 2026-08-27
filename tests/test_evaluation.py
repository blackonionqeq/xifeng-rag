import pytest

from xifeng_rag.evaluation import (
    evidence_coverage_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_retrieval_metrics() -> None:
    ranked = ["a", "b", "c"]
    relevant = {"b", "c"}
    assert precision_at_k(ranked, relevant, 2) == 0.5
    assert recall_at_k(ranked, relevant, 2) == 0.5
    assert reciprocal_rank(ranked, relevant) == 0.5
    assert ndcg_at_k(ranked, relevant, 3) == pytest.approx(
        (1 / 1.584962500721156 + 1 / 2) / (1 + 1 / 1.584962500721156)
    )


def test_abstention_recall_is_one_for_no_relevant_documents() -> None:
    assert recall_at_k(["a"], set(), 5) == 1.0
    assert ndcg_at_k(["a"], set(), 5) == 1.0


def test_evidence_coverage_uses_stable_units() -> None:
    evidence = [("version-a", 2, 4), ("version-a", 8, 8)]
    retrieved = [("version-a", 1, 2), ("version-b", 2, 8), ("version-a", 4, 6)]
    assert evidence_coverage_at_k(retrieved, evidence, 2) == 0.25
    assert evidence_coverage_at_k(retrieved, evidence, 3) == 0.5
