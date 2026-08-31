from __future__ import annotations

import pytest

from xifeng_rag.domain import (
    EvidenceAssessmentStatus,
    QueryMatch,
    RetrievalCandidate,
    RetrievalScores,
)
from xifeng_rag.evidence_assessment import assess_evidence


def _candidate(
    chunk_id: str,
    *,
    article_id: str,
    score: float | None,
    source: list[str],
    query_kinds: list[str] | None = None,
) -> RetrievalCandidate:
    query_matches = [
        QueryMatch(
            query_index=index,
            query_kind=kind,
            query_text=f"query-{kind}",
            rank=1,
            weight=1.0,
            retrieval_scores=RetrievalScores(),
        )
        for index, kind in enumerate(query_kinds or [])
    ]
    return RetrievalCandidate(
        chunk_id=chunk_id,
        article_id=article_id,
        article_version=f"version-{article_id}",
        article_title=f"title-{article_id}",
        original_text="synthetic evidence",
        paragraph_range=(1, 2),
        retrieval_scores=RetrievalScores(reranker=score),
        source=source,
        query_matches=query_matches,
    )


def test_assessment_calculates_rank_source_and_query_agreement_signals() -> None:
    candidates = [
        _candidate(
            "c1",
            article_id="a1",
            score=0.9,
            source=["dense", "bm25"],
            query_kinds=["original", "rewrite"],
        ),
        _candidate(
            "c2",
            article_id="a1",
            score=0.6,
            source=["dense"],
            query_kinds=["original"],
        ),
        _candidate(
            "c3",
            article_id="a2",
            score=0.2,
            source=["dense", "bm25"],
            query_kinds=["rewrite"],
        ),
    ]

    result = assess_evidence(candidates, 3)

    assert result.status == EvidenceAssessmentStatus.SUCCESS
    assert result.calculation_version == "evidence-assessment-v1"
    assert result.candidate_count == 3
    assert result.observed_count == 3
    assert result.article_count_at_k == 2
    assert result.top1_reranker_score == pytest.approx(0.9)
    assert result.top1_top2_margin == pytest.approx(0.3)
    assert result.top1_top3_margin == pytest.approx(0.7)
    assert result.top1_dual_retrieval is True
    assert result.dual_retrieval_rate_at_k == pytest.approx(2 / 3)
    assert result.shared_by_original_and_rewrite_rate_at_k == pytest.approx(1 / 3)
    assert result.top1_shared_by_original_and_rewrite is True


def test_assessment_uses_null_for_unavailable_stages_and_short_lists() -> None:
    result = assess_evidence(
        [_candidate("c1", article_id="a1", score=None, source=["bm25"])], 5
    )

    assert result.status == EvidenceAssessmentStatus.SUCCESS
    assert result.top1_reranker_score is None
    assert result.top1_top2_margin is None
    assert result.top1_top3_margin is None
    assert result.top1_dual_retrieval is False
    assert result.dual_retrieval_rate_at_k == 0.0
    assert result.shared_by_original_and_rewrite_rate_at_k is None
    assert result.top1_shared_by_original_and_rewrite is None


def test_empty_candidates_are_a_successful_observation() -> None:
    result = assess_evidence([], 5)

    assert result.status == EvidenceAssessmentStatus.SUCCESS
    assert result.candidate_count == 0
    assert result.observed_count == 0
    assert result.article_count_at_k == 0
    assert result.top1_dual_retrieval is None
    assert result.dual_retrieval_rate_at_k is None


def test_invalid_observation_depth_is_isolated_without_leaking_input() -> None:
    result = assess_evidence([], 0)

    assert result.status == EvidenceAssessmentStatus.FAILED
    assert result.observation_k == 1
    assert result.diagnostic == "evidence_assessment_failed:ValueError"
