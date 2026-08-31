from __future__ import annotations

from collections.abc import Sequence

from xifeng_rag.domain import (
    EvidenceAssessment,
    EvidenceAssessmentStatus,
    RetrievalCandidate,
)

CALCULATION_VERSION = "evidence-assessment-v1"


def assess_evidence(
    candidates: Sequence[RetrievalCandidate], observation_k: int
) -> EvidenceAssessment:
    """Calculate deterministic signals from the final ranked candidates."""
    try:
        return _assess_evidence(candidates, observation_k)
    except Exception as exc:  # pragma: no cover - defensive isolation boundary
        return EvidenceAssessment(
            status=EvidenceAssessmentStatus.FAILED,
            calculation_version=CALCULATION_VERSION,
            observation_k=max(1, observation_k),
            observed_count=0,
            candidate_count=0,
            article_count_at_k=0,
            diagnostic=f"evidence_assessment_failed:{type(exc).__name__}",
        )


def _assess_evidence(
    candidates: Sequence[RetrievalCandidate], observation_k: int
) -> EvidenceAssessment:
    if observation_k <= 0:
        raise ValueError("observation_k must be positive")

    observed = list(candidates[:observation_k])
    scores = [item.retrieval_scores.reranker for item in observed]
    dual_retrieval = [_is_dual_retrieval(item) for item in observed]
    has_rewrite_trace = any(
        match.query_kind == "rewrite"
        for item in observed
        for match in item.query_matches
    )
    shared_by_query = [_is_shared_by_original_and_rewrite(item) for item in observed]

    return EvidenceAssessment(
        status=EvidenceAssessmentStatus.SUCCESS,
        calculation_version=CALCULATION_VERSION,
        observation_k=observation_k,
        observed_count=len(observed),
        candidate_count=len(candidates),
        article_count_at_k=len({item.article_id for item in observed}),
        top1_reranker_score=scores[0] if scores else None,
        top1_top2_margin=_score_margin(scores, 1),
        top1_top3_margin=_score_margin(scores, 2),
        top1_dual_retrieval=dual_retrieval[0] if dual_retrieval else None,
        dual_retrieval_rate_at_k=(
            sum(dual_retrieval) / len(observed) if observed else None
        ),
        shared_by_original_and_rewrite_rate_at_k=(
            sum(shared_by_query) / len(observed)
            if observed and has_rewrite_trace
            else None
        ),
        top1_shared_by_original_and_rewrite=(
            shared_by_query[0] if observed and has_rewrite_trace else None
        ),
    )


def _score_margin(scores: Sequence[float | None], comparison_index: int) -> float | None:
    if len(scores) <= comparison_index:
        return None
    first = scores[0]
    comparison = scores[comparison_index]
    if first is None or comparison is None:
        return None
    return first - comparison


def _is_dual_retrieval(candidate: RetrievalCandidate) -> bool:
    sources = set(candidate.source)
    return "dense" in sources and "bm25" in sources


def _is_shared_by_original_and_rewrite(candidate: RetrievalCandidate) -> bool:
    kinds = {match.query_kind for match in candidate.query_matches}
    return "original" in kinds and "rewrite" in kinds
