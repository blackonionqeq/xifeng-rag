from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from xifeng_rag.domain import (
    AssemblyErrorCode,
    AssemblyExclusionReason,
    AssemblyResult,
    AssemblyStatus,
    AssemblyUsage,
    CompressionMethod,
    ContextAssemblyConfigSnapshot,
    EvidenceItem,
    ExcludedCandidate,
    RetrievalCandidate,
)

EMPTY_CONTEXT = '{"evidence":[]}'


class ContextAssemblyError(ValueError):
    def __init__(self, code: AssemblyErrorCode, detail: str) -> None:
        self.code = code
        super().__init__(f"{code.value}: {detail}")


def _error(code: AssemblyErrorCode, detail: str) -> ContextAssemblyError:
    return ContextAssemblyError(code, detail)


def _validate_config(config: ContextAssemblyConfigSnapshot) -> None:
    if config.max_evidence_items <= 0:
        raise _error(AssemblyErrorCode.INVALID_CONFIG, "max_evidence_items 必须大于 0")
    if config.max_total_chars < len(EMPTY_CONTEXT):
        raise _error(
            AssemblyErrorCode.INVALID_CONFIG,
            f"max_total_chars 必须至少为 {len(EMPTY_CONTEXT)}",
        )


def _validate_candidate(candidate: RetrievalCandidate, rank: int) -> None:
    required = {
        "chunk_id": candidate.chunk_id,
        "article_id": candidate.article_id,
        "article_version": candidate.article_version,
        "article_title": candidate.article_title,
        "original_text": candidate.original_text,
    }
    missing = [name for name, value in required.items() if not value.strip()]
    start, end = candidate.paragraph_range
    if missing or start < 1 or end < start or not candidate.source:
        chunk_label = candidate.chunk_id.strip() or f"rank-{rank}"
        raise _error(
            AssemblyErrorCode.INVALID_CANDIDATE,
            f"候选 {chunk_label}（rank={rank}）结构非法",
        )


def _evidence(candidate: RetrievalCandidate, input_rank: int, citation_number: int) -> EvidenceItem:
    return EvidenceItem(
        citation_id=f"E{citation_number}",
        source_chunk_id=candidate.chunk_id,
        article_id=candidate.article_id,
        article_version=candidate.article_version,
        article_title=candidate.article_title,
        original_paragraph_range=candidate.paragraph_range,
        selected_paragraph_range=candidate.paragraph_range,
        original_text=candidate.original_text,
        retrieval_scores=candidate.retrieval_scores.model_copy(deep=True),
        input_rank=input_rank,
        reranker_rank=candidate.retrieval_scores.reranker_rank,
        topic_title=candidate.topic_title,
        context_summary=candidate.context_summary,
        compression_method=CompressionMethod.NONE,
        matched_ranges=list(candidate.matched_ranges),
        matched_child_ids=list(candidate.matched_child_ids),
        matched_child_ranks=list(candidate.matched_child_ranks),
        expanded_range=candidate.expanded_range,
        neighbor_trimmed=candidate.neighbor_trimmed,
    )


def _without_neighbor_expansion(candidate: RetrievalCandidate) -> RetrievalCandidate:
    if candidate.matched_original_text is None or candidate.matched_bounding_range is None:
        return candidate
    return candidate.model_copy(
        update={
            "original_text": candidate.matched_original_text,
            "paragraph_range": candidate.matched_bounding_range,
            "expanded_range": candidate.matched_bounding_range,
            "neighbor_trimmed": True,
        }
    )


def _rendered_item(item: EvidenceItem, include_ai_metadata: bool) -> dict[str, object]:
    rendered: dict[str, object] = {
        "citation_id": item.citation_id,
        "article_title": item.article_title,
        "article_version": item.article_version,
        "paragraph_range": list(item.selected_paragraph_range),
        "original_text": item.original_text,
    }
    if include_ai_metadata:
        rendered["ai_generated_retrieval_metadata"] = {
            "topic_title": item.topic_title,
            "context_summary": item.context_summary,
        }
    return rendered


def render_context(items: Sequence[EvidenceItem], include_ai_metadata: bool) -> str:
    payload = {
        "evidence": [_rendered_item(item, include_ai_metadata) for item in items]
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _individual_rendered_chars(item: EvidenceItem, include_ai_metadata: bool) -> int:
    return len(
        json.dumps(
            _rendered_item(item, include_ai_metadata),
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


def _assert_invariants(result: AssemblyResult) -> None:
    expected_ids = [f"E{number}" for number in range(1, len(result.evidence_items) + 1)]
    rebuilt = render_context(
        result.evidence_items, result.config_snapshot.include_ai_metadata
    )
    valid = (
        [item.citation_id for item in result.evidence_items] == expected_ids
        and rebuilt == result.rendered_context
        and len(rebuilt) == result.usage.rendered_context_chars
        and len(rebuilt) <= result.config_snapshot.max_total_chars
        and result.usage.selected_count == len(result.evidence_items)
        and result.usage.excluded_count == len(result.excluded_candidates)
        and result.usage.input_candidate_count
        == result.usage.selected_count + result.usage.excluded_count
    )
    if not valid:
        raise _error(
            AssemblyErrorCode.RENDER_INVARIANT_FAILED,
            "组装结果未通过内部一致性校验",
        )


def assemble_context(
    ranked_candidates: Sequence[RetrievalCandidate],
    config: ContextAssemblyConfigSnapshot,
) -> AssemblyResult:
    _validate_config(config)
    for rank, candidate in enumerate(ranked_candidates, start=1):
        _validate_candidate(candidate, rank)

    selected: list[EvidenceItem] = []
    excluded: list[ExcludedCandidate] = []
    seen_chunks: set[str] = set()
    seen_ranges: set[tuple[str, tuple[int, int]]] = set()

    for rank, candidate in enumerate(ranked_candidates, start=1):
        item = _evidence(candidate, rank, len(selected) + 1)
        item_chars = _individual_rendered_chars(item, config.include_ai_metadata)
        range_identity = (candidate.article_version, candidate.paragraph_range)
        if candidate.chunk_id in seen_chunks or range_identity in seen_ranges:
            excluded.append(
                ExcludedCandidate(
                    source_chunk_id=candidate.chunk_id,
                    original_rank=rank,
                    reason=AssemblyExclusionReason.DUPLICATE,
                    rendered_chars=item_chars,
                )
            )
            continue
        seen_chunks.add(candidate.chunk_id)
        seen_ranges.add(range_identity)

        if len(selected) >= config.max_evidence_items:
            excluded.append(
                ExcludedCandidate(
                    source_chunk_id=candidate.chunk_id,
                    original_rank=rank,
                    reason=AssemblyExclusionReason.ITEM_LIMIT,
                    rendered_chars=item_chars,
                )
            )
            continue

        hypothetical = render_context([*selected, item], config.include_ai_metadata)
        if len(hypothetical) > config.max_total_chars:
            trimmed_candidate = _without_neighbor_expansion(candidate)
            if trimmed_candidate is not candidate:
                trimmed_item = _evidence(trimmed_candidate, rank, len(selected) + 1)
                trimmed_hypothetical = render_context(
                    [*selected, trimmed_item], config.include_ai_metadata
                )
                if len(trimmed_hypothetical) <= config.max_total_chars:
                    item = trimmed_item
                    item_chars = _individual_rendered_chars(
                        item, config.include_ai_metadata
                    )
                    hypothetical = trimmed_hypothetical
        if len(hypothetical) > config.max_total_chars:
            if not selected:
                raise _error(
                    AssemblyErrorCode.BUDGET_TOO_SMALL,
                    f"排名第一的有效候选无法放入 {config.max_total_chars} 字符预算",
                )
            excluded.append(
                ExcludedCandidate(
                    source_chunk_id=candidate.chunk_id,
                    original_rank=rank,
                    reason=AssemblyExclusionReason.CHAR_BUDGET,
                    rendered_chars=item_chars,
                )
            )
            continue
        selected.append(item)

    rendered = render_context(selected, config.include_ai_metadata)
    usage = AssemblyUsage(
        input_candidate_count=len(ranked_candidates),
        selected_count=len(selected),
        excluded_count=len(excluded),
        original_text_chars=sum(len(item.original_text) for item in selected),
        ai_metadata_chars=(
            sum(len(item.topic_title or "") + len(item.context_summary or "") for item in selected)
            if config.include_ai_metadata
            else 0
        ),
        rendered_context_chars=len(rendered),
    )
    result = AssemblyResult(
        status=AssemblyStatus.READY if selected else AssemblyStatus.NO_EVIDENCE,
        evidence_items=selected,
        excluded_candidates=excluded,
        usage=usage,
        config_snapshot=config.model_copy(deep=True),
        rendered_context=rendered,
        rendered_context_sha256=hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
    )
    _assert_invariants(result)
    return result
