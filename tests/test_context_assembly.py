from __future__ import annotations

import hashlib
import json

import pytest

from xifeng_rag.context_assembly import (
    ContextAssemblyError,
    assemble_context,
    render_context,
)
from xifeng_rag.domain import (
    AssemblyErrorCode,
    AssemblyExclusionReason,
    AssemblyStatus,
    ContextAssemblyConfigSnapshot,
    RetrievalCandidate,
    RetrievalScores,
)


def candidate(
    number: int,
    text: str = "自造证据",
    *,
    chunk_id: str | None = None,
    version: str = "version-1",
    paragraph_range: tuple[int, int] | None = None,
) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=chunk_id or f"chunk-{number}",
        article_id="article-1",
        article_version=version,
        article_title="自造文章",
        original_text=text,
        paragraph_range=paragraph_range or (number, number),
        retrieval_scores=RetrievalScores(
            rrf=0.1 / number,
            rrf_rank=number,
            reranker=1.0 / number,
            reranker_rank=number,
        ),
        source=["bm25", "dense"],
        topic_title=f"AI 话题 {number}",
        context_summary=f"AI 概括 {number}",
    )


def config(
    *, max_items: int = 5, max_chars: int = 50_000, include_ai: bool = False
) -> ContextAssemblyConfigSnapshot:
    return ContextAssemblyConfigSnapshot(
        max_evidence_items=max_items,
        max_total_chars=max_chars,
        include_ai_metadata=include_ai,
    )


def test_assembly_preserves_order_provenance_and_input() -> None:
    candidates = [candidate(1), candidate(2)]
    before = [item.model_dump() for item in candidates]

    result = assemble_context(candidates, config())

    assert result.status == AssemblyStatus.READY
    assert [item.citation_id for item in result.evidence_items] == ["E1", "E2"]
    assert [item.source_chunk_id for item in result.evidence_items] == ["chunk-1", "chunk-2"]
    assert [item.input_rank for item in result.evidence_items] == [1, 2]
    assert result.evidence_items[0].selected_paragraph_range == (1, 1)
    assert [item.model_dump() for item in candidates] == before
    assert result.rendered_context_sha256 == hashlib.sha256(
        result.rendered_context.encode("utf-8")
    ).hexdigest()


def test_ai_metadata_is_internal_by_default_and_configurable_in_rendering() -> None:
    without_ai = assemble_context([candidate(1)], config())
    with_ai = assemble_context([candidate(1)], config(include_ai=True))

    assert without_ai.evidence_items[0].topic_title == "AI 话题 1"
    assert "AI 话题" not in without_ai.rendered_context
    payload = json.loads(with_ai.rendered_context)
    metadata = payload["evidence"][0]["ai_generated_retrieval_metadata"]
    assert metadata == {"topic_title": "AI 话题 1", "context_summary": "AI 概括 1"}
    assert with_ai.usage.ai_metadata_chars > 0
    assert with_ai.rendered_context_sha256 != without_ai.rendered_context_sha256


def test_exact_duplicates_and_item_limit_have_explicit_reasons() -> None:
    candidates = [
        candidate(1),
        candidate(2, chunk_id="chunk-1"),
        candidate(3, version="version-1", paragraph_range=(1, 1)),
        candidate(4),
    ]

    result = assemble_context(candidates, config(max_items=1))

    assert [item.source_chunk_id for item in result.evidence_items] == ["chunk-1"]
    assert [item.reason for item in result.excluded_candidates] == [
        AssemblyExclusionReason.DUPLICATE,
        AssemblyExclusionReason.DUPLICATE,
        AssemblyExclusionReason.ITEM_LIMIT,
    ]


def test_char_budget_skips_large_middle_candidate_and_keeps_later_small_one() -> None:
    candidates = [candidate(1, "甲" * 20), candidate(2, "乙" * 500), candidate(3, "丙")]
    generous = assemble_context(candidates, config())
    first, _, third = generous.evidence_items
    third = third.model_copy(update={"citation_id": "E2"})
    budget = len(render_context([first, third], False))

    result = assemble_context(candidates, config(max_chars=budget))

    assert [item.source_chunk_id for item in result.evidence_items] == ["chunk-1", "chunk-3"]
    assert result.excluded_candidates[0].source_chunk_id == "chunk-2"
    assert result.excluded_candidates[0].reason == AssemblyExclusionReason.CHAR_BUDGET
    assert result.usage.rendered_context_chars <= budget


def test_first_candidate_over_budget_is_a_classified_error_without_body() -> None:
    private_body = "不能出现在错误里的自造正文"
    with pytest.raises(ContextAssemblyError) as raised:
        assemble_context([candidate(1, private_body)], config(max_chars=20))

    assert raised.value.code == AssemblyErrorCode.BUDGET_TOO_SMALL
    assert private_body not in str(raised.value)


def test_empty_candidates_return_deterministic_no_evidence() -> None:
    first = assemble_context([], config())
    second = assemble_context([], config())

    assert first.status == AssemblyStatus.NO_EVIDENCE
    assert first.evidence_items == []
    assert first.rendered_context == '{"evidence":[]}'
    assert first == second


@pytest.mark.parametrize(
    ("bad_config", "code"),
    [
        (config(max_items=0), AssemblyErrorCode.INVALID_CONFIG),
        (config(max_chars=1), AssemblyErrorCode.INVALID_CONFIG),
    ],
)
def test_invalid_config_has_stable_error_code(
    bad_config: ContextAssemblyConfigSnapshot, code: AssemblyErrorCode
) -> None:
    with pytest.raises(ContextAssemblyError) as raised:
        assemble_context([], bad_config)
    assert raised.value.code == code


def test_invalid_candidate_error_does_not_expose_original_text() -> None:
    private_body = "不能泄露的自造正文"
    bad = candidate(1, private_body).model_copy(update={"article_title": ""})

    with pytest.raises(ContextAssemblyError) as raised:
        assemble_context([bad], config())

    assert raised.value.code == AssemblyErrorCode.INVALID_CANDIDATE
    assert private_body not in str(raised.value)


def test_json_round_trips_chinese_quotes_and_newlines_deterministically() -> None:
    body = '第一行“引号”\n第二行\\路径'
    first = assemble_context([candidate(1, body)], config())
    second = assemble_context([candidate(1, body)], config())

    assert json.loads(first.rendered_context)["evidence"][0]["original_text"] == body
    assert first.rendered_context == second.rendered_context
    assert first.rendered_context_sha256 == second.rendered_context_sha256
