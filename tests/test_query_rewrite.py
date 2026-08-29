from __future__ import annotations

import json

import pytest

from xifeng_rag.domain import RetrievalCandidate, RetrievalScores
from xifeng_rag.query_rewrite import (
    OllamaQueryRewriter,
    QueryRewriteStatus,
    augmented_rerank_query,
    build_query_variants,
    filter_variants,
    weighted_query_fusion,
)
from xifeng_rag.structured_generation import GenerationResult


def _candidate(chunk_id: str, rank: int) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=chunk_id,
        article_id=f"article-{chunk_id}",
        article_version=f"version-{chunk_id}",
        article_title=f"文章 {chunk_id}",
        original_text=f"自造证据 {chunk_id}",
        paragraph_range=(rank, rank),
        retrieval_scores=RetrievalScores(rrf=1 / (60 + rank), rrf_rank=rank),
        source=["bm25", "dense"],
        candidate_rank=rank,
    )


def test_filter_variants_removes_empty_duplicate_original_and_excess() -> None:
    assert filter_variants(
        "原问题",
        ["", " 原问题 ", "改写一", "改写一", "改写二", "改写三"],
        2,
    ) == ["改写一", "改写二"]
    with pytest.raises(ValueError, match="不能小于"):
        filter_variants("问题", [], -1)


def test_build_query_variants_shares_fixed_rewrite_weight() -> None:
    variants = build_query_variants(
        "原问题",
        ["改写一", "改写二"],
        original_weight=1.0,
        rewrite_total_weight=0.7,
    )

    assert [item.kind for item in variants] == ["original", "rewrite", "rewrite"]
    assert [item.weight for item in variants] == pytest.approx([1.0, 0.35, 0.35])
    rerank_query = augmented_rerank_query("原问题", variants)
    assert rerank_query.startswith("原问题（主要语义约束）：原问题")
    assert "改写一" in rerank_query and "改写二" in rerank_query


def test_weighted_query_fusion_prefers_agreement_and_preserves_audit_matches() -> None:
    variants = build_query_variants(
        "原问题",
        ["保守改写"],
        original_weight=1.0,
        rewrite_total_weight=0.7,
    )
    fused = weighted_query_fusion(
        [
            (variants[0], [_candidate("a", 1), _candidate("b", 2)]),
            (variants[1], [_candidate("b", 1), _candidate("c", 2)]),
        ],
        rrf_k=60,
        candidate_limit=3,
    )

    assert [item.chunk_id for item in fused] == ["b", "a", "c"]
    assert [match.query_kind for match in fused[0].query_matches] == [
        "original",
        "rewrite",
    ]
    assert fused[0].retrieval_scores.query_fusion_rank == 1
    assert fused[0].candidate_rank == 1
    second_score = fused[1].retrieval_scores.query_fusion
    third_score = fused[2].retrieval_scores.query_fusion
    assert second_score is not None and third_score is not None
    assert second_score > third_score


class FakeOllamaClient:
    def __init__(self, content: str) -> None:
        self.content = content
        self.started = 0
        self.prompts: list[str] = []
        self.schemas: list[dict[str, object]] = []

    def start_server(self) -> None:
        self.started += 1

    def installed_models(self) -> list[str]:
        return ["test-model:latest"]

    def generate(self, prompt: str, schema: dict[str, object]) -> GenerationResult:
        self.prompts.append(prompt)
        self.schemas.append(schema)
        return GenerationResult(content=self.content, metadata={"eval_count": 7})


def test_ollama_rewriter_uses_structured_query_only_prompt_and_filters_output() -> None:
    client = FakeOllamaClient(
        json.dumps(
            {"variants": ["作者本人先后担任过哪些职位？", "原问题", "多余改写"]},
            ensure_ascii=False,
        )
    )
    rewriter = OllamaQueryRewriter(client, "test-model", max_retries=0)

    result = rewriter.rewrite("原问题", 1)

    assert result.status == QueryRewriteStatus.SUCCESS
    assert result.variants == ["作者本人先后担任过哪些职位？"]
    assert result.metadata == {"eval_count": 7}
    assert client.started == 1
    assert "原问题" in client.prompts[0]
    assert "不要用“或、以及、/”" in client.prompts[0]
    assert "付费正文" not in client.prompts[0]
    assert client.schemas[0]["required"] == ["variants"]


class FailingOllamaClient(FakeOllamaClient):
    def start_server(self) -> None:
        raise FileNotFoundError("自造的本地服务不可用")


def test_ollama_rewriter_failure_returns_diagnostic_without_raising() -> None:
    result = OllamaQueryRewriter(FailingOllamaClient("{}"), "test-model").rewrite(
        "问题", 1
    )

    assert result.status == QueryRewriteStatus.FAILED
    assert result.variants == []
    assert result.diagnostic == "FileNotFoundError"
