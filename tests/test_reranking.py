from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from xifeng_rag.domain import RetrievalCandidate, RetrievalScores
from xifeng_rag.reranking import (
    INPUT_TEMPLATE_VERSION,
    RerankScore,
    SentenceTransformersReranker,
    rerank_candidates,
)


def _candidate(chunk_id: str, rrf_rank: int) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=chunk_id,
        article_id="article-test",
        article_version="version-test",
        article_title="自造标题",
        original_text="只用于测试的短正文",
        paragraph_range=(1, 1),
        retrieval_scores=RetrievalScores(rrf=0.1, rrf_rank=rrf_rank),
        source=["bm25", "dense"],
    )


class FakeReranker:
    def __init__(self, results: list[RerankScore]) -> None:
        self.results = results

    @property
    def metadata(self) -> dict[str, object]:
        return {"provider": "fake"}

    def rerank(
        self, query: str, candidates: list[RetrievalCandidate]
    ) -> list[RerankScore]:
        del query, candidates
        return self.results

    def close(self) -> None:
        pass


def test_rerank_candidates_uses_stable_ties_and_preserves_stage_scores() -> None:
    candidates = [_candidate("a", 1), _candidate("b", 2), _candidate("c", 3)]
    reranked = rerank_candidates(
        "测试问题",
        candidates,
        FakeReranker(
            [
                RerankScore(0.2, False),
                RerankScore(0.9, True),
                RerankScore(0.9, False),
            ]
        ),
        top_k=2,
    )
    assert [item.chunk_id for item in reranked] == ["b", "c"]
    assert reranked[0].retrieval_scores.rrf == 0.1
    assert reranked[0].retrieval_scores.reranker == pytest.approx(0.9)
    assert reranked[0].retrieval_scores.reranker_rank == 1
    assert reranked[0].retrieval_scores.reranker_truncated is True


def test_rerank_candidates_rejects_invalid_output() -> None:
    candidates = [_candidate("a", 1)]
    with pytest.raises(ValueError, match="输出数量"):
        rerank_candidates("问题", candidates, FakeReranker([]), top_k=1)
    with pytest.raises(ValueError, match="非有限"):
        rerank_candidates(
            "问题", candidates, FakeReranker([RerankScore(float("nan"), False)]), top_k=1
        )


def test_sentence_transformers_provider_batches_and_marks_pair_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: dict[str, object] = {}

    class FakeTokenizer:
        def __call__(self, query: str, passage: str, **kwargs):
            assert query == "问题"
            assert kwargs == {"add_special_tokens": True, "truncation": False}
            return {"input_ids": list(range(20 if "很长" in passage else 5))}

    class FakeCrossEncoder:
        def __init__(self, model_id: str, **kwargs) -> None:
            created["model_id"] = model_id
            created["kwargs"] = kwargs
            self.tokenizer = FakeTokenizer()
            self.model = SimpleNamespace(
                config=SimpleNamespace(_commit_hash="revision-test"),
                device="cpu",
            )

        def predict(self, pairs, **kwargs):
            created["pairs"] = pairs
            created["predict_kwargs"] = kwargs
            return np.asarray([0.25, 0.75], dtype=np.float32)

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(CrossEncoder=FakeCrossEncoder),
    )
    provider = SentenceTransformersReranker(
        "fake/model", revision=None, device="cpu", batch_size=2, max_pair_tokens=10
    )
    candidates = [_candidate("a", 1), _candidate("b", 2)]
    candidates[1].original_text = "很长的自造正文"
    results = provider.rerank("问题", candidates)

    assert results == [RerankScore(0.25, False), RerankScore(0.75, True)]
    assert created["kwargs"] == {
        "revision": None,
        "cache_folder": None,
        "device": "cpu",
        "max_length": 10,
    }
    assert created["predict_kwargs"] == {
        "batch_size": 2,
        "show_progress_bar": False,
        "convert_to_numpy": True,
    }
    assert provider.metadata["model_revision"] == "revision-test"
    assert provider.metadata["input_template_version"] == INPUT_TEMPLATE_VERSION
    assert provider.metadata["truncated_candidate_count"] == 1
    provider.close()
