from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from sqlalchemy import select

from xifeng_rag.chunking import GenerationResult, approve_manifest, export_review, propose_chunks
from xifeng_rag.corpus import scan_corpus
from xifeng_rag.domain import RetrievalMode
from xifeng_rag.indexing import (
    IndexBuildError,
    RankedHit,
    SentenceTransformersEmbeddingProvider,
    build_hybrid_index,
    build_index,
    build_layered_index,
    reciprocal_rank_fusion,
    record_build_failure,
    search,
    search_layered,
    search_layered_with_query_variants,
    search_with_query_variants,
)
from xifeng_rag.query_rewrite import build_query_variants
from xifeng_rag.reranking import RerankScore
from xifeng_rag.storage import ArticleVersionRow, Database, IndexBuildRow


def test_sentence_transformers_embedding_loads_from_local_cache_only(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeSentenceTransformer:
        max_seq_length = 512
        device = "cpu"

        def __init__(self, model_id: str, **kwargs: object) -> None:
            captured["model_id"] = model_id
            captured.update(kwargs)

        def get_embedding_dimension(self) -> int:
            return 2

        def _first_module(self) -> object:
            return SimpleNamespace(
                auto_model=SimpleNamespace(
                    config=SimpleNamespace(_commit_hash="local-test-revision")
                )
            )

    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer", FakeSentenceTransformer
    )

    provider = SentenceTransformersEmbeddingProvider("local-model")

    assert captured["local_files_only"] is True
    assert provider.metadata["model_revision"] == "local-test-revision"


class TwoTopicGenerator:
    def generate(self, prompt: str, schema: dict[str, object]) -> GenerationResult:
        return GenerationResult(
            content=json.dumps(
                {
                    "article_summary": "文章分别讨论苹果种植和香蕉销售。",
                    "topics": [
                        {
                            "end_unit": 2,
                            "title": "苹果",
                            "summary": "苹果种植",
                            "keywords": ["苹果", "果园"],
                            "content_role": "substantive",
                        },
                        {
                            "end_unit": 4,
                            "title": "航天",
                            "summary": "火箭发射",
                            "keywords": ["火箭", "卫星"],
                            "content_role": "substantive",
                        },
                    ]
                },
                ensure_ascii=False,
            )
        )


class FakeEmbeddingProvider:
    dimension = 2

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "provider": "fake",
            "model_id": "fake-v1",
            "model_revision": "test",
            "dimension": 2,
            "max_seq_length": 512,
            "normalize_embeddings": True,
            "query_instruction": "test",
        }

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            [[0.0, 1.0] if "火箭" in text else [1.0, 0.0] for text in texts],
            dtype=np.float32,
        )

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            [[1.0, 0.0] if "水果" in text else [0.0, 1.0] for text in texts],
            dtype=np.float32,
        )

    def close(self) -> None:
        pass


def _approve_fixture(database: Database, tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "fixture.md").write_text(
        "# 示例\n\n果园种植苹果。\n\n红苹果成熟丰收。\n\n火箭发射卫星。",
        encoding="utf-8",
    )
    with database.session() as session:
        scan_corpus(session, corpus)
    with database.session() as session:
        version = session.scalar(select(ArticleVersionRow))
        assert version is not None
        propose_chunks(session, version, TwoTopicGenerator(), "fake", 0)
        version_id = version.id
    with database.session() as session:
        review, _ = export_review(session, version_id, tmp_path / "reviews")
        approve_manifest(session, review)


def test_build_publish_and_search_returns_original_citation(
    database: Database, tmp_path: Path
) -> None:
    _approve_fixture(database, tmp_path)
    index_root = tmp_path / "indexes"
    with database.session() as session:
        outcome = build_index(session, index_root)
        assert outcome.chunk_count == 2
        assert outcome.path.is_dir()
    with database.session() as session:
        candidates = search(session, "卫星火箭", 2)
        assert candidates[0].topic_title == "航天"
        assert "火箭发射卫星" in candidates[0].original_text
        assert candidates[0].paragraph_range == (3, 4)
        active = session.scalar(select(IndexBuildRow).where(IndexBuildRow.active.is_(True)))
        assert active is not None


def test_rrf_combines_ranks_and_has_stable_order() -> None:
    fused = reciprocal_rank_fusion(
        [RankedHit("a", 8.0, 1), RankedHit("b", 7.0, 2)],
        [RankedHit("b", 0.9, 1), RankedHit("c", 0.8, 2)],
        rrf_k=60,
    )
    assert [item.chunk_id for item in fused] == ["b", "a", "c"]
    assert fused[0].bm25 is not None and fused[0].dense is not None


def test_hybrid_build_supports_dense_and_rrf_search(database: Database, tmp_path: Path) -> None:
    _approve_fixture(database, tmp_path)
    provider = FakeEmbeddingProvider()
    with database.session() as session:
        outcome = build_hybrid_index(session, tmp_path / "hybrid-indexes", provider)
        assert (outcome.path / "qdrant").is_dir()
        metadata = json.loads((outcome.path / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["embedding"]["model_id"] == "fake-v1"
        assert metadata["retrieval"] == {
            "bm25_candidate_k": 5,
            "dense_candidate_k": 5,
            "rrf_k": 60,
        }
        variant = build_hybrid_index(
            session,
            tmp_path / "hybrid-indexes",
            provider,
            bm25_candidate_k=6,
        )
        assert variant.build_id != outcome.build_id
    with database.session() as session:
        dense = search(
            session,
            "水果怎么种",
            2,
            build_id=outcome.build_id,
            mode=RetrievalMode.DENSE,
            embedding_provider=provider,
        )
        hybrid = search(
            session,
            "水果苹果",
            2,
            build_id=outcome.build_id,
            mode=RetrievalMode.HYBRID,
            embedding_provider=provider,
        )
    assert dense[0].topic_title == "苹果"
    assert dense[0].retrieval_scores.dense_rank == 1
    assert dense[0].source == ["dense"]
    assert hybrid[0].retrieval_scores.rrf is not None
    assert hybrid[0].source == ["bm25", "dense"]


def test_hybrid_search_can_rerank_wide_candidates(database: Database, tmp_path: Path) -> None:
    _approve_fixture(database, tmp_path)
    provider = FakeEmbeddingProvider()

    class ReverseReranker:
        @property
        def metadata(self) -> dict[str, object]:
            return {"provider": "fake"}

        def rerank(self, query, candidates):
            del query
            return [
                RerankScore(float(index), False)
                for index, _candidate in enumerate(candidates, start=1)
            ]

        def close(self) -> None:
            pass

    with database.session() as session:
        outcome = build_hybrid_index(session, tmp_path / "reranked-indexes", provider)
        baseline = search(
            session,
            "水果苹果",
            1,
            build_id=outcome.build_id,
            mode=RetrievalMode.HYBRID,
            embedding_provider=provider,
        )
        reranked = search(
            session,
            "水果苹果",
            1,
            build_id=outcome.build_id,
            mode=RetrievalMode.HYBRID,
            embedding_provider=provider,
            reranker=ReverseReranker(),
            rerank_candidate_k=2,
        )
    assert reranked[0].chunk_id != baseline[0].chunk_id
    assert reranked[0].retrieval_scores.reranker_rank == 1
    assert reranked[0].retrieval_scores.rrf_rank == 2


def test_query_variants_fuse_before_one_augmented_query_rerank(
    database: Database, tmp_path: Path
) -> None:
    _approve_fixture(database, tmp_path)
    provider = FakeEmbeddingProvider()

    class RecordingReranker:
        def __init__(self) -> None:
            self.queries: list[str] = []

        @property
        def metadata(self) -> dict[str, object]:
            return {"provider": "fake"}

        def rerank(self, query, candidates):
            self.queries.append(query)
            return [
                RerankScore(float(len(candidates) - index), False)
                for index in range(len(candidates))
            ]

        def close(self) -> None:
            pass

    variants = build_query_variants(
        "水果苹果",
        ["苹果怎样种植"],
        original_weight=1.0,
        rewrite_total_weight=0.7,
    )
    reranker = RecordingReranker()
    with database.session() as session:
        outcome = build_hybrid_index(session, tmp_path / "multi-query-indexes", provider)
        results = search_with_query_variants(
            session,
            "水果苹果",
            variants,
            build_id=outcome.build_id,
            embedding_provider=provider,
            reranker=reranker,
            top_k=2,
            rerank_candidate_k=2,
        )

    assert len(reranker.queries) == 1
    assert reranker.queries[0].startswith("原问题（主要语义约束）：水果苹果")
    assert "苹果怎样种植" in reranker.queries[0]
    assert results
    assert results[0].query_matches
    assert results[0].retrieval_scores.query_fusion_rank is not None


def test_layered_build_stays_opt_in_and_search_expands_within_parent(
    database: Database, tmp_path: Path
) -> None:
    _approve_fixture(database, tmp_path)
    provider = FakeEmbeddingProvider()

    class AppleReranker:
        @property
        def metadata(self) -> dict[str, object]:
            return {"provider": "fake"}

        def rerank(self, query, candidates):
            del query
            return [
                RerankScore(10.0 if "红苹果" in item.original_text else 0.0, False)
                for item in candidates
            ]

        def close(self) -> None:
            pass

    with database.session() as session:
        baseline = build_hybrid_index(session, tmp_path / "layered-indexes", provider)
        layered = build_layered_index(
            session,
            tmp_path / "layered-indexes",
            provider,
            target_chars=1,
            max_chars=100,
            tail_merge_chars=0,
            parent_candidate_k=1,
            child_candidate_k=4,
            candidate_limit=10,
        )
        active = session.scalar(select(IndexBuildRow).where(IndexBuildRow.active.is_(True)))
        assert active is not None and active.id == baseline.build_id
        layered_row = session.get(IndexBuildRow, layered.build_id)
        assert layered_row is not None and layered_row.status == "ready"
        assert (layered.path / "parent" / "bm25").is_dir()
        assert (layered.path / "child" / "qdrant").is_dir()
        manifest_text = (layered.path / "children.json").read_text(encoding="utf-8")
        assert "original_text" not in manifest_text

        results = search_layered(
            session,
            "红苹果",
            build_id=layered.build_id,
            embedding_provider=provider,
            reranker=AppleReranker(),
            top_k=1,
            parent_candidate_k=1,
            child_candidate_k=4,
            candidate_limit=10,
            neighbor_window=1,
        )
        child_only = search_layered(
            session,
            "红苹果",
            build_id=layered.build_id,
            embedding_provider=provider,
            reranker=AppleReranker(),
            top_k=1,
            parent_candidate_k=1,
            child_candidate_k=4,
            candidate_limit=10,
            neighbor_window=0,
            layer_mode="child",
        )
        rewritten = search_layered_with_query_variants(
            session,
            "红苹果",
            build_query_variants(
                "红苹果",
                ["成熟的红色苹果"],
                original_weight=1.0,
                rewrite_total_weight=0.7,
            ),
            build_id=layered.build_id,
            embedding_provider=provider,
            reranker=AppleReranker(),
            top_k=1,
            parent_candidate_k=1,
            child_candidate_k=4,
            candidate_limit=10,
            query_candidate_limit=10,
            neighbor_window=0,
        )
        without_neighbors = search_layered(
            session,
            "红苹果",
            build_id=layered.build_id,
            embedding_provider=provider,
            reranker=AppleReranker(),
            top_k=4,
            parent_candidate_k=1,
            child_candidate_k=4,
            candidate_limit=10,
            neighbor_window=0,
        )
        with_neighbors = search_layered(
            session,
            "红苹果",
            build_id=layered.build_id,
            embedding_provider=provider,
            reranker=AppleReranker(),
            top_k=4,
            parent_candidate_k=1,
            child_candidate_k=4,
            candidate_limit=10,
            neighbor_window=1,
        )
        with pytest.raises(ValueError, match="Index Profile 不一致"):
            search_layered(
                session,
                "红苹果",
                build_id=layered.build_id,
                embedding_provider=provider,
                reranker=AppleReranker(),
                top_k=1,
                parent_candidate_k=2,
                child_candidate_k=4,
                candidate_limit=10,
            )

    assert len(results) == 1
    assert results[0].parent_chunk_id is not None
    assert results[0].matched_ranges == [(3, 3)]
    assert results[0].expanded_range == (3, 4)
    assert results[0].paragraph_range == (3, 4)
    assert "红苹果成熟丰收" in results[0].original_text
    assert results[0].layer_sources
    assert child_only[0].layer_sources == ["child"]
    assert child_only[0].expanded_range == child_only[0].matched_ranges[0]
    assert rewritten[0].query_matches
    assert {item.query_kind for item in rewritten[0].query_matches} == {
        "original",
        "rewrite",
    }
    assert sorted(
        ranked for item in without_neighbors for ranked in item.matched_child_ranks
    ) == sorted(ranked for item in with_neighbors for ranked in item.matched_child_ranks)


def test_layered_build_reuses_unchanged_parent_and_child_embeddings(
    database: Database, tmp_path: Path
) -> None:
    _approve_fixture(database, tmp_path)

    class CountingProvider(FakeEmbeddingProvider):
        def __init__(self) -> None:
            self.encoded_batches: list[list[str]] = []

        def encode_documents(self, texts: list[str]) -> np.ndarray:
            self.encoded_batches.append(list(texts))
            return super().encode_documents(texts)

    provider = CountingProvider()
    with database.session() as session:
        build_layered_index(
            session,
            tmp_path / "incremental-indexes",
            provider,
            target_chars=1,
            max_chars=100,
            tail_merge_chars=0,
            parent_candidate_k=1,
            child_candidate_k=4,
            candidate_limit=20,
        )
    provider.encoded_batches.clear()

    corpus = tmp_path / "corpus"
    (corpus / "second.md").write_text(
        "# 第二篇\n\n果园种植苹果。\n\n红苹果成熟丰收。\n\n火箭发射卫星。",
        encoding="utf-8",
    )
    with database.session() as session:
        scan_corpus(session, corpus)
        version = session.scalar(
            select(ArticleVersionRow).where(ArticleVersionRow.status == "pending_chunking")
        )
        assert version is not None
        propose_chunks(session, version, TwoTopicGenerator(), "fake", 0)
        version_id = version.id
    with database.session() as session:
        review, _ = export_review(session, version_id, tmp_path / "second-reviews")
        approve_manifest(session, review)
        outcome = build_layered_index(
            session,
            tmp_path / "incremental-indexes",
            provider,
            target_chars=1,
            max_chars=100,
            tail_merge_chars=0,
            parent_candidate_k=1,
            child_candidate_k=4,
            candidate_limit=20,
        )
        metadata = json.loads((outcome.path / "metadata.json").read_text(encoding="utf-8"))

    assert metadata["reused_parent_embedding_count"] == 2
    assert metadata["reused_child_embedding_count"] == 4
    assert sum(len(batch) for batch in provider.encoded_batches) == 6


def test_failed_layered_child_index_keeps_default_build_active(
    database: Database, tmp_path: Path
) -> None:
    _approve_fixture(database, tmp_path)
    with database.session() as session:
        baseline = build_hybrid_index(
            session, tmp_path / "layered-rollback", FakeEmbeddingProvider()
        )

    class FailSecondLayerProvider(FakeEmbeddingProvider):
        def __init__(self) -> None:
            self.calls = 0

        def encode_documents(self, texts: list[str]) -> np.ndarray:
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("synthetic child embedding failure")
            return super().encode_documents(texts)

    with pytest.raises(IndexBuildError) as raised, database.session() as session:
        build_layered_index(
            session,
            tmp_path / "layered-rollback",
            FailSecondLayerProvider(),
            candidate_limit=30,
        )

    assert not raised.value.final_path.exists()
    with database.session() as session:
        active = session.scalar(select(IndexBuildRow).where(IndexBuildRow.active.is_(True)))
        assert active is not None and active.id == baseline.build_id


def test_reranker_rejects_single_retrieval_modes(database: Database, tmp_path: Path) -> None:
    _approve_fixture(database, tmp_path)
    with database.session() as session:
        outcome = build_index(session, tmp_path / "bm25-only")
        with pytest.raises(ValueError, match="只支持 hybrid"):
            search(
                session,
                "问题",
                build_id=outcome.build_id,
                mode=RetrievalMode.BM25,
                reranker=object(),  # type: ignore[arg-type]
            )


def test_failed_dense_build_keeps_artifacts_unpublished(database: Database, tmp_path: Path) -> None:
    _approve_fixture(database, tmp_path)

    class FailingProvider(FakeEmbeddingProvider):
        def encode_documents(self, texts: list[str]) -> np.ndarray:
            del texts
            raise RuntimeError("synthetic dense failure")

    try:
        with database.session() as session:
            build_hybrid_index(session, tmp_path / "failed-hybrid", FailingProvider())
    except IndexBuildError as exc:
        assert exc.profile_id != "approved-topic-original-bm25-jieba-v1"
        assert not exc.final_path.exists()
        with database.session() as session:
            record_build_failure(session, exc)
    else:
        raise AssertionError("synthetic dense build must fail")


def test_build_rejects_empty_index(database: Database, tmp_path: Path) -> None:
    with database.session() as session:
        try:
            build_index(session, tmp_path / "indexes")
        except ValueError as exc:
            assert "拒绝建立空索引" in str(exc)
        else:
            raise AssertionError("empty index must fail")


def test_bm25_build_contains_only_topics_included_by_review(
    database: Database, tmp_path: Path
) -> None:
    corpus = tmp_path / "role-corpus"
    corpus.mkdir()
    (corpus / "fixture.md").write_text(
        "# 示例\n这是一段足够长的有效正文，包含作者的观点、解释、案例以及完整结论。"
        "它应当作为实质内容进入索引。\n阅读付费提示。",
        encoding="utf-8",
    )
    response = json.dumps(
        {
            "article_summary": "文章包含正文与不进入索引的提示内容。",
            "topics": [
                {
                    "end_unit": 3,
                    "title": "正文",
                    "summary": "有效内容",
                    "keywords": ["正文"],
                },
                {
                    "end_unit": 4,
                    "title": "提示",
                    "summary": "阅读提示",
                    "keywords": ["提示"],
                },
            ]
        },
        ensure_ascii=False,
    )
    with database.session() as session:
        scan_corpus(session, corpus)
    with database.session() as session:
        version = session.scalar(select(ArticleVersionRow))
        assert version is not None

        class RoleGenerator:
            def generate(self, prompt: str, schema: dict[str, object]) -> GenerationResult:
                return GenerationResult(content=response)

        propose_chunks(
            session,
            version,
            RoleGenerator(),
            "role-fake",
            0,
        )
        version_id = version.id
    with database.session() as session:
        review, _ = export_review(session, version_id, tmp_path / "role-reviews")
        payload = json.loads(review.read_text(encoding="utf-8"))
        payload["topics"][1]["include_in_index"] = False
        review.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        approve_manifest(session, review)
        outcome = build_index(session, tmp_path / "role-indexes")
        assert outcome.chunk_count == 1


def test_failed_build_is_not_published_and_diagnostic_can_be_recorded(
    database: Database, tmp_path: Path, monkeypatch
) -> None:
    _approve_fixture(database, tmp_path)
    index_root = tmp_path / "indexes"

    def fail_index(*args, **kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr("bm25s.BM25.index", fail_index)
    try:
        with database.session() as session:
            build_index(session, index_root)
    except IndexBuildError as exc:
        with database.session() as session:
            record_build_failure(session, exc)
    else:
        raise AssertionError("synthetic build must fail")

    with database.session() as session:
        row = session.scalar(select(IndexBuildRow))
        assert row is not None
        assert row.status == "failed"
        assert row.error_type == "RuntimeError"
        assert not row.active
        assert not Path(row.path).exists()
