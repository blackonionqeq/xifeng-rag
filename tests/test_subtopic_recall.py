from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import select

from xifeng_rag.domain import RetrievalMode, SubtopicSetStatus
from xifeng_rag.indexing import (
    RankedHit,
    reciprocal_rank_fusion,
    weighted_reciprocal_rank_fusion,
)
from xifeng_rag.layered_article_search import ArticleDiscoveryLayer
from xifeng_rag.storage import (
    ApprovedManifestRow,
    ArticleIndexBuildRow,
    ArticleRow,
    ArticleVersionRow,
    ChunkProposalRow,
    Database,
    ParagraphRow,
    SubtopicSetRow,
    TopicChunkRow,
)
from xifeng_rag.subtopic_recall import (
    SUBTOPIC_BM25_CHANNEL,
    SUBTOPIC_DENSE_CHANNEL,
    build_subtopic_chunk_map,
    build_subtopic_context,
    nominate_subtopic_chunks,
    read_build_universe,
    search_with_subtopic_candidates,
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


class ListReranker:
    """按候选顺序打分的重排器，保持融合排序不变，便于断言。"""

    @property
    def metadata(self) -> dict[str, object]:
        return {"provider": "fake-reranker"}

    def rerank(self, query, candidates):
        del query
        from xifeng_rag.reranking import RerankScore

        return [
            RerankScore(float(index), False)
            for index in range(len(candidates), 0, -1)
        ]

    def close(self) -> None:
        pass


def test_weighted_rrf_matches_legacy_two_channel_order() -> None:
    bm25_hits = [RankedHit("a", 8.0, 1), RankedHit("b", 7.0, 2)]
    dense_hits = [RankedHit("b", 0.9, 1), RankedHit("c", 0.8, 2)]
    legacy = reciprocal_rank_fusion(bm25_hits, dense_hits, rrf_k=60)
    weighted = weighted_reciprocal_rank_fusion(
        {"bm25": (bm25_hits, 1.0), "dense": (dense_hits, 1.0)}, rrf_k=60
    )
    assert [item.chunk_id for item in weighted] == [item.chunk_id for item in legacy]
    assert weighted[0].rrf_score == pytest.approx(legacy[0].rrf_score)
    assert weighted[0].channels["bm25"].rank == 2
    assert weighted[0].channels["dense"].rank == 1


def test_weighted_rrf_four_channel_math() -> None:
    fused = weighted_reciprocal_rank_fusion(
        {
            "bm25": ([RankedHit("a", 5.0, 1)], 1.0),
            "dense": ([RankedHit("b", 0.5, 1)], 1.0),
            SUBTOPIC_BM25_CHANNEL: ([RankedHit("a", 3.0, 2)], 2.0),
            SUBTOPIC_DENSE_CHANNEL: ([RankedHit("c", 0.2, 3)], 1.0),
        },
        rrf_k=60,
    )
    by_id = {item.chunk_id: item for item in fused}
    assert by_id["a"].rrf_score == pytest.approx(1.0 / 61 + 2.0 / 62)
    assert by_id["b"].rrf_score == pytest.approx(1.0 / 61)
    assert by_id["c"].rrf_score == pytest.approx(1.0 / 63)
    assert by_id["a"].channels.keys() == {"bm25", SUBTOPIC_BM25_CHANNEL}
    assert [item.chunk_id for item in fused] == ["a", "b", "c"]


def test_weighted_rrf_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="权重"):
        weighted_reciprocal_rank_fusion(
            {"bm25": ([RankedHit("a", 1.0, 1)], 0.0)}, rrf_k=60
        )
    with pytest.raises(ValueError, match="重复候选"):
        weighted_reciprocal_rank_fusion(
            {
                "bm25": (
                    [RankedHit("a", 1.0, 1), RankedHit("a", 0.5, 2)],
                    1.0,
                )
            },
            rrf_k=60,
        )


def _add_subtopic_article(
    database: Database,
    ordinal: int,
    *,
    title: str,
    topic_title: str,
    paragraphs: list[str],
    subtopics: list[dict[str, object]] | None,
    status: str = SubtopicSetStatus.APPROVED.value,
) -> tuple[str, str, str]:
    article_id = f"recall-article-{ordinal}"
    version_id = f"recall-version-{ordinal}"
    proposal_id = f"recall-proposal-{ordinal}"
    manifest_id = f"recall-manifest-{ordinal}"
    chunk_id = f"recall-chunk-{ordinal}"
    topics = {
        "topics": [
            {
                "start_paragraph": 1,
                "end_paragraph": len(paragraphs),
                "title": topic_title,
                "summary": f"块级概括 {ordinal}",
                "keywords": [topic_title],
                "include_in_index": True,
            }
        ]
    }
    with database.session() as session:
        session.add(ArticleRow(id=article_id, source_path=f"recall-{ordinal}.md", title=title))
        session.flush()
        session.add(
            ArticleVersionRow(
                id=version_id,
                article_id=article_id,
                source_hash=str(ordinal) * 64,
                canonical_hash=str(ordinal) * 64,
                cleaner_version="test",
                canonical_text="\n\n".join(paragraphs),
                status="indexed",
            )
        )
        session.flush()
        for number, text in enumerate(paragraphs, start=1):
            session.add(
                ParagraphRow(
                    version_id=version_id,
                    number=number,
                    text=text,
                    source_start=number,
                    source_end=number,
                )
            )
        session.add(
            ChunkProposalRow(
                id=proposal_id,
                version_id=version_id,
                status="approved",
                provider_id="test",
                model_id="test",
                prompt_version="test",
                schema_version="test",
                topics_json=json.dumps(topics, ensure_ascii=False),
            )
        )
        session.flush()
        session.add(
            ApprovedManifestRow(
                id=manifest_id,
                proposal_id=proposal_id,
                version_id=version_id,
                topics_json=json.dumps(topics, ensure_ascii=False),
            )
        )
        session.flush()
        session.add(
            TopicChunkRow(
                id=chunk_id,
                manifest_id=manifest_id,
                version_id=version_id,
                article_id=article_id,
                start_paragraph=1,
                end_paragraph=len(paragraphs),
                topic_title=topic_title,
                context_summary=f"块级概括 {ordinal}",
                keywords_json=json.dumps([topic_title], ensure_ascii=False),
                original_text="\n\n".join(paragraphs),
            )
        )
        if subtopics is not None:
            payload = {
                "topics": [
                    {
                        "parent_start_unit": 1,
                        "parent_end_unit": len(paragraphs),
                        "parent_title": topic_title,
                        "subtopics": subtopics,
                    }
                ]
            }
            session.add(
                SubtopicSetRow(
                    id=f"recall-subtopics-{ordinal}",
                    version_id=version_id,
                    manifest_id=manifest_id,
                    status=status,
                    provider_id="test",
                    model_id="test",
                    prompt_version="test",
                    schema_version="test",
                    subtopics_json=json.dumps(payload, ensure_ascii=False),
                )
            )
    return article_id, version_id, chunk_id


def _units(
    entries: list[tuple[str, str, str, int, int, str]],
) -> dict[str, dict[str, object]]:
    """构造 subtopics 层单元清单。

    元组为 (version_id, parent_topic, article_title, start, end, text)。
    真实套件的单元 title 是文章标题, 细话题标题只出现在 search_text 中,
    对齐因此依赖 (version_id, start_unit, end_unit)。
    """
    units: dict[str, dict[str, object]] = {}
    for index, (version_id, parent_topic, title, start, end, _text) in enumerate(
        entries, start=1
    ):
        units[f"unit-{index}"] = {
            "unit_id": f"unit-{index}",
            "unit_type": ArticleDiscoveryLayer.SUBTOPICS.value,
            "article_id": "unused",
            "version_id": version_id,
            "title": title,
            "parent_topic": parent_topic,
            "parent_chunk_id": None,
            "start_unit": start,
            "end_unit": end,
            "search_text": title,
        }
    return units


def test_map_joins_by_parent_range_and_ignores_superseded(
    database: Database,
) -> None:
    _add_subtopic_article(
        database,
        1,
        title="苹果与火箭",
        topic_title="种植与发射",
        paragraphs=["果园种植苹果。", "火箭发射卫星。"],
        subtopics=[
            {"title": "苹果种植", "start_unit": 1, "end_unit": 1, "summary": "苹果。"},
            {"title": "火箭发射", "start_unit": 2, "end_unit": 2, "summary": "火箭。"},
        ],
        status=SubtopicSetStatus.SUPERSEDED.value,
    )
    _add_subtopic_article(
        database,
        2,
        title="协作系统",
        topic_title="商业协作",
        paragraphs=["商业协作案例。"],
        subtopics=[
            {"title": "协作机制", "start_unit": 1, "end_unit": 1, "summary": "协作。"}
        ],
    )
    units = _units(
        [
            ("recall-version-1", "种植与发射", "苹果种植", 1, 1, "苹果种植"),
            ("recall-version-1", "种植与发射", "火箭发射", 2, 2, "火箭发射"),
            ("recall-version-2", "商业协作", "协作机制", 1, 1, "协作机制"),
        ]
    )
    with database.session() as session:
        chunk_map = build_subtopic_chunk_map(
            session, units, frozenset({"recall-chunk-2"})
        )
    # superseded 文章的细话题不进映射, 其父块也不在 universe, 计入丢弃.
    assert chunk_map.mapped_unit_count == 1
    assert chunk_map.unit_key_to_chunk == {
        ("recall-version-2", 1, 1): "recall-chunk-2"
    }
    assert chunk_map.covered_chunk_ids == frozenset({"recall-chunk-2"})
    # superseded 集合整体被忽略(不进映射也不计入丢弃); 丢弃计数只统计
    # approved 集合中父块不在 universe 的细话题.
    assert chunk_map.dropped_unit_count == 0
    assert chunk_map.suite_unit_count == 3


def test_map_raises_when_suite_and_build_have_no_overlap(database: Database) -> None:
    _add_subtopic_article(
        database,
        1,
        title="独立文章",
        topic_title="独立话题",
        paragraphs=["独立内容。"],
        subtopics=[
            {"title": "独立细话题", "start_unit": 1, "end_unit": 1, "summary": "独立。"}
        ],
    )
    units = _units([("recall-version-1", "独立话题", "独立细话题", 1, 1, "独立")])
    with database.session() as session, pytest.raises(ValueError, match="没有共同候选块"):
        build_subtopic_chunk_map(session, units, frozenset({"other-chunk"}))


def test_latest_approved_wins_for_same_version(database: Database) -> None:
    _add_subtopic_article(
        database,
        1,
        title="同版本多集合",
        topic_title="同版本话题",
        paragraphs=["第一批内容。", "第二批内容。"],
        subtopics=[
            {"title": "旧细话题", "start_unit": 1, "end_unit": 1, "summary": "旧。"}
        ],
    )
    # 再插一条更晚创建的 approved 集合, 标题不同.
    with database.session() as session:
        previous = session.get(SubtopicSetRow, "recall-subtopics-1")
        assert previous is not None
        session.add(
            SubtopicSetRow(
                id="recall-subtopics-1-later",
                version_id="recall-version-1",
                manifest_id="recall-manifest-1",
                status=SubtopicSetStatus.APPROVED.value,
                provider_id="test",
                model_id="test",
                prompt_version="test",
                schema_version="test",
                created_at=previous.created_at + timedelta(seconds=1),
                subtopics_json=json.dumps(
                    {
                        "topics": [
                            {
                                "parent_start_unit": 1,
                                "parent_end_unit": 2,
                                "parent_title": "同版本话题",
                                "subtopics": [
                                    {
                                        "title": "新细话题",
                                        "start_unit": 2,
                                        "end_unit": 2,
                                        "summary": "新。",
                                    }
                                ],
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
            )
        )
    units = _units(
        [
            ("recall-version-1", "同版本话题", "旧细话题", 1, 1, "旧"),
            ("recall-version-1", "同版本话题", "新细话题", 2, 2, "新"),
        ]
    )
    with database.session() as session:
        chunk_map = build_subtopic_chunk_map(
            session, units, frozenset({"recall-chunk-1"})
        )
    # 同一 version 只保留最新 approved 集合的条目。
    assert ("recall-version-1", 2, 2) in chunk_map.unit_key_to_chunk
    assert ("recall-version-1", 1, 1) not in chunk_map.unit_key_to_chunk


def test_nominate_maps_unit_hits_and_drops_out_of_universe(
    database: Database, tmp_path: Path
) -> None:
    _add_subtopic_article(
        database,
        1,
        title="命中文章",
        topic_title="命中话题",
        paragraphs=["命中内容一。", "命中内容二。"],
        subtopics=[
            {"title": "命中细话题", "start_unit": 1, "end_unit": 1, "summary": "命中。"},
            {"title": "宇宙外细话题", "start_unit": 2, "end_unit": 2, "summary": "外部。"},
        ],
    )
    # 第二条细话题刻意从映射中剔除, 模拟 universe 外丢弃.
    units = _units(
        [
            ("recall-version-1", "命中话题", "命中细话题", 1, 1, "命中"),
            ("recall-version-1", "命中话题", "宇宙外细话题", 2, 2, "外部"),
        ]
    )
    with database.session() as session:
        chunk_map = build_subtopic_chunk_map(session, units, frozenset({"recall-chunk-1"}))
    assert chunk_map.mapped_unit_count == 2  # 同一父块, 两个细话题都映射到它

    class StubProvider:
        dimension = 2

        @property
        def metadata(self) -> dict[str, object]:
            return {"provider": "stub"}

        def encode_queries(self, texts: list[str]) -> np.ndarray:
            del texts
            return np.asarray([[0.0, 1.0]], dtype=np.float32)

        def close(self) -> None:
            pass

    # 手工写一个只含两个单元的 BM25 索引目录.
    index_dir = tmp_path / "subtopic-layer"
    index_dir.mkdir()
    import bm25s

    corpus_ids = [
        {
            "unit_id": "unit-1",
            "version_id": "recall-version-1",
            "parent_topic": "命中话题",
            "title": "命中细话题",
            "start_unit": 1,
            "end_unit": 1,
        },
        {
            "unit_id": "unit-2",
            "version_id": "recall-version-1",
            "parent_topic": "命中话题",
            "title": "宇宙外细话题",
            "start_unit": 2,
            "end_unit": 2,
        },
    ]
    retriever = bm25s.BM25(corpus=corpus_ids)
    retriever.index([["命中"], ["外部"]], show_progress=False)
    retriever.save(str(index_dir / "bm25"), corpus=corpus_ids)
    # 让第二个单元的 key 不在映射中(模拟 universe 外丢弃).
    with database.session() as session:
        del_key = ("recall-version-1", 2, 2)
        pruned = {k: v for k, v in chunk_map.unit_key_to_chunk.items() if k != del_key}
        from dataclasses import replace

        pruned_map = replace(chunk_map, unit_key_to_chunk=pruned)
    hits, audit = nominate_subtopic_chunks(
        index_dir,
        units,
        pruned_map,
        "命中 外部",
        mode=RetrievalMode.BM25,
        limit=2,
        provider=None,
    )
    assert [item.chunk_id for item in hits] == ["recall-chunk-1"]
    assert audit.dropped_hit_count >= 1
    assert audit.nominated_chunk_ids == ("recall-chunk-1",)


def _write_hybrid_build(
    database: Database,
    tmp_path: Path,
    provider: FakeEmbeddingProvider,
    ordinal: int,
    topic_title: str,
    paragraphs: list[str],
) -> Path:
    """直接复用 build_hybrid_index 造一个真实话题块 Build。"""
    from xifeng_rag.indexing import build_hybrid_index

    with database.session() as session:
        outcome = build_hybrid_index(
            session, tmp_path / f"build-{ordinal}", provider
        )
        return outcome.path


def test_search_with_subtopic_candidates_records_provenance(
    database: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 建一篇文章(一个话题块), 并手工造 subtopics 层索引目录与套件.
    _add_subtopic_article(
        database,
        1,
        title="火箭文章",
        topic_title="火箭",
        paragraphs=["果园种植苹果。", "火箭发射卫星。"],
        subtopics=[
            {"title": "火箭发射", "start_unit": 2, "end_unit": 2, "summary": "火箭。"}
        ],
    )
    provider = FakeEmbeddingProvider()
    build_path = _write_hybrid_build(database, tmp_path, provider, 1, "火箭", [])

    units = _units(
        [("recall-version-1", "火箭", "火箭发射", 2, 2, "火箭发射卫星")]
    )
    suite_dir = tmp_path / "subtopic-layer"
    suite_dir.mkdir()

    # 对 nominate_subtopic_chunks 打桩: BM25 路提名 recall-chunk-1 第 1 名, Dense 路为空.
    def fake_nominate(path, unit_dict, chunk_map, query, *, mode, limit, provider):
        from xifeng_rag.subtopic_recall import SubtopicQueryAudit

        if mode == RetrievalMode.BM25:
            return [RankedHit("recall-chunk-1", 3.0, 1)], SubtopicQueryAudit(
                nominated_chunk_ids=("recall-chunk-1",),
                nominated_subtopic_titles=("火箭发射",),
                dropped_hit_count=0,
            )
        return [], SubtopicQueryAudit((), (), 0)

    monkeypatch.setattr(
        "xifeng_rag.subtopic_recall.nominate_subtopic_chunks", fake_nominate
    )
    with database.session() as session:
        universe = read_build_universe(build_path)
        assert "recall-chunk-1" in universe
        chunk_map = build_subtopic_chunk_map(session, units, universe)
        assert chunk_map.mapped_unit_count == 1
        from dataclasses import replace

        from xifeng_rag.subtopic_recall import SubtopicRecallContext

        context = SubtopicRecallContext(
            build_id="layered-article-index_test",
            build_path=suite_dir,
            units=units,
            chunk_map=chunk_map,
            corpus_snapshot="snapshot-test",
            embedding_metadata={},
        )
        context = replace(context, chunk_map=chunk_map)
        candidates, audit = search_with_subtopic_candidates(
            session,
            "火箭发射",
            top_k=2,
            embedding_provider=provider,
            bm25_candidate_k=5,
            dense_candidate_k=5,
            rrf_k=60,
            reranker=ListReranker(),
            rerank_candidate_k=5,
            context=context,
            build_path=build_path,
            subtopic_candidate_k=10,
            subtopic_weight=1.0,
        )
    assert audit.nominated_chunk_ids == ("recall-chunk-1",)
    subtopic_nominated = [item for item in candidates if item.chunk_id == "recall-chunk-1"]
    assert subtopic_nominated, candidates
    first = subtopic_nominated[0]
    assert SUBTOPIC_BM25_CHANNEL in first.source
    scores = first.retrieval_scores
    assert scores.subtopic_bm25 == pytest.approx(3.0)
    assert scores.subtopic_bm25_rank == 1
    assert scores.subtopic_dense is None


def test_build_subtopic_context_validates_embedding_metadata(
    database: Database, tmp_path: Path
) -> None:
    _add_subtopic_article(
        database,
        1,
        title="嵌入校验文章",
        topic_title="嵌入校验话题",
        paragraphs=["嵌入校验内容。"],
        subtopics=[
            {"title": "嵌入校验细话题", "start_unit": 1, "end_unit": 1, "summary": "嵌入。"}
        ],
    )
    units = _units(
        [("recall-version-1", "嵌入校验话题", "嵌入校验细话题", 1, 1, "嵌入校验")]
    )
    layer_dir = tmp_path / "subtopic-layer"
    layer_dir.mkdir()
    (layer_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile_version": "layered-article-discovery-v1",
                "layer": "subtopics",
                "corpus_snapshot": "snapshot-test",
                "unit_count": 1,
                "article_count": 1,
                "candidate_k": 20,
                "embedding": {"provider": "other", "model_id": "other-v1"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (layer_dir / "units.json").write_text(
        json.dumps(list(units.values()), ensure_ascii=False), encoding="utf-8"
    )
    build_dir = tmp_path / "article-indexes" / "layered-article-index_test"
    build_dir.mkdir(parents=True)
    with database.session() as session:
        session.add(
            ArticleIndexBuildRow(
                id="layered-article-index_test",
                profile_id="profile-test",
                corpus_snapshot="snapshot-test",
                path=str(layer_dir),
                status="ready",
                active=False,
            )
        )
    suite_path = tmp_path / "latest-layered-suite.json"
    suite_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "suite_id": "suite-test",
                "corpus_snapshot": "snapshot-test",
                "builds": {
                    layer.value: {"build_id": "layered-article-index_test"}
                    for layer in ArticleDiscoveryLayer
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    with database.session() as session, pytest.raises(ValueError, match="Embedding Profile"):
        build_subtopic_context(
            session,
            suite_path=suite_path,
            universe_chunk_ids=frozenset({"recall-chunk-1"}),
            embedding_metadata={"provider": "fake", "model_id": "fake-v1"},
        )


def _frozen_dataset_with_two_chunks(database: Database, tmp_path: Path) -> str:
    """沿用 curation 测试的轻量做法：直接落库一个可运行的冻结评测集。"""
    from xifeng_rag.evaluation_curation import (
        AnswerPoint,
        DatasetPurpose,
        EvidenceRange,
        QuestionDraft,
        ReviewedAnswer,
        create_dataset,
        freeze_dataset,
        save_question,
        save_reviewed_answer,
    )

    _add_subtopic_article(
        database,
        1,
        title="火箭文章",
        topic_title="火箭",
        paragraphs=["果园种植苹果。", "火箭发射卫星。"],
        subtopics=None,
    )
    version_id = "recall-version-1"
    import hashlib

    from xifeng_rag.storage import IndexBuildRow

    with database.session() as session:
        chunk_ids = [
            row.id
            for row in session.scalars(
                select(TopicChunkRow).order_by(TopicChunkRow.id)
            ).all()
        ]
        snapshot = hashlib.sha256("\n".join(chunk_ids).encode()).hexdigest()
        session.add(
            IndexBuildRow(
                id="index_recall_test",
                profile_id="profile-test",
                corpus_snapshot=snapshot,
                path=str(tmp_path / "index"),
                status="active",
                active=True,
            )
        )
    with database.session() as session:
        dataset = create_dataset(session, "细话题召回测试", "v0.1", DatasetPurpose.PILOT)
        dataset_id = dataset.id
        question = save_question(
            session,
            dataset_id,
            QuestionDraft(query="火箭发射了什么？", version_id=version_id),
        )
        save_reviewed_answer(
            session,
            question.id,
            ReviewedAnswer(
                reference_answer="发射了卫星。",
                answer_points=[
                    AnswerPoint(
                        claim_group="point-1",
                        claim="发射卫星",
                        evidence_ranges=[
                            EvidenceRange(start_paragraph=2, end_paragraph=2)
                        ],
                        uncertainty="",
                    )
                ],
                reviewed=True,
            ),
        )
        freeze_dataset(session, dataset_id)
    return dataset_id


def _details_of(database: Database) -> list[dict[str, object]]:
    from xifeng_rag.storage import EvaluationRunRow

    with database.session() as session:
        run = session.scalar(
            select(EvaluationRunRow).order_by(EvaluationRunRow.created_at.desc())
        )
        assert run is not None
        return json.loads(run.details_json)


def test_run_frozen_evaluation_without_subtopics_keeps_details_unchanged(
    database: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from xifeng_rag.domain import RetrievalCandidate, RetrievalMode, RetrievalScores
    from xifeng_rag.evaluation import run_frozen_evaluation

    dataset_id = _frozen_dataset_with_two_chunks(database, tmp_path)

    class FakeReranker:
        @property
        def metadata(self) -> dict[str, object]:
            return {"provider": "fake-reranker"}

    def fake_search(session, query, top_k, build_id=None, **kwargs):
        del session, kwargs
        assert query == "火箭发射了什么？"
        return [
            RetrievalCandidate(
                chunk_id="recall-chunk-1",
                article_id="recall-article-1",
                article_version="recall-version-1",
                article_title="自造文章",
                original_text="不写入运行详情的自造原文",
                paragraph_range=(1, 2),
                retrieval_scores=RetrievalScores(bm25=1.0, bm25_rank=1),
            )
        ]

    monkeypatch.setattr("xifeng_rag.evaluation.search", fake_search)
    with database.session() as session:
        outcome = run_frozen_evaluation(
            session,
            dataset_id,
            k=1,
            mode=RetrievalMode.HYBRID,
            reranker=FakeReranker(),  # type: ignore[arg-type]
        )
    details = _details_of(database)
    assert set(details[0].keys()) == {
        "question_id",
        "split",
        "ranked_chunk_ids",
        "retrieval_candidates",
        "mapped_relevant_chunk_ids",
        "precision_at_k",
        "recall_at_k",
        "mrr",
        "ndcg_at_k",
        "evidence_coverage_at_k",
        "expanded_evidence_coverage_at_k",
        "latency_ms",
        "should_abstain",
        "retrieval_mode",
        "layered",
        "query_rewrite",
        "layered_config",
        "subtopic_recall",
    }
    assert details[0]["subtopic_recall"] is None
    assert outcome["subtopics"] is None


def test_run_frozen_evaluation_subtopics_records_audit(
    database: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from xifeng_rag.domain import RetrievalCandidate, RetrievalMode, RetrievalScores
    from xifeng_rag.evaluation import run_frozen_evaluation
    from xifeng_rag.subtopic_recall import SubtopicQueryAudit, SubtopicRecallContext

    dataset_id = _frozen_dataset_with_two_chunks(database, tmp_path)

    class FakeReranker:
        @property
        def metadata(self) -> dict[str, object]:
            return {"provider": "fake-reranker"}

    def fake_subtopic_search(session, query, top_k, **kwargs):
        del session, kwargs
        candidates = [
            RetrievalCandidate(
                chunk_id="recall-chunk-1",
                article_id="recall-article-1",
                article_version="recall-version-1",
                article_title="自造文章",
                original_text="不写入运行详情的自造原文",
                paragraph_range=(1, 2),
                retrieval_scores=RetrievalScores(bm25=1.0, bm25_rank=1),
            )
        ]
        audit = SubtopicQueryAudit(
            nominated_chunk_ids=("recall-chunk-1",),
            nominated_subtopic_titles=("火箭发射",),
            dropped_hit_count=2,
        )
        return candidates, audit

    def fake_context(session, **kwargs):
        del session, kwargs
        from xifeng_rag.subtopic_recall import SubtopicChunkMap

        return SubtopicRecallContext(
            build_id="layered-article-index_test",
            build_path=tmp_path / "layer",
            units={},
            chunk_map=SubtopicChunkMap(
                unit_key_to_chunk={},
                universe_chunk_ids=frozenset({"recall-chunk-1"}),
                suite_unit_count=3,
                mapped_unit_count=2,
                dropped_unit_count=1,
                covered_chunk_ids=frozenset({"recall-chunk-1"}),
                covered_version_ids=frozenset({"recall-version-1"}),
                covered_article_count=1,
            ),
            corpus_snapshot="snapshot-test",
            embedding_metadata={},
        )

    def fake_universe(build_path):
        del build_path
        return frozenset({"recall-chunk-1"})

    monkeypatch.setattr(
        "xifeng_rag.evaluation.search_with_subtopic_candidates", fake_subtopic_search
    )
    monkeypatch.setattr("xifeng_rag.evaluation.build_subtopic_context", fake_context)
    monkeypatch.setattr("xifeng_rag.evaluation.read_build_universe", fake_universe)
    with database.session() as session:
        outcome = run_frozen_evaluation(
            session,
            dataset_id,
            k=1,
            build_id="index_recall_test",
            mode=RetrievalMode.HYBRID,
            embedding_provider=FakeEmbeddingProvider(),
            reranker=FakeReranker(),  # type: ignore[arg-type]
            subtopics=True,
            subtopic_suite=tmp_path / "latest-layered-suite.json",
        )
    subtopics_summary = outcome["subtopics"]
    assert isinstance(subtopics_summary, dict)
    assert subtopics_summary["build_id"] == "layered-article-index_test"
    details = _details_of(database)
    assert details[0]["subtopic_recall"] == {
        "nominated_chunk_ids": ["recall-chunk-1"],
        "nominated_subtopic_titles": ["火箭发射"],
        "dropped_hit_count": 2,
    }


def test_run_frozen_evaluation_subtopics_validates_combinations(
    database: Database, tmp_path: Path
) -> None:
    from xifeng_rag.domain import RetrievalMode
    from xifeng_rag.evaluation import run_frozen_evaluation

    dataset_id = _frozen_dataset_with_two_chunks(database, tmp_path)
    with database.session() as session:
        with pytest.raises(ValueError, match="Hybrid"):
            run_frozen_evaluation(session, dataset_id, k=1, subtopics=True)
        with pytest.raises(ValueError, match="Reranker"):
            run_frozen_evaluation(
                session,
                dataset_id,
                k=1,
                mode=RetrievalMode.HYBRID,
                build_id="index_recall_test",
                subtopics=True,
            )


def test_cli_eval_run_subtopics_validation(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from xifeng_rag.cli import app

    runner = CliRunner()
    config = tmp_path / "config.yaml"
    config.write_text(
        f"project_root: {tmp_path.as_posix()}\n"
        "corpus:\n"
        "  source_dir: corpus\n"
        "storage:\n"
        "  data_dir: data\n",
        encoding="utf-8",
    )

    def invoke(*args: str):
        return runner.invoke(
            app, ["--config", str(config), "eval", "run", "evalset-x", *args]
        )

    no_rerank = invoke("--subtopics", "--build", "index-x", "--mode", "hybrid")
    assert no_rerank.exit_code != 0
    assert "--subtopics 必须同时启用 --rerank" in no_rerank.output

    no_build = invoke("--subtopics", "--mode", "hybrid", "--rerank")
    assert no_build.exit_code != 0
    assert "--subtopics 必须同时用 --build 指定冻结 Build" in no_build.output

    with_layered = invoke(
        "--subtopics", "--layered", "--build", "index-x", "--mode", "hybrid", "--rerank"
    )
    assert with_layered.exit_code != 0
    assert "--subtopics 不能与 --layered 同时使用" in with_layered.output

    with_rewrite = invoke(
        "--subtopics", "--build", "index-x", "--mode", "hybrid", "--rerank", "--rewrite"
    )
    assert with_rewrite.exit_code != 0
    assert "--subtopics 第一版不能与 --rewrite 同时使用" in with_rewrite.output

    lone_suite = invoke("--subtopic-suite", "suite.json")
    assert lone_suite.exit_code != 0
    assert "--subtopic-suite 和 --subtopic-candidate-k 只能与 --subtopics 一起使用" in (
        lone_suite.output
    )
