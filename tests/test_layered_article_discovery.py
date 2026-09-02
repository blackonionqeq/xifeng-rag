from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from xifeng_rag.article_evaluation import ArticleDiscoveryDataset
from xifeng_rag.domain import (
    OverviewStatus,
    RetrievalMode,
    SubtopicSetStatus,
    VersionStatus,
)
from xifeng_rag.layered_article_evaluation import (
    FallbackPolicy,
    compare_layered_article_evaluations,
    run_layered_article_evaluation,
    save_layered_article_evaluation,
    simulate_layered_fallback,
    validate_layered_comparison,
)
from xifeng_rag.layered_article_search import (
    ArticleAggregation,
    ArticleDiscoveryLayer,
    UnitSearchHit,
    aggregate_unit_hits,
    build_layered_article_indexes,
    load_layered_suite,
    search_article_layer,
    selected_layer_units,
)
from xifeng_rag.storage import (
    ApprovedManifestRow,
    ArticleOverviewRow,
    ArticleRow,
    ArticleVersionRow,
    ChunkProposalRow,
    Database,
    ParagraphRow,
    SubtopicSetRow,
    TopicChunkRow,
)


class LayerEmbeddingProvider:
    dimension = 4

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "provider": "test",
            "model_id": "layer-test",
            "model_revision": "v1",
            "dimension": self.dimension,
            "normalize_embeddings": True,
        }

    @staticmethod
    def _vector(text: str) -> list[float]:
        values = [
            float(text.count("阶段") + text.count("参照")),
            float(text.count("商业") + text.count("协作")),
            float(text.count("案例")),
            1.0,
        ]
        vector = np.asarray(values, dtype=np.float32)
        return (vector / np.linalg.norm(vector)).tolist()

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return np.asarray([self._vector(text) for text in texts], dtype=np.float32)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return np.asarray([self._vector(text) for text in texts], dtype=np.float32)

    def close(self) -> None:
        pass


def _add_layered_article(
    database: Database,
    ordinal: int,
    *,
    title: str,
    summary: str,
    topic_title: str,
    keyword: str,
    paragraphs: list[str],
    subtopics: list[dict[str, object]],
) -> str:
    article_id = f"layer-article-{ordinal}"
    version_id = f"layer-version-{ordinal}"
    proposal_id = f"layer-proposal-{ordinal}"
    manifest_id = f"layer-manifest-{ordinal}"
    topics = {
        "topics": [
            {
                "start_paragraph": 1,
                "end_paragraph": len(paragraphs),
                "title": topic_title,
                "summary": f"块级秘密概括 {ordinal}",
                "keywords": [keyword],
                "include_in_index": True,
            }
        ]
    }
    with database.session() as session:
        session.add(
            ArticleRow(id=article_id, source_path=f"layer-{ordinal}.md", title=title)
        )
        session.flush()
        session.add(
            ArticleVersionRow(
                id=version_id,
                article_id=article_id,
                source_hash=str(ordinal) * 64,
                canonical_hash=str(ordinal) * 64,
                cleaner_version="test",
                canonical_text="\n\n".join(paragraphs),
                status=VersionStatus.INDEXED.value,
            )
        )
        session.flush()
        for number, text in enumerate(paragraphs, start=1):
            session.add(
                ParagraphRow(
                    version_id=version_id,
                    number=number,
                    text=text,
                    source_start=number - 1,
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
                id=f"layer-chunk-{ordinal}",
                manifest_id=manifest_id,
                version_id=version_id,
                article_id=article_id,
                start_paragraph=1,
                end_paragraph=len(paragraphs),
                topic_title=topic_title,
                context_summary=f"块级秘密概括 {ordinal}",
                keywords_json=json.dumps([keyword], ensure_ascii=False),
                original_text="\n\n".join(paragraphs),
            )
        )
        session.add(
            ArticleOverviewRow(
                id=f"layer-overview-{ordinal}",
                version_id=version_id,
                manifest_id=manifest_id,
                article_summary=summary,
                status=OverviewStatus.APPROVED.value,
                provider_id="test",
                model_id="test",
                prompt_version="test",
            )
        )
        if subtopics:
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
                    id=f"layer-subtopics-{ordinal}",
                    version_id=version_id,
                    manifest_id=manifest_id,
                    status=SubtopicSetStatus.APPROVED.value,
                    provider_id="test",
                    model_id="test",
                    prompt_version="test",
                    schema_version="test",
                    subtopics_json=json.dumps(payload, ensure_ascii=False),
                )
            )
    return article_id


@pytest.fixture
def layered_articles(database: Database) -> tuple[str, str]:
    first = _add_layered_article(
        database,
        1,
        title="时代位置",
        summary="概括秘密：社会阶段改变个人参照。",
        topic_title="阶段判断",
        keyword="参照系",
        paragraphs=["阶段原文案例甲很长很长。", "阶段原文案例乙也很长很长。"],
        subtopics=[
            {
                "title": "参照案例甲",
                "start_unit": 1,
                "end_unit": 1,
                "summary": "细话题秘密甲。",
            },
            {
                "title": "参照案例乙",
                "start_unit": 2,
                "end_unit": 2,
                "summary": "细话题秘密乙。",
            },
        ],
    )
    second = _add_layered_article(
        database,
        2,
        title="系统协作",
        summary="概括秘密：技术工作需要理解商业协作。",
        topic_title="商业全局",
        keyword="协作关系",
        paragraphs=["商业协作原文案例很长很长。"],
        subtopics=[],
    )
    return first, second


def test_selected_units_keep_four_layers_isolated_and_track_ranges(
    database: Database, layered_articles: tuple[str, str]
) -> None:
    first, second = layered_articles
    with database.session() as session:
        layers = selected_layer_units(
            session, target_chars=5, max_chars=20, tail_merge_chars=0
        )

    overview = next(
        item
        for item in layers[ArticleDiscoveryLayer.OVERVIEW]
        if item.article_id == first
    )
    keywords = next(
        item
        for item in layers[ArticleDiscoveryLayer.KEYWORDS]
        if item.article_id == first
    )
    assert "概括秘密" in overview.search_text
    assert "阶段判断" not in overview.search_text
    assert "阶段判断" in keywords.search_text
    assert "参照系" in keywords.search_text
    assert "概括秘密" not in keywords.search_text
    assert "细话题秘密" not in keywords.search_text
    assert len(layers[ArticleDiscoveryLayer.SUBTOPICS]) == 2
    assert {item.article_id for item in layers[ArticleDiscoveryLayer.SUBTOPICS]} == {first}
    assert second not in {item.article_id for item in layers[ArticleDiscoveryLayer.SUBTOPICS]}
    originals = [
        item
        for item in layers[ArticleDiscoveryLayer.ORIGINAL]
        if item.article_id == first
    ]
    assert [(item.start_unit, item.end_unit) for item in originals] == [(1, 1), (2, 2)]
    assert all("原文" in item.search_text for item in originals)
    assert all("概括秘密" not in item.search_text for item in originals)


def test_top_n_rank_fusion_caps_long_article_contribution() -> None:
    units = {
        **{
            f"a-{number}": {
                "unit_id": f"a-{number}",
                "article_id": "a",
                "version_id": "va",
                "title": "长文章",
                "parent_topic": None,
                "start_unit": number,
                "end_unit": number,
            }
            for number in range(1, 6)
        },
        "b-1": {
            "unit_id": "b-1",
            "article_id": "b",
            "version_id": "vb",
            "title": "短文章",
            "parent_topic": None,
            "start_unit": 1,
            "end_unit": 1,
        },
    }
    hits = [
        UnitSearchHit(units[unit_id], 1.0 / rank, rank)
        for rank, unit_id in enumerate(["a-1", "a-2", "b-1", "a-3", "a-4", "a-5"], start=1)
    ]
    fused = aggregate_unit_hits(
        hits,
        units,
        aggregation=ArticleAggregation.TOP_N_RANK_FUSION,
        top_n=3,
        rrf_k=60,
        top_k=2,
    )
    long_article = next(item for item in fused if item.article_id == "a")
    assert long_article.unit_count == 5
    assert [item[0] for item in long_article.contributing_units] == [
        "a-1",
        "a-2",
        "a-3",
    ]


def test_layered_build_evaluation_is_safe_and_supports_offline_fallback(
    database: Database,
    tmp_path: Path,
    layered_articles: tuple[str, str],
) -> None:
    first, second = layered_articles
    provider = LayerEmbeddingProvider()
    with database.session() as session:
        suite = build_layered_article_indexes(
            session,
            tmp_path / "article-indexes",
            provider,
            candidate_k=20,
            target_chars=5,
            max_chars=20,
            tail_merge_chars=0,
        )
    assert len(suite.builds) == 4
    assert len({item.profile_id for item in suite.builds.values()}) == 4
    assert all(
        "秘密" not in item.path.joinpath("units.json").read_text(encoding="utf-8")
        for item in suite.builds.values()
    )
    build_ids = load_layered_suite(suite.manifest_path)

    with database.session() as session:
        hits, _ = search_article_layer(
            session,
            build_ids[ArticleDiscoveryLayer.ORIGINAL],
            ArticleDiscoveryLayer.ORIGINAL,
            "阶段原文案例乙",
            mode=RetrievalMode.BM25,
            candidate_k=20,
            top_k=2,
        )
    assert hits[0].article_id == first
    assert (hits[0].start_unit, hits[0].end_unit) == (2, 2)

    dataset = ArticleDiscoveryDataset.model_validate(
        {
            "schema_version": 1,
            "queries": [
                {
                    "id": "layer-q1",
                    "query": "阶段参照案例",
                    "expected_article_id": first,
                    "review_status": "confirmed",
                },
                {
                    "id": "layer-q2",
                    "query": "商业协作",
                    "expected_article_id": second,
                    "review_status": "confirmed",
                },
            ],
        }
    )
    with database.session() as session:
        result = run_layered_article_evaluation(
            session,
            dataset,
            tmp_path / "dataset.json",
            provider,
            build_ids,
            candidate_k=20,
        )
    assert len(result.channels) == 12
    subtopic_channels = [
        item for item in result.channels if item.layer == ArticleDiscoveryLayer.SUBTOPICS
    ]
    assert all(item.metrics.unavailable_question_count == 1 for item in subtopic_channels)
    assert all(
        question.bm25_dense_top1_agree is not None
        for channel in result.channels
        for question in channel.questions
        if question.status == "evaluated"
    )
    saved = save_layered_article_evaluation(result, tmp_path / "runs")
    payload = saved.read_text(encoding="utf-8")
    assert "概括秘密" not in payload
    assert "细话题秘密" not in payload
    assert "阶段原文" not in payload

    fallback = simulate_layered_fallback(
        result,
        policy=FallbackPolicy.MARGIN_CASCADE,
        margin_ratio_threshold=1.0,
    )
    assert fallback.source_run_id == result.run_id
    assert fallback.layer_trigger_rate["overview"] == 1.0
    assert fallback.latency_max_ms >= fallback.latency_mean_ms

    incompatible = result.model_copy(update={"candidate_k": 99}, deep=True)
    with pytest.raises(ValueError, match="候选深度"):
        validate_layered_comparison(result, incompatible)

    current = result.model_copy(update={"run_id": "new-run"}, deep=True)
    current.channels[0].metrics.mrr = 0.0
    comparison = compare_layered_article_evaluations(result, current)
    baseline_mrr = result.channels[0].metrics.mrr
    assert baseline_mrr is not None
    assert comparison.channels[0].metrics["mrr"].delta == -baseline_mrr
