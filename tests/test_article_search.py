from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from xifeng_rag.article_search import (
    build_article_index,
    search_articles,
    selected_article_documents,
)
from xifeng_rag.domain import OverviewStatus, RetrievalMode, VersionStatus
from xifeng_rag.storage import (
    ApprovedManifestRow,
    ArticleIndexBuildRow,
    ArticleOverviewRow,
    ArticleRow,
    ArticleVersionRow,
    ChunkProposalRow,
    Database,
)


class ThemeEmbeddingProvider:
    dimension = 2

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "provider": "fake",
            "model_id": "theme-fake-v1",
            "model_revision": "test",
            "dimension": 2,
            "max_seq_length": 512,
            "normalize_embeddings": True,
            "query_instruction": "test",
        }

    @staticmethod
    def _vector(text: str) -> list[float]:
        if any(word in text for word in ("社会阶段", "阶层", "时代位置")):
            return [1.0, 0.0]
        if any(word in text for word in ("做事", "做人", "商业全局", "系统协作")):
            return [0.0, 1.0]
        return [0.5, 0.5]

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return np.asarray([self._vector(text) for text in texts], dtype=np.float32)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return np.asarray([self._vector(text) for text in texts], dtype=np.float32)

    def close(self) -> None:
        pass


def _add_article(
    database: Database,
    ordinal: int,
    *,
    title: str,
    article_summary: str,
    topic_titles: list[str],
    keywords: list[str],
    overview_status: OverviewStatus = OverviewStatus.APPROVED,
) -> str:
    article_id = f"article-{ordinal}"
    version_id = f"version-{ordinal}"
    proposal_id = f"proposal-{ordinal}"
    manifest_id = f"manifest-{ordinal}"
    topics = {
        "topics": [
            {
                "start_paragraph": index,
                "end_paragraph": index,
                "title": topic_title,
                "summary": f"不应写入文章级索引的块级概括 {index}",
                "keywords": keywords if index == 1 else [f"辅助词{index}"],
                "include_in_index": True,
            }
            for index, topic_title in enumerate(topic_titles, start=1)
        ]
    }
    with database.session() as session:
        article = ArticleRow(
            id=article_id,
            source_path=f"article-{ordinal}.md",
            title=title,
        )
        session.add(article)
        session.flush()
        version = ArticleVersionRow(
            id=version_id,
            article_id=article_id,
            source_hash=str(ordinal) * 64,
            canonical_hash=str(ordinal) * 64,
            cleaner_version="test",
            canonical_text="自造测试正文",
            status=VersionStatus.INDEXED.value,
        )
        session.add(version)
        session.flush()
        proposal = ChunkProposalRow(
            id=proposal_id,
            version_id=version_id,
            status="approved",
            provider_id="test",
            model_id="test",
            prompt_version="test",
            schema_version="test",
            topics_json=json.dumps(topics, ensure_ascii=False),
        )
        session.add(proposal)
        session.flush()
        manifest = ApprovedManifestRow(
            id=manifest_id,
            proposal_id=proposal_id,
            version_id=version_id,
            topics_json=json.dumps(topics, ensure_ascii=False),
        )
        session.add(manifest)
        session.flush()
        session.add(
            ArticleOverviewRow(
                id=f"overview-{ordinal}",
                version_id=version_id,
                manifest_id=manifest_id,
                article_summary=article_summary,
                status=overview_status.value,
                provider_id="test",
                model_id="test",
                prompt_version="test",
            )
        )
    return article_id


def _article_fixture(database: Database) -> tuple[str, str]:
    stage_id = _add_article(
        database,
        1,
        title="时代位置与个人选择",
        article_summary="讨论社会阶段如何改变个人处境与职业选择。",
        topic_titles=["生产组织的变化", "阶段与个人选择"],
        keywords=["社会阶段", "阶层", "组织变迁"],
    )
    system_id = _add_article(
        database,
        2,
        title="技术工作与系统协作",
        article_summary="分析只关注技术执行为何容易忽略商业全局。",
        topic_titles=["岗位与系统", "承担风险"],
        keywords=["做事", "做人", "商业模式"],
    )
    _add_article(
        database,
        3,
        title="不应进入正式索引的草稿",
        article_summary="这是一条尚未批准的文章概括。",
        topic_titles=["待审主题"],
        keywords=["待审"],
        overview_status=OverviewStatus.PENDING_REVIEW,
    )
    return stage_id, system_id


def test_article_document_is_concise_and_requires_approved_overview(
    database: Database,
) -> None:
    _article_fixture(database)
    with database.session() as session:
        documents = selected_article_documents(session)
    assert len(documents) == 2
    stage = next(item for item in documents if item.article_id == "article-1")
    assert "不应写入文章级索引的块级概括" not in stage.search_text
    assert "社会阶段" not in stage.supplemental_keywords
    assert "阶层" in stage.supplemental_keywords
    assert "组织变迁" in stage.supplemental_keywords


def test_two_article_discovery_queries_rank_expected_article_first(
    database: Database, tmp_path: Path
) -> None:
    stage_id, system_id = _article_fixture(database)
    provider = ThemeEmbeddingProvider()
    with database.session() as session:
        outcome = build_article_index(session, tmp_path / "article-indexes", provider)
        assert outcome.article_count == 2
        build = session.get(ArticleIndexBuildRow, outcome.build_id)
        assert build is not None
        assert build.active is True

    cases = [
        ("哪篇文章讨论社会阶段而不是静态阶层", stage_id),
        ("作者在哪篇文章谈做事的人和商业全局", system_id),
    ]
    with database.session() as session:
        for query, expected_article_id in cases:
            results = search_articles(
                session,
                query,
                2,
                mode=RetrievalMode.HYBRID,
                embedding_provider=provider,
            )
            assert results[0].article_id == expected_article_id
            assert results[0].sources == ["bm25", "dense"]
            assert results[0].rrf_score is not None
