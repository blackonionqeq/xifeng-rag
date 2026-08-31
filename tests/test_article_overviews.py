from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import select

from xifeng_rag.article_overviews import (
    approve_article_overview,
    overview_source,
    propose_article_overview,
    render_overview_prompt,
    revise_pending_article_overview,
)
from xifeng_rag.corpus import scan_corpus
from xifeng_rag.domain import OverviewStatus, ProposalStatus, VersionStatus
from xifeng_rag.storage import (
    ApprovedManifestRow,
    ArticleOverviewRow,
    ArticleVersionRow,
    ChunkProposalRow,
    Database,
)
from xifeng_rag.structured_generation import GenerationResult


class FakeGenerator:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str, schema: dict[str, object]) -> GenerationResult:
        self.prompts.append(prompt)
        assert schema["required"] == ["article_summary"]
        return GenerationResult(
            content=json.dumps(
                {"article_summary": "文章从两个话题分析自造问题，并给出整体结论。"},
                ensure_ascii=False,
            ),
            metadata={"eval_count": 8},
        )


def _approved_version(database: Database, tmp_path: Path) -> str:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "article.md").write_text("# 自造文章\n\n第一段。\n\n第二段。", encoding="utf-8")
    with database.session() as session:
        scan_corpus(session, corpus)
        version = session.scalar(select(ArticleVersionRow))
        assert version is not None
        version.status = VersionStatus.INDEXED.value
        proposal = ChunkProposalRow(
            id="proposal-overview",
            version_id=version.id,
            status=ProposalStatus.APPROVED.value,
            provider_id="test",
            model_id="test",
            prompt_version="test",
            schema_version="test",
            topics_json="{}",
        )
        session.add(proposal)
        session.flush()
        topics = {
            "topics": [
                {
                    "start_paragraph": 1,
                    "end_paragraph": 1,
                    "title": "导言",
                    "summary": "提出自造问题",
                    "keywords": ["问题"],
                    "include_in_index": False,
                },
                {
                    "start_paragraph": 2,
                    "end_paragraph": 3,
                    "title": "分析",
                    "summary": "分析问题并给出结论",
                    "keywords": ["分析", "结论"],
                    "include_in_index": True,
                },
            ]
        }
        proposal.topics_json = json.dumps(topics, ensure_ascii=False)
        session.add(
            ApprovedManifestRow(
                id="manifest-overview",
                proposal_id=proposal.id,
                version_id=version.id,
                topics_json=json.dumps(topics, ensure_ascii=False),
            )
        )
        return version.id


def test_overview_uses_all_manifest_topics_without_original_text(
    database: Database, tmp_path: Path
) -> None:
    version_id = _approved_version(database, tmp_path)
    generator = FakeGenerator()
    with database.session() as session:
        source = overview_source(session, version_id)
        assert [topic.include_in_index for topic in source.topics] == [False, True]
        prompt = render_overview_prompt(source)
        assert "提出自造问题" in prompt
        assert "include_in_index" in prompt
        assert "第一段" not in prompt
        assert "优选 80 至 140 字，最长不超过 200 字" in prompt
        assert "本文旨在" in prompt
        row = propose_article_overview(session, version_id, generator, "fake-model")
        overview_id = row.id
        assert row.status == OverviewStatus.PENDING_REVIEW.value
        assert row.generation_json == '{"eval_count": 8}'

    with database.session() as session:
        approved = approve_article_overview(session, overview_id, "人工修改后的整体概括。")
        assert approved.status == OverviewStatus.APPROVED.value
        assert approved.article_summary == "人工修改后的整体概括。"
        assert approved.approved_at is not None
        stored = session.scalar(
            select(ArticleOverviewRow).where(ArticleOverviewRow.id == overview_id)
        )
        assert stored is not None


def test_revise_pending_overview_preserves_history_and_rejects_approved(
    database: Database, tmp_path: Path
) -> None:
    version_id = _approved_version(database, tmp_path)
    generator = FakeGenerator()
    with database.session() as session:
        previous = propose_article_overview(session, version_id, generator, "local-model")
        previous_id = previous.id

    with database.session() as session:
        replacement = revise_pending_article_overview(
            session,
            previous_id,
            generator,
            "deepseek-v4-flash",
            provider_id="deepseek",
        )
        replacement_id = replacement.id
        assert replacement_id != previous_id
        assert replacement.provider_id == "deepseek"
        assert replacement.status == OverviewStatus.PENDING_REVIEW.value
        prior = session.get(ArticleOverviewRow, previous_id)
        assert prior is not None
        assert prior.status == OverviewStatus.SUPERSEDED.value

    with database.session() as session:
        approve_article_overview(session, replacement_id)
        try:
            revise_pending_article_overview(
                session,
                replacement_id,
                generator,
                "deepseek-v4-flash",
                provider_id="deepseek",
            )
        except ValueError as exc:
            assert "只允许精简待审概括" in str(exc)
        else:
            raise AssertionError("已批准概括不应允许批量精简")
