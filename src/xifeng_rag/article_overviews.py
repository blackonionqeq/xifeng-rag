from __future__ import annotations

import json
from dataclasses import dataclass

from pydantic import ValidationError
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from xifeng_rag.corpus import sha256_text, stable_id
from xifeng_rag.domain import GeneratedArticleOverview, OverviewStatus, Topic, TopicProposal
from xifeng_rag.storage import (
    ApprovedManifestRow,
    ArticleOverviewRow,
    ArticleRow,
    ArticleVersionRow,
    ChunkProposalRow,
    utcnow,
)
from xifeng_rag.structured_generation import StructuredGenerator

PROMPT_VERSION = "article-overview-v2-concise-from-topic-metadata"
SCHEMA_VERSION = "article-overview-v1"

DEFAULT_INSTRUCTION = """请根据文章标题和完整的话题分块元数据，生成一段忠实的文章整体概括。

要求：
1. 优先说明核心判断，再交代主要论证路径和重要延伸主题。
2. 综合各话题之间的关系，不要逐块机械抄写或简单罗列。
3. 只能使用给定元数据，不补充外部事实，不复制或假装引用作者原话。
4. 保留对按主题反查文章有帮助的概念和近义表达。
5. 使用一段简洁中文，优选 80 至 140 字，最长不超过 200 字，最多 2 至 3 句。
6. 不使用“本文旨在”“文章指出”“整体框架揭示”等空泛套话，不重复结论。
7. 只返回 article_summary。"""


def article_overview_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {"article_summary": {"type": "string"}},
        "required": ["article_summary"],
    }


@dataclass(frozen=True)
class OverviewSource:
    version_id: str
    article_title: str
    source_kind: str
    source_id: str
    topics: list[Topic]


def overview_source(
    session: Session, version_id: str, proposal_id: str | None = None
) -> OverviewSource:
    version = session.get(ArticleVersionRow, version_id)
    if version is None or not version.active:
        raise ValueError("文章版本不存在或已停用")
    article = session.get(ArticleRow, version.article_id)
    if article is None or not article.active:
        raise ValueError("文章不存在或来源已缺失")
    if proposal_id is not None:
        proposal = session.get(ChunkProposalRow, proposal_id)
        if proposal is None or proposal.version_id != version.id:
            raise ValueError("分块提案不存在或不属于该文章版本")
        parsed = TopicProposal.model_validate_json(proposal.topics_json)
        return OverviewSource(version.id, article.title, "proposal", proposal.id, parsed.topics)
    manifest = session.scalar(
        select(ApprovedManifestRow)
        .where(ApprovedManifestRow.version_id == version.id)
        .order_by(ApprovedManifestRow.approved_at.desc(), ApprovedManifestRow.id.desc())
    )
    if manifest is None or not manifest.topics_json:
        raise ValueError("文章版本没有包含完整话题元数据的批准清单")
    parsed = TopicProposal.model_validate_json(manifest.topics_json)
    return OverviewSource(version.id, article.title, "manifest", manifest.id, parsed.topics)


def render_overview_prompt(source: OverviewSource) -> str:
    topics = [
        {
            "ordinal": index,
            "title": topic.title,
            "summary": topic.summary,
            "keywords": topic.keywords,
            "include_in_index": topic.include_in_index,
        }
        for index, topic in enumerate(source.topics, start=1)
    ]
    return (
        f"{DEFAULT_INSTRUCTION}\n\n文章标题：{source.article_title}\n\n"
        "完整话题元数据：\n"
        + json.dumps(topics, ensure_ascii=False, indent=2)
    )


def propose_article_overview(
    session: Session,
    version_id: str,
    generator: StructuredGenerator,
    model_id: str,
    *,
    provider_id: str = "ollama",
    proposal_id: str | None = None,
) -> ArticleOverviewRow:
    source = overview_source(session, version_id, proposal_id)
    identity = stable_id(
        "overview",
        source.version_id,
        source.source_kind,
        source.source_id,
        provider_id,
        model_id,
        PROMPT_VERSION,
        length=32,
    )
    existing = session.get(ArticleOverviewRow, identity)
    if existing is not None and existing.status in {
        OverviewStatus.PENDING_REVIEW.value,
        OverviewStatus.APPROVED.value,
    }:
        return existing
    row = existing or ArticleOverviewRow(
        id=identity,
        version_id=source.version_id,
        proposal_id=source.source_id if source.source_kind == "proposal" else None,
        manifest_id=source.source_id if source.source_kind == "manifest" else None,
        article_summary="生成中",
        status=OverviewStatus.GENERATING.value,
        provider_id=provider_id,
        model_id=model_id,
        prompt_version=PROMPT_VERSION,
    )
    session.add(row)
    try:
        result = generator.generate(render_overview_prompt(source), article_overview_schema())
        generated = GeneratedArticleOverview.model_validate_json(result.content)
    except (ValueError, ValidationError, RuntimeError) as exc:
        row.status = OverviewStatus.FAILED.value
        row.article_summary = "生成失败，请重新生成或人工填写"
        row.generation_json = json.dumps(
            {"error_type": type(exc).__name__}, ensure_ascii=False
        )
        return row
    row.article_summary = generated.article_summary
    row.status = OverviewStatus.PENDING_REVIEW.value
    row.generation_json = json.dumps(result.metadata, ensure_ascii=False)
    return row


def revise_pending_article_overview(
    session: Session,
    overview_id: str,
    generator: StructuredGenerator,
    model_id: str,
    *,
    provider_id: str,
) -> ArticleOverviewRow:
    """Generate a replacement draft while preserving the prior draft for audit."""
    previous = session.get(ArticleOverviewRow, overview_id)
    if previous is None:
        raise ValueError("文章概括不存在")
    if previous.status != OverviewStatus.PENDING_REVIEW.value:
        raise ValueError(f"只允许精简待审概括：{previous.status}")
    replacement = propose_article_overview(
        session,
        previous.version_id,
        generator,
        model_id,
        provider_id=provider_id,
        proposal_id=previous.proposal_id,
    )
    if replacement.status != OverviewStatus.PENDING_REVIEW.value:
        raise RuntimeError(f"精简概括生成失败：{replacement.status}")
    if replacement.id != previous.id:
        previous.status = OverviewStatus.SUPERSEDED.value
    return replacement


def approve_article_overview(
    session: Session, overview_id: str, article_summary: str | None = None
) -> ArticleOverviewRow:
    row = session.get(ArticleOverviewRow, overview_id)
    if row is None:
        raise ValueError("文章概括不存在")
    if row.status != OverviewStatus.PENDING_REVIEW.value:
        raise ValueError(f"文章概括状态不可批准：{row.status}")
    summary = (article_summary if article_summary is not None else row.article_summary).strip()
    GeneratedArticleOverview(article_summary=summary)
    session.execute(
        update(ArticleOverviewRow)
        .where(
            ArticleOverviewRow.version_id == row.version_id,
            ArticleOverviewRow.status == OverviewStatus.APPROVED.value,
            ArticleOverviewRow.id != row.id,
        )
        .values(status=OverviewStatus.SUPERSEDED.value)
    )
    row.article_summary = summary
    row.status = OverviewStatus.APPROVED.value
    row.approved_at = utcnow()
    return row


def overview_input_hash(source: OverviewSource) -> str:
    """Expose a stable audit key without storing a rendered model request."""
    return sha256_text(render_overview_prompt(source))
