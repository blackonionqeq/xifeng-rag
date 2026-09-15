from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from xifeng_rag.chunking import ProposalOutcome, propose_chunks
from xifeng_rag.domain import VersionStatus
from xifeng_rag.storage import ArticleRow, ArticleVersionRow, Database, ParagraphRow
from xifeng_rag.structured_generation import StructuredGenerator


@dataclass(frozen=True)
class ChunkRetryTarget:
    article_id: str
    article_title: str
    version_id: str
    character_count: int
    unit_count: int


def chunk_retry_target(
    session: Session,
    article_id: str,
    *,
    expected_version_id: str | None = None,
) -> ChunkRetryTarget:
    article = session.get(ArticleRow, article_id)
    if article is None or not article.active:
        raise ValueError("文章不存在或来源已缺失")
    version = session.scalar(
        select(ArticleVersionRow)
        .where(
            ArticleVersionRow.article_id == article_id,
            ArticleVersionRow.active.is_(True),
        )
        .order_by(ArticleVersionRow.created_at.desc(), ArticleVersionRow.id.desc())
    )
    if version is None:
        raise ValueError("指定文章没有 active 版本")
    if expected_version_id is not None and version.id != expected_version_id:
        raise ValueError("文章最新版本已变化，请刷新管理页后重试")
    if version.status not in {
        VersionStatus.PENDING_CHUNKING.value,
        VersionStatus.PENDING_REVIEW.value,
    }:
        raise ValueError("指定文章没有可重试的 active 待分块版本")
    unit_count = session.scalar(
        select(func.count()).select_from(ParagraphRow).where(ParagraphRow.version_id == version.id)
    )
    return ChunkRetryTarget(
        article_id=article.id,
        article_title=article.title,
        version_id=version.id,
        character_count=len(version.canonical_text),
        unit_count=unit_count or 0,
    )


def run_codex_chunk_retry(
    database: Database,
    article_id: str,
    expected_version_id: str,
    generator_factory: Callable[[str], StructuredGenerator],
    model: str,
    max_retries: int,
    *,
    custom_prompt: str | None = None,
) -> ProposalOutcome:
    with database.session() as session:
        chunk_retry_target(
            session,
            article_id,
            expected_version_id=expected_version_id,
        )
    generator = generator_factory(model)
    with database.session() as session:
        target = chunk_retry_target(
            session,
            article_id,
            expected_version_id=expected_version_id,
        )
        version = session.get(ArticleVersionRow, target.version_id)
        if version is None:
            raise RuntimeError(f"待分块版本已消失：{target.version_id}")
        return propose_chunks(
            session,
            version,
            generator,
            model,
            max_retries,
            provider_id="codex-cli",
            custom_prompt=custom_prompt,
            retry_id=uuid4().hex,
        )
