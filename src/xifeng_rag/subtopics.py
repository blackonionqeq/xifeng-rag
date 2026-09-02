from __future__ import annotations

import json
from datetime import UTC, datetime

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from xifeng_rag.corpus import stable_id
from xifeng_rag.domain import (
    ArticleSubtopics,
    SubtopicSetStatus,
    TopicProposal,
    TopicSubtopics,
)
from xifeng_rag.storage import (
    ApprovedManifestRow,
    ArticleRow,
    ArticleVersionRow,
    ParagraphRow,
    SubtopicSetRow,
)
from xifeng_rag.structured_generation import StructuredGenerator

PROMPT_VERSION = "optional-subtopics-v2-parent-limit"
SCHEMA_VERSION = "optional-subtopics-v1"


def subtopic_schema() -> dict[str, object]:
    subtopic = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "start_unit": {"type": "integer"},
            "end_unit": {"type": "integer"},
            "summary": {"type": "string"},
        },
        "required": ["title", "start_unit", "end_unit", "summary"],
    }
    parent = {
        "type": "object",
        "properties": {
            "parent_start_unit": {"type": "integer"},
            "parent_end_unit": {"type": "integer"},
            "parent_title": {"type": "string"},
            "subtopics": {"type": "array", "items": subtopic},
        },
        "required": [
            "parent_start_unit",
            "parent_end_unit",
            "parent_title",
            "subtopics",
        ],
    }
    return {
        "type": "object",
        "properties": {"topics": {"type": "array", "items": parent}},
        "required": ["topics"],
    }


def _manifest_topics(session: Session, manifest_id: str) -> TopicProposal:
    manifest = session.get(ApprovedManifestRow, manifest_id)
    if manifest is None or not manifest.topics_json:
        raise ValueError("批准分块清单不存在或缺少话题数据")
    return TopicProposal.model_validate_json(manifest.topics_json)


def active_manifest(session: Session, version_id: str) -> ApprovedManifestRow:
    manifest = session.scalar(
        select(ApprovedManifestRow)
        .where(ApprovedManifestRow.version_id == version_id)
        .order_by(ApprovedManifestRow.approved_at.desc())
    )
    if manifest is None:
        raise ValueError("文章版本没有批准分块，不能补充细话题")
    return manifest


def validate_for_manifest(
    proposal: ArticleSubtopics,
    manifest_topics: TopicProposal,
) -> ArticleSubtopics:
    expected = {
        (topic.start_paragraph, topic.end_paragraph): topic
        for topic in manifest_topics.topics
        if topic.include_in_index
    }
    seen: set[tuple[int, int]] = set()
    normalized: list[TopicSubtopics] = []
    for parent in proposal.topics:
        key = (parent.parent_start_unit, parent.parent_end_unit)
        topic = expected.get(key)
        if topic is None:
            raise ValueError(f"细话题引用未知或不索引的父范围：{key[0]}–{key[1]}")
        if key in seen:
            raise ValueError(f"细话题父范围重复：{key[0]}–{key[1]}")
        seen.add(key)
        normalized.append(parent.model_copy(update={"parent_title": topic.title}, deep=True))
    return ArticleSubtopics(topics=normalized)


def generation_prompt(
    article_title: str,
    manifest_topics: TopicProposal,
    paragraphs: list[ParagraphRow],
) -> str:
    included = [topic for topic in manifest_topics.topics if topic.include_in_index]
    boundaries = "\n".join(
        f"- {topic.start_paragraph}–{topic.end_paragraph}：{topic.title}" for topic in included
    )
    allowed = {
        number
        for topic in included
        for number in range(topic.start_paragraph, topic.end_paragraph + 1)
    }
    numbered = "\n\n".join(
        f"[{item.number}] {item.text}" for item in paragraphs if item.number in allowed
    )
    return f"""为已批准分块的文章补充可选细话题。父话题边界已经人工批准，不得修改。

细话题是用户可能单独记住并用于反查文章的具体案例、局部概念、机制或特殊表述。每个父话题
最多返回 4 个细话题，也可以返回空 subtopics；整篇文章不设细话题总数上限。只保留检索辨识度
最高的候选，不要为了凑数逐段复述，也不要求覆盖父话题。相邻、范围重叠或表达同一机制的候选
必须合并为一个细话题。每个细话题只返回简短 title、精确 start_unit/end_unit 和一句忠实
summary，不复制长篇原文，不补充外部事实。每个范围必须完全位于对应父话题内。只处理下面
列出的父话题，并原样返回父范围和标题。

文章标题：{article_title}

已批准父话题：
{boundaries}

带编号单元：
{numbered}
"""


def generate_for_existing_article(
    session: Session,
    version_id: str,
    generator: StructuredGenerator,
    *,
    provider_id: str,
    model_id: str,
) -> SubtopicSetRow:
    version = session.get(ArticleVersionRow, version_id)
    if version is None or not version.active:
        raise ValueError("文章版本不存在或已停用")
    article = session.get(ArticleRow, version.article_id)
    if article is None or not article.active:
        raise ValueError("文章不存在或已停用")
    manifest = active_manifest(session, version_id)
    manifest_topics = _manifest_topics(session, manifest.id)
    paragraphs = list(
        session.scalars(
            select(ParagraphRow)
            .where(ParagraphRow.version_id == version_id)
            .order_by(ParagraphRow.number)
        ).all()
    )
    result = generator.generate(
        generation_prompt(article.title, manifest_topics, paragraphs), subtopic_schema()
    )
    parsed = ArticleSubtopics.model_validate_json(result.content)
    normalized = validate_for_manifest(parsed, manifest_topics)
    proposal_id = stable_id(
        "subtopics",
        version_id,
        manifest.id,
        provider_id,
        model_id,
        PROMPT_VERSION,
        datetime.now(UTC).isoformat(),
        length=40,
    )
    row = SubtopicSetRow(
        id=proposal_id,
        version_id=version_id,
        manifest_id=manifest.id,
        source_proposal_id=None,
        status=SubtopicSetStatus.PENDING_REVIEW.value,
        provider_id=provider_id,
        model_id=model_id,
        prompt_version=PROMPT_VERSION,
        schema_version=SCHEMA_VERSION,
        subtopics_json=normalized.model_dump_json(),
        generation_json=json.dumps(result.metadata, ensure_ascii=False),
    )
    session.add(row)
    session.flush()
    return row


def current_approved_subtopics(session: Session, version_id: str) -> SubtopicSetRow | None:
    return session.scalar(
        select(SubtopicSetRow)
        .where(
            SubtopicSetRow.version_id == version_id,
            SubtopicSetRow.status == SubtopicSetStatus.APPROVED.value,
        )
        .order_by(SubtopicSetRow.approved_at.desc())
    )


def pending_subtopics(session: Session, version_id: str) -> SubtopicSetRow | None:
    return session.scalar(
        select(SubtopicSetRow)
        .where(
            SubtopicSetRow.version_id == version_id,
            SubtopicSetRow.status == SubtopicSetStatus.PENDING_REVIEW.value,
        )
        .order_by(SubtopicSetRow.created_at.desc())
    )


def approve_subtopics(
    session: Session,
    set_id: str,
    payload: object,
) -> SubtopicSetRow:
    row = session.get(SubtopicSetRow, set_id)
    if row is None or row.status != SubtopicSetStatus.PENDING_REVIEW.value:
        raise ValueError("细话题草案不存在或已处理")
    parsed = ArticleSubtopics.model_validate(payload)
    normalized = validate_for_manifest(parsed, _manifest_topics(session, row.manifest_id))
    session.execute(
        update(SubtopicSetRow)
        .where(
            SubtopicSetRow.version_id == row.version_id,
            SubtopicSetRow.status == SubtopicSetStatus.APPROVED.value,
        )
        .values(status=SubtopicSetStatus.SUPERSEDED.value)
    )
    row.subtopics_json = normalized.model_dump_json()
    row.status = SubtopicSetStatus.APPROVED.value
    row.approved_at = datetime.now(UTC)
    session.flush()
    return row


def create_joint_approved_subtopics(
    session: Session,
    *,
    version_id: str,
    manifest_id: str,
    source_proposal_id: str,
    provider_id: str,
    model_id: str,
    proposal: TopicProposal,
) -> SubtopicSetRow | None:
    parents = [
        TopicSubtopics(
            parent_start_unit=topic.start_paragraph,
            parent_end_unit=topic.end_paragraph,
            parent_title=topic.title,
            subtopics=topic.subtopics,
        )
        for topic in proposal.topics
        if topic.include_in_index and topic.subtopics
    ]
    if not parents:
        return None
    row_id = stable_id("joint-subtopics", manifest_id, source_proposal_id, length=40)
    existing = session.get(SubtopicSetRow, row_id)
    if existing is not None:
        return existing
    row = SubtopicSetRow(
        id=row_id,
        version_id=version_id,
        manifest_id=manifest_id,
        source_proposal_id=source_proposal_id,
        status=SubtopicSetStatus.APPROVED.value,
        provider_id=provider_id,
        model_id=model_id,
        prompt_version=PROMPT_VERSION,
        schema_version=SCHEMA_VERSION,
        subtopics_json=ArticleSubtopics(topics=parents).model_dump_json(),
        approved_at=datetime.now(UTC),
    )
    session.add(row)
    session.flush()
    return row
