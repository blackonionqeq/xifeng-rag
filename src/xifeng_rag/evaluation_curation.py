from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from xifeng_rag.corpus import stable_id
from xifeng_rag.storage import (
    ArticleRow,
    ArticleVersionRow,
    EvaluationAIProposalRow,
    EvaluationDatasetRow,
    EvaluationEvidenceRow,
    EvaluationQuestionRow,
    EvaluationSnapshotArticleRow,
    IndexBuildRow,
    ParagraphRow,
    TopicChunkRow,
    utcnow,
)
from xifeng_rag.structured_generation import StructuredGenerator

PROMPT_VERSION = "evaluation-evidence-v1"
SCHEMA_VERSION = "evaluation-candidate-v1"


class DatasetPurpose(StrEnum):
    PILOT = "pilot"
    BENCHMARK = "benchmark"


class DatasetStatus(StrEnum):
    DRAFT = "draft"
    FROZEN = "frozen"


class ReviewStatus(StrEnum):
    DRAFT = "draft"
    REVIEWED = "reviewed"


class AIProposalStatus(StrEnum):
    PENDING_REVIEW = "pending_review"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class EvidenceRange(BaseModel):
    start_paragraph: int = Field(ge=1)
    end_paragraph: int = Field(ge=1)

    @model_validator(mode="after")
    def ordered(self) -> EvidenceRange:
        if self.end_paragraph < self.start_paragraph:
            raise ValueError("证据结束单元不能早于开始单元")
        return self


class AnswerPoint(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    claim_group: str = Field(min_length=1, max_length=80)
    claim: str = Field(min_length=1, max_length=2000)
    evidence_ranges: list[EvidenceRange] = Field(min_length=1)
    uncertainty: str = Field(max_length=1000)


class CandidateAnswer(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    candidate_answer: str = Field(min_length=1, max_length=10000)
    answer_points: list[AnswerPoint] = Field(min_length=1, max_length=30)


class QuestionDraft(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=2000)
    version_id: str = Field(min_length=1)
    split: str = Field(default="dev", pattern="^(dev|holdout)$")


class ReviewedAnswer(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    reference_answer: str = Field(min_length=1, max_length=10000)
    answer_points: list[AnswerPoint] = Field(min_length=1, max_length=30)
    reviewed: bool = False


@dataclass(frozen=True)
class FreezeOutcome:
    dataset_id: str
    content_hash: str
    corpus_snapshot: str
    article_count: int
    question_count: int


def create_dataset(
    session: Session, name: str, version: str, purpose: DatasetPurpose = DatasetPurpose.PILOT
) -> EvaluationDatasetRow:
    clean_name = name.strip()
    clean_version = version.strip()
    if not clean_name or not clean_version:
        raise ValueError("评测集名称和版本不能为空")
    dataset_id = stable_id("evalset", clean_name, clean_version, purpose.value, length=32)
    if session.get(EvaluationDatasetRow, dataset_id) is not None:
        raise ValueError("相同名称、版本和用途的评测集已存在")
    row = EvaluationDatasetRow(
        id=dataset_id,
        name=clean_name,
        version=clean_version,
        purpose=purpose.value,
        status=DatasetStatus.DRAFT.value,
    )
    session.add(row)
    session.flush()
    return row


def indexed_articles(session: Session) -> list[dict[str, object]]:
    rows = session.execute(
        select(ArticleRow, ArticleVersionRow, func.count(TopicChunkRow.id))
        .join(ArticleVersionRow, ArticleVersionRow.article_id == ArticleRow.id)
        .join(TopicChunkRow, TopicChunkRow.version_id == ArticleVersionRow.id)
        .where(ArticleRow.active.is_(True), ArticleVersionRow.status == "indexed")
        .group_by(ArticleRow.id, ArticleVersionRow.id)
        .order_by(ArticleRow.title)
    ).all()
    return [
        {
            "article_id": article.id,
            "title": article.title,
            "version_id": version.id,
            "canonical_hash": version.canonical_hash,
            "character_count": len(version.canonical_text),
            "unit_count": session.scalar(
                select(func.count())
                .select_from(ParagraphRow)
                .where(ParagraphRow.version_id == version.id)
            )
            or 0,
            "chunk_count": chunk_count,
        }
        for article, version, chunk_count in rows
    ]


def _require_draft_dataset(session: Session, dataset_id: str) -> EvaluationDatasetRow:
    dataset = session.get(EvaluationDatasetRow, dataset_id)
    if dataset is None:
        raise ValueError("评测集不存在")
    if dataset.status != DatasetStatus.DRAFT.value:
        raise ValueError("冻结评测集不可修改")
    return dataset


def save_question(
    session: Session,
    dataset_id: str,
    value: QuestionDraft,
    question_id: str | None = None,
) -> EvaluationQuestionRow:
    dataset = _require_draft_dataset(session, dataset_id)
    version = session.get(ArticleVersionRow, value.version_id)
    if version is None or version.status != "indexed":
        raise ValueError("只能选择当前已索引文章版本")
    if session.get(ArticleRow, version.article_id) is None:
        raise ValueError("文章不存在")
    if not session.scalar(
        select(func.count())
        .select_from(TopicChunkRow)
        .where(TopicChunkRow.version_id == version.id)
    ):
        raise ValueError("文章版本没有可检索话题块")
    if dataset.purpose == DatasetPurpose.PILOT.value and value.split != "dev":
        raise ValueError("pilot 数据集只允许 dev 问题")
    if question_id is None:
        ordinal = (
            session.scalar(
                select(func.max(EvaluationQuestionRow.ordinal)).where(
                    EvaluationQuestionRow.dataset_id == dataset_id
                )
            )
            or 0
        ) + 1
        question_id = stable_id("evalq", dataset_id, str(ordinal), value.query, length=32)
        row = EvaluationQuestionRow(
            id=question_id,
            dataset_id=dataset_id,
            ordinal=ordinal,
            query=value.query,
            split=value.split,
            scope="article",
            article_id=version.article_id,
            version_id=version.id,
            answerable=True,
            review_status=ReviewStatus.DRAFT.value,
        )
        session.add(row)
    else:
        row = session.get(EvaluationQuestionRow, question_id)
        if row is None or row.dataset_id != dataset_id:
            raise ValueError("问题不存在或不属于当前评测集")
        row.query = value.query
        row.split = value.split
        row.article_id = version.article_id
        row.version_id = version.id
        row.reference_answer = None
        row.review_status = ReviewStatus.DRAFT.value
        session.execute(
            delete(EvaluationEvidenceRow).where(EvaluationEvidenceRow.question_id == row.id)
        )
    row.updated_at = utcnow()
    dataset.updated_at = utcnow()
    session.flush()
    return row


def question_document(session: Session, question_id: str) -> dict[str, object]:
    question = session.get(EvaluationQuestionRow, question_id)
    if question is None:
        raise ValueError("问题不存在")
    dataset = session.get(EvaluationDatasetRow, question.dataset_id)
    article = session.get(ArticleRow, question.article_id)
    version = session.get(ArticleVersionRow, question.version_id)
    if dataset is None or article is None or version is None:
        raise ValueError("问题关联数据不完整")
    paragraphs = session.scalars(
        select(ParagraphRow)
        .where(ParagraphRow.version_id == version.id)
        .order_by(ParagraphRow.number)
    ).all()
    evidence = session.scalars(
        select(EvaluationEvidenceRow)
        .where(EvaluationEvidenceRow.question_id == question.id)
        .order_by(EvaluationEvidenceRow.ordinal)
    ).all()
    proposals = session.scalars(
        select(EvaluationAIProposalRow)
        .where(EvaluationAIProposalRow.question_id == question.id)
        .order_by(EvaluationAIProposalRow.created_at.desc())
    ).all()
    return {
        "dataset": {
            "id": dataset.id,
            "name": dataset.name,
            "version": dataset.version,
            "purpose": dataset.purpose,
            "status": dataset.status,
        },
        "question": {
            "id": question.id,
            "query": question.query,
            "split": question.split,
            "scope": question.scope,
            "answerable": question.answerable,
            "reference_answer": question.reference_answer or "",
            "review_status": question.review_status,
        },
        "article": {
            "id": article.id,
            "title": article.title,
            "version_id": version.id,
            "canonical_hash": version.canonical_hash,
            "character_count": len(version.canonical_text),
            "unit_count": len(paragraphs),
        },
        "paragraphs": [{"number": item.number, "text": item.text} for item in paragraphs],
        "answer_points": _evidence_to_points(evidence),
        "ai_proposals": [
            {
                "id": item.id,
                "provider_id": item.provider_id,
                "model_id": item.model_id,
                "status": item.status,
                "proposal": json.loads(item.proposal_json),
                "created_at": item.created_at.isoformat(),
            }
            for item in proposals
        ],
    }


def _evidence_to_points(rows: Sequence[EvaluationEvidenceRow]) -> list[dict[str, object]]:
    points: dict[str, dict[str, object]] = {}
    for row in rows:
        point = points.setdefault(
            row.claim_group,
            {
                "claim_group": row.claim_group,
                "claim": row.claim_text,
                "uncertainty": row.uncertainty,
                "evidence_ranges": [],
            },
        )
        ranges = point["evidence_ranges"]
        assert isinstance(ranges, list)
        ranges.append(
            {"start_paragraph": row.start_paragraph, "end_paragraph": row.end_paragraph}
        )
    return list(points.values())


def save_reviewed_answer(
    session: Session, question_id: str, value: ReviewedAnswer
) -> EvaluationQuestionRow:
    question = session.get(EvaluationQuestionRow, question_id)
    if question is None:
        raise ValueError("问题不存在")
    dataset = _require_draft_dataset(session, question.dataset_id)
    paragraph_count = session.scalar(
        select(func.count())
        .select_from(ParagraphRow)
        .where(ParagraphRow.version_id == question.version_id)
    ) or 0
    seen_groups: set[str] = set()
    for point in value.answer_points:
        if point.claim_group in seen_groups:
            raise ValueError(f"答案要点标识重复：{point.claim_group}")
        seen_groups.add(point.claim_group)
        for evidence_range in point.evidence_ranges:
            if evidence_range.end_paragraph > paragraph_count:
                raise ValueError(f"证据结束单元 {evidence_range.end_paragraph} 越界")
    session.execute(
        delete(EvaluationEvidenceRow).where(EvaluationEvidenceRow.question_id == question.id)
    )
    ordinal = 0
    for point in value.answer_points:
        for evidence_range in point.evidence_ranges:
            ordinal += 1
            session.add(
                EvaluationEvidenceRow(
                    id=stable_id(
                        "evidence",
                        question.id,
                        point.claim_group,
                        str(evidence_range.start_paragraph),
                        str(evidence_range.end_paragraph),
                        length=40,
                    ),
                    question_id=question.id,
                    ordinal=ordinal,
                    article_id=question.article_id,
                    version_id=question.version_id,
                    start_paragraph=evidence_range.start_paragraph,
                    end_paragraph=evidence_range.end_paragraph,
                    claim_group=point.claim_group,
                    claim_text=point.claim,
                    uncertainty=point.uncertainty,
                )
            )
    question.reference_answer = value.reference_answer
    question.review_status = (
        ReviewStatus.REVIEWED.value if value.reviewed else ReviewStatus.DRAFT.value
    )
    question.updated_at = utcnow()
    dataset.updated_at = utcnow()
    session.flush()
    return question


def candidate_schema() -> dict[str, object]:
    return CandidateAnswer.model_json_schema()


def _candidate_prompt(document: dict[str, object]) -> str:
    question = document["question"]
    article = document["article"]
    paragraphs = document["paragraphs"]
    assert isinstance(question, dict) and isinstance(article, dict) and isinstance(paragraphs, list)
    numbered = "\n".join(f"[{item['number']}] {item['text']}" for item in paragraphs)
    return f"""你辅助人工为一条检索评测问题建立候选参考答案和证据范围。
问题由人工提供，不得改写。只能依据所给文章回答，不能使用外部知识，也不能调用检索。
将答案拆成最少必要的独立要点。每个要点给出唯一简短 claim_group，并标出直接支持该要点
的最小充分原子单元范围。多个不连续范围必须分别列出。若原文存在不确定性，在 uncertainty
中说明。不要复制长篇原文。

问题：{question['query']}
文章标题：{article['title']}
文章版本：{article['version_id']}

编号正文：
{numbered}"""


def generate_ai_proposal(
    session: Session,
    question_id: str,
    generator: StructuredGenerator,
    model_id: str,
) -> EvaluationAIProposalRow:
    question = session.get(EvaluationQuestionRow, question_id)
    if question is None:
        raise ValueError("问题不存在")
    _require_draft_dataset(session, question.dataset_id)
    document = question_document(session, question_id)
    result = generator.generate(_candidate_prompt(document), candidate_schema())
    candidate = CandidateAnswer.model_validate_json(result.content)
    paragraph_count = int(document["article"]["unit_count"])  # type: ignore[index]
    for point in candidate.answer_points:
        for evidence_range in point.evidence_ranges:
            if evidence_range.end_paragraph > paragraph_count:
                raise ValueError(f"Codex 证据结束单元 {evidence_range.end_paragraph} 越界")
    proposal_id = stable_id(
        "evalproposal", question.id, model_id, str(utcnow().timestamp()), length=32
    )
    row = EvaluationAIProposalRow(
        id=proposal_id,
        question_id=question.id,
        provider_id="codex-cli",
        model_id=model_id,
        prompt_version=PROMPT_VERSION,
        schema_version=SCHEMA_VERSION,
        status=AIProposalStatus.PENDING_REVIEW.value,
        proposal_json=candidate.model_dump_json(),
    )
    session.add(row)
    session.flush()
    return row


def review_ai_proposal(
    session: Session, proposal_id: str, action: str
) -> EvaluationAIProposalRow:
    proposal = session.get(EvaluationAIProposalRow, proposal_id)
    if proposal is None:
        raise ValueError("AI 候选不存在")
    question = session.get(EvaluationQuestionRow, proposal.question_id)
    if question is None:
        raise ValueError("AI 候选对应的问题不存在")
    _require_draft_dataset(session, question.dataset_id)
    if proposal.status != AIProposalStatus.PENDING_REVIEW.value:
        raise ValueError("AI 候选已经处理")
    if action not in {AIProposalStatus.ACCEPTED.value, AIProposalStatus.REJECTED.value}:
        raise ValueError("AI 候选操作必须为 accepted 或 rejected")
    proposal.status = action
    proposal.reviewed_at = utcnow()
    return proposal


def apply_ai_proposal(session: Session, proposal_id: str, accept: bool) -> EvaluationAIProposalRow:
    proposal = session.get(EvaluationAIProposalRow, proposal_id)
    if proposal is None:
        raise ValueError("AI 候选不存在")
    candidate = CandidateAnswer.model_validate_json(proposal.proposal_json)
    if accept:
        save_reviewed_answer(
            session,
            proposal.question_id,
            ReviewedAnswer(
                reference_answer=candidate.candidate_answer,
                answer_points=candidate.answer_points,
                reviewed=False,
            ),
        )
    return review_ai_proposal(
        session,
        proposal_id,
        AIProposalStatus.ACCEPTED.value if accept else AIProposalStatus.REJECTED.value,
    )


def freeze_dataset(session: Session, dataset_id: str) -> FreezeOutcome:
    dataset = _require_draft_dataset(session, dataset_id)
    questions = list(
        session.scalars(
            select(EvaluationQuestionRow)
            .where(EvaluationQuestionRow.dataset_id == dataset.id)
            .order_by(EvaluationQuestionRow.ordinal)
        ).all()
    )
    if not questions:
        raise ValueError("空评测集不能冻结")
    incomplete = [
        item.id for item in questions if item.review_status != ReviewStatus.REVIEWED.value
    ]
    if incomplete:
        raise ValueError(f"仍有 {len(incomplete)} 道问题未完成人工审核")
    pending = session.scalar(
        select(func.count())
        .select_from(EvaluationAIProposalRow)
        .join(
            EvaluationQuestionRow,
            EvaluationAIProposalRow.question_id == EvaluationQuestionRow.id,
        )
        .where(
            EvaluationQuestionRow.dataset_id == dataset.id,
            EvaluationAIProposalRow.status == AIProposalStatus.PENDING_REVIEW.value,
        )
    ) or 0
    if pending:
        raise ValueError(f"仍有 {pending} 份 AI 候选待处理")
    build = session.scalar(select(IndexBuildRow).where(IndexBuildRow.active.is_(True)))
    if build is None:
        raise ValueError("没有活动索引，无法冻结语料快照")
    indexed_chunks = list(
        session.scalars(
            select(TopicChunkRow)
            .join(ArticleVersionRow, ArticleVersionRow.id == TopicChunkRow.version_id)
            .where(ArticleVersionRow.status == "indexed")
            .order_by(TopicChunkRow.id)
        ).all()
    )
    actual_snapshot = hashlib.sha256(
        "\n".join(item.id for item in indexed_chunks).encode()
    ).hexdigest()
    if actual_snapshot != build.corpus_snapshot:
        raise ValueError("活动索引记录与当前已索引话题块不一致")
    members = session.execute(
        select(
            TopicChunkRow.article_id,
            TopicChunkRow.version_id,
            ArticleVersionRow.canonical_hash,
            TopicChunkRow.manifest_id,
        )
        .join(ArticleVersionRow, ArticleVersionRow.id == TopicChunkRow.version_id)
        .where(ArticleVersionRow.status == "indexed")
        .distinct()
        .order_by(TopicChunkRow.article_id)
    ).all()
    if not members or not indexed_chunks:
        raise ValueError("活动索引没有可冻结的文章")
    member_versions = {item.version_id for item in members}
    missing = [item.id for item in questions if item.version_id not in member_versions]
    if missing:
        raise ValueError("部分问题的目标文章版本不属于当前活动索引")
    session.execute(
        delete(EvaluationSnapshotArticleRow).where(
            EvaluationSnapshotArticleRow.dataset_id == dataset.id
        )
    )
    snapshot_values: list[dict[str, str]] = []
    for member in members:
        row = EvaluationSnapshotArticleRow(
            id=stable_id("evalsnapshot", dataset.id, member.version_id, length=40),
            dataset_id=dataset.id,
            article_id=member.article_id,
            version_id=member.version_id,
            canonical_hash=member.canonical_hash,
            manifest_id=member.manifest_id,
        )
        session.add(row)
        snapshot_values.append(
            {
                "article_id": row.article_id,
                "version_id": row.version_id,
                "canonical_hash": row.canonical_hash,
                "manifest_id": row.manifest_id,
            }
        )
    question_values: list[dict[str, object]] = []
    for question in questions:
        evidence = session.scalars(
            select(EvaluationEvidenceRow)
            .where(EvaluationEvidenceRow.question_id == question.id)
            .order_by(EvaluationEvidenceRow.ordinal)
        ).all()
        if question.answerable and not evidence:
            raise ValueError(f"问题 {question.id} 没有人工确认的证据")
        question_values.append(
            {
                "id": question.id,
                "ordinal": question.ordinal,
                "query": question.query,
                "split": question.split,
                "scope": question.scope,
                "article_id": question.article_id,
                "version_id": question.version_id,
                "answerable": question.answerable,
                "reference_answer": question.reference_answer,
                "evidence": [
                    {
                        "start": item.start_paragraph,
                        "end": item.end_paragraph,
                        "claim_group": item.claim_group,
                        "claim_text": item.claim_text,
                        "uncertainty": item.uncertainty,
                    }
                    for item in evidence
                ],
            }
        )
    canonical = json.dumps(
        {
            "dataset": {
                "id": dataset.id,
                "name": dataset.name,
                "version": dataset.version,
                "purpose": dataset.purpose,
            },
            "corpus_snapshot": build.corpus_snapshot,
            "articles": snapshot_values,
            "questions": question_values,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    content_hash = hashlib.sha256(canonical.encode()).hexdigest()
    frozen_at = utcnow()
    dataset.content_hash = content_hash
    dataset.index_corpus_snapshot = build.corpus_snapshot
    dataset.status = DatasetStatus.FROZEN.value
    dataset.frozen_at = frozen_at
    dataset.updated_at = frozen_at
    session.flush()
    return FreezeOutcome(
        dataset_id=dataset.id,
        content_hash=content_hash,
        corpus_snapshot=build.corpus_snapshot,
        article_count=len(members),
        question_count=len(questions),
    )
