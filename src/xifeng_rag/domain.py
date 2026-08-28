from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class VersionStatus(StrEnum):
    DISCOVERED = "discovered"
    NEEDS_CLEANUP_REVIEW = "needs_cleanup_review"
    PENDING_CHUNKING = "pending_chunking"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    INDEXED = "indexed"
    INACTIVE = "inactive"


class ProposalStatus(StrEnum):
    GENERATING = "generating"
    PENDING_REVIEW = "pending_review"
    FAILED = "failed"
    APPROVED = "approved"


class ContentRole(StrEnum):
    SUBSTANTIVE = "substantive"
    BOILERPLATE = "boilerplate"


class BuildStatus(StrEnum):
    BUILDING = "building"
    ACTIVE = "active"
    FAILED = "failed"
    SUPERSEDED = "superseded"


class RetrievalMode(StrEnum):
    BM25 = "bm25"
    DENSE = "dense"
    HYBRID = "hybrid"


class AssemblyStatus(StrEnum):
    READY = "ready"
    NO_EVIDENCE = "no_evidence"


class AssemblyExclusionReason(StrEnum):
    DUPLICATE = "duplicate"
    ITEM_LIMIT = "item_limit"
    CHAR_BUDGET = "char_budget"


class CompressionMethod(StrEnum):
    NONE = "none"


class AssemblyErrorCode(StrEnum):
    INVALID_CONFIG = "invalid_config"
    INVALID_CANDIDATE = "invalid_candidate"
    BUDGET_TOO_SMALL = "budget_too_small"
    RENDER_INVARIANT_FAILED = "render_invariant_failed"


class ParagraphData(BaseModel):
    number: int = Field(ge=1)
    text: str = Field(min_length=1)
    source_start: int = Field(ge=0)
    source_end: int = Field(gt=0)


class Topic(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    start_paragraph: int = Field(ge=1)
    end_paragraph: int = Field(ge=1)
    title: str = Field(min_length=1, max_length=200)
    summary: str = Field(min_length=1, max_length=2000)
    keywords: list[str] = Field(min_length=1, max_length=20)
    include_in_index: bool = True
    # Compatibility input only. New reviews and manifests persist the explicit
    # indexing decision instead of a semantic content classification.
    content_role: ContentRole | None = Field(default=None, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def map_legacy_content_role(cls, value: object) -> object:
        if not isinstance(value, dict) or "include_in_index" in value:
            return value
        normalized = dict(value)
        role = normalized.get("content_role")
        if role is not None:
            normalized["include_in_index"] = role != ContentRole.BOILERPLATE.value
        return normalized

    @model_validator(mode="after")
    def valid_range(self) -> Topic:
        if self.end_paragraph < self.start_paragraph:
            raise ValueError("话题块结束段落不能早于开始段落")
        if any(not keyword.strip() for keyword in self.keywords):
            raise ValueError("关键词不能为空")
        return self


class TopicProposal(BaseModel):
    topics: list[Topic] = Field(min_length=1, max_length=20)


class GeneratedTopic(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    end_unit: int = Field(ge=1)
    title: str = Field(min_length=1, max_length=200)
    summary: str = Field(min_length=1, max_length=2000)
    keywords: list[str] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def nonempty_keywords(self) -> GeneratedTopic:
        if any(not keyword.strip() for keyword in self.keywords):
            raise ValueError("关键词不能为空")
        return self


class GeneratedTopicProposal(BaseModel):
    topics: list[GeneratedTopic] = Field(min_length=1, max_length=20)


class RetrievalScores(BaseModel):
    bm25: float | None = None
    bm25_rank: int | None = Field(default=None, ge=1)
    dense: float | None = None
    dense_rank: int | None = Field(default=None, ge=1)
    rrf: float | None = None
    rrf_rank: int | None = Field(default=None, ge=1)
    reranker: float | None = None
    reranker_rank: int | None = Field(default=None, ge=1)
    reranker_truncated: bool | None = None


class RetrievalCandidate(BaseModel):
    chunk_id: str
    article_id: str
    article_version: str
    article_title: str
    original_text: str
    paragraph_range: tuple[int, int]
    retrieval_scores: RetrievalScores
    source: list[str] = Field(default_factory=lambda: ["bm25"])
    topic_title: str | None = None
    context_summary: str | None = None


class ContextAssemblyConfigSnapshot(BaseModel):
    max_evidence_items: int
    max_total_chars: int
    include_ai_metadata: bool


class EvidenceItem(BaseModel):
    citation_id: str = Field(pattern=r"^E[1-9][0-9]*$")
    source_chunk_id: str = Field(min_length=1)
    article_id: str = Field(min_length=1)
    article_version: str = Field(min_length=1)
    article_title: str = Field(min_length=1)
    original_paragraph_range: tuple[int, int]
    selected_paragraph_range: tuple[int, int]
    original_text: str = Field(min_length=1)
    retrieval_scores: RetrievalScores
    input_rank: int = Field(ge=1)
    reranker_rank: int | None = Field(default=None, ge=1)
    topic_title: str | None = None
    context_summary: str | None = None
    compression_method: CompressionMethod = CompressionMethod.NONE

    @model_validator(mode="after")
    def valid_evidence_ranges(self) -> EvidenceItem:
        original_start, original_end = self.original_paragraph_range
        selected_start, selected_end = self.selected_paragraph_range
        if original_start < 1 or original_end < original_start:
            raise ValueError("原始证据范围非法")
        if selected_start < original_start or selected_end > original_end:
            raise ValueError("选中证据范围必须位于原始范围内")
        if selected_end < selected_start:
            raise ValueError("选中证据范围非法")
        if self.compression_method == CompressionMethod.NONE and (
            self.selected_paragraph_range != self.original_paragraph_range
        ):
            raise ValueError("未压缩证据必须保留完整原始范围")
        return self


class ExcludedCandidate(BaseModel):
    source_chunk_id: str
    original_rank: int = Field(ge=1)
    reason: AssemblyExclusionReason
    rendered_chars: int = Field(ge=0)


class AssemblyUsage(BaseModel):
    input_candidate_count: int = Field(ge=0)
    selected_count: int = Field(ge=0)
    excluded_count: int = Field(ge=0)
    original_text_chars: int = Field(ge=0)
    ai_metadata_chars: int = Field(ge=0)
    rendered_context_chars: int = Field(ge=0)


class AssemblyResult(BaseModel):
    status: AssemblyStatus
    evidence_items: list[EvidenceItem]
    excluded_candidates: list[ExcludedCandidate]
    usage: AssemblyUsage
    config_snapshot: ContextAssemblyConfigSnapshot
    rendered_context: str
    rendered_context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def validate_topic_coverage(proposal: TopicProposal, paragraph_count: int) -> list[str]:
    errors: list[str] = []
    expected = 1
    for index, topic in enumerate(proposal.topics, start=1):
        if topic.start_paragraph != expected:
            errors.append(f"第 {index} 块应从段落 {expected} 开始，实际为 {topic.start_paragraph}")
        if topic.end_paragraph > paragraph_count:
            errors.append(f"第 {index} 块结束段落 {topic.end_paragraph} 越界")
        expected = topic.end_paragraph + 1
    if expected != paragraph_count + 1:
        errors.append(f"分块未覆盖到最后段落 {paragraph_count}")
    return errors
