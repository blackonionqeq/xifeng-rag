from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    event,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class ArticleRow(Base):
    __tablename__ = "articles"
    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    source_path: Mapped[str] = mapped_column(Text, unique=True)
    title: Mapped[str] = mapped_column(Text)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    captured_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    source_modified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    source_size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_character_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    metadata_synced_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ArticleVersionRow(Base):
    __tablename__ = "article_versions"
    id: Mapped[str] = mapped_column(String(96), primary_key=True)
    article_id: Mapped[str] = mapped_column(ForeignKey("articles.id"), index=True)
    source_hash: Mapped[str] = mapped_column(String(64), index=True)
    canonical_hash: Mapped[str] = mapped_column(String(64), index=True)
    cleaner_version: Mapped[str] = mapped_column(String(40))
    canonical_text: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(40), index=True)
    removed_ranges_json: Mapped[str] = mapped_column(Text, default="[]")
    warnings_json: Mapped[str] = mapped_column(Text, default="[]")
    links_json: Mapped[str] = mapped_column(Text, default="[]")
    active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ParagraphRow(Base):
    __tablename__ = "paragraphs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version_id: Mapped[str] = mapped_column(ForeignKey("article_versions.id"), index=True)
    number: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    source_start: Mapped[int] = mapped_column(Integer)
    source_end: Mapped[int] = mapped_column(Integer)


class ChunkProposalRow(Base):
    __tablename__ = "chunk_proposals"
    id: Mapped[str] = mapped_column(String(96), primary_key=True)
    version_id: Mapped[str] = mapped_column(ForeignKey("article_versions.id"), index=True)
    status: Mapped[str] = mapped_column(String(40), index=True)
    provider_id: Mapped[str] = mapped_column(Text, default="ollama")
    model_id: Mapped[str] = mapped_column(Text)
    prompt_version: Mapped[str] = mapped_column(String(40))
    prompt_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    prompt_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    retry_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    schema_version: Mapped[str] = mapped_column(String(40))
    generation_json: Mapped[str] = mapped_column(Text, default="{}")
    topics_json: Mapped[str] = mapped_column(Text, default="[]")
    diagnostics_json: Mapped[str] = mapped_column(Text, default="[]")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ApprovedManifestRow(Base):
    __tablename__ = "approved_manifests"
    id: Mapped[str] = mapped_column(String(96), primary_key=True)
    proposal_id: Mapped[str] = mapped_column(ForeignKey("chunk_proposals.id"), unique=True)
    version_id: Mapped[str] = mapped_column(ForeignKey("article_versions.id"), index=True)
    manifest_version: Mapped[int] = mapped_column(Integer, default=1)
    topics_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    approved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class TopicChunkRow(Base):
    __tablename__ = "topic_chunks"
    id: Mapped[str] = mapped_column(String(120), primary_key=True)
    manifest_id: Mapped[str] = mapped_column(ForeignKey("approved_manifests.id"), index=True)
    version_id: Mapped[str] = mapped_column(ForeignKey("article_versions.id"), index=True)
    article_id: Mapped[str] = mapped_column(ForeignKey("articles.id"), index=True)
    start_paragraph: Mapped[int] = mapped_column(Integer)
    end_paragraph: Mapped[int] = mapped_column(Integer)
    topic_title: Mapped[str] = mapped_column(Text)
    context_summary: Mapped[str] = mapped_column(Text)
    keywords_json: Mapped[str] = mapped_column(Text)
    original_text: Mapped[str] = mapped_column(Text)


class IndexBuildRow(Base):
    __tablename__ = "index_builds"
    id: Mapped[str] = mapped_column(String(96), primary_key=True)
    profile_id: Mapped[str] = mapped_column(String(96), index=True)
    corpus_snapshot: Mapped[str] = mapped_column(String(64))
    path: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(40), index=True)
    active: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    error_type: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class EvaluationRunRow(Base):
    __tablename__ = "evaluation_runs"
    id: Mapped[str] = mapped_column(String(96), primary_key=True)
    dataset_id: Mapped[str] = mapped_column(Text)
    build_id: Mapped[str] = mapped_column(ForeignKey("index_builds.id"))
    metrics_json: Mapped[str] = mapped_column(Text)
    details_json: Mapped[str] = mapped_column(Text)
    dataset_content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    corpus_snapshot: Mapped[str | None] = mapped_column(String(64), nullable=True)
    profile_id: Mapped[str | None] = mapped_column(String(96), nullable=True)
    k: Mapped[int | None] = mapped_column(Integer, nullable=True)
    code_commit: Mapped[str | None] = mapped_column(String(64), nullable=True)
    worktree_dirty: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    retrieval_mode: Mapped[str | None] = mapped_column(String(20), nullable=True)
    bm25_candidate_k: Mapped[int | None] = mapped_column(Integer, nullable=True)
    dense_candidate_k: Mapped[int | None] = mapped_column(Integer, nullable=True)
    rrf_k: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class EvaluationDatasetRow(Base):
    __tablename__ = "evaluation_datasets"
    id: Mapped[str] = mapped_column(String(96), primary_key=True)
    name: Mapped[str] = mapped_column(Text)
    version: Mapped[str] = mapped_column(String(40))
    purpose: Mapped[str] = mapped_column(String(20), index=True)
    status: Mapped[str] = mapped_column(String(20), index=True)
    index_corpus_snapshot: Mapped[str | None] = mapped_column(String(64), nullable=True)
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    frozen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class EvaluationSnapshotArticleRow(Base):
    __tablename__ = "evaluation_snapshot_articles"
    id: Mapped[str] = mapped_column(String(120), primary_key=True)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("evaluation_datasets.id"), index=True)
    article_id: Mapped[str] = mapped_column(ForeignKey("articles.id"), index=True)
    version_id: Mapped[str] = mapped_column(ForeignKey("article_versions.id"), index=True)
    canonical_hash: Mapped[str] = mapped_column(String(64))
    manifest_id: Mapped[str] = mapped_column(ForeignKey("approved_manifests.id"))


class EvaluationQuestionRow(Base):
    __tablename__ = "evaluation_questions"
    id: Mapped[str] = mapped_column(String(96), primary_key=True)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("evaluation_datasets.id"), index=True)
    ordinal: Mapped[int] = mapped_column(Integer)
    query: Mapped[str] = mapped_column(Text)
    split: Mapped[str] = mapped_column(String(20), index=True)
    scope: Mapped[str] = mapped_column(String(20))
    article_id: Mapped[str] = mapped_column(ForeignKey("articles.id"), index=True)
    version_id: Mapped[str] = mapped_column(ForeignKey("article_versions.id"), index=True)
    answerable: Mapped[bool] = mapped_column(Boolean, default=True)
    reference_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    review_status: Mapped[str] = mapped_column(String(20), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class EvaluationEvidenceRow(Base):
    __tablename__ = "evaluation_evidence"
    id: Mapped[str] = mapped_column(String(120), primary_key=True)
    question_id: Mapped[str] = mapped_column(ForeignKey("evaluation_questions.id"), index=True)
    ordinal: Mapped[int] = mapped_column(Integer)
    article_id: Mapped[str] = mapped_column(ForeignKey("articles.id"), index=True)
    version_id: Mapped[str] = mapped_column(ForeignKey("article_versions.id"), index=True)
    start_paragraph: Mapped[int] = mapped_column(Integer)
    end_paragraph: Mapped[int] = mapped_column(Integer)
    claim_group: Mapped[str] = mapped_column(String(80))
    claim_text: Mapped[str] = mapped_column(Text)
    uncertainty: Mapped[str] = mapped_column(Text, default="")


class EvaluationAIProposalRow(Base):
    __tablename__ = "evaluation_ai_proposals"
    id: Mapped[str] = mapped_column(String(96), primary_key=True)
    question_id: Mapped[str] = mapped_column(ForeignKey("evaluation_questions.id"), index=True)
    provider_id: Mapped[str] = mapped_column(String(40))
    model_id: Mapped[str] = mapped_column(Text)
    prompt_version: Mapped[str] = mapped_column(String(40))
    schema_version: Mapped[str] = mapped_column(String(40))
    status: Mapped[str] = mapped_column(String(20), index=True)
    proposal_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class CleanupDecisionRow(Base):
    __tablename__ = "cleanup_decisions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version_id: Mapped[str] = mapped_column(ForeignKey("article_versions.id"), index=True)
    action: Mapped[str] = mapped_column(String(40))
    truncate_from_paragraph: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


def create_db_engine(database_path: Path) -> Engine:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite+pysqlite:///{database_path}")

    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection: object, _record: object) -> None:
        cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    return engine


class Database:
    def __init__(self, path: Path) -> None:
        self.engine = create_db_engine(path)
        self._sessions = sessionmaker(self.engine, expire_on_commit=False)

    def initialize(self) -> None:
        migration_dir = Path(__file__).parent / "migrations"
        config = Config()
        config.set_main_option("script_location", str(migration_dir))
        config.set_main_option(
            "sqlalchemy.url", self.engine.url.render_as_string(hide_password=False)
        )
        command.upgrade(config, "head")

    @contextmanager
    def session(self) -> Iterator[Session]:
        with self._sessions.begin() as session:
            yield session
