from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import bm25s
import jieba
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from xifeng_rag.corpus import stable_id
from xifeng_rag.domain import (
    BuildStatus,
    RetrievalCandidate,
    RetrievalScores,
    VersionStatus,
)
from xifeng_rag.storage import ArticleRow, ArticleVersionRow, IndexBuildRow, TopicChunkRow

TOKENIZER_VERSION = "jieba-search-v1"
PROFILE_ID = "approved-topic-original-bm25-jieba-v1"
TOKEN_PATTERN = re.compile(r"[\w\u3400-\u9fff]+", re.UNICODE)


class ChineseTokenizer:
    def __init__(self) -> None:
        jieba.setLogLevel(logging.WARNING)
        self._tokenizer = jieba.Tokenizer()
        self._tokenizer.initialize()

    def tokenize(self, text: str) -> list[str]:
        tokens = self._tokenizer.lcut_for_search(text.casefold(), HMM=True)
        return [token for token in tokens if TOKEN_PATTERN.fullmatch(token)]


@dataclass(frozen=True)
class BuildOutcome:
    build_id: str
    chunk_count: int
    path: Path
    duration_ms: int


class IndexBuildError(RuntimeError):
    def __init__(
        self,
        build_id: str,
        snapshot: str,
        final_path: Path,
        error_type: str,
        duration_ms: int,
    ) -> None:
        super().__init__(f"BM25 索引构建失败：{error_type}")
        self.build_id = build_id
        self.snapshot = snapshot
        self.final_path = final_path
        self.error_type = error_type
        self.duration_ms = duration_ms


def _selected_chunks(session: Session) -> list[TopicChunkRow]:
    rows = session.execute(
        select(TopicChunkRow, ArticleVersionRow)
        .join(ArticleVersionRow, TopicChunkRow.version_id == ArticleVersionRow.id)
        .join(ArticleRow, TopicChunkRow.article_id == ArticleRow.id)
        .where(
            ArticleRow.active.is_(True),
            ArticleVersionRow.active.is_(True),
            ArticleVersionRow.status.in_(
                [VersionStatus.APPROVED.value, VersionStatus.INDEXED.value]
            ),
        )
        .order_by(ArticleVersionRow.created_at.desc(), TopicChunkRow.start_paragraph)
    ).all()
    selected_versions: dict[str, str] = {}
    chunks: list[TopicChunkRow] = []
    for chunk, version in rows:
        chosen = selected_versions.setdefault(version.article_id, version.id)
        if chosen == version.id:
            chunks.append(chunk)
    return sorted(chunks, key=lambda item: item.id)


def build_index(session: Session, index_root: Path) -> BuildOutcome:
    started = time.perf_counter()
    chunks = _selected_chunks(session)
    if not chunks:
        raise ValueError("没有已批准的话题块，拒绝建立空索引")
    snapshot = hashlib.sha256("\n".join(item.id for item in chunks).encode()).hexdigest()
    build_id = stable_id("index", PROFILE_ID, snapshot, length=32)
    final_path = index_root / build_id
    existing = session.get(IndexBuildRow, build_id)
    if existing is not None and final_path.is_dir():
        _activate_build(session, existing, chunks)
        return BuildOutcome(build_id, len(chunks), final_path, existing.duration_ms or 0)

    index_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".building-", dir=index_root))
    row = existing or IndexBuildRow(
        id=build_id,
        profile_id=PROFILE_ID,
        corpus_snapshot=snapshot,
        path=str(final_path),
        status=BuildStatus.BUILDING.value,
        active=False,
    )
    session.add(row)
    session.flush()
    tokenizer = ChineseTokenizer()
    corpus = [{"chunk_id": item.id} for item in chunks]
    try:
        tokenized = [tokenizer.tokenize(item.original_text) for item in chunks]
        retriever = bm25s.BM25(corpus=corpus)
        retriever.index(tokenized, show_progress=False)
        retriever.save(temporary / "bm25", corpus=corpus)
        metadata = {
            "build_id": build_id,
            "profile_id": PROFILE_ID,
            "tokenizer_version": TOKENIZER_VERSION,
            "corpus_snapshot": snapshot,
            "chunk_count": len(chunks),
        }
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        loaded = bm25s.BM25.load(temporary / "bm25", load_corpus=True)
        loaded.retrieve([tokenizer.tokenize(chunks[0].original_text[:80])], k=1)
        if final_path.exists():
            raise FileExistsError(f"不可变索引目录已存在但数据库记录无效：{final_path}")
        os.replace(temporary, final_path)
        duration_ms = round((time.perf_counter() - started) * 1000)
        row.duration_ms = duration_ms
        _activate_build(session, row, chunks)
        return BuildOutcome(build_id, len(chunks), final_path, duration_ms)
    except Exception as exc:
        row.status = BuildStatus.FAILED.value
        row.error_type = type(exc).__name__
        if temporary.exists():
            shutil.rmtree(temporary)
        raise IndexBuildError(
            build_id,
            snapshot,
            final_path,
            type(exc).__name__,
            round((time.perf_counter() - started) * 1000),
        ) from exc


def record_build_failure(session: Session, error: IndexBuildError) -> None:
    row = session.get(IndexBuildRow, error.build_id)
    if row is None:
        row = IndexBuildRow(
            id=error.build_id,
            profile_id=PROFILE_ID,
            corpus_snapshot=error.snapshot,
            path=str(error.final_path),
            status=BuildStatus.FAILED.value,
            active=False,
        )
        session.add(row)
    row.status = BuildStatus.FAILED.value
    row.active = False
    row.error_type = error.error_type
    row.duration_ms = error.duration_ms


def _activate_build(session: Session, row: IndexBuildRow, chunks: list[TopicChunkRow]) -> None:
    session.execute(
        update(IndexBuildRow)
        .where(IndexBuildRow.active.is_(True), IndexBuildRow.id != row.id)
        .values(active=False, status=BuildStatus.SUPERSEDED.value)
    )
    row.active = True
    row.status = BuildStatus.ACTIVE.value
    version_ids = {chunk.version_id for chunk in chunks}
    article_ids = {chunk.article_id for chunk in chunks}
    session.execute(
        update(ArticleVersionRow)
        .where(
            ArticleVersionRow.article_id.in_(article_ids),
            ArticleVersionRow.id.not_in(version_ids),
            ArticleVersionRow.active.is_(True),
        )
        .values(active=False, status=VersionStatus.INACTIVE.value)
    )
    session.execute(
        update(ArticleVersionRow)
        .where(ArticleVersionRow.id.in_(version_ids))
        .values(status=VersionStatus.INDEXED.value, active=True)
    )


def search(
    session: Session, query: str, top_k: int = 5, build_id: str | None = None
) -> list[RetrievalCandidate]:
    if not query.strip():
        raise ValueError("查询不能为空")
    if build_id is None:
        build = session.scalar(
            select(IndexBuildRow).where(
                IndexBuildRow.active.is_(True),
                IndexBuildRow.status == BuildStatus.ACTIVE.value,
            )
        )
    else:
        build = session.get(IndexBuildRow, build_id)
        if build is not None and build.status not in {
            BuildStatus.ACTIVE.value,
            BuildStatus.SUPERSEDED.value,
        }:
            build = None
    if build is None:
        raise ValueError("没有可用的指定索引，请先执行 `xifeng index build`")
    build_path = Path(build.path)
    if not build_path.is_dir():
        raise FileNotFoundError(f"活动索引目录不存在：{build_path}")
    tokenizer = ChineseTokenizer()
    retriever = bm25s.BM25.load(build_path / "bm25", load_corpus=True)
    results, scores = retriever.retrieve([tokenizer.tokenize(query)], k=top_k)
    candidates: list[RetrievalCandidate] = []
    for metadata, score in zip(results[0], scores[0], strict=True):
        chunk_id = metadata["chunk_id"] if isinstance(metadata, dict) else str(metadata)
        chunk = session.get(TopicChunkRow, chunk_id)
        if chunk is None:
            raise RuntimeError(f"索引引用了不存在的话题块：{chunk_id}")
        version = session.get(ArticleVersionRow, chunk.version_id)
        article = session.get(ArticleRow, chunk.article_id)
        if version is None or article is None:
            raise RuntimeError(f"话题块缺少文章元数据：{chunk_id}")
        candidates.append(
            RetrievalCandidate(
                chunk_id=chunk.id,
                article_id=chunk.article_id,
                article_version=chunk.version_id,
                article_title=article.title,
                original_text=chunk.original_text,
                paragraph_range=(chunk.start_paragraph, chunk.end_paragraph),
                retrieval_scores=RetrievalScores(bm25=float(score)),
                topic_title=chunk.topic_title,
                context_summary=chunk.context_summary,
            )
        )
    return candidates
