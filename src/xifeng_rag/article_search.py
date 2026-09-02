from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import bm25s
import numpy as np
from qdrant_client import QdrantClient, models
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from xifeng_rag.corpus import stable_id
from xifeng_rag.domain import (
    ArticleSubtopics,
    BuildStatus,
    OverviewStatus,
    RetrievalMode,
    TopicProposal,
    VersionStatus,
)
from xifeng_rag.indexing import (
    ChineseTokenizer,
    EmbeddingProvider,
    RankedHit,
    reciprocal_rank_fusion,
)
from xifeng_rag.storage import (
    ApprovedManifestRow,
    ArticleIndexBuildRow,
    ArticleOverviewRow,
    ArticleRow,
    ArticleVersionRow,
    SubtopicSetRow,
)

ARTICLE_PROFILE_VERSION = "approved-article-overview-topics-subtopics-bge-bm25-rrf-v3"
ARTICLE_COLLECTION = "article_documents"
DOCUMENTS_FILE = "documents.json"


@dataclass(frozen=True)
class ArticleSearchDocument:
    document_id: str
    article_id: str
    version_id: str
    overview_id: str
    manifest_id: str
    title: str
    published_at: str | None
    article_summary: str
    topic_titles: list[str]
    topic_keywords: dict[str, list[str]]
    supplemental_keywords: list[str]
    subtopics: list[dict[str, object]]
    search_text: str


@dataclass(frozen=True)
class ArticleSearchResult:
    article_id: str
    version_id: str
    title: str
    published_at: str | None
    article_summary: str
    matched_topics: list[str]
    matched_subtopics: list[str]
    sources: list[str]
    rrf_score: float | None
    bm25_score: float | None
    bm25_rank: int | None
    dense_score: float | None
    dense_rank: int | None


@dataclass(frozen=True)
class ArticleBuildOutcome:
    build_id: str
    article_count: int
    path: Path
    duration_ms: int


class ArticleIndexBuildError(RuntimeError):
    def __init__(
        self,
        build_id: str,
        profile_id: str,
        snapshot: str,
        final_path: Path,
        error_type: str,
        duration_ms: int,
    ) -> None:
        super().__init__(f"文章级索引构建失败：{error_type}")
        self.build_id = build_id
        self.profile_id = profile_id
        self.snapshot = snapshot
        self.final_path = final_path
        self.error_type = error_type
        self.duration_ms = duration_ms


def _isoformat(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _supplemental_keywords(
    title: str,
    article_summary: str,
    topic_titles: list[str],
    keywords: list[str],
) -> list[str]:
    represented = "\n".join([title, article_summary, *topic_titles]).casefold()
    selected: list[str] = []
    seen: set[str] = set()
    for value in keywords:
        keyword = value.strip()
        normalized = keyword.casefold()
        if not keyword or normalized in seen or normalized in represented:
            continue
        seen.add(normalized)
        selected.append(keyword)
    return selected


def _search_text(
    title: str,
    article_summary: str,
    topic_titles: list[str],
    supplemental_keywords: list[str],
    subtopics: list[dict[str, object]],
) -> str:
    subtopic_parts = [
        str(value)
        for item in subtopics
        for value in (item.get("title", ""), item.get("summary", ""))
    ]
    parts = [
        title,
        article_summary,
        *topic_titles,
        *supplemental_keywords,
        *subtopic_parts,
    ]
    return "\n".join(item.strip() for item in parts if item.strip())


def selected_article_documents(session: Session) -> list[ArticleSearchDocument]:
    rows = session.execute(
        select(
            ArticleOverviewRow,
            ArticleVersionRow,
            ArticleRow,
            ApprovedManifestRow,
        )
        .join(ArticleVersionRow, ArticleOverviewRow.version_id == ArticleVersionRow.id)
        .join(ArticleRow, ArticleVersionRow.article_id == ArticleRow.id)
        .join(ApprovedManifestRow, ArticleOverviewRow.manifest_id == ApprovedManifestRow.id)
        .where(
            ArticleOverviewRow.status == OverviewStatus.APPROVED.value,
            ArticleVersionRow.active.is_(True),
            ArticleVersionRow.status.in_(
                [VersionStatus.APPROVED.value, VersionStatus.INDEXED.value]
            ),
            ArticleRow.active.is_(True),
        )
        .order_by(
            ArticleRow.id,
            ArticleVersionRow.created_at.desc(),
            ArticleOverviewRow.approved_at.desc(),
        )
    ).all()
    selected: dict[str, ArticleSearchDocument] = {}
    for overview, version, article, manifest in rows:
        if article.id in selected or not manifest.topics_json:
            continue
        proposal = TopicProposal.model_validate_json(manifest.topics_json)
        topic_titles = [topic.title.strip() for topic in proposal.topics if topic.title.strip()]
        topic_keywords = {
            topic.title.strip(): list(topic.keywords)
            for topic in proposal.topics
            if topic.title.strip()
        }
        keywords = [keyword for topic in proposal.topics for keyword in topic.keywords]
        supplemental = _supplemental_keywords(
            article.title, overview.article_summary, topic_titles, keywords
        )
        approved_subtopics = session.scalar(
            select(SubtopicSetRow)
            .where(
                SubtopicSetRow.version_id == version.id,
                SubtopicSetRow.manifest_id == manifest.id,
                SubtopicSetRow.status == "approved",
            )
            .order_by(SubtopicSetRow.approved_at.desc())
        )
        subtopics: list[dict[str, object]] = []
        if approved_subtopics is not None:
            parsed_subtopics = ArticleSubtopics.model_validate_json(
                approved_subtopics.subtopics_json
            )
            subtopics = [
                {
                    "title": item.title,
                    "summary": item.summary,
                    "start_unit": item.start_unit,
                    "end_unit": item.end_unit,
                    "parent_title": parent.parent_title,
                }
                for parent in parsed_subtopics.topics
                for item in parent.subtopics
            ]
        text = _search_text(
            article.title,
            overview.article_summary,
            topic_titles,
            supplemental,
            subtopics,
        )
        document_id = stable_id(
            "article-document",
            article.id,
            version.id,
            overview.id,
            manifest.id,
            text,
            length=48,
        )
        selected[article.id] = ArticleSearchDocument(
            document_id=document_id,
            article_id=article.id,
            version_id=version.id,
            overview_id=overview.id,
            manifest_id=manifest.id,
            title=article.title,
            published_at=_isoformat(article.published_at),
            article_summary=overview.article_summary,
            topic_titles=topic_titles,
            topic_keywords=topic_keywords,
            supplemental_keywords=supplemental,
            subtopics=subtopics,
            search_text=text,
        )
    return sorted(selected.values(), key=lambda item: item.document_id)


def _profile_id(
    provider: EmbeddingProvider,
    bm25_candidate_k: int,
    dense_candidate_k: int,
    rrf_k: int,
) -> str:
    semantic = json.dumps(
        {
            "profile_version": ARTICLE_PROFILE_VERSION,
            "tokenizer_version": "jieba-search-v1",
            "embedding": provider.metadata,
            "bm25_candidate_k": bm25_candidate_k,
            "dense_candidate_k": dense_candidate_k,
            "rrf_k": rrf_k,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return stable_id("article-profile", semantic, length=48)


def _point_id(document_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"xifeng-rag:article:{document_id}"))


def _activate_build(session: Session, row: ArticleIndexBuildRow) -> None:
    session.execute(
        update(ArticleIndexBuildRow)
        .where(ArticleIndexBuildRow.active.is_(True), ArticleIndexBuildRow.id != row.id)
        .values(active=False, status=BuildStatus.SUPERSEDED.value)
    )
    row.active = True
    row.status = BuildStatus.ACTIVE.value
    row.error_type = None


def build_article_index(
    session: Session,
    index_root: Path,
    provider: EmbeddingProvider,
    *,
    bm25_candidate_k: int = 5,
    dense_candidate_k: int = 5,
    rrf_k: int = 60,
) -> ArticleBuildOutcome:
    started = time.perf_counter()
    documents = selected_article_documents(session)
    if not documents:
        raise ValueError("没有带已批准文章概括的活动文章，拒绝建立空文章索引")
    profile_id = _profile_id(provider, bm25_candidate_k, dense_candidate_k, rrf_k)
    snapshot = hashlib.sha256(
        "\n".join(item.document_id for item in documents).encode("utf-8")
    ).hexdigest()
    build_id = stable_id("article-index", profile_id, snapshot, length=32)
    final_path = index_root / build_id
    existing = session.get(ArticleIndexBuildRow, build_id)
    if existing is not None and final_path.is_dir():
        _activate_build(session, existing)
        return ArticleBuildOutcome(build_id, len(documents), final_path, existing.duration_ms or 0)

    index_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".building-", dir=index_root))
    row = existing or ArticleIndexBuildRow(
        id=build_id,
        profile_id=profile_id,
        corpus_snapshot=snapshot,
        path=str(final_path),
        status=BuildStatus.BUILDING.value,
        active=False,
    )
    session.add(row)
    session.flush()
    qdrant: QdrantClient | None = None
    try:
        tokenizer = ChineseTokenizer()
        corpus = [{"document_id": item.document_id} for item in documents]
        retriever = bm25s.BM25(corpus=corpus)
        retriever.index(
            [tokenizer.tokenize(item.search_text) for item in documents],
            show_progress=False,
        )
        retriever.save(temporary / "bm25", corpus=corpus)

        vectors = np.asarray(
            provider.encode_documents([item.search_text for item in documents]),
            dtype=np.float32,
        )
        if vectors.shape != (len(documents), provider.dimension):
            raise ValueError(
                f"Embedding 输出形状错误：期望 {(len(documents), provider.dimension)}，"
                f"实际 {vectors.shape}"
            )
        qdrant = QdrantClient(path=str(temporary / "qdrant"))
        qdrant.create_collection(
            collection_name=ARTICLE_COLLECTION,
            vectors_config=models.VectorParams(
                size=provider.dimension, distance=models.Distance.COSINE
            ),
        )
        qdrant.upload_points(
            collection_name=ARTICLE_COLLECTION,
            points=[
                models.PointStruct(
                    id=_point_id(document.document_id),
                    vector=vector.tolist(),
                    payload={
                        "document_id": document.document_id,
                        "article_id": document.article_id,
                        "version_id": document.version_id,
                    },
                )
                for document, vector in zip(documents, vectors, strict=True)
            ],
        )
        if qdrant.count(ARTICLE_COLLECTION, exact=True).count != len(documents):
            raise RuntimeError("文章 Dense point 数量不一致")
        qdrant.close()
        qdrant = None

        (temporary / DOCUMENTS_FILE).write_text(
            json.dumps([asdict(item) for item in documents], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        metadata = {
            "build_id": build_id,
            "profile_id": profile_id,
            "profile_version": ARTICLE_PROFILE_VERSION,
            "corpus_snapshot": snapshot,
            "article_count": len(documents),
            "embedding": provider.metadata,
            "retrieval": {
                "bm25_candidate_k": bm25_candidate_k,
                "dense_candidate_k": dense_candidate_k,
                "rrf_k": rrf_k,
            },
        }
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        loaded = bm25s.BM25.load(temporary / "bm25", load_corpus=True)
        loaded.retrieve(
            [tokenizer.tokenize(documents[0].search_text[:80])],
            k=1,
            show_progress=False,
        )
        if final_path.exists():
            raise FileExistsError(f"不可变文章索引目录已存在：{final_path}")
        os.replace(temporary, final_path)
        duration_ms = round((time.perf_counter() - started) * 1000)
        row.duration_ms = duration_ms
        _activate_build(session, row)
        return ArticleBuildOutcome(build_id, len(documents), final_path, duration_ms)
    except Exception as exc:
        if qdrant is not None:
            qdrant.close()
        row.status = BuildStatus.FAILED.value
        row.error_type = type(exc).__name__
        if temporary.exists():
            shutil.rmtree(temporary)
        raise ArticleIndexBuildError(
            build_id,
            profile_id,
            snapshot,
            final_path,
            type(exc).__name__,
            round((time.perf_counter() - started) * 1000),
        ) from exc


def record_article_build_failure(session: Session, error: ArticleIndexBuildError) -> None:
    row = session.get(ArticleIndexBuildRow, error.build_id)
    if row is None:
        row = ArticleIndexBuildRow(
            id=error.build_id,
            profile_id=error.profile_id,
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


def _load_documents(path: Path) -> dict[str, ArticleSearchDocument]:
    payload = json.loads((path / DOCUMENTS_FILE).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("文章索引文档清单格式错误")
    documents = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("文章索引文档格式错误")
        compatible = dict(item)
        compatible.setdefault("subtopics", [])
        documents.append(ArticleSearchDocument(**compatible))
    return {item.document_id: item for item in documents}


def _bm25_search(path: Path, query: str, limit: int) -> list[RankedHit]:
    retriever = bm25s.BM25.load(path / "bm25", load_corpus=True)
    available = int(retriever.scores["num_docs"])
    results, scores = retriever.retrieve(
        [ChineseTokenizer().tokenize(query)],
        k=min(limit, available),
        show_progress=False,
    )
    hits: list[RankedHit] = []
    for rank, (metadata, score) in enumerate(zip(results[0], scores[0], strict=True), start=1):
        document_id = metadata.get("document_id") if isinstance(metadata, dict) else None
        if not isinstance(document_id, str):
            raise RuntimeError("文章 BM25 结果缺少 document_id")
        hits.append(RankedHit(document_id, float(score), rank))
    return hits


def _dense_search(
    path: Path, query: str, limit: int, provider: EmbeddingProvider
) -> list[RankedHit]:
    vectors = np.asarray(provider.encode_queries([query]), dtype=np.float32)
    if vectors.shape != (1, provider.dimension):
        raise ValueError(
            f"查询 Embedding 输出形状错误：期望 {(1, provider.dimension)}，实际 {vectors.shape}"
        )
    client = QdrantClient(path=str(path / "qdrant"))
    try:
        points = client.query_points(
            collection_name=ARTICLE_COLLECTION,
            query=vectors[0].tolist(),
            limit=limit,
            with_payload=True,
        ).points
    finally:
        client.close()
    hits: list[RankedHit] = []
    for rank, point in enumerate(points, start=1):
        payload = point.payload or {}
        document_id = payload.get("document_id")
        if not isinstance(document_id, str):
            raise RuntimeError("文章 Dense 结果缺少 document_id")
        hits.append(RankedHit(document_id, float(point.score), rank))
    return hits


def _matched_topics(query: str, document: ArticleSearchDocument) -> list[str]:
    tokenizer = ChineseTokenizer()
    query_tokens = set(tokenizer.tokenize(query))
    scored: list[tuple[int, str]] = []
    for title in document.topic_titles:
        searchable = " ".join([title, *document.topic_keywords.get(title, [])])
        topic_tokens = set(tokenizer.tokenize(searchable))
        score = len(query_tokens & topic_tokens)
        if title.casefold() in query.casefold() or query.casefold() in title.casefold():
            score += 2
        if score:
            scored.append((score, title))
    return [title for _, title in sorted(scored, key=lambda item: (-item[0], item[1]))[:3]]


def _matched_subtopics(query: str, document: ArticleSearchDocument) -> list[str]:
    tokenizer = ChineseTokenizer()
    query_tokens = set(tokenizer.tokenize(query))
    scored: list[tuple[int, str]] = []
    for item in document.subtopics:
        title = str(item.get("title", "")).strip()
        summary = str(item.get("summary", "")).strip()
        if not title:
            continue
        tokens = set(tokenizer.tokenize(f"{title} {summary}"))
        score = len(query_tokens & tokens)
        if title.casefold() in query.casefold() or query.casefold() in title.casefold():
            score += 2
        if score:
            scored.append((score, title))
    return [title for _, title in sorted(scored, key=lambda item: (-item[0], item[1]))[:3]]


def search_articles(
    session: Session,
    query: str,
    top_k: int,
    *,
    mode: RetrievalMode = RetrievalMode.HYBRID,
    embedding_provider: EmbeddingProvider | None = None,
    build_id: str | None = None,
    bm25_candidate_k: int = 5,
    dense_candidate_k: int = 5,
    rrf_k: int = 60,
) -> list[ArticleSearchResult]:
    if not query.strip():
        raise ValueError("查询不能为空")
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")
    selected_mode = RetrievalMode(mode)
    if build_id is None:
        build = session.scalar(
            select(ArticleIndexBuildRow).where(
                ArticleIndexBuildRow.active.is_(True),
                ArticleIndexBuildRow.status == BuildStatus.ACTIVE.value,
            )
        )
    else:
        build = session.get(ArticleIndexBuildRow, build_id)
        if build is not None and build.status not in {
            BuildStatus.ACTIVE.value,
            BuildStatus.SUPERSEDED.value,
        }:
            build = None
    if build is None:
        raise ValueError("没有可用的活动文章级索引")
    path = Path(build.path)
    if not path.is_dir():
        raise FileNotFoundError(f"文章索引目录不存在：{path}")
    documents = _load_documents(path)
    bm25_hits = (
        _bm25_search(path, query, min(len(documents), max(top_k, bm25_candidate_k)))
        if selected_mode in {RetrievalMode.BM25, RetrievalMode.HYBRID}
        else []
    )
    dense_hits = (
        _dense_search(
            path,
            query,
            min(len(documents), max(top_k, dense_candidate_k)),
            embedding_provider,
        )
        if selected_mode in {RetrievalMode.DENSE, RetrievalMode.HYBRID}
        and embedding_provider is not None
        else []
    )
    if selected_mode in {RetrievalMode.DENSE, RetrievalMode.HYBRID} and embedding_provider is None:
        raise ValueError("Dense 或 Hybrid 文章检索需要 Embedding provider")
    if selected_mode == RetrievalMode.BM25:
        fused = [(hit.chunk_id, None, hit, None) for hit in bm25_hits]
    elif selected_mode == RetrievalMode.DENSE:
        fused = [(hit.chunk_id, None, None, hit) for hit in dense_hits]
    else:
        fused = [
            (hit.chunk_id, hit.rrf_score, hit.bm25, hit.dense)
            for hit in reciprocal_rank_fusion(bm25_hits, dense_hits, rrf_k)
        ]
    results: list[ArticleSearchResult] = []
    for document_id, rrf_score, bm25, dense in fused[:top_k]:
        document = documents.get(document_id)
        if document is None:
            raise RuntimeError(f"文章索引结果引用未知文档：{document_id}")
        sources = []
        if bm25 is not None:
            sources.append("bm25")
        if dense is not None:
            sources.append("dense")
        results.append(
            ArticleSearchResult(
                article_id=document.article_id,
                version_id=document.version_id,
                title=document.title,
                published_at=document.published_at,
                article_summary=document.article_summary,
                matched_topics=_matched_topics(query, document),
                matched_subtopics=_matched_subtopics(query, document),
                sources=sources,
                rrf_score=rrf_score,
                bm25_score=bm25.score if bm25 is not None else None,
                bm25_rank=bm25.rank if bm25 is not None else None,
                dense_score=dense.score if dense is not None else None,
                dense_rank=dense.rank if dense is not None else None,
            )
        )
    return results
