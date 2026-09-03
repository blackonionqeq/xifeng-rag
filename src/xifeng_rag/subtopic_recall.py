"""正查评测的细话题召回通道。

问题命中已批准细话题单元后，提名其父 TopicChunk 进入候选池，
与常规 BM25/Dense 通道做加权 RRF 融合。仅用于 `xifeng eval run --subtopics`
实验链路，不改变生产检索默认行为。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import bm25s
from sqlalchemy import select
from sqlalchemy.orm import Session

from xifeng_rag.domain import ArticleSubtopics, RetrievalMode
from xifeng_rag.indexing import (
    EmbeddingProvider,
    RankedHit,
    RetrievalCandidate,
    RetrievalScores,
    _bm25_search,
    _dense_search,
    rerank_candidates,
    weighted_reciprocal_rank_fusion,
)
from xifeng_rag.layered_article_search import (
    METADATA_FILE,
    ArticleDiscoveryLayer,
    _load_build,
    _unit_hits,
    load_layered_suite,
)
from xifeng_rag.reranking import Reranker
from xifeng_rag.storage import (
    ArticleIndexBuildRow,
    ArticleRow,
    ArticleVersionRow,
    SubtopicSetRow,
    TopicChunkRow,
)

SUBTOPIC_BM25_CHANNEL = "subtopic-bm25"
SUBTOPIC_DENSE_CHANNEL = "subtopic-dense"

# 套件单元的 title 是文章标题而非细话题标题, 因此用
# (version_id, start_unit, end_unit) 对齐; 同一父话题下细话题范围不重叠。
UnitKey = tuple[str, int, int]


@dataclass(frozen=True)
class SubtopicChunkMap:
    """细话题单元到候选宇宙内 TopicChunk 的映射及审计信息。"""

    unit_key_to_chunk: dict[UnitKey, str]
    universe_chunk_ids: frozenset[str]
    suite_unit_count: int
    mapped_unit_count: int
    dropped_unit_count: int
    covered_chunk_ids: frozenset[str]
    covered_version_ids: frozenset[str]
    covered_article_count: int


@dataclass(frozen=True)
class SubtopicQueryAudit:
    """单次查询的细话题通道审计。"""

    nominated_chunk_ids: tuple[str, ...]
    nominated_subtopic_titles: tuple[str, ...]
    dropped_hit_count: int


@dataclass(frozen=True)
class SubtopicRecallContext:
    """运行开始时一次性构建的细话题通道上下文。"""

    build_id: str
    build_path: Path
    units: dict[str, dict[str, object]]
    chunk_map: SubtopicChunkMap
    corpus_snapshot: str
    embedding_metadata: dict[str, object]


def read_build_universe(build_path: Path) -> frozenset[str]:
    """读取话题块 Build 的 BM25 corpus，得到候选宇宙 chunk_id 集合。"""
    retriever = bm25s.BM25.load(build_path / "bm25", load_corpus=True)
    corpus = retriever.corpus
    if corpus is None:
        raise ValueError(f"Build 缺少 BM25 corpus：{build_path}")
    return frozenset(
        item["chunk_id"] if isinstance(item, dict) else str(item) for item in corpus
    )


def load_subtopic_build(
    session: Session, suite_path: Path
) -> tuple[ArticleIndexBuildRow, Path, dict[str, dict[str, object]]]:
    """从分层套件清单解析 subtopics 层 Build。"""
    if not suite_path.is_file():
        raise FileNotFoundError(
            "分层文章反查套件清单不存在，请运行 "
            "`uv run xifeng articles build-layered-index --candidate-k 20`："
            f"{suite_path}"
        )
    build_id = load_layered_suite(suite_path)[ArticleDiscoveryLayer.SUBTOPICS]
    build, _metadata, units = _load_build(
        session, build_id, ArticleDiscoveryLayer.SUBTOPICS
    )
    path = Path(build.path)
    if not path.is_dir():
        raise FileNotFoundError(f"细话题层索引目录不存在：{path}")
    return build, path, units


def _unit_key(version_id: str, start_unit: int, end_unit: int) -> UnitKey:
    return (version_id, start_unit, end_unit)


def build_subtopic_chunk_map(
    session: Session,
    units: dict[str, dict[str, object]],
    universe_chunk_ids: frozenset[str],
) -> SubtopicChunkMap:
    """把已批准细话题条目映射到候选宇宙内的父 TopicChunk。

    join 键：(version_id, parent_start_unit, parent_end_unit) 精确匹配
    TopicChunkRow 的 (version_id, start_paragraph, end_paragraph)。
    父块不在候选宇宙内的条目丢弃并计数；套件非空但映射为空时报错。
    """
    chunk_rows = list(
        session.scalars(
            select(TopicChunkRow).where(TopicChunkRow.id.in_(universe_chunk_ids))
        ).all()
    )
    by_range: dict[tuple[str, int, int], str] = {
        (row.version_id, row.start_paragraph, row.end_paragraph): row.id
        for row in chunk_rows
    }
    approved_rows = list(
        session.scalars(
            select(SubtopicSetRow).where(SubtopicSetRow.status == "approved")
        ).all()
    )
    latest_per_version: dict[str, SubtopicSetRow] = {}
    for row in approved_rows:
        current = latest_per_version.get(row.version_id)
        if current is None or row.created_at > current.created_at:
            latest_per_version[row.version_id] = row

    mapped: dict[UnitKey, str] = {}
    dropped_unit_count = 0
    covered_chunks: set[str] = set()
    covered_versions: set[str] = set()
    for version_id, row in latest_per_version.items():
        article = ArticleSubtopics.model_validate_json(row.subtopics_json)
        for topic in article.topics:
            chunk_id = by_range.get(
                (version_id, topic.parent_start_unit, topic.parent_end_unit)
            )
            if chunk_id is None:
                dropped_unit_count += len(topic.subtopics)
                continue
            covered_chunks.add(chunk_id)
            covered_versions.add(version_id)
            for subtopic in topic.subtopics:
                mapped[_unit_key(version_id, subtopic.start_unit, subtopic.end_unit)] = (
                    chunk_id
                )

    suite_unit_count = len(units)
    if units and not mapped:
        raise ValueError(
            "细话题套件与指定 Build 没有共同候选块，请检查套件与 --build 是否匹配，"
            "或运行 `uv run xifeng articles build-layered-index --candidate-k 20` 重建套件"
        )
    article_ids = {row.article_id for row in chunk_rows if row.id in covered_chunks}
    return SubtopicChunkMap(
        unit_key_to_chunk=mapped,
        universe_chunk_ids=frozenset(universe_chunk_ids),
        suite_unit_count=suite_unit_count,
        mapped_unit_count=len(mapped),
        dropped_unit_count=dropped_unit_count,
        covered_chunk_ids=frozenset(covered_chunks),
        covered_version_ids=frozenset(covered_versions),
        covered_article_count=len(article_ids),
    )


def _unit_hit_key(unit: dict[str, object]) -> UnitKey | None:
    version_id = unit.get("version_id")
    start_unit = unit.get("start_unit")
    end_unit = unit.get("end_unit")
    if (
        not isinstance(version_id, str)
        or not isinstance(start_unit, int)
        or not isinstance(end_unit, int)
    ):
        return None
    return _unit_key(version_id, start_unit, end_unit)


def nominate_subtopic_chunks(
    subtopic_path: Path,
    units: dict[str, dict[str, object]],
    chunk_map: SubtopicChunkMap,
    query: str,
    *,
    mode: RetrievalMode,
    limit: int,
    provider: EmbeddingProvider | None,
) -> tuple[list[RankedHit], SubtopicQueryAudit]:
    """跑一层细话题检索单元，把宇宙内命中映射为父 chunk 提名。"""
    hits = _unit_hits(subtopic_path, units, query, mode, limit, provider)
    ranked: list[RankedHit] = []
    nominated_chunk_ids: list[str] = []
    nominated_titles: list[str] = []
    dropped_hit_count = 0
    seen_chunks: set[str] = set()
    for hit in hits:
        key = _unit_hit_key(hit.unit)
        chunk_id = chunk_map.unit_key_to_chunk.get(key) if key is not None else None
        if chunk_id is None:
            dropped_hit_count += 1
            continue
        # 套件单元的 title 是文章标题, 审计改记父话题标题定位命中来源.
        nominated_titles.append(str(hit.unit.get("parent_topic", "")))
        if chunk_id in seen_chunks:
            # 同一父 chunk 被多个细话题命中时只保留首个(rank 最优)提名。
            continue
        seen_chunks.add(chunk_id)
        nominated_chunk_ids.append(chunk_id)
        ranked.append(
            RankedHit(chunk_id=chunk_id, score=hit.score, rank=len(ranked) + 1)
        )
    return ranked, SubtopicQueryAudit(
        nominated_chunk_ids=tuple(nominated_chunk_ids),
        nominated_subtopic_titles=tuple(nominated_titles),
        dropped_hit_count=dropped_hit_count,
    )


def build_subtopic_context(
    session: Session,
    *,
    suite_path: Path,
    universe_chunk_ids: frozenset[str],
    embedding_metadata: dict[str, object] | None,
) -> SubtopicRecallContext:
    """加载套件、校验 Embedding 一致性并构建映射。"""
    build, path, units = load_subtopic_build(session, suite_path)
    metadata = json.loads((path / METADATA_FILE).read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("细话题层 Build metadata 格式错误")
    if (
        embedding_metadata is not None
        and metadata.get("embedding") != embedding_metadata
    ):
        raise ValueError("细话题层 Embedding Profile 与当前配置不一致")
    chunk_map = build_subtopic_chunk_map(session, units, universe_chunk_ids)
    return SubtopicRecallContext(
        build_id=build.id,
        build_path=path,
        units=units,
        chunk_map=chunk_map,
        corpus_snapshot=build.corpus_snapshot,
        embedding_metadata=metadata.get("embedding") or {},
    )


def search_with_subtopic_candidates(
    session: Session,
    query: str,
    top_k: int,
    *,
    embedding_provider: EmbeddingProvider,
    bm25_candidate_k: int,
    dense_candidate_k: int,
    rrf_k: int,
    reranker: Reranker,
    rerank_candidate_k: int,
    context: SubtopicRecallContext,
    build_path: Path,
    subtopic_candidate_k: int,
    subtopic_weight: float,
) -> tuple[list[RetrievalCandidate], SubtopicQueryAudit]:
    """常规 BM25/Dense 通道叠加细话题提名的四路加权 RRF 检索。"""
    if not query.strip():
        raise ValueError("查询不能为空")
    if subtopic_candidate_k <= 0:
        raise ValueError("subtopic_candidate_k 必须大于 0")
    if subtopic_weight <= 0:
        raise ValueError("subtopic_weight 必须大于 0")
    candidate_k = max(top_k, rerank_candidate_k)
    bm25_hits = _bm25_search(build_path, query, max(candidate_k, bm25_candidate_k))
    dense_hits = _dense_search(
        build_path, query, max(candidate_k, dense_candidate_k), embedding_provider
    )
    subtopic_bm25_hits, bm25_audit = nominate_subtopic_chunks(
        context.build_path,
        context.units,
        context.chunk_map,
        query,
        mode=RetrievalMode.BM25,
        limit=subtopic_candidate_k,
        provider=None,
    )
    subtopic_dense_hits, dense_audit = nominate_subtopic_chunks(
        context.build_path,
        context.units,
        context.chunk_map,
        query,
        mode=RetrievalMode.DENSE,
        limit=subtopic_candidate_k,
        provider=embedding_provider,
    )
    fused = weighted_reciprocal_rank_fusion(
        {
            "bm25": (bm25_hits, 1.0),
            "dense": (dense_hits, 1.0),
            SUBTOPIC_BM25_CHANNEL: (subtopic_bm25_hits, subtopic_weight),
            SUBTOPIC_DENSE_CHANNEL: (subtopic_dense_hits, subtopic_weight),
        },
        rrf_k=rrf_k,
    )
    candidates: list[RetrievalCandidate] = []
    for rrf_rank, hit in enumerate(fused[:candidate_k], start=1):
        chunk = session.get(TopicChunkRow, hit.chunk_id)
        if chunk is None:
            raise RuntimeError(f"索引引用了不存在的话题块：{hit.chunk_id}")
        version = session.get(ArticleVersionRow, chunk.version_id)
        article = session.get(ArticleRow, chunk.article_id)
        if version is None or article is None:
            raise RuntimeError(f"话题块缺少文章元数据：{hit.chunk_id}")
        sources: list[str] = []
        if "bm25" in hit.channels:
            sources.append("bm25")
        if "dense" in hit.channels:
            sources.append("dense")
        if SUBTOPIC_BM25_CHANNEL in hit.channels:
            sources.append(SUBTOPIC_BM25_CHANNEL)
        if SUBTOPIC_DENSE_CHANNEL in hit.channels:
            sources.append(SUBTOPIC_DENSE_CHANNEL)
        subtopic_bm25 = hit.channels.get(SUBTOPIC_BM25_CHANNEL)
        subtopic_dense = hit.channels.get(SUBTOPIC_DENSE_CHANNEL)
        bm25_hit = hit.channels.get("bm25")
        dense_hit = hit.channels.get("dense")
        candidates.append(
            RetrievalCandidate(
                chunk_id=chunk.id,
                article_id=chunk.article_id,
                article_version=chunk.version_id,
                article_title=article.title,
                original_text=chunk.original_text,
                paragraph_range=(chunk.start_paragraph, chunk.end_paragraph),
                retrieval_scores=RetrievalScores(
                    bm25=bm25_hit.score if bm25_hit else None,
                    bm25_rank=bm25_hit.rank if bm25_hit else None,
                    dense=dense_hit.score if dense_hit else None,
                    dense_rank=dense_hit.rank if dense_hit else None,
                    rrf=hit.rrf_score,
                    rrf_rank=rrf_rank,
                    subtopic_bm25=subtopic_bm25.score if subtopic_bm25 else None,
                    subtopic_bm25_rank=subtopic_bm25.rank if subtopic_bm25 else None,
                    subtopic_dense=subtopic_dense.score if subtopic_dense else None,
                    subtopic_dense_rank=subtopic_dense.rank if subtopic_dense else None,
                ),
                source=sources,
                topic_title=chunk.topic_title,
                context_summary=chunk.context_summary,
            )
        )
    audit = SubtopicQueryAudit(
        nominated_chunk_ids=(
            bm25_audit.nominated_chunk_ids + dense_audit.nominated_chunk_ids
        ),
        nominated_subtopic_titles=(
            bm25_audit.nominated_subtopic_titles
            + dense_audit.nominated_subtopic_titles
        ),
        dropped_hit_count=bm25_audit.dropped_hit_count + dense_audit.dropped_hit_count,
    )
    reranked = rerank_candidates(query, candidates, reranker, top_k)
    return reranked, audit
