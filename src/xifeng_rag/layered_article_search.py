from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import bm25s
import numpy as np
from qdrant_client import QdrantClient, models
from sqlalchemy import select
from sqlalchemy.orm import Session

from xifeng_rag.article_search import selected_article_documents
from xifeng_rag.corpus import stable_id
from xifeng_rag.domain import BuildStatus, RetrievalMode
from xifeng_rag.indexing import ChineseTokenizer, EmbeddingProvider, RankedHit
from xifeng_rag.layered_retrieval import AtomicUnit, ParentChunk, split_parent_chunk
from xifeng_rag.storage import (
    ArticleIndexBuildRow,
    ParagraphRow,
    TopicChunkRow,
)

PROFILE_VERSION = "layered-article-discovery-v1"
COLLECTION = "article_discovery_units"
UNITS_FILE = "units.json"
METADATA_FILE = "metadata.json"
LATEST_SUITE_FILE = "latest-layered-suite.json"


class ArticleDiscoveryLayer(StrEnum):
    OVERVIEW = "overview"
    KEYWORDS = "keywords"
    SUBTOPICS = "subtopics"
    ORIGINAL = "original"


class ArticleAggregation(StrEnum):
    BEST_UNIT = "best_unit"
    TOP_N_RANK_FUSION = "top_n_rank_fusion"


@dataclass(frozen=True)
class ArticleLayerUnit:
    unit_id: str
    unit_type: ArticleDiscoveryLayer
    article_id: str
    version_id: str
    title: str
    parent_topic: str | None
    parent_chunk_id: str | None
    start_unit: int | None
    end_unit: int | None
    search_text: str

    def safe_payload(self) -> dict[str, object]:
        return {
            "unit_id": self.unit_id,
            "unit_type": self.unit_type.value,
            "article_id": self.article_id,
            "version_id": self.version_id,
            "title": self.title,
            "parent_topic": self.parent_topic,
            "parent_chunk_id": self.parent_chunk_id,
            "start_unit": self.start_unit,
            "end_unit": self.end_unit,
        }


@dataclass(frozen=True)
class LayerBuildOutcome:
    layer: ArticleDiscoveryLayer
    build_id: str
    profile_id: str
    corpus_snapshot: str
    unit_count: int
    article_count: int
    path: Path
    duration_ms: int


@dataclass(frozen=True)
class LayeredBuildSuite:
    suite_id: str
    corpus_snapshot: str
    builds: dict[ArticleDiscoveryLayer, LayerBuildOutcome]
    manifest_path: Path


@dataclass(frozen=True)
class UnitSearchHit:
    unit: dict[str, object]
    score: float
    rank: int


@dataclass(frozen=True)
class AggregatedArticleHit:
    article_id: str
    version_id: str
    title: str
    score: float
    rank: int
    best_unit_id: str
    best_unit_score: float
    best_unit_rank: int
    parent_topic: str | None
    start_unit: int | None
    end_unit: int | None
    unit_count: int
    contributing_units: tuple[tuple[str, int, float], ...]


def _deduplicated_keywords(
    title: str, topic_titles: list[str], topic_keywords: dict[str, list[str]]
) -> list[str]:
    represented = "\n".join([title, *topic_titles]).casefold()
    result: list[str] = []
    seen: set[str] = set()
    for topic_title in topic_titles:
        for value in topic_keywords.get(topic_title, []):
            keyword = value.strip()
            normalized = keyword.casefold()
            if not keyword or normalized in seen or normalized in represented:
                continue
            seen.add(normalized)
            result.append(keyword)
    return result


def selected_layer_units(
    session: Session,
    *,
    splitter_version: str = "atomic-greedy-v1",
    target_chars: int = 400,
    max_chars: int = 600,
    tail_merge_chars: int = 200,
) -> dict[ArticleDiscoveryLayer, list[ArticleLayerUnit]]:
    documents = selected_article_documents(session)
    by_layer = {layer: [] for layer in ArticleDiscoveryLayer}
    for document in documents:
        overview_text = "\n".join((document.title, document.article_summary))
        by_layer[ArticleDiscoveryLayer.OVERVIEW].append(
            ArticleLayerUnit(
                unit_id=stable_id(
                    "article-layer-overview",
                    document.article_id,
                    document.version_id,
                    document.overview_id,
                    overview_text,
                    length=48,
                ),
                unit_type=ArticleDiscoveryLayer.OVERVIEW,
                article_id=document.article_id,
                version_id=document.version_id,
                title=document.title,
                parent_topic=None,
                parent_chunk_id=None,
                start_unit=None,
                end_unit=None,
                search_text=overview_text,
            )
        )
        keywords = _deduplicated_keywords(
            document.title, document.topic_titles, document.topic_keywords
        )
        keyword_text = "\n".join([document.title, *document.topic_titles, *keywords])
        by_layer[ArticleDiscoveryLayer.KEYWORDS].append(
            ArticleLayerUnit(
                unit_id=stable_id(
                    "article-layer-keywords",
                    document.article_id,
                    document.version_id,
                    document.manifest_id,
                    keyword_text,
                    length=48,
                ),
                unit_type=ArticleDiscoveryLayer.KEYWORDS,
                article_id=document.article_id,
                version_id=document.version_id,
                title=document.title,
                parent_topic=None,
                parent_chunk_id=None,
                start_unit=None,
                end_unit=None,
                search_text=keyword_text,
            )
        )
        for item in document.subtopics:
            subtopic_title = str(item["title"])
            parent_title = str(item["parent_title"])
            start_unit = int(str(item["start_unit"]))
            end_unit = int(str(item["end_unit"]))
            text = "\n".join(
                (document.title, parent_title, subtopic_title, str(item["summary"]))
            )
            by_layer[ArticleDiscoveryLayer.SUBTOPICS].append(
                ArticleLayerUnit(
                    unit_id=stable_id(
                        "article-layer-subtopic",
                        document.article_id,
                        document.version_id,
                        document.manifest_id,
                        parent_title,
                        subtopic_title,
                        f"{start_unit}:{end_unit}",
                        text,
                        length=48,
                    ),
                    unit_type=ArticleDiscoveryLayer.SUBTOPICS,
                    article_id=document.article_id,
                    version_id=document.version_id,
                    title=document.title,
                    parent_topic=parent_title,
                    parent_chunk_id=None,
                    start_unit=start_unit,
                    end_unit=end_unit,
                    search_text=text,
                )
            )

    versions = {item.version_id: item for item in documents}
    chunks = list(
        session.scalars(
            select(TopicChunkRow)
            .where(TopicChunkRow.version_id.in_(versions))
            .order_by(TopicChunkRow.version_id, TopicChunkRow.start_paragraph)
        ).all()
    )
    paragraphs = list(
        session.scalars(
            select(ParagraphRow)
            .where(ParagraphRow.version_id.in_(versions))
            .order_by(ParagraphRow.version_id, ParagraphRow.number)
        ).all()
    )
    by_version: dict[str, list[ParagraphRow]] = {}
    for paragraph in paragraphs:
        by_version.setdefault(paragraph.version_id, []).append(paragraph)
    for chunk in chunks:
        document = versions.get(chunk.version_id)
        if document is None:
            continue
        units = [
            AtomicUnit(number=item.number, text=item.text)
            for item in by_version.get(chunk.version_id, [])
            if chunk.start_paragraph <= item.number <= chunk.end_paragraph
        ]
        children = split_parent_chunk(
            ParentChunk(
                id=chunk.id,
                article_id=chunk.article_id,
                version_id=chunk.version_id,
                start_paragraph=chunk.start_paragraph,
                end_paragraph=chunk.end_paragraph,
            ),
            units,
            splitter_version=splitter_version,
            target_chars=target_chars,
            max_chars=max_chars,
            tail_merge_chars=tail_merge_chars,
        )
        for child in children:
            by_layer[ArticleDiscoveryLayer.ORIGINAL].append(
                ArticleLayerUnit(
                    unit_id=stable_id(
                        "article-layer-original", child.id, document.title, length=48
                    ),
                    unit_type=ArticleDiscoveryLayer.ORIGINAL,
                    article_id=child.article_id,
                    version_id=child.article_version,
                    title=document.title,
                    parent_topic=chunk.topic_title,
                    parent_chunk_id=chunk.id,
                    start_unit=child.start_paragraph,
                    end_unit=child.end_paragraph,
                    search_text=f"{document.title}\n{child.original_text}",
                )
            )
    return {
        layer: sorted(units, key=lambda item: item.unit_id)
        for layer, units in by_layer.items()
    }


def corpus_snapshot_for_units(
    units_by_layer: dict[ArticleDiscoveryLayer, list[ArticleLayerUnit]],
) -> str:
    semantic = [
        f"{layer.value}:{unit.unit_id}"
        for layer in ArticleDiscoveryLayer
        for unit in units_by_layer[layer]
    ]
    return hashlib.sha256("\n".join(semantic).encode("utf-8")).hexdigest()


def _profile_id(
    layer: ArticleDiscoveryLayer,
    provider: EmbeddingProvider,
    *,
    candidate_k: int,
    splitter_version: str,
    target_chars: int,
    max_chars: int,
    tail_merge_chars: int,
) -> str:
    semantic: dict[str, object] = {
        "profile_version": PROFILE_VERSION,
        "layer": layer.value,
        "tokenizer": "jieba-search-v1",
        "embedding": provider.metadata,
        "candidate_k": candidate_k,
    }
    if layer == ArticleDiscoveryLayer.ORIGINAL:
        semantic["splitter"] = {
            "version": splitter_version,
            "target_chars": target_chars,
            "max_chars": max_chars,
            "tail_merge_chars": tail_merge_chars,
        }
    return stable_id(
        "layered-article-profile",
        json.dumps(semantic, ensure_ascii=False, sort_keys=True),
        length=48,
    )


def _point_id(unit_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"xifeng-rag:article-layer:{unit_id}"))


def _build_layer(
    session: Session,
    index_root: Path,
    provider: EmbeddingProvider,
    layer: ArticleDiscoveryLayer,
    units: list[ArticleLayerUnit],
    corpus_snapshot: str,
    *,
    candidate_k: int,
    splitter_version: str,
    target_chars: int,
    max_chars: int,
    tail_merge_chars: int,
) -> LayerBuildOutcome:
    started = time.perf_counter()
    profile_id = _profile_id(
        layer,
        provider,
        candidate_k=candidate_k,
        splitter_version=splitter_version,
        target_chars=target_chars,
        max_chars=max_chars,
        tail_merge_chars=tail_merge_chars,
    )
    build_id = stable_id(
        "layered-article-index", profile_id, corpus_snapshot, length=32
    )
    final_path = index_root / build_id
    existing = session.get(ArticleIndexBuildRow, build_id)
    if existing is not None and final_path.is_dir():
        return LayerBuildOutcome(
            layer,
            build_id,
            profile_id,
            corpus_snapshot,
            len(units),
            len({item.article_id for item in units}),
            final_path,
            existing.duration_ms or 0,
        )

    index_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".building-{layer.value}-", dir=index_root))
    row = existing or ArticleIndexBuildRow(
        id=build_id,
        profile_id=profile_id,
        corpus_snapshot=corpus_snapshot,
        path=str(final_path),
        status=BuildStatus.BUILDING.value,
        active=False,
    )
    session.add(row)
    session.flush()
    client: QdrantClient | None = None
    try:
        if units:
            corpus = [{"unit_id": item.unit_id} for item in units]
            retriever = bm25s.BM25(corpus=corpus)
            tokenizer = ChineseTokenizer()
            retriever.index(
                [tokenizer.tokenize(item.search_text) for item in units],
                show_progress=False,
            )
            retriever.save(temporary / "bm25", corpus=corpus)
            vectors = np.asarray(
                provider.encode_documents([item.search_text for item in units]),
                dtype=np.float32,
            )
            if vectors.shape != (len(units), provider.dimension):
                raise ValueError(
                    f"Embedding 输出形状错误：期望 {(len(units), provider.dimension)}，"
                    f"实际 {vectors.shape}"
                )
            client = QdrantClient(path=str(temporary / "qdrant"))
            client.create_collection(
                collection_name=COLLECTION,
                vectors_config=models.VectorParams(
                    size=provider.dimension, distance=models.Distance.COSINE
                ),
            )
            client.upload_points(
                collection_name=COLLECTION,
                points=[
                    models.PointStruct(
                        id=_point_id(unit.unit_id),
                        vector=vector.tolist(),
                        payload={"unit_id": unit.unit_id},
                    )
                    for unit, vector in zip(units, vectors, strict=True)
                ],
            )
            if client.count(COLLECTION, exact=True).count != len(units):
                raise RuntimeError("文章发现层 Dense point 数量不一致")
            client.close()
            client = None
        (temporary / UNITS_FILE).write_text(
            json.dumps(
                [item.safe_payload() for item in units], ensure_ascii=False, indent=2
            ),
            encoding="utf-8",
        )
        metadata = {
            "build_id": build_id,
            "profile_id": profile_id,
            "profile_version": PROFILE_VERSION,
            "layer": layer.value,
            "corpus_snapshot": corpus_snapshot,
            "unit_count": len(units),
            "article_count": len({item.article_id for item in units}),
            "candidate_k": candidate_k,
            "embedding": provider.metadata,
            "splitter": (
                {
                    "version": splitter_version,
                    "target_chars": target_chars,
                    "max_chars": max_chars,
                    "tail_merge_chars": tail_merge_chars,
                }
                if layer == ArticleDiscoveryLayer.ORIGINAL
                else None
            ),
        }
        (temporary / METADATA_FILE).write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if final_path.exists():
            raise FileExistsError(f"不可变分层文章索引目录已存在：{final_path}")
        os.replace(temporary, final_path)
        duration_ms = round((time.perf_counter() - started) * 1000)
        row.status = BuildStatus.READY.value
        row.active = False
        row.error_type = None
        row.duration_ms = duration_ms
        return LayerBuildOutcome(
            layer,
            build_id,
            profile_id,
            corpus_snapshot,
            len(units),
            len({item.article_id for item in units}),
            final_path,
            duration_ms,
        )
    except Exception as exc:
        if client is not None:
            client.close()
        row.status = BuildStatus.FAILED.value
        row.active = False
        row.error_type = type(exc).__name__
        row.duration_ms = round((time.perf_counter() - started) * 1000)
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def build_layered_article_indexes(
    session: Session,
    index_root: Path,
    provider: EmbeddingProvider,
    *,
    candidate_k: int = 20,
    splitter_version: str = "atomic-greedy-v1",
    target_chars: int = 400,
    max_chars: int = 600,
    tail_merge_chars: int = 200,
) -> LayeredBuildSuite:
    units_by_layer = selected_layer_units(
        session,
        splitter_version=splitter_version,
        target_chars=target_chars,
        max_chars=max_chars,
        tail_merge_chars=tail_merge_chars,
    )
    if not units_by_layer[ArticleDiscoveryLayer.OVERVIEW]:
        raise ValueError("没有带已批准文章概括的活动文章，拒绝建立空分层文章索引")
    snapshot = corpus_snapshot_for_units(units_by_layer)
    outcomes: dict[ArticleDiscoveryLayer, LayerBuildOutcome] = {}
    created_paths: list[Path] = []
    try:
        for layer in ArticleDiscoveryLayer:
            profile_id = _profile_id(
                layer,
                provider,
                candidate_k=candidate_k,
                splitter_version=splitter_version,
                target_chars=target_chars,
                max_chars=max_chars,
                tail_merge_chars=tail_merge_chars,
            )
            expected_path = index_root / stable_id(
                "layered-article-index", profile_id, snapshot, length=32
            )
            path_existed = expected_path.exists()
            outcomes[layer] = _build_layer(
                session,
                index_root,
                provider,
                layer,
                units_by_layer[layer],
                snapshot,
                candidate_k=candidate_k,
                splitter_version=splitter_version,
                target_chars=target_chars,
                max_chars=max_chars,
                tail_merge_chars=tail_merge_chars,
            )
            if not path_existed:
                created_paths.append(expected_path)
    except Exception:
        for path in created_paths:
            if path.is_dir():
                shutil.rmtree(path)
        raise
    suite_id = stable_id(
        "layered-article-suite",
        snapshot,
        *(outcomes[layer].build_id for layer in ArticleDiscoveryLayer),
        length=32,
    )
    payload = {
        "schema_version": 1,
        "suite_id": suite_id,
        "corpus_snapshot": snapshot,
        "builds": {
            layer.value: {
                "build_id": outcome.build_id,
                "profile_id": outcome.profile_id,
                "unit_count": outcome.unit_count,
                "article_count": outcome.article_count,
            }
            for layer, outcome in outcomes.items()
        },
    }
    index_root.mkdir(parents=True, exist_ok=True)
    manifest_path = index_root / f"layered-suite-{suite_id}.json"
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary_latest = index_root / f".{LATEST_SUITE_FILE}.tmp"
    temporary_latest.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary_latest, index_root / LATEST_SUITE_FILE)
    return LayeredBuildSuite(suite_id, snapshot, outcomes, manifest_path)


def load_layered_suite(path: Path) -> dict[ArticleDiscoveryLayer, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("builds"), dict):
        raise ValueError("分层文章索引套件清单格式错误")
    result: dict[ArticleDiscoveryLayer, str] = {}
    for layer in ArticleDiscoveryLayer:
        item = payload["builds"].get(layer.value)
        build_id = item.get("build_id") if isinstance(item, dict) else None
        if not isinstance(build_id, str):
            raise ValueError(f"分层文章索引套件缺少 {layer.value} Build")
        result[layer] = build_id
    return result


def _load_build(
    session: Session, build_id: str, expected_layer: ArticleDiscoveryLayer
) -> tuple[ArticleIndexBuildRow, dict[str, object], dict[str, dict[str, object]]]:
    build = session.get(ArticleIndexBuildRow, build_id)
    if build is None or build.status not in {
        BuildStatus.READY.value,
        BuildStatus.ACTIVE.value,
        BuildStatus.SUPERSEDED.value,
    }:
        raise ValueError(f"分层文章索引不可用：{build_id}")
    path = Path(build.path)
    metadata = json.loads((path / METADATA_FILE).read_text(encoding="utf-8"))
    if metadata.get("layer") != expected_layer.value:
        raise ValueError(
            f"Build {build_id} 层不匹配：{metadata.get('layer')} != {expected_layer.value}"
        )
    payload = json.loads((path / UNITS_FILE).read_text(encoding="utf-8"))
    units = {str(item["unit_id"]): item for item in payload}
    if len(units) != len(payload):
        raise ValueError("分层文章索引单元清单包含重复 ID")
    return build, metadata, units


def _unit_hits(
    path: Path,
    units: dict[str, dict[str, object]],
    query: str,
    mode: RetrievalMode,
    limit: int,
    provider: EmbeddingProvider | None,
) -> list[UnitSearchHit]:
    if not units:
        return []
    if mode == RetrievalMode.BM25:
        retriever = bm25s.BM25.load(path / "bm25", load_corpus=True)
        available = int(retriever.scores["num_docs"])
        results, scores = retriever.retrieve(
            [ChineseTokenizer().tokenize(query)],
            k=min(limit, available),
            show_progress=False,
        )
        ranked = [
            RankedHit(str(item["unit_id"]), float(score), rank)
            for rank, (item, score) in enumerate(
                zip(results[0], scores[0], strict=True), start=1
            )
        ]
    elif mode == RetrievalMode.DENSE:
        if provider is None:
            raise ValueError("Dense 分层文章检索需要 Embedding provider")
        vector = np.asarray(provider.encode_queries([query]), dtype=np.float32)
        if vector.shape != (1, provider.dimension):
            raise ValueError("查询 Embedding 输出形状错误")
        client = QdrantClient(path=str(path / "qdrant"))
        try:
            points = client.query_points(
                collection_name=COLLECTION,
                query=vector[0].tolist(),
                limit=min(limit, len(units)),
                with_payload=True,
            ).points
        finally:
            client.close()
        ranked = [
            RankedHit(str((point.payload or {})["unit_id"]), float(point.score), rank)
            for rank, point in enumerate(points, start=1)
        ]
    else:
        raise ValueError("隔离层基础通道只支持 bm25 或 dense")
    return [UnitSearchHit(units[item.chunk_id], item.score, item.rank) for item in ranked]


def aggregate_unit_hits(
    hits: list[UnitSearchHit],
    all_units: dict[str, dict[str, object]],
    *,
    aggregation: ArticleAggregation,
    top_n: int = 3,
    rrf_k: int = 60,
    top_k: int = 5,
) -> list[AggregatedArticleHit]:
    if top_n <= 0 or rrf_k <= 0 or top_k <= 0:
        raise ValueError("聚合参数必须大于 0")
    unit_counts: dict[str, int] = {}
    for unit in all_units.values():
        article_id = str(unit["article_id"])
        unit_counts[article_id] = unit_counts.get(article_id, 0) + 1
    grouped: dict[str, list[UnitSearchHit]] = {}
    for hit in hits:
        grouped.setdefault(str(hit.unit["article_id"]), []).append(hit)
    scored: list[tuple[float, UnitSearchHit, tuple[tuple[str, int, float], ...]]] = []
    for article_hits in grouped.values():
        ordered = sorted(article_hits, key=lambda item: item.rank)
        best = ordered[0]
        contributors = ordered[:top_n]
        score = (
            best.score
            if aggregation == ArticleAggregation.BEST_UNIT
            else sum(1.0 / (rrf_k + item.rank) for item in contributors)
        )
        scored.append(
            (
                score,
                best,
                tuple(
                    (str(item.unit["unit_id"]), item.rank, item.score)
                    for item in contributors
                ),
            )
        )
    scored.sort(key=lambda item: (-item[0], item[1].rank, str(item[1].unit["article_id"])))
    return [
        AggregatedArticleHit(
            article_id=str(best.unit["article_id"]),
            version_id=str(best.unit["version_id"]),
            title=str(best.unit["title"]),
            score=score,
            rank=rank,
            best_unit_id=str(best.unit["unit_id"]),
            best_unit_score=best.score,
            best_unit_rank=best.rank,
            parent_topic=(
                str(best.unit["parent_topic"])
                if best.unit.get("parent_topic") is not None
                else None
            ),
            start_unit=(
                int(str(best.unit["start_unit"]))
                if best.unit.get("start_unit") is not None
                else None
            ),
            end_unit=(
                int(str(best.unit["end_unit"]))
                if best.unit.get("end_unit") is not None
                else None
            ),
            unit_count=unit_counts[str(best.unit["article_id"])],
            contributing_units=contributors,
        )
        for rank, (score, best, contributors) in enumerate(scored[:top_k], start=1)
    ]


def search_article_layer(
    session: Session,
    build_id: str,
    layer: ArticleDiscoveryLayer,
    query: str,
    *,
    mode: RetrievalMode,
    aggregation: ArticleAggregation = ArticleAggregation.BEST_UNIT,
    embedding_provider: EmbeddingProvider | None = None,
    candidate_k: int = 20,
    top_k: int = 5,
    top_n: int = 3,
    rrf_k: int = 60,
) -> tuple[list[AggregatedArticleHit], dict[str, object]]:
    if not query.strip():
        raise ValueError("查询不能为空")
    build, metadata, units = _load_build(session, build_id, layer)
    configured_candidate_k = int(str(metadata.get("candidate_k")))
    if candidate_k != configured_candidate_k:
        raise ValueError(
            f"候选深度与 Build Profile 不一致：{candidate_k} != {configured_candidate_k}"
        )
    if mode == RetrievalMode.DENSE and (
        embedding_provider is None
        or embedding_provider.metadata != metadata.get("embedding")
    ):
        raise ValueError("Dense provider 与 Build Profile 不一致")
    started = time.perf_counter()
    unit_hits = _unit_hits(
        Path(build.path), units, query, RetrievalMode(mode), candidate_k, embedding_provider
    )
    retrieved_at = time.perf_counter()
    articles = aggregate_unit_hits(
        unit_hits,
        units,
        aggregation=aggregation,
        top_n=top_n,
        rrf_k=rrf_k,
        top_k=top_k,
    )
    completed_at = time.perf_counter()
    return articles, {
        "latency_ms": (completed_at - started) * 1000,
        "retrieval_latency_ms": (retrieved_at - started) * 1000,
        "aggregation_latency_ms": (completed_at - retrieved_at) * 1000,
        "raw_unit_hit_count": len(unit_hits),
        "unit_count": len(units),
        "article_count": int(str(metadata["article_count"])),
        "profile_id": build.profile_id,
        "corpus_snapshot": build.corpus_snapshot,
        "embedding": metadata.get("embedding"),
        "splitter": metadata.get("splitter"),
    }
