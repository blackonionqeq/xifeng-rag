from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import bm25s
import jieba
import numpy as np
from qdrant_client import QdrantClient, models
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from xifeng_rag.corpus import stable_id
from xifeng_rag.domain import (
    BuildStatus,
    LayeredRetrievalMode,
    RetrievalCandidate,
    RetrievalMode,
    RetrievalScores,
    VersionStatus,
)
from xifeng_rag.layered_retrieval import (
    AtomicUnit,
    ChildChunk,
    ChildManifestEntry,
    ParentChunk,
    read_child_manifest,
    split_parent_chunk,
    validate_candidate_limit,
    write_child_manifest,
)
from xifeng_rag.reranking import Reranker, rerank_candidates
from xifeng_rag.storage import (
    ArticleRow,
    ArticleVersionRow,
    IndexBuildRow,
    ParagraphRow,
    TopicChunkRow,
)

TOKENIZER_VERSION = "jieba-search-v1"
BM25_PROFILE_ID = "approved-topic-original-bm25-jieba-v1"
PROFILE_ID = BM25_PROFILE_ID
HYBRID_PROFILE_VERSION = "approved-topic-original-bge-bm25-rrf-v1"
QDRANT_COLLECTION = "topic_chunks"
LAYERED_PROFILE_VERSION = "approved-parent-child-original-bge-bm25-rrf-v1"
TOKEN_PATTERN = re.compile(r"[\w\u3400-\u9fff]+", re.UNICODE)


class EmbeddingProvider(Protocol):
    @property
    def dimension(self) -> int: ...

    @property
    def metadata(self) -> dict[str, object]: ...

    def encode_documents(self, texts: list[str]) -> np.ndarray: ...

    def encode_queries(self, texts: list[str]) -> np.ndarray: ...

    def close(self) -> None: ...


class SentenceTransformersEmbeddingProvider:
    def __init__(
        self,
        model_id: str,
        *,
        revision: str | None = None,
        cache_dir: Path | None = None,
        device: str = "auto",
        batch_size: int = 16,
        normalize_embeddings: bool = True,
        query_instruction: str = "为这个句子生成表示以用于检索相关文章：",
    ) -> None:
        from sentence_transformers import SentenceTransformer

        resolved_device = None if device == "auto" else device
        self._model: Any = SentenceTransformer(
            model_id,
            revision=revision,
            cache_folder=str(cache_dir) if cache_dir is not None else None,
            device=resolved_device,
        )
        dimension = self._model.get_embedding_dimension()
        if dimension is None or dimension <= 0:
            raise ValueError("Embedding 模型未提供有效向量维度")
        self._dimension = dimension
        self._batch_size = batch_size
        self._normalize = normalize_embeddings
        self._query_instruction = query_instruction
        loaded_revision = revision
        try:
            first_module = self._model._first_module()
            loaded_revision = loaded_revision or first_module.auto_model.config._commit_hash
        except (AttributeError, IndexError, TypeError):
            loaded_revision = loaded_revision or "unresolved"
        self._metadata: dict[str, object] = {
            "provider": "sentence-transformers",
            "model_id": model_id,
            "model_revision": loaded_revision,
            "dimension": dimension,
            "max_seq_length": int(self._model.max_seq_length),
            "device": str(self._model.device),
            "normalize_embeddings": normalize_embeddings,
            "query_instruction": query_instruction,
            "batch_size": batch_size,
        }

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def metadata(self) -> dict[str, object]:
        return dict(self._metadata)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode_document(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self._normalize,
        )
        return self._validate(vectors, len(texts))

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        instructed = [f"{self._query_instruction}{text}" for text in texts]
        vectors = self._model.encode_query(
            instructed,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self._normalize,
        )
        return self._validate(vectors, len(texts))

    def _validate(self, vectors: object, count: int) -> np.ndarray:
        array = np.asarray(vectors, dtype=np.float32)
        if array.shape != (count, self.dimension):
            raise ValueError(
                f"Embedding 输出形状错误：期望 {(count, self.dimension)}，实际 {array.shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError("Embedding 输出包含非有限值")
        return array

    def close(self) -> None:
        self._model = None  # type: ignore[assignment]


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
        profile_id: str = BM25_PROFILE_ID,
    ) -> None:
        super().__init__(f"索引构建失败：{error_type}")
        self.build_id = build_id
        self.snapshot = snapshot
        self.final_path = final_path
        self.error_type = error_type
        self.duration_ms = duration_ms
        self.profile_id = profile_id


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


def _hybrid_profile_id(
    provider: EmbeddingProvider,
    bm25_candidate_k: int,
    dense_candidate_k: int,
    rrf_k: int,
) -> str:
    metadata = provider.metadata
    semantic = json.dumps(
        {
            key: metadata.get(key)
            for key in (
                "provider",
                "model_id",
                "model_revision",
                "dimension",
                "max_seq_length",
                "normalize_embeddings",
                "query_instruction",
            )
        }
        | {
            "bm25_candidate_k": bm25_candidate_k,
            "dense_candidate_k": dense_candidate_k,
            "rrf_k": rrf_k,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return stable_id("profile", HYBRID_PROFILE_VERSION, TOKENIZER_VERSION, semantic, length=48)


def _qdrant_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"xifeng-rag:{chunk_id}"))


def _stored_qdrant_chunk_ids(client: QdrantClient) -> set[str]:
    stored: set[str] = set()
    offset: Any = None
    while True:
        points, offset = client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            chunk_id = payload.get("chunk_id")
            if not isinstance(chunk_id, str):
                raise RuntimeError("Dense 索引 point 缺少 chunk_id")
            stored.add(chunk_id)
        if offset is None:
            return stored


def _write_hybrid_layer(
    path: Path,
    ids: list[str],
    texts: list[str],
    payloads: list[dict[str, object]],
    provider: EmbeddingProvider,
    reusable_vectors: dict[str, np.ndarray] | None = None,
) -> None:
    tokenizer = ChineseTokenizer()
    corpus = [{"chunk_id": item_id} for item_id in ids]
    retriever = bm25s.BM25(corpus=corpus)
    retriever.index([tokenizer.tokenize(text) for text in texts], show_progress=False)
    retriever.save(path / "bm25", corpus=corpus)
    reusable_vectors = reusable_vectors or {}
    missing_indexes = [
        index for index, item_id in enumerate(ids) if item_id not in reusable_vectors
    ]
    encoded: dict[str, np.ndarray] = {}
    if missing_indexes:
        new_vectors = np.asarray(
            provider.encode_documents([texts[index] for index in missing_indexes]),
            dtype=np.float32,
        )
        if new_vectors.shape != (len(missing_indexes), provider.dimension):
            raise ValueError(
                "Embedding 输出形状错误："
                f"期望 {(len(missing_indexes), provider.dimension)}，实际 {new_vectors.shape}"
            )
        encoded = {
            ids[index]: vector
            for index, vector in zip(missing_indexes, new_vectors, strict=True)
        }
    vectors = np.asarray(
        [reusable_vectors.get(item_id, encoded.get(item_id)) for item_id in ids],
        dtype=np.float32,
    )
    if vectors.shape != (len(ids), provider.dimension):
        raise ValueError(
            f"Embedding 输出形状错误：期望 {(len(ids), provider.dimension)}，实际 {vectors.shape}"
        )
    client = QdrantClient(path=str(path / "qdrant"))
    try:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=models.VectorParams(
                size=provider.dimension,
                distance=models.Distance.COSINE,
            ),
        )
        client.upload_points(
            collection_name=QDRANT_COLLECTION,
            points=[
                models.PointStruct(
                    id=_qdrant_point_id(item_id),
                    vector=vector.tolist(),
                    payload=payload,
                )
                for item_id, vector, payload in zip(ids, vectors, payloads, strict=True)
            ],
        )
        if client.count(QDRANT_COLLECTION, exact=True).count != len(ids):
            raise RuntimeError("Dense point 数量与分层清单不一致")
        if _stored_qdrant_chunk_ids(client) != set(ids):
            raise RuntimeError("Dense point ID 与分层清单不一致")
    finally:
        client.close()
    loaded = bm25s.BM25.load(path / "bm25", load_corpus=True)
    loaded.retrieve([tokenizer.tokenize(texts[0][:80])], k=1)


def _load_reusable_vectors(
    session: Session,
    profile_id: str,
    layer: str,
    ids: list[str],
    dimension: int,
) -> dict[str, np.ndarray]:
    needed = set(ids)
    reused: dict[str, np.ndarray] = {}
    rows = list(
        session.scalars(
            select(IndexBuildRow)
            .where(
                IndexBuildRow.profile_id == profile_id,
                IndexBuildRow.status.in_(
                    [
                        BuildStatus.READY.value,
                        BuildStatus.ACTIVE.value,
                        BuildStatus.SUPERSEDED.value,
                    ]
                ),
            )
            .order_by(IndexBuildRow.created_at.desc())
        ).all()
    )
    for row in rows:
        qdrant_path = Path(row.path) / layer / "qdrant"
        if not qdrant_path.is_dir() or not needed:
            continue
        client = QdrantClient(path=str(qdrant_path))
        try:
            requested = sorted(needed)
            for offset in range(0, len(requested), 256):
                points = client.retrieve(
                    collection_name=QDRANT_COLLECTION,
                    ids=[_qdrant_point_id(item_id) for item_id in requested[offset : offset + 256]],
                    with_payload=True,
                    with_vectors=True,
                )
                for point in points:
                    payload = point.payload or {}
                    item_id = payload.get("chunk_id")
                    if not isinstance(item_id, str) or item_id not in needed:
                        continue
                    vector = np.asarray(point.vector, dtype=np.float32)
                    if vector.shape == (dimension,) and np.isfinite(vector).all():
                        reused[item_id] = vector
                        needed.remove(item_id)
        finally:
            client.close()
    return reused


def _layered_profile_id(
    provider: EmbeddingProvider,
    *,
    splitter_version: str,
    target_chars: int,
    max_chars: int,
    tail_merge_chars: int,
    parent_candidate_k: int,
    child_candidate_k: int,
    candidate_limit: int,
    rrf_k: int,
) -> str:
    semantic = json.dumps(
        {
            "profile_version": LAYERED_PROFILE_VERSION,
            "tokenizer_version": TOKENIZER_VERSION,
            "embedding": {
                key: provider.metadata.get(key)
                for key in (
                    "provider",
                    "model_id",
                    "model_revision",
                    "dimension",
                    "max_seq_length",
                    "normalize_embeddings",
                    "query_instruction",
                )
            },
            "splitter_version": splitter_version,
            "target_chars": target_chars,
            "max_chars": max_chars,
            "tail_merge_chars": tail_merge_chars,
            "parent_candidate_k": parent_candidate_k,
            "child_candidate_k": child_candidate_k,
            "candidate_limit": candidate_limit,
            "rrf_k": rrf_k,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return stable_id("profile", semantic, length=48)


def build_layered_index(
    session: Session,
    index_root: Path,
    provider: EmbeddingProvider,
    *,
    splitter_version: str = "atomic-greedy-v1",
    target_chars: int = 400,
    max_chars: int = 600,
    tail_merge_chars: int = 200,
    parent_candidate_k: int = 5,
    child_candidate_k: int = 20,
    candidate_limit: int = 128,
    rrf_k: int = 60,
) -> BuildOutcome:
    started = time.perf_counter()
    parents = _selected_chunks(session)
    if not parents:
        raise ValueError("没有已批准的话题块，拒绝建立空索引")
    version_ids = {item.version_id for item in parents}
    paragraphs = list(
        session.scalars(
            select(ParagraphRow)
            .where(ParagraphRow.version_id.in_(version_ids))
            .order_by(ParagraphRow.version_id, ParagraphRow.number)
        ).all()
    )
    by_version: dict[str, list[ParagraphRow]] = {}
    for paragraph in paragraphs:
        by_version.setdefault(paragraph.version_id, []).append(paragraph)
    children: list[ChildChunk] = []
    for parent in parents:
        units = [
            item
            for item in by_version.get(parent.version_id, [])
            if parent.start_paragraph <= item.number <= parent.end_paragraph
        ]
        children.extend(
            split_parent_chunk(
                ParentChunk(
                    id=parent.id,
                    article_id=parent.article_id,
                    version_id=parent.version_id,
                    start_paragraph=parent.start_paragraph,
                    end_paragraph=parent.end_paragraph,
                ),
                [AtomicUnit(number=item.number, text=item.text) for item in units],
                splitter_version=splitter_version,
                target_chars=target_chars,
                max_chars=max_chars,
                tail_merge_chars=tail_merge_chars,
            )
        )
    worst_case = validate_candidate_limit(
        children,
        parent_candidate_k=parent_candidate_k,
        child_candidate_k=child_candidate_k,
        candidate_limit=candidate_limit,
    )
    snapshot = hashlib.sha256("\n".join(item.id for item in parents).encode()).hexdigest()
    profile_id = _layered_profile_id(
        provider,
        splitter_version=splitter_version,
        target_chars=target_chars,
        max_chars=max_chars,
        tail_merge_chars=tail_merge_chars,
        parent_candidate_k=parent_candidate_k,
        child_candidate_k=child_candidate_k,
        candidate_limit=candidate_limit,
        rrf_k=rrf_k,
    )
    build_id = stable_id("index", profile_id, snapshot, length=32)
    final_path = index_root / build_id
    existing = session.get(IndexBuildRow, build_id)
    if existing is not None and final_path.is_dir():
        return BuildOutcome(build_id, len(children), final_path, existing.duration_ms or 0)

    index_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".building-", dir=index_root))
    row = existing or IndexBuildRow(
        id=build_id,
        profile_id=profile_id,
        corpus_snapshot=snapshot,
        path=str(final_path),
        status=BuildStatus.BUILDING.value,
        active=False,
    )
    session.add(row)
    session.flush()
    try:
        parent_path = temporary / "parent"
        child_path = temporary / "child"
        parent_path.mkdir()
        child_path.mkdir()
        reusable_parent_vectors = _load_reusable_vectors(
            session,
            profile_id,
            "parent",
            [item.id for item in parents],
            provider.dimension,
        )
        reusable_child_vectors = _load_reusable_vectors(
            session,
            profile_id,
            "child",
            [item.id for item in children],
            provider.dimension,
        )
        _write_hybrid_layer(
            parent_path,
            [item.id for item in parents],
            [item.original_text for item in parents],
            [
                {
                    "chunk_id": item.id,
                    "article_id": item.article_id,
                    "article_version": item.version_id,
                    "manifest_id": item.manifest_id,
                    "start_paragraph": item.start_paragraph,
                    "end_paragraph": item.end_paragraph,
                }
                for item in parents
            ],
            provider,
            reusable_parent_vectors,
        )
        _write_hybrid_layer(
            child_path,
            [item.id for item in children],
            [item.original_text for item in children],
            [
                {
                    "chunk_id": item.id,
                    "parent_chunk_id": item.parent_chunk_id,
                    "article_id": item.article_id,
                    "article_version": item.article_version,
                    "start_paragraph": item.start_paragraph,
                    "end_paragraph": item.end_paragraph,
                }
                for item in children
            ],
            provider,
            reusable_child_vectors,
        )
        write_child_manifest(temporary / "children.json", children)
        if {item.id for item in read_child_manifest(temporary / "children.json")} != {
            item.id for item in children
        }:
            raise RuntimeError("子块清单写入后集合不一致")
        metadata = {
            "build_id": build_id,
            "profile_id": profile_id,
            "profile_version": LAYERED_PROFILE_VERSION,
            "retrieval_granularity": "parent-child-v1",
            "corpus_snapshot": snapshot,
            "parent_count": len(parents),
            "child_count": len(children),
            "worst_case_candidates": worst_case,
            "reused_parent_embedding_count": len(reusable_parent_vectors),
            "reused_child_embedding_count": len(reusable_child_vectors),
            "embedding": provider.metadata,
            "splitter": {
                "version": splitter_version,
                "target_chars": target_chars,
                "max_chars": max_chars,
                "tail_merge_chars": tail_merge_chars,
            },
            "retrieval": {
                "parent_candidate_k": parent_candidate_k,
                "child_candidate_k": child_candidate_k,
                "candidate_limit": candidate_limit,
                "rrf_k": rrf_k,
            },
        }
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if final_path.exists():
            raise FileExistsError(f"不可变索引目录已存在但数据库记录无效：{final_path}")
        os.replace(temporary, final_path)
        duration_ms = round((time.perf_counter() - started) * 1000)
        row.duration_ms = duration_ms
        row.status = BuildStatus.READY.value
        row.active = False
        return BuildOutcome(build_id, len(children), final_path, duration_ms)
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
            profile_id,
        ) from exc


def build_hybrid_index(
    session: Session,
    index_root: Path,
    provider: EmbeddingProvider,
    *,
    bm25_candidate_k: int = 5,
    dense_candidate_k: int = 5,
    rrf_k: int = 60,
) -> BuildOutcome:
    started = time.perf_counter()
    chunks = _selected_chunks(session)
    if not chunks:
        raise ValueError("没有已批准的话题块，拒绝建立空索引")
    snapshot = hashlib.sha256("\n".join(item.id for item in chunks).encode()).hexdigest()
    profile_id = _hybrid_profile_id(
        provider, bm25_candidate_k, dense_candidate_k, rrf_k
    )
    build_id = stable_id("index", profile_id, snapshot, length=32)
    final_path = index_root / build_id
    existing = session.get(IndexBuildRow, build_id)
    if existing is not None and final_path.is_dir():
        _activate_build(session, existing, chunks)
        return BuildOutcome(build_id, len(chunks), final_path, existing.duration_ms or 0)

    index_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".building-", dir=index_root))
    row = existing or IndexBuildRow(
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
        corpus = [{"chunk_id": item.id} for item in chunks]
        tokenized = [tokenizer.tokenize(item.original_text) for item in chunks]
        retriever = bm25s.BM25(corpus=corpus)
        retriever.index(tokenized, show_progress=False)
        retriever.save(temporary / "bm25", corpus=corpus)

        vectors = np.asarray(
            provider.encode_documents([item.original_text for item in chunks]), dtype=np.float32
        )
        if vectors.shape != (len(chunks), provider.dimension):
            raise ValueError(
                f"Embedding 输出形状错误：期望 {(len(chunks), provider.dimension)}，"
                f"实际 {vectors.shape}"
            )
        qdrant_path = temporary / "qdrant"
        qdrant = QdrantClient(path=str(qdrant_path))
        qdrant.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=models.VectorParams(
                size=provider.dimension,
                distance=models.Distance.COSINE,
            ),
        )
        qdrant.upload_points(
            collection_name=QDRANT_COLLECTION,
            points=[
                models.PointStruct(
                    id=_qdrant_point_id(chunk.id),
                    vector=vector.tolist(),
                    payload={
                        "chunk_id": chunk.id,
                        "article_id": chunk.article_id,
                        "article_version": chunk.version_id,
                        "manifest_id": chunk.manifest_id,
                        "start_paragraph": chunk.start_paragraph,
                        "end_paragraph": chunk.end_paragraph,
                    },
                )
                for chunk, vector in zip(chunks, vectors, strict=True)
            ],
        )
        count = qdrant.count(QDRANT_COLLECTION, exact=True).count
        if count != len(chunks):
            raise RuntimeError(f"Dense point 数量不一致：期望 {len(chunks)}，实际 {count}")
        expected_chunk_ids = {chunk.id for chunk in chunks}
        stored_chunk_ids = _stored_qdrant_chunk_ids(qdrant)
        if stored_chunk_ids != expected_chunk_ids:
            raise RuntimeError("Dense point 集合与 TopicChunk 集合不一致")
        probe = qdrant.query_points(
            collection_name=QDRANT_COLLECTION,
            query=vectors[0].tolist(),
            limit=1,
            with_payload=True,
        ).points
        if not probe or probe[0].payload is None or probe[0].payload.get("chunk_id") is None:
            raise RuntimeError("Dense 索引自检未返回 chunk_id")
        qdrant.close()
        qdrant = None

        metadata = {
            "build_id": build_id,
            "profile_id": profile_id,
            "profile_version": HYBRID_PROFILE_VERSION,
            "tokenizer_version": TOKENIZER_VERSION,
            "corpus_snapshot": snapshot,
            "chunk_count": len(chunks),
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
        loaded.retrieve([tokenizer.tokenize(chunks[0].original_text[:80])], k=1)
        if final_path.exists():
            raise FileExistsError(f"不可变索引目录已存在但数据库记录无效：{final_path}")
        os.replace(temporary, final_path)
        duration_ms = round((time.perf_counter() - started) * 1000)
        row.duration_ms = duration_ms
        _activate_build(session, row, chunks)
        return BuildOutcome(build_id, len(chunks), final_path, duration_ms)
    except Exception as exc:
        if qdrant is not None:
            qdrant.close()
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
            profile_id,
        ) from exc


def record_build_failure(session: Session, error: IndexBuildError) -> None:
    row = session.get(IndexBuildRow, error.build_id)
    if row is None:
        row = IndexBuildRow(
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


@dataclass(frozen=True)
class RankedHit:
    chunk_id: str
    score: float
    rank: int


@dataclass(frozen=True)
class FusedHit:
    chunk_id: str
    rrf_score: float
    bm25: RankedHit | None
    dense: RankedHit | None


@dataclass
class _ExpansionGroup:
    parent_id: str
    start_index: int
    end_index: int
    candidates: list[RetrievalCandidate]


def reciprocal_rank_fusion(
    bm25_hits: list[RankedHit], dense_hits: list[RankedHit], rrf_k: int = 60
) -> list[FusedHit]:
    if rrf_k <= 0:
        raise ValueError("rrf_k 必须大于 0")
    by_id: dict[str, dict[str, RankedHit]] = {}
    for source, hits in (("bm25", bm25_hits), ("dense", dense_hits)):
        for hit in hits:
            by_id.setdefault(hit.chunk_id, {})[source] = hit
    fused = [
        FusedHit(
            chunk_id=chunk_id,
            rrf_score=sum(1.0 / (rrf_k + item.rank) for item in sources.values()),
            bm25=sources.get("bm25"),
            dense=sources.get("dense"),
        )
        for chunk_id, sources in by_id.items()
    ]
    return sorted(
        fused,
        key=lambda item: (
            -item.rrf_score,
            min(
                item.bm25.rank if item.bm25 is not None else 10**9,
                item.dense.rank if item.dense is not None else 10**9,
            ),
            item.chunk_id,
        ),
    )


def _bm25_search(build_path: Path, query: str, limit: int) -> list[RankedHit]:
    tokenizer = ChineseTokenizer()
    retriever = bm25s.BM25.load(build_path / "bm25", load_corpus=True)
    available = int(retriever.scores["num_docs"])
    results, scores = retriever.retrieve(
        [tokenizer.tokenize(query)], k=min(limit, available), show_progress=False
    )
    hits: list[RankedHit] = []
    for rank, (metadata, score) in enumerate(
        zip(results[0], scores[0], strict=True), start=1
    ):
        chunk_id = metadata["chunk_id"] if isinstance(metadata, dict) else str(metadata)
        hits.append(RankedHit(chunk_id=chunk_id, score=float(score), rank=rank))
    return hits


def _dense_search(
    build_path: Path, query: str, limit: int, provider: EmbeddingProvider
) -> list[RankedHit]:
    qdrant_path = build_path / "qdrant"
    if not qdrant_path.is_dir():
        raise ValueError("指定 Index Build 不包含 Dense 索引")
    vectors = np.asarray(provider.encode_queries([query]), dtype=np.float32)
    if vectors.shape != (1, provider.dimension):
        raise ValueError(
            f"查询 Embedding 输出形状错误：期望 {(1, provider.dimension)}，实际 {vectors.shape}"
        )
    client = QdrantClient(path=str(qdrant_path))
    try:
        points = client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=vectors[0].tolist(),
            limit=limit,
            with_payload=True,
        ).points
    finally:
        client.close()
    hits: list[RankedHit] = []
    for rank, point in enumerate(points, start=1):
        payload = point.payload or {}
        chunk_id = payload.get("chunk_id")
        if not isinstance(chunk_id, str):
            raise RuntimeError("Dense 索引 point 缺少 chunk_id")
        hits.append(RankedHit(chunk_id=chunk_id, score=float(point.score), rank=rank))
    return hits


def _hybrid_layer_hits(
    layer_path: Path,
    query: str,
    limit: int,
    provider: EmbeddingProvider,
    rrf_k: int,
) -> list[FusedHit]:
    return reciprocal_rank_fusion(
        _bm25_search(layer_path, query, limit),
        _dense_search(layer_path, query, limit, provider),
        rrf_k,
    )[:limit]


def _hit_scores(hit: FusedHit, rank: int) -> RetrievalScores:
    return RetrievalScores(
        bm25=hit.bm25.score if hit.bm25 is not None else None,
        bm25_rank=hit.bm25.rank if hit.bm25 is not None else None,
        dense=hit.dense.score if hit.dense is not None else None,
        dense_rank=hit.dense.rank if hit.dense is not None else None,
        rrf=hit.rrf_score,
        rrf_rank=rank,
    )


def _hit_sources(hit: FusedHit) -> list[str]:
    sources: list[str] = []
    if hit.bm25 is not None:
        sources.append("bm25")
    if hit.dense is not None:
        sources.append("dense")
    return sources


def _paragraph_text(
    paragraphs: dict[tuple[str, int], str], version_id: str, start: int, end: int
) -> str:
    try:
        return "\n\n".join(paragraphs[(version_id, number)] for number in range(start, end + 1))
    except KeyError as exc:
        raise RuntimeError(
            f"子块引用了不存在的规范原子单元：{version_id} ¶{start}–{end}"
        ) from exc


def _expand_layered_neighbors(
    candidates: list[RetrievalCandidate],
    entries: list[ChildManifestEntry],
    paragraphs: dict[tuple[str, int], str],
    neighbor_window: int,
) -> list[RetrievalCandidate]:
    if neighbor_window < 0:
        raise ValueError("neighbor_window 不能小于 0")
    by_parent: dict[str, list[ChildManifestEntry]] = {}
    by_id: dict[str, tuple[ChildManifestEntry, int]] = {}
    for entry in entries:
        by_parent.setdefault(entry.parent_chunk_id, []).append(entry)
    for parent_entries in by_parent.values():
        parent_entries.sort(key=lambda item: (item.start_paragraph, item.id))
        for index, entry in enumerate(parent_entries):
            by_id[entry.id] = (entry, index)

    groups: list[_ExpansionGroup] = []
    for candidate in candidates:
        found = by_id.get(candidate.chunk_id)
        if found is None:
            raise RuntimeError(f"重排结果引用了不存在的子块：{candidate.chunk_id}")
        entry, index = found
        parent_entries = by_parent[entry.parent_chunk_id]
        start_index = max(0, index - neighbor_window)
        end_index = min(len(parent_entries) - 1, index + neighbor_window)
        touching = [
            group
            for group in groups
            if group.parent_id == entry.parent_chunk_id
            and start_index <= group.end_index + 1
            and end_index + 1 >= group.start_index
        ]
        if not touching:
            groups.append(
                _ExpansionGroup(
                    parent_id=entry.parent_chunk_id,
                    start_index=start_index,
                    end_index=end_index,
                    candidates=[candidate],
                )
            )
            continue
        primary = touching[0]
        primary.start_index = min(primary.start_index, start_index)
        primary.end_index = max(primary.end_index, end_index)
        primary.candidates.append(candidate)
        for extra in touching[1:]:
            primary.start_index = min(primary.start_index, extra.start_index)
            primary.end_index = max(primary.end_index, extra.end_index)
            primary.candidates.extend(extra.candidates)
            groups.remove(extra)

    expanded: list[RetrievalCandidate] = []
    for group in groups:
        typed_members = group.candidates
        base = min(
            typed_members,
            key=lambda item: item.retrieval_scores.reranker_rank or 10**9,
        )
        parent_entries = by_parent[group.parent_id]
        selected_entries = parent_entries[
            group.start_index : group.end_index + 1
        ]
        start = selected_entries[0].start_paragraph
        end = selected_entries[-1].end_paragraph
        matched_ranges = sorted(
            {
                matched
                for member in typed_members
                for matched in (member.matched_ranges or [member.paragraph_range])
            }
        )
        matched_start = min(item[0] for item in matched_ranges)
        matched_end = max(item[1] for item in matched_ranges)
        expanded.append(
            base.model_copy(
                update={
                    "original_text": _paragraph_text(
                        paragraphs, base.article_version, start, end
                    ),
                    "paragraph_range": (start, end),
                    "expanded_range": (start, end),
                    "matched_ranges": matched_ranges,
                    "matched_child_ids": sorted(
                        {
                            child_id
                            for member in typed_members
                            for child_id in (member.matched_child_ids or [member.chunk_id])
                        }
                    ),
                    "matched_child_ranks": sorted(
                        [
                            ranked
                            for member in typed_members
                            for ranked in (
                                member.matched_child_ranks
                                or [
                                    (
                                        member.chunk_id,
                                        member.retrieval_scores.reranker_rank or 10**9,
                                    )
                                ]
                            )
                        ],
                        key=lambda item: (item[1], item[0]),
                    ),
                    "matched_original_text": _paragraph_text(
                        paragraphs, base.article_version, matched_start, matched_end
                    ),
                    "matched_bounding_range": (matched_start, matched_end),
                    "layer_sources": sorted(
                        {
                            source
                            for member in typed_members
                            for source in member.layer_sources
                        }
                    ),
                }
            )
        )
    return sorted(
        expanded,
        key=lambda item: item.retrieval_scores.reranker_rank or 10**9,
    )


def search_layered(
    session: Session,
    query: str,
    *,
    build_id: str,
    embedding_provider: EmbeddingProvider,
    reranker: Reranker,
    top_k: int = 5,
    parent_candidate_k: int = 5,
    child_candidate_k: int = 20,
    candidate_limit: int = 128,
    rrf_k: int = 60,
    neighbor_window: int = 1,
    layer_mode: LayeredRetrievalMode | str = LayeredRetrievalMode.PARENT_CHILD,
) -> list[RetrievalCandidate]:
    if not query.strip():
        raise ValueError("查询不能为空")
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")
    build = session.get(IndexBuildRow, build_id)
    if build is None or build.status not in {
        BuildStatus.READY.value,
        BuildStatus.ACTIVE.value,
        BuildStatus.SUPERSEDED.value,
    }:
        raise ValueError("没有可用的分层 Index Build")
    build_path = Path(build.path)
    metadata_path = build_path / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"分层索引元数据不存在：{metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("retrieval_granularity") != "parent-child-v1":
        raise ValueError("指定 Build 不是分层父子索引")
    configured_retrieval = metadata.get("retrieval")
    expected_retrieval = {
        "parent_candidate_k": parent_candidate_k,
        "child_candidate_k": child_candidate_k,
        "candidate_limit": candidate_limit,
        "rrf_k": rrf_k,
    }
    if configured_retrieval != expected_retrieval:
        raise ValueError("查询分层参数与 Index Profile 不一致，请重建或使用匹配配置")
    entries = read_child_manifest(build_path / "children.json")
    worst_case = validate_candidate_limit(
        entries,
        parent_candidate_k=parent_candidate_k,
        child_candidate_k=child_candidate_k,
        candidate_limit=candidate_limit,
    )
    selected_layer_mode = LayeredRetrievalMode(layer_mode)
    parent_hits = (
        _hybrid_layer_hits(
            build_path / "parent", query, parent_candidate_k, embedding_provider, rrf_k
        )
        if selected_layer_mode == LayeredRetrievalMode.PARENT_CHILD
        else []
    )
    child_hits = _hybrid_layer_hits(
        build_path / "child", query, child_candidate_k, embedding_provider, rrf_k
    )
    children_by_parent: dict[str, list[ChildManifestEntry]] = {}
    for entry in entries:
        children_by_parent.setdefault(entry.parent_chunk_id, []).append(entry)
    for values in children_by_parent.values():
        values.sort(key=lambda item: (item.start_paragraph, item.id))

    parent_by_child: dict[str, tuple[FusedHit, int]] = {}
    for rank, hit in enumerate(parent_hits, start=1):
        for child in children_by_parent.get(hit.chunk_id, []):
            parent_by_child.setdefault(child.id, (hit, rank))
    child_by_id = {hit.chunk_id: (hit, rank) for rank, hit in enumerate(child_hits, start=1)}
    ordered_ids = [hit.chunk_id for hit in child_hits]
    for hit in parent_hits:
        ordered_ids.extend(child.id for child in children_by_parent.get(hit.chunk_id, []))
    ordered_ids = list(dict.fromkeys(ordered_ids))
    if len(ordered_ids) > candidate_limit or len(ordered_ids) > worst_case:
        raise RuntimeError("分层候选数量超过已验证上限")

    entries_by_id = {item.id: item for item in entries}
    version_ids = {item.article_version for item in entries}
    paragraph_rows = list(
        session.scalars(select(ParagraphRow).where(ParagraphRow.version_id.in_(version_ids))).all()
    )
    paragraphs = {(item.version_id, item.number): item.text for item in paragraph_rows}
    candidates: list[RetrievalCandidate] = []
    for candidate_rank, child_id in enumerate(ordered_ids, start=1):
        entry = entries_by_id.get(child_id)
        if entry is None:
            raise RuntimeError(f"索引引用了不存在的子块：{child_id}")
        parent = session.get(TopicChunkRow, entry.parent_chunk_id)
        article = session.get(ArticleRow, entry.article_id)
        if parent is None or article is None:
            raise RuntimeError(f"子块缺少父块或文章元数据：{child_id}")
        direct = child_by_id.get(child_id)
        nominated = parent_by_child.get(child_id)
        child_scores = _hit_scores(*direct) if direct is not None else None
        parent_scores = _hit_scores(*nominated) if nominated is not None else None
        primary_scores = child_scores or parent_scores or RetrievalScores()
        primary_scores = primary_scores.model_copy(update={"rrf_rank": candidate_rank})
        layer_sources = []
        sources: list[str] = []
        if nominated is not None:
            layer_sources.append("parent")
            sources.extend(_hit_sources(nominated[0]))
        if direct is not None:
            layer_sources.append("child")
            sources.extend(_hit_sources(direct[0]))
        child_range = (entry.start_paragraph, entry.end_paragraph)
        candidates.append(
            RetrievalCandidate(
                chunk_id=entry.id,
                parent_chunk_id=entry.parent_chunk_id,
                article_id=entry.article_id,
                article_version=entry.article_version,
                article_title=article.title,
                original_text=_paragraph_text(
                    paragraphs, entry.article_version, *child_range
                ),
                paragraph_range=child_range,
                matched_ranges=[child_range],
                matched_child_ids=[entry.id],
                expanded_range=child_range,
                retrieval_scores=primary_scores,
                parent_retrieval_scores=parent_scores,
                child_retrieval_scores=child_scores,
                source=list(dict.fromkeys(sources)),
                layer_sources=layer_sources,
                candidate_rank=candidate_rank,
                topic_title=parent.topic_title,
                context_summary=parent.context_summary,
            )
        )
    reranked = rerank_candidates(query, candidates, reranker, top_k)
    return _expand_layered_neighbors(reranked, entries, paragraphs, neighbor_window)


def search(
    session: Session,
    query: str,
    top_k: int = 5,
    build_id: str | None = None,
    *,
    mode: RetrievalMode | str = RetrievalMode.BM25,
    embedding_provider: EmbeddingProvider | None = None,
    bm25_candidate_k: int = 5,
    dense_candidate_k: int = 5,
    rrf_k: int = 60,
    reranker: Reranker | None = None,
    rerank_candidate_k: int = 20,
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
    selected_mode = RetrievalMode(mode)
    if reranker is not None and selected_mode != RetrievalMode.HYBRID:
        raise ValueError("Reranker 第一版只支持 hybrid 检索模式")
    if rerank_candidate_k <= 0:
        raise ValueError("rerank_candidate_k 必须大于 0")
    candidate_floor = max(top_k, rerank_candidate_k) if reranker is not None else top_k
    bm25_hits = (
        _bm25_search(build_path, query, max(candidate_floor, bm25_candidate_k))
        if selected_mode in {RetrievalMode.BM25, RetrievalMode.HYBRID}
        else []
    )
    if selected_mode in {RetrievalMode.DENSE, RetrievalMode.HYBRID}:
        if embedding_provider is None:
            raise ValueError("Dense 检索需要本地 Embedding Provider")
        dense_hits = _dense_search(
            build_path,
            query,
            max(candidate_floor, dense_candidate_k),
            embedding_provider,
        )
    else:
        dense_hits = []
    if selected_mode == RetrievalMode.BM25:
        fused = [FusedHit(item.chunk_id, 0.0, item, None) for item in bm25_hits]
    elif selected_mode == RetrievalMode.DENSE:
        fused = [FusedHit(item.chunk_id, 0.0, None, item) for item in dense_hits]
    else:
        fused = reciprocal_rank_fusion(bm25_hits, dense_hits, rrf_k)
    candidates: list[RetrievalCandidate] = []
    for rrf_rank, hit in enumerate(fused[:candidate_floor], start=1):
        chunk = session.get(TopicChunkRow, hit.chunk_id)
        if chunk is None:
            raise RuntimeError(f"索引引用了不存在的话题块：{hit.chunk_id}")
        version = session.get(ArticleVersionRow, chunk.version_id)
        article = session.get(ArticleRow, chunk.article_id)
        if version is None or article is None:
            raise RuntimeError(f"话题块缺少文章元数据：{hit.chunk_id}")
        sources = []
        if hit.bm25 is not None:
            sources.append("bm25")
        if hit.dense is not None:
            sources.append("dense")
        candidates.append(
            RetrievalCandidate(
                chunk_id=chunk.id,
                article_id=chunk.article_id,
                article_version=chunk.version_id,
                article_title=article.title,
                original_text=chunk.original_text,
                paragraph_range=(chunk.start_paragraph, chunk.end_paragraph),
                retrieval_scores=RetrievalScores(
                    bm25=hit.bm25.score if hit.bm25 is not None else None,
                    bm25_rank=hit.bm25.rank if hit.bm25 is not None else None,
                    dense=hit.dense.score if hit.dense is not None else None,
                    dense_rank=hit.dense.rank if hit.dense is not None else None,
                    rrf=hit.rrf_score if selected_mode == RetrievalMode.HYBRID else None,
                    rrf_rank=rrf_rank if selected_mode == RetrievalMode.HYBRID else None,
                ),
                source=sources,
                topic_title=chunk.topic_title,
                context_summary=chunk.context_summary,
            )
        )
    if reranker is not None:
        return rerank_candidates(query, candidates, reranker, top_k)
    return candidates[:top_k]
