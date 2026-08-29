from __future__ import annotations

import json
import math
import statistics
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, Field, model_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from xifeng_rag.corpus import stable_id
from xifeng_rag.domain import LayeredRetrievalMode, RetrievalMode
from xifeng_rag.indexing import (
    EmbeddingProvider,
    search,
    search_layered,
    search_layered_with_query_variants,
    search_with_query_variants,
)
from xifeng_rag.layered_retrieval import read_child_manifest
from xifeng_rag.query_rewrite import (
    QueryRewriter,
    QueryRewriteStatus,
    build_query_variants,
)
from xifeng_rag.reranking import Reranker
from xifeng_rag.storage import (
    EvaluationDatasetRow,
    EvaluationEvidenceRow,
    EvaluationQuestionRow,
    EvaluationRunRow,
    IndexBuildRow,
    TopicChunkRow,
)


class EvaluationQuestion(BaseModel):
    question_id: str
    query: str = Field(min_length=1)
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    should_abstain: bool = False

    @model_validator(mode="after")
    def relevance_rules(self) -> EvaluationQuestion:
        if self.should_abstain and self.relevant_chunk_ids:
            raise ValueError("应拒答问题不能同时标注相关块")
        if not self.should_abstain and not self.relevant_chunk_ids:
            raise ValueError("普通问题必须标注至少一个相关块")
        return self


class EvaluationSet(BaseModel):
    dataset_id: str
    split: str
    questions: list[EvaluationQuestion] = Field(min_length=1)


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k 必须大于 0")
    return sum(item in relevant for item in ranked[:k]) / k


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if not relevant else sum(item in relevant for item in ranked[:k]) / len(relevant)


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    return next((1.0 / rank for rank, item in enumerate(ranked, start=1) if item in relevant), 0.0)


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, item in enumerate(ranked[:k], start=1)
        if item in relevant
    )
    ideal_hits = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 1.0


def evidence_coverage_at_k(
    retrieved_ranges: list[tuple[str, int, int]],
    evidence_ranges: list[tuple[str, int, int]],
    k: int,
) -> float:
    gold = {
        (version_id, unit)
        for version_id, start, end in evidence_ranges
        for unit in range(start, end + 1)
    }
    if not gold:
        return 1.0
    covered = {
        (version_id, unit)
        for version_id, start, end in retrieved_ranges[:k]
        for unit in range(start, end + 1)
        if (version_id, unit) in gold
    }
    return len(covered) / len(gold)


def _git_state() -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if commit.returncode != 0 or status.returncode != 0:
            return None, None
        return commit.stdout.strip()[:64], bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return None, None


def run_frozen_evaluation(
    session: Session,
    dataset_id: str,
    k: int = 5,
    build_id: str | None = None,
    *,
    mode: RetrievalMode | str = RetrievalMode.BM25,
    embedding_provider: EmbeddingProvider | None = None,
    bm25_candidate_k: int = 5,
    dense_candidate_k: int = 5,
    rrf_k: int = 60,
    reranker: Reranker | None = None,
    rerank_candidate_k: int = 20,
    layered: bool = False,
    layered_parent_candidate_k: int = 5,
    layered_child_candidate_k: int = 20,
    layered_candidate_limit: int = 128,
    layered_neighbor_window: int = 1,
    layered_mode: LayeredRetrievalMode | str = LayeredRetrievalMode.PARENT_CHILD,
    query_rewriter: QueryRewriter | None = None,
    rewrite_max_variants: int = 1,
    rewrite_original_weight: float = 1.0,
    rewrite_total_weight: float = 0.7,
    rewrite_fusion_rrf_k: int = 60,
    rewrite_candidate_limit: int = 128,
) -> dict[str, object]:
    if k <= 0:
        raise ValueError("k 必须大于 0")
    dataset = session.get(EvaluationDatasetRow, dataset_id)
    if dataset is None:
        raise ValueError("评测集不存在")
    if dataset.status != "frozen" or not dataset.content_hash:
        raise ValueError("只能运行已冻结评测集")
    build = (
        session.get(IndexBuildRow, build_id)
        if build_id is not None
        else session.scalar(select(IndexBuildRow).where(IndexBuildRow.active.is_(True)))
    )
    if build is None:
        raise ValueError("没有活动索引")
    if build.corpus_snapshot != dataset.index_corpus_snapshot:
        raise ValueError("活动索引语料快照与冻结评测集不一致")
    questions = list(
        session.scalars(
            select(EvaluationQuestionRow)
            .where(EvaluationQuestionRow.dataset_id == dataset.id)
            .order_by(EvaluationQuestionRow.ordinal)
        ).all()
    )
    if not questions:
        raise ValueError("冻结评测集没有问题")
    child_manifest = (
        read_child_manifest(Path(build.path) / "children.json") if layered else []
    )
    if layered and (embedding_provider is None or reranker is None or build_id is None):
        raise ValueError("分层评测需要显式 Build、Embedding Provider 和 Reranker")
    if query_rewriter is not None and (
        RetrievalMode(mode) != RetrievalMode.HYBRID
        or embedding_provider is None
        or reranker is None
    ):
        raise ValueError("查询改写评测需要 Hybrid、Embedding Provider 和 Reranker")
    details: list[dict[str, object]] = []
    for question in questions:
        evidence = list(
            session.scalars(
                select(EvaluationEvidenceRow)
                .where(EvaluationEvidenceRow.question_id == question.id)
                .order_by(EvaluationEvidenceRow.ordinal)
            ).all()
        )
        evidence_ranges = [
            (item.version_id, item.start_paragraph, item.end_paragraph) for item in evidence
        ]
        relevant_chunks = list(
            session.scalars(
                select(TopicChunkRow).where(
                    TopicChunkRow.version_id.in_({item.version_id for item in evidence}),
                )
            ).all()
        ) if evidence else []
        relevant = {
            chunk.id
            for chunk in relevant_chunks
            if any(
                chunk.version_id == version_id
                and chunk.start_paragraph <= end
                and chunk.end_paragraph >= start
                for version_id, start, end in evidence_ranges
            )
        }
        started = time.perf_counter()
        rewrite_result = (
            query_rewriter.rewrite(question.query, rewrite_max_variants)
            if query_rewriter is not None
            else None
        )
        query_variants = (
            build_query_variants(
                question.query,
                rewrite_result.variants,
                original_weight=rewrite_original_weight,
                rewrite_total_weight=rewrite_total_weight,
            )
            if rewrite_result is not None and rewrite_result.variants
            else None
        )
        if layered and query_variants is not None:
            assert embedding_provider is not None and reranker is not None
            candidates = search_layered_with_query_variants(
                session,
                question.query,
                query_variants,
                build_id=build.id,
                embedding_provider=embedding_provider,
                reranker=reranker,
                top_k=k,
                parent_candidate_k=layered_parent_candidate_k,
                child_candidate_k=layered_child_candidate_k,
                candidate_limit=layered_candidate_limit,
                rrf_k=rrf_k,
                neighbor_window=layered_neighbor_window,
                layer_mode=layered_mode,
                query_fusion_rrf_k=rewrite_fusion_rrf_k,
                query_candidate_limit=rewrite_candidate_limit,
            )
        elif layered:
            assert embedding_provider is not None and reranker is not None
            candidates = search_layered(
                session,
                question.query,
                build_id=build.id,
                embedding_provider=embedding_provider,
                reranker=reranker,
                top_k=k,
                parent_candidate_k=layered_parent_candidate_k,
                child_candidate_k=layered_child_candidate_k,
                candidate_limit=layered_candidate_limit,
                rrf_k=rrf_k,
                neighbor_window=layered_neighbor_window,
                layer_mode=layered_mode,
            )
        elif query_variants is not None:
            assert embedding_provider is not None and reranker is not None
            candidates = search_with_query_variants(
                session,
                question.query,
                query_variants,
                top_k=k,
                build_id=build.id,
                mode=mode,
                embedding_provider=embedding_provider,
                bm25_candidate_k=bm25_candidate_k,
                dense_candidate_k=dense_candidate_k,
                rrf_k=rrf_k,
                reranker=reranker,
                rerank_candidate_k=rerank_candidate_k,
                query_fusion_rrf_k=rewrite_fusion_rrf_k,
                query_candidate_limit=rewrite_candidate_limit,
            )
        else:
            candidates = search(
                session,
                question.query,
                top_k=k,
                build_id=build.id,
                mode=mode,
                embedding_provider=embedding_provider,
                bm25_candidate_k=bm25_candidate_k,
                dense_candidate_k=dense_candidate_k,
                rrf_k=rrf_k,
                reranker=reranker,
                rerank_candidate_k=rerank_candidate_k,
            )
        latency_ms = (time.perf_counter() - started) * 1000
        ranked = [item.chunk_id for item in candidates]
        retrieved_ranges = [
            (item.article_version, item.paragraph_range[0], item.paragraph_range[1])
            for item in candidates
        ]
        matched_ranges = [
            (item.article_version, start, end)
            for item in candidates
            for start, end in (item.matched_ranges or [item.paragraph_range])
        ]
        if layered:
            relevant_child_ids = {
                item.id
                for item in child_manifest
                if any(
                    item.article_version == version_id
                    and item.start_paragraph <= end
                    and item.end_paragraph >= start
                    for version_id, start, end in evidence_ranges
                )
            }
            ranked_children = [
                child_id
                for child_id, _rank in sorted(
                    [
                        ranked_child
                        for item in candidates
                        for ranked_child in item.matched_child_ranks
                    ],
                    key=lambda item: (item[1], item[0]),
                )
            ]
            question_precision = precision_at_k(ranked_children, relevant_child_ids, k)
            question_recall = recall_at_k(ranked_children, relevant_child_ids, k)
            question_mrr = reciprocal_rank(ranked_children, relevant_child_ids)
            question_ndcg = ndcg_at_k(ranked_children, relevant_child_ids, k)
        else:
            question_precision = precision_at_k(ranked, relevant, k)
            question_recall = recall_at_k(ranked, relevant, k)
            question_mrr = reciprocal_rank(ranked, relevant)
            question_ndcg = ndcg_at_k(ranked, relevant, k)
        details.append(
            {
                "question_id": question.id,
                "split": question.split,
                "ranked_chunk_ids": ranked,
                "retrieval_candidates": [
                    {
                        "chunk_id": item.chunk_id,
                        "source": item.source,
                        "scores": item.retrieval_scores.model_dump(),
                        "layer_sources": item.layer_sources,
                        "parent_scores": (
                            item.parent_retrieval_scores.model_dump()
                            if item.parent_retrieval_scores is not None
                            else None
                        ),
                        "child_scores": (
                            item.child_retrieval_scores.model_dump()
                            if item.child_retrieval_scores is not None
                            else None
                        ),
                        "matched_child_ids": item.matched_child_ids,
                        "matched_child_ranks": item.matched_child_ranks,
                        "matched_ranges": item.matched_ranges,
                        "expanded_range": item.expanded_range,
                        "query_matches": [
                            match.model_dump(mode="json") for match in item.query_matches
                        ],
                    }
                    for item in candidates
                ],
                "mapped_relevant_chunk_ids": sorted(relevant),
                "precision_at_k": question_precision,
                "recall_at_k": question_recall,
                "mrr": question_mrr,
                "ndcg_at_k": question_ndcg,
                "evidence_coverage_at_k": evidence_coverage_at_k(
                    matched_ranges if layered else retrieved_ranges, evidence_ranges, k
                ),
                "expanded_evidence_coverage_at_k": (
                    evidence_coverage_at_k(retrieved_ranges, evidence_ranges, k)
                    if layered
                    else None
                ),
                "latency_ms": latency_ms,
                "should_abstain": not question.answerable,
                "retrieval_mode": str(RetrievalMode(mode)),
                "layered": layered,
                "query_rewrite": (
                    rewrite_result.model_dump(mode="json")
                    if rewrite_result is not None
                    else None
                ),
                "layered_config": (
                    {
                        "parent_candidate_k": layered_parent_candidate_k,
                        "child_candidate_k": layered_child_candidate_k,
                        "candidate_limit": layered_candidate_limit,
                        "neighbor_window": layered_neighbor_window,
                        "layer_mode": str(LayeredRetrievalMode(layered_mode)),
                    }
                    if layered
                    else None
                ),
            }
        )
    metric_names = [
        "precision_at_k",
        "recall_at_k",
        "mrr",
        "ndcg_at_k",
        "evidence_coverage_at_k",
    ]
    metrics = {
        name: statistics.fmean(float(str(item[name])) for item in details)
        for name in metric_names
    }
    if layered:
        metrics["expanded_evidence_coverage_at_k"] = statistics.fmean(
            float(str(item["expanded_evidence_coverage_at_k"])) for item in details
        )
    latencies = [float(str(item["latency_ms"])) for item in details]
    metrics["latency_mean_ms"] = statistics.fmean(latencies)
    metrics["latency_p95_ms"] = sorted(latencies)[
        max(0, math.ceil(len(latencies) * 0.95) - 1)
    ]
    rewrite_details: list[dict[str, object]] = []
    for item in details:
        rewrite_detail = item.get("query_rewrite")
        if isinstance(rewrite_detail, dict):
            rewrite_details.append(rewrite_detail)
    if rewrite_details:
        rewrite_latencies = [float(str(item["latency_ms"])) for item in rewrite_details]
        metrics["rewrite_latency_mean_ms"] = statistics.fmean(rewrite_latencies)
        metrics["rewrite_failure_rate"] = sum(
            item["status"] == QueryRewriteStatus.FAILED.value for item in rewrite_details
        ) / len(rewrite_details)
    run_id = stable_id("eval", dataset.id, build.id, str(time.time_ns()), length=32)
    code_commit, worktree_dirty = _git_state()
    reranker_metadata = dict(reranker.metadata) if reranker is not None else {}
    if reranker is not None:
        reranker_metadata["candidate_k"] = rerank_candidate_k
    truncated_count = _metadata_int(reranker_metadata, "truncated_candidate_count") or 0
    session.add(
        EvaluationRunRow(
            id=run_id,
            dataset_id=dataset.id,
            build_id=build.id,
            metrics_json=json.dumps(metrics),
            details_json=json.dumps(details, ensure_ascii=False),
            dataset_content_hash=dataset.content_hash,
            corpus_snapshot=build.corpus_snapshot,
            profile_id=build.profile_id,
            k=k,
            code_commit=code_commit,
            worktree_dirty=worktree_dirty,
            retrieval_mode=str(RetrievalMode(mode)),
            bm25_candidate_k=bm25_candidate_k,
            dense_candidate_k=dense_candidate_k,
            rrf_k=rrf_k,
            reranker_enabled=reranker is not None,
            reranker_provider=_metadata_text(reranker_metadata, "provider"),
            reranker_model_id=_metadata_text(reranker_metadata, "model_id"),
            reranker_model_revision=_metadata_text(reranker_metadata, "model_revision"),
            reranker_device=_metadata_text(reranker_metadata, "device"),
            reranker_batch_size=_metadata_int(reranker_metadata, "batch_size"),
            reranker_max_pair_tokens=_metadata_int(reranker_metadata, "max_pair_tokens"),
            reranker_candidate_k=rerank_candidate_k if reranker is not None else None,
            reranker_input_template_version=_metadata_text(
                reranker_metadata, "input_template_version"
            ),
            reranker_load_duration_ms=_metadata_float(
                reranker_metadata, "load_duration_ms"
            ),
            reranker_truncated_candidate_count=(
                truncated_count if reranker is not None else None
            ),
            reranker_process_memory_delta_bytes=_metadata_int(
                reranker_metadata, "process_memory_delta_bytes"
            ),
            reranker_cuda_peak_memory_bytes=_metadata_int(
                reranker_metadata, "cuda_peak_memory_bytes"
            ),
        )
    )
    return {
        "run_id": run_id,
        "dataset_id": dataset.id,
        "retrieval_mode": str(RetrievalMode(mode)),
        "reranker": reranker_metadata if reranker is not None else None,
        "query_rewrite": (
            {
                "provider": rewrite_details[0]["provider"],
                "max_variants": rewrite_max_variants,
                "original_weight": rewrite_original_weight,
                "rewrite_total_weight": rewrite_total_weight,
                "fusion_rrf_k": rewrite_fusion_rrf_k,
                "candidate_limit": rewrite_candidate_limit,
            }
            if rewrite_details
            else None
        ),
        "layered": (
            {
                "parent_candidate_k": layered_parent_candidate_k,
                "child_candidate_k": layered_child_candidate_k,
                "candidate_limit": layered_candidate_limit,
                "neighbor_window": layered_neighbor_window,
                "layer_mode": str(LayeredRetrievalMode(layered_mode)),
            }
            if layered
            else None
        ),
        "metrics": metrics,
    }


def _metadata_text(metadata: Mapping[str, object], key: str) -> str | None:
    value = metadata.get(key)
    return None if value is None else str(value)


def _metadata_int(metadata: Mapping[str, object], key: str) -> int | None:
    value = metadata.get(key)
    return int(value) if isinstance(value, (int, float)) else None


def _metadata_float(metadata: Mapping[str, object], key: str) -> float | None:
    value = metadata.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def run_evaluation(session: Session, dataset_path: Path, k: int = 5) -> dict[str, object]:
    dataset = EvaluationSet.model_validate_json(dataset_path.read_text(encoding="utf-8"))
    build = session.scalar(select(IndexBuildRow).where(IndexBuildRow.active.is_(True)))
    if build is None:
        raise ValueError("没有活动索引")
    details: list[dict[str, object]] = []
    for question in dataset.questions:
        started = time.perf_counter()
        candidates = search(session, question.query, top_k=k)
        latency_ms = (time.perf_counter() - started) * 1000
        ranked = [item.chunk_id for item in candidates]
        relevant = set(question.relevant_chunk_ids)
        details.append(
            {
                "question_id": question.question_id,
                "ranked_chunk_ids": ranked,
                "precision_at_k": precision_at_k(ranked, relevant, k),
                "recall_at_k": recall_at_k(ranked, relevant, k),
                "mrr": reciprocal_rank(ranked, relevant),
                "ndcg_at_k": ndcg_at_k(ranked, relevant, k),
                "latency_ms": latency_ms,
                "should_abstain": question.should_abstain,
            }
        )
    metric_names = ["precision_at_k", "recall_at_k", "mrr", "ndcg_at_k"]
    metrics: dict[str, float] = {
        name: statistics.fmean(float(str(item[name])) for item in details) for name in metric_names
    }
    latencies = [float(str(item["latency_ms"])) for item in details]
    metrics["latency_mean_ms"] = statistics.fmean(latencies)
    metrics["latency_p95_ms"] = sorted(latencies)[max(0, math.ceil(len(latencies) * 0.95) - 1)]
    run_id = stable_id("eval", dataset.dataset_id, build.id, str(time.time_ns()), length=32)
    session.add(
        EvaluationRunRow(
            id=run_id,
            dataset_id=dataset.dataset_id,
            build_id=build.id,
            metrics_json=json.dumps(metrics),
            details_json=json.dumps(details, ensure_ascii=False),
        )
    )
    return {"run_id": run_id, "dataset_id": dataset.dataset_id, "metrics": metrics}
