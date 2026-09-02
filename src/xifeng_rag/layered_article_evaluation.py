from __future__ import annotations

import math
import os
import tempfile
from collections import defaultdict
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel
from sqlalchemy.orm import Session

from xifeng_rag.article_evaluation import (
    ArticleDiscoveryDataset,
    _git_state,
)
from xifeng_rag.corpus import stable_id
from xifeng_rag.domain import RetrievalMode
from xifeng_rag.indexing import EmbeddingProvider
from xifeng_rag.layered_article_search import (
    ArticleAggregation,
    ArticleDiscoveryLayer,
    article_rank_fusion,
    is_layer_clear,
    search_article_layer,
)


class LayeredCandidateResult(BaseModel):
    rank: int
    article_id: str
    version_id: str
    title: str
    score: float
    best_unit_id: str
    best_unit_score: float
    best_unit_rank: int
    parent_topic: str | None
    start_unit: int | None
    end_unit: int | None
    unit_count: int
    contributing_units: list[tuple[str, int, float]]


class LayeredQuestionResult(BaseModel):
    question_id: str
    query: str
    expected_article_id: str
    expected_title: str | None
    status: Literal["evaluated", "expected_article_not_in_layer"]
    expected_rank: int | None
    reciprocal_rank: float | None
    hit_at_1: bool | None
    hit_at_3: bool | None
    hit_at_5: bool | None
    latency_ms: float | None
    retrieval_latency_ms: float | None
    aggregation_latency_ms: float | None
    top1_top2_score_margin: float | None
    top1_top2_score_ratio: float | None
    top1_top2_rank_gap: int | None
    expected_article_unit_count: int | None
    top1_article_unit_count: int | None
    bm25_dense_top1_agree: bool | None = None
    bm25_dense_top_k_overlap: float | None = None
    candidates: list[LayeredCandidateResult]


class UnitCountGroupMetrics(BaseModel):
    question_count: int
    mrr: float
    target_top1_count: int


class LayeredChannelMetrics(BaseModel):
    evaluated_question_count: int
    unavailable_question_count: int
    hit_at_1: float | None
    hit_at_3: float | None
    hit_at_5: float | None
    mrr: float | None
    latency_mean_ms: float | None
    latency_p95_ms: float | None
    unit_count_groups: dict[str, UnitCountGroupMetrics]


class LayeredChannelResult(BaseModel):
    layer: ArticleDiscoveryLayer
    mode: RetrievalMode
    aggregation: ArticleAggregation
    build_id: str
    profile_id: str
    unit_count: int
    article_count: int
    article_unit_counts: dict[str, int]
    metrics: LayeredChannelMetrics
    questions: list[LayeredQuestionResult]


class LayeredArticleEvaluationResult(BaseModel):
    schema_version: Literal[1] = 1
    task: Literal["layered_article_discovery"] = "layered_article_discovery"
    run_id: str
    created_at: datetime
    dataset_path: str
    dataset_content_hash: str
    confirmed_question_count: int
    skipped_question_count: int
    corpus_snapshot: str
    build_ids: dict[str, str]
    profile_ids: dict[str, str]
    embedding: dict[str, object]
    splitter: dict[str, object] | None
    code_commit: str | None
    worktree_dirty: bool | None
    evaluation_k: int
    candidate_k: int
    aggregation_top_n: int
    aggregation_rrf_k: int
    channels: list[LayeredChannelResult]


def _safe_ratio(first: float, second: float) -> float | None:
    if second == 0:
        return None
    return first / second


def _group_metrics(questions: list[LayeredQuestionResult]) -> dict[str, UnitCountGroupMetrics]:
    groups: dict[str, list[LayeredQuestionResult]] = defaultdict(list)
    for question in questions:
        if question.status != "evaluated" or question.expected_article_unit_count is None:
            continue
        count = question.expected_article_unit_count
        bucket = "1" if count == 1 else "2-3" if count <= 3 else "4+"
        groups[bucket].append(question)
    return {
        bucket: UnitCountGroupMetrics(
            question_count=len(items),
            mrr=sum(item.reciprocal_rank or 0.0 for item in items) / len(items),
            target_top1_count=sum(item.expected_rank == 1 for item in items),
        )
        for bucket, items in sorted(groups.items())
    }


def _metrics(questions: list[LayeredQuestionResult]) -> LayeredChannelMetrics:
    evaluated = [item for item in questions if item.status == "evaluated"]
    latencies = [item.latency_ms for item in evaluated if item.latency_ms is not None]

    def rate(field: str) -> float | None:
        if not evaluated:
            return None
        return sum(bool(getattr(item, field)) for item in evaluated) / len(evaluated)

    return LayeredChannelMetrics(
        evaluated_question_count=len(evaluated),
        unavailable_question_count=len(questions) - len(evaluated),
        hit_at_1=rate("hit_at_1"),
        hit_at_3=rate("hit_at_3"),
        hit_at_5=rate("hit_at_5"),
        mrr=(
            sum(item.reciprocal_rank or 0.0 for item in evaluated) / len(evaluated)
            if evaluated
            else None
        ),
        latency_mean_ms=sum(latencies) / len(latencies) if latencies else None,
        latency_p95_ms=(
            sorted(latencies)[max(0, math.ceil(len(latencies) * 0.95) - 1)]
            if latencies
            else None
        ),
        unit_count_groups=_group_metrics(evaluated),
    )


def _aggregations_for(layer: ArticleDiscoveryLayer) -> tuple[ArticleAggregation, ...]:
    if layer in {ArticleDiscoveryLayer.SUBTOPICS, ArticleDiscoveryLayer.ORIGINAL}:
        return (
            ArticleAggregation.BEST_UNIT,
            ArticleAggregation.TOP_N_RANK_FUSION,
        )
    return (ArticleAggregation.BEST_UNIT,)


def run_layered_article_evaluation(
    session: Session,
    dataset: ArticleDiscoveryDataset,
    dataset_path: Path,
    embedding_provider: EmbeddingProvider,
    build_ids: dict[ArticleDiscoveryLayer, str],
    *,
    evaluation_k: int = 5,
    candidate_k: int = 20,
    aggregation_top_n: int = 3,
    aggregation_rrf_k: int = 60,
) -> LayeredArticleEvaluationResult:
    if evaluation_k < 5:
        raise ValueError("分层文章反查评测需要 evaluation_k 至少为 5")
    missing = set(ArticleDiscoveryLayer) - set(build_ids)
    if missing:
        raise ValueError("缺少分层 Build：" + ", ".join(sorted(item.value for item in missing)))
    channels: list[LayeredChannelResult] = []
    snapshots: set[str] = set()
    embeddings: list[dict[str, object]] = []
    splitter: dict[str, object] | None = None
    for layer in ArticleDiscoveryLayer:
        for mode in (RetrievalMode.BM25, RetrievalMode.DENSE):
            for aggregation in _aggregations_for(layer):
                questions: list[LayeredQuestionResult] = []
                channel_metadata: dict[str, object] | None = None
                safe_units = _all_safe_units(session, build_ids[layer], layer)
                available_unit_counts: dict[str, int] = defaultdict(int)
                for item in safe_units:
                    available_unit_counts[str(item["article_id"])] += 1
                available_article_ids = set(available_unit_counts)
                for question in dataset.confirmed_questions:
                    hits, metadata = search_article_layer(
                        session,
                        build_ids[layer],
                        layer,
                        question.query,
                        mode=mode,
                        aggregation=aggregation,
                        embedding_provider=(
                            embedding_provider if mode == RetrievalMode.DENSE else None
                        ),
                        candidate_k=candidate_k,
                        top_k=evaluation_k,
                        top_n=aggregation_top_n,
                        rrf_k=aggregation_rrf_k,
                    )
                    channel_metadata = metadata
                    available_in_layer = question.expected_article_id in available_article_ids
                    if not available_in_layer:
                        questions.append(
                            LayeredQuestionResult(
                                question_id=question.id,
                                query=question.query,
                                expected_article_id=question.expected_article_id,
                                expected_title=question.expected_title,
                                status="expected_article_not_in_layer",
                                expected_rank=None,
                                reciprocal_rank=None,
                                hit_at_1=None,
                                hit_at_3=None,
                                hit_at_5=None,
                                latency_ms=None,
                                retrieval_latency_ms=None,
                                aggregation_latency_ms=None,
                                top1_top2_score_margin=None,
                                top1_top2_score_ratio=None,
                                top1_top2_rank_gap=None,
                                expected_article_unit_count=None,
                                top1_article_unit_count=None,
                                candidates=[],
                            )
                        )
                        continue
                    expected_rank = next(
                        (
                            item.rank
                            for item in hits
                            if item.article_id == question.expected_article_id
                        ),
                        None,
                    )
                    first = hits[0].score if hits else None
                    second = hits[1].score if len(hits) > 1 else None
                    questions.append(
                        LayeredQuestionResult(
                            question_id=question.id,
                            query=question.query,
                            expected_article_id=question.expected_article_id,
                            expected_title=question.expected_title,
                            status="evaluated",
                            expected_rank=expected_rank,
                            reciprocal_rank=(
                                1.0 / expected_rank if expected_rank is not None else 0.0
                            ),
                            hit_at_1=expected_rank == 1,
                            hit_at_3=expected_rank is not None and expected_rank <= 3,
                            hit_at_5=expected_rank is not None and expected_rank <= 5,
                            latency_ms=float(str(metadata["latency_ms"])),
                            retrieval_latency_ms=float(
                                str(metadata["retrieval_latency_ms"])
                            ),
                            aggregation_latency_ms=float(
                                str(metadata["aggregation_latency_ms"])
                            ),
                            top1_top2_score_margin=(
                                first - second
                                if first is not None and second is not None
                                else None
                            ),
                            top1_top2_score_ratio=(
                                _safe_ratio(first, second)
                                if first is not None and second is not None
                                else None
                            ),
                            top1_top2_rank_gap=(
                                abs(hits[0].best_unit_rank - hits[1].best_unit_rank)
                                if len(hits) > 1
                                else None
                            ),
                            expected_article_unit_count=available_unit_counts[
                                question.expected_article_id
                            ],
                            top1_article_unit_count=(hits[0].unit_count if hits else None),
                            candidates=[
                                LayeredCandidateResult(
                                    rank=item.rank,
                                    article_id=item.article_id,
                                    version_id=item.version_id,
                                    title=item.title,
                                    score=item.score,
                                    best_unit_id=item.best_unit_id,
                                    best_unit_score=item.best_unit_score,
                                    best_unit_rank=item.best_unit_rank,
                                    parent_topic=item.parent_topic,
                                    start_unit=item.start_unit,
                                    end_unit=item.end_unit,
                                    unit_count=item.unit_count,
                                    contributing_units=list(item.contributing_units),
                                )
                                for item in hits
                            ],
                        )
                    )
                if channel_metadata is None:
                    raise ValueError("文章反查评测集没有确认题")
                snapshots.add(str(channel_metadata["corpus_snapshot"]))
                embedding = channel_metadata.get("embedding")
                if isinstance(embedding, dict):
                    embeddings.append(embedding)
                if layer == ArticleDiscoveryLayer.ORIGINAL:
                    raw_splitter = channel_metadata.get("splitter")
                    splitter = raw_splitter if isinstance(raw_splitter, dict) else None
                channels.append(
                    LayeredChannelResult(
                        layer=layer,
                        mode=mode,
                        aggregation=aggregation,
                        build_id=build_ids[layer],
                        profile_id=str(channel_metadata["profile_id"]),
                        unit_count=int(str(channel_metadata["unit_count"])),
                        article_count=int(str(channel_metadata["article_count"])),
                        article_unit_counts=dict(available_unit_counts),
                        metrics=_metrics(questions),
                        questions=questions,
                    )
                )
    if len(snapshots) != 1:
        raise ValueError("分层 Build 的语料快照不一致，拒绝评测")
    if embeddings and any(item != embeddings[0] for item in embeddings[1:]):
        raise ValueError("分层 Build 的 Embedding 配置不一致，拒绝单变量比较")
    _add_cross_mode_signals(channels, evaluation_k)
    created_at = datetime.now(UTC)
    commit, dirty = _git_state()
    snapshot = next(iter(snapshots))
    run_id = stable_id(
        "layered-article-evaluation",
        dataset.content_hash,
        snapshot,
        created_at.isoformat(),
        length=32,
    )
    return LayeredArticleEvaluationResult(
        run_id=run_id,
        created_at=created_at,
        dataset_path=str(dataset_path.resolve()),
        dataset_content_hash=dataset.content_hash,
        confirmed_question_count=len(dataset.confirmed_questions),
        skipped_question_count=len(dataset.queries) - len(dataset.confirmed_questions),
        corpus_snapshot=snapshot,
        build_ids={layer.value: value for layer, value in build_ids.items()},
        profile_ids={item.layer.value: item.profile_id for item in channels},
        embedding=embeddings[0] if embeddings else {},
        splitter=splitter,
        code_commit=commit,
        worktree_dirty=dirty,
        evaluation_k=evaluation_k,
        candidate_k=candidate_k,
        aggregation_top_n=aggregation_top_n,
        aggregation_rrf_k=aggregation_rrf_k,
        channels=channels,
    )


def _all_safe_units(
    session: Session, build_id: str, layer: ArticleDiscoveryLayer
) -> list[dict[str, object]]:
    from xifeng_rag.layered_article_search import _load_build

    _, _, units = _load_build(session, build_id, layer)
    return list(units.values())


def _add_cross_mode_signals(
    channels: list[LayeredChannelResult], top_k: int
) -> None:
    keyed = {(item.layer, item.mode, item.aggregation): item for item in channels}
    for layer in ArticleDiscoveryLayer:
        for aggregation in _aggregations_for(layer):
            bm25 = keyed[(layer, RetrievalMode.BM25, aggregation)]
            dense = keyed[(layer, RetrievalMode.DENSE, aggregation)]
            dense_questions = {item.question_id: item for item in dense.questions}
            for question in bm25.questions:
                peer = dense_questions[question.question_id]
                if question.status != "evaluated" or peer.status != "evaluated":
                    continue
                bm25_ids = [item.article_id for item in question.candidates[:top_k]]
                dense_ids = [item.article_id for item in peer.candidates[:top_k]]
                agree = bool(bm25_ids and dense_ids and bm25_ids[0] == dense_ids[0])
                union = set(bm25_ids) | set(dense_ids)
                overlap = len(set(bm25_ids) & set(dense_ids)) / len(union) if union else 0.0
                question.bm25_dense_top1_agree = agree
                peer.bm25_dense_top1_agree = agree
                question.bm25_dense_top_k_overlap = overlap
                peer.bm25_dense_top_k_overlap = overlap


def save_layered_article_evaluation(
    result: LayeredArticleEvaluationResult, output_dir: Path
) -> Path:
    resolved = output_dir.resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    final_path = resolved / f"{result.run_id}.json"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=resolved,
            prefix=".layered-article-",
            suffix=".tmp",
            delete=False,
        ) as stream:
            stream.write(result.model_dump_json(indent=2))
            temporary = Path(stream.name)
        os.replace(temporary, final_path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return final_path


def load_layered_article_evaluation(path: Path) -> LayeredArticleEvaluationResult:
    return LayeredArticleEvaluationResult.model_validate_json(
        path.resolve().read_text(encoding="utf-8")
    )


def validate_layered_comparison(
    baseline: LayeredArticleEvaluationResult,
    current: LayeredArticleEvaluationResult,
) -> None:
    fields = {
        "题集哈希": "dataset_content_hash",
        "语料快照": "corpus_snapshot",
        "层 Profile": "profile_ids",
        "候选深度": "candidate_k",
        "结果深度": "evaluation_k",
        "聚合 Top-N": "aggregation_top_n",
        "聚合 RRF k": "aggregation_rrf_k",
    }
    issues = [
        label
        for label, field in fields.items()
        if getattr(baseline, field) != getattr(current, field)
    ]
    baseline_channels = {
        (item.layer, item.mode, item.aggregation) for item in baseline.channels
    }
    current_channels = {
        (item.layer, item.mode, item.aggregation) for item in current.channels
    }
    if baseline_channels != current_channels:
        issues.append("通道集合")
    if issues:
        raise ValueError("分层文章反查结果不可直接比较：" + "、".join(issues) + "不同")


class LayeredMetricDelta(BaseModel):
    baseline: float | None
    current: float | None
    delta: float | None


class LayeredQuestionRankDelta(BaseModel):
    question_id: str
    baseline_rank: int | None
    current_rank: int | None
    status: Literal["improved", "regressed", "unchanged", "incompatible"]


class LayeredChannelComparison(BaseModel):
    layer: ArticleDiscoveryLayer
    mode: RetrievalMode
    aggregation: ArticleAggregation
    metrics: dict[str, LayeredMetricDelta]
    questions: list[LayeredQuestionRankDelta]


class LayeredEvaluationComparison(BaseModel):
    schema_version: Literal[1] = 1
    task: Literal["layered_article_discovery_comparison"] = (
        "layered_article_discovery_comparison"
    )
    baseline_run_id: str
    current_run_id: str
    dataset_content_hash: str
    corpus_snapshot: str
    profile_ids: dict[str, str]
    channels: list[LayeredChannelComparison]


def _rank_status(
    baseline: LayeredQuestionResult, current: LayeredQuestionResult
) -> Literal["improved", "regressed", "unchanged", "incompatible"]:
    if baseline.status != "evaluated" or current.status != "evaluated":
        return "incompatible"
    if baseline.expected_rank == current.expected_rank:
        return "unchanged"
    if current.expected_rank is None:
        return "regressed"
    if baseline.expected_rank is None or current.expected_rank < baseline.expected_rank:
        return "improved"
    return "regressed"


def compare_layered_article_evaluations(
    baseline: LayeredArticleEvaluationResult,
    current: LayeredArticleEvaluationResult,
) -> LayeredEvaluationComparison:
    validate_layered_comparison(baseline, current)
    current_channels = {
        (item.layer, item.mode, item.aggregation): item for item in current.channels
    }
    metric_fields = (
        "hit_at_1",
        "hit_at_3",
        "hit_at_5",
        "mrr",
        "latency_mean_ms",
        "latency_p95_ms",
    )
    comparisons: list[LayeredChannelComparison] = []
    for base_channel in baseline.channels:
        current_channel = current_channels[
            (base_channel.layer, base_channel.mode, base_channel.aggregation)
        ]
        metrics: dict[str, LayeredMetricDelta] = {}
        for field in metric_fields:
            base_value = getattr(base_channel.metrics, field)
            current_value = getattr(current_channel.metrics, field)
            metrics[field] = LayeredMetricDelta(
                baseline=base_value,
                current=current_value,
                delta=(
                    current_value - base_value
                    if base_value is not None and current_value is not None
                    else None
                ),
            )
        current_questions = {item.question_id: item for item in current_channel.questions}
        if {item.question_id for item in base_channel.questions} != set(current_questions):
            raise ValueError("分层文章反查结果不可直接比较：题目 ID 不同")
        comparisons.append(
            LayeredChannelComparison(
                layer=base_channel.layer,
                mode=base_channel.mode,
                aggregation=base_channel.aggregation,
                metrics=metrics,
                questions=[
                    LayeredQuestionRankDelta(
                        question_id=item.question_id,
                        baseline_rank=item.expected_rank,
                        current_rank=current_questions[item.question_id].expected_rank,
                        status=_rank_status(item, current_questions[item.question_id]),
                    )
                    for item in base_channel.questions
                ],
            )
        )
    return LayeredEvaluationComparison(
        baseline_run_id=baseline.run_id,
        current_run_id=current.run_id,
        dataset_content_hash=baseline.dataset_content_hash,
        corpus_snapshot=baseline.corpus_snapshot,
        profile_ids=baseline.profile_ids,
        channels=comparisons,
    )


class FallbackPolicy(StrEnum):
    MARGIN_CASCADE = "margin_cascade"
    DISAGREEMENT_CASCADE = "disagreement_cascade"


class FallbackQuestionResult(BaseModel):
    question_id: str
    expected_article_id: str
    activated_layers: list[ArticleDiscoveryLayer]
    final_rank: int | None
    reciprocal_rank: float
    latency_estimate_ms: float
    improved_vs_overview: bool
    regressed_vs_overview: bool


class FallbackSimulationResult(BaseModel):
    schema_version: Literal[1] = 1
    task: Literal["layered_article_fallback_simulation"] = (
        "layered_article_fallback_simulation"
    )
    source_run_id: str
    dataset_content_hash: str
    corpus_snapshot: str
    policy: FallbackPolicy
    margin_ratio_threshold: float
    overlap_threshold: float
    fusion_rrf_k: int
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    mrr: float
    layer_trigger_rate: dict[str, float]
    latency_mean_ms: float
    latency_max_ms: float
    improved_question_ids: list[str]
    regressed_question_ids: list[str]
    questions: list[FallbackQuestionResult]


def _channel(
    result: LayeredArticleEvaluationResult,
    layer: ArticleDiscoveryLayer,
    mode: RetrievalMode,
) -> LayeredChannelResult:
    return next(
        item
        for item in result.channels
        if item.layer == layer
        and item.mode == mode
        and item.aggregation == ArticleAggregation.BEST_UNIT
    )


def _question(channel: LayeredChannelResult, question_id: str) -> LayeredQuestionResult:
    return next(item for item in channel.questions if item.question_id == question_id)


def _is_clear(
    bm25: LayeredQuestionResult,
    dense: LayeredQuestionResult,
    margin_threshold: float,
    overlap_threshold: float,
) -> bool:
    if bm25.status != "evaluated" or dense.status != "evaluated":
        return False
    if not bm25.bm25_dense_top1_agree:
        return False
    margins = []
    for item in (bm25, dense):
        if item.top1_top2_score_margin is None or not item.candidates:
            continue
        denominator = abs(item.candidates[0].score)
        if denominator:
            margins.append(item.top1_top2_score_margin / denominator)
    return is_layer_clear(
        top1_agree=True,
        relative_margins=margins,
        top_k_overlap=bm25.bm25_dense_top_k_overlap or 0.0,
        margin_threshold=margin_threshold,
        overlap_threshold=overlap_threshold,
    )


def _fuse_channels(
    channel_questions: list[LayeredQuestionResult], rrf_k: int
) -> list[str]:
    rankings = [
        [candidate.article_id for candidate in question.candidates]
        for question in channel_questions
        if question.status == "evaluated"
    ]
    return [article_id for article_id, _ in article_rank_fusion(rankings, rrf_k)]


def simulate_layered_fallback(
    result: LayeredArticleEvaluationResult,
    *,
    policy: FallbackPolicy = FallbackPolicy.MARGIN_CASCADE,
    margin_ratio_threshold: float = 0.1,
    overlap_threshold: float = 0.5,
    fusion_rrf_k: int = 60,
) -> FallbackSimulationResult:
    if not 0 <= margin_ratio_threshold <= 1 or not 0 <= overlap_threshold <= 1:
        raise ValueError("降级阈值必须位于 0 到 1")
    overview_bm25 = _channel(result, ArticleDiscoveryLayer.OVERVIEW, RetrievalMode.BM25)
    overview_dense = _channel(result, ArticleDiscoveryLayer.OVERVIEW, RetrievalMode.DENSE)
    outputs: list[FallbackQuestionResult] = []
    triggers: dict[ArticleDiscoveryLayer, int] = defaultdict(int)
    improved: list[str] = []
    regressed: list[str] = []
    for base in overview_bm25.questions:
        qid = base.question_id
        activated = [ArticleDiscoveryLayer.OVERVIEW]
        selected = [
            _question(overview_bm25, qid),
            _question(overview_dense, qid),
        ]
        overview_fused = _fuse_channels(selected, fusion_rrf_k)
        overview_rank = (
            overview_fused.index(base.expected_article_id) + 1
            if base.expected_article_id in overview_fused
            else None
        )
        overview_clear = _is_clear(
            selected[0], selected[1], margin_ratio_threshold, overlap_threshold
        )
        if not overview_clear:
            next_layer = (
                ArticleDiscoveryLayer.SUBTOPICS
                if policy == FallbackPolicy.DISAGREEMENT_CASCADE
                and selected[0].bm25_dense_top1_agree is False
                else ArticleDiscoveryLayer.KEYWORDS
            )
            activated.append(next_layer)
            selected.extend(
                [
                    _question(_channel(result, next_layer, RetrievalMode.BM25), qid),
                    _question(_channel(result, next_layer, RetrievalMode.DENSE), qid),
                ]
            )
            if not _is_clear(
                selected[-2], selected[-1], margin_ratio_threshold, overlap_threshold
            ):
                remaining = (
                    ArticleDiscoveryLayer.SUBTOPICS
                    if next_layer == ArticleDiscoveryLayer.KEYWORDS
                    else ArticleDiscoveryLayer.ORIGINAL
                )
                activated.append(remaining)
                selected.extend(
                    [
                        _question(_channel(result, remaining, RetrievalMode.BM25), qid),
                        _question(_channel(result, remaining, RetrievalMode.DENSE), qid),
                    ]
                )
                if remaining != ArticleDiscoveryLayer.ORIGINAL and not _is_clear(
                    selected[-2], selected[-1], margin_ratio_threshold, overlap_threshold
                ):
                    activated.append(ArticleDiscoveryLayer.ORIGINAL)
                    selected.extend(
                        [
                            _question(
                                _channel(
                                    result,
                                    ArticleDiscoveryLayer.ORIGINAL,
                                    RetrievalMode.BM25,
                                ),
                                qid,
                            ),
                            _question(
                                _channel(
                                    result,
                                    ArticleDiscoveryLayer.ORIGINAL,
                                    RetrievalMode.DENSE,
                                ),
                                qid,
                            ),
                        ]
                    )
        fused = _fuse_channels(selected, fusion_rrf_k)
        final_rank = (
            fused.index(base.expected_article_id) + 1
            if base.expected_article_id in fused
            else None
        )
        is_improved = final_rank is not None and (
            overview_rank is None or final_rank < overview_rank
        )
        is_regressed = overview_rank is not None and (
            final_rank is None or final_rank > overview_rank
        )
        if is_improved:
            improved.append(qid)
        if is_regressed:
            regressed.append(qid)
        for layer in activated:
            triggers[layer] += 1
        latency = sum(item.latency_ms or 0.0 for item in selected)
        outputs.append(
            FallbackQuestionResult(
                question_id=qid,
                expected_article_id=base.expected_article_id,
                activated_layers=activated,
                final_rank=final_rank,
                reciprocal_rank=1.0 / final_rank if final_rank is not None else 0.0,
                latency_estimate_ms=latency,
                improved_vs_overview=is_improved,
                regressed_vs_overview=is_regressed,
            )
        )
    count = len(outputs)
    return FallbackSimulationResult(
        source_run_id=result.run_id,
        dataset_content_hash=result.dataset_content_hash,
        corpus_snapshot=result.corpus_snapshot,
        policy=policy,
        margin_ratio_threshold=margin_ratio_threshold,
        overlap_threshold=overlap_threshold,
        fusion_rrf_k=fusion_rrf_k,
        hit_at_1=sum(item.final_rank == 1 for item in outputs) / count,
        hit_at_3=sum(item.final_rank is not None and item.final_rank <= 3 for item in outputs)
        / count,
        hit_at_5=sum(item.final_rank is not None and item.final_rank <= 5 for item in outputs)
        / count,
        mrr=sum(item.reciprocal_rank for item in outputs) / count,
        layer_trigger_rate={
            layer.value: triggers[layer] / count for layer in ArticleDiscoveryLayer
        },
        latency_mean_ms=sum(item.latency_estimate_ms for item in outputs) / count,
        latency_max_ms=max(item.latency_estimate_ms for item in outputs),
        improved_question_ids=improved,
        regressed_question_ids=regressed,
        questions=outputs,
    )
