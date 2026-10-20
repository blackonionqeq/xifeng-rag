from __future__ import annotations

import json
import statistics
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from xifeng_rag.answering import answer_with_context, prepare_answer_context
from xifeng_rag.corpus import stable_id
from xifeng_rag.domain import (
    AssemblyResult,
    ContextAssemblyConfigSnapshot,
    RetrievalCandidate,
    RetrievalMode,
)
from xifeng_rag.generation import (
    AnswerGenerationError,
    AnswerStatus,
    Generator,
    GeneratorProviderError,
    SupplementPolicy,
)
from xifeng_rag.storage import (
    AnswerEvaluationItemRow,
    AnswerEvaluationRunRow,
    EvaluationDatasetRow,
    EvaluationEvidenceRow,
    EvaluationQuestionRow,
    IndexBuildRow,
)

Retriever = Callable[[Session, str], list[RetrievalCandidate]]


@dataclass(frozen=True)
class GoldRange:
    version_id: str
    start: int
    end: int
    claim_group: str


@dataclass(frozen=True)
class PreparedAnswerItem:
    question_id: str
    ordinal: int
    query: str
    answerable: bool
    difficulty: str
    origin: str
    context: AssemblyResult
    retrieved_ranges: tuple[tuple[str, int, int], ...]
    gold_ranges: tuple[GoldRange, ...]
    retrieval_latency_ms: float
    assembly_latency_ms: float


@dataclass(frozen=True)
class PreparedAnswerBatch:
    dataset_id: str
    dataset_name: str
    dataset_version: str
    dataset_content_hash: str
    corpus_snapshot: str
    build_id: str
    split: str
    mode: RetrievalMode
    k: int
    items: tuple[PreparedAnswerItem, ...]

    @property
    def external_question_count(self) -> int:
        return sum(bool(item.context.evidence_items) for item in self.items)

    @property
    def answerable_question_count(self) -> int:
        return sum(item.answerable for item in self.items)

    @property
    def evidence_item_count(self) -> int:
        return sum(item.context.usage.selected_count for item in self.items)

    @property
    def evidence_character_count(self) -> int:
        return sum(item.context.usage.original_text_chars for item in self.items)


def prepare_answer_batch(
    session: Session,
    dataset_id: str,
    *,
    split: str,
    mode: RetrievalMode,
    k: int,
    build_id: str | None,
    retriever: Retriever,
    assembly_config: ContextAssemblyConfigSnapshot,
) -> PreparedAnswerBatch:
    if split not in {"dev", "holdout"}:
        raise ValueError("--split 必须显式指定 dev 或 holdout")
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
        raise ValueError("索引语料快照与冻结评测集不一致")
    questions = list(
        session.scalars(
            select(EvaluationQuestionRow)
            .where(
                EvaluationQuestionRow.dataset_id == dataset.id,
                EvaluationQuestionRow.split == split,
            )
            .order_by(EvaluationQuestionRow.ordinal)
        ).all()
    )
    if not questions:
        raise ValueError(f"冻结评测集没有 {split} 问题")

    items: list[PreparedAnswerItem] = []
    for question in questions:
        evidence = list(
            session.scalars(
                select(EvaluationEvidenceRow)
                .where(EvaluationEvidenceRow.question_id == question.id)
                .order_by(EvaluationEvidenceRow.ordinal)
            ).all()
        )
        retrieval_started = time.perf_counter()
        candidates = retriever(session, question.query)
        retrieval_latency_ms = (time.perf_counter() - retrieval_started) * 1000
        assembly_started = time.perf_counter()
        context = prepare_answer_context(candidates, assembly_config)
        assembly_latency_ms = (time.perf_counter() - assembly_started) * 1000
        items.append(
            PreparedAnswerItem(
                question_id=question.id,
                ordinal=question.ordinal,
                query=question.query,
                answerable=question.answerable,
                difficulty=question.difficulty,
                origin=question.origin,
                context=context,
                retrieved_ranges=tuple(
                    (
                        candidate.article_version,
                        candidate.paragraph_range[0],
                        candidate.paragraph_range[1],
                    )
                    for candidate in candidates
                ),
                gold_ranges=tuple(
                    GoldRange(
                        version_id=row.version_id,
                        start=row.start_paragraph,
                        end=row.end_paragraph,
                        claim_group=row.claim_group,
                    )
                    for row in evidence
                ),
                retrieval_latency_ms=retrieval_latency_ms,
                assembly_latency_ms=assembly_latency_ms,
            )
        )
    return PreparedAnswerBatch(
        dataset_id=dataset.id,
        dataset_name=dataset.name,
        dataset_version=dataset.version,
        dataset_content_hash=dataset.content_hash,
        corpus_snapshot=build.corpus_snapshot,
        build_id=build.id,
        split=split,
        mode=mode,
        k=k,
        items=tuple(items),
    )


def _unit_coverage(
    ranges: Sequence[tuple[str, int, int]], gold_ranges: Sequence[GoldRange]
) -> float:
    gold = {
        (item.version_id, unit)
        for item in gold_ranges
        for unit in range(item.start, item.end + 1)
    }
    if not gold:
        return 1.0
    covered = {
        (version_id, unit)
        for version_id, start, end in ranges
        for unit in range(start, end + 1)
        if (version_id, unit) in gold
    }
    return len(covered) / len(gold)


def _overlaps_gold(version_id: str, start: int, end: int, gold: Sequence[GoldRange]) -> bool:
    return any(
        item.version_id == version_id and start <= item.end and end >= item.start
        for item in gold
    )


def _git_state() -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5
        )
        if commit.returncode != 0 or status.returncode != 0:
            return None, None
        return commit.stdout.strip()[:64], bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return None, None


def _score_item(
    item: PreparedAnswerItem, result
) -> tuple[dict[str, object], list[str], list[dict[str, object]]]:
    answer = result.answer
    predicted_abstain = answer.status == AnswerStatus.INSUFFICIENT_EVIDENCE
    cited_ids = {
        evidence_id
        for section in answer.grounded_sections
        for evidence_id in section.evidence_ids
    }
    evidence_by_id = {evidence.citation_id: evidence for evidence in item.context.evidence_items}
    valid_ids = cited_ids & evidence_by_id.keys()
    citation_ranges = [
        (
            evidence_by_id[citation_id].article_version,
            evidence_by_id[citation_id].selected_paragraph_range[0],
            evidence_by_id[citation_id].selected_paragraph_range[1],
        )
        for citation_id in sorted(valid_ids)
    ]
    on_gold_count = sum(
        _overlaps_gold(version, start, end, item.gold_ranges)
        for version, start, end in citation_ranges
    )
    claim_groups = {gold.claim_group for gold in item.gold_ranges}
    covered_claim_groups = {
        gold.claim_group
        for gold in item.gold_ranges
        if any(
            version == gold.version_id and start <= gold.end and end >= gold.start
            for version, start, end in citation_ranges
        )
    }
    retrieval_coverage = _unit_coverage(item.retrieved_ranges, item.gold_ranges)
    context_ranges = tuple(
        (
            evidence.article_version,
            evidence.selected_paragraph_range[0],
            evidence.selected_paragraph_range[1],
        )
        for evidence in item.context.evidence_items
    )
    context_coverage = _unit_coverage(context_ranges, item.gold_ranges)
    citation_coverage = _unit_coverage(citation_ranges, item.gold_ranges)
    metrics: dict[str, object] = {
        "answerable": item.answerable,
        "difficulty": item.difficulty,
        "origin": item.origin,
        "predicted_abstain": predicted_abstain,
        "correct_refusal": (not item.answerable and predicted_abstain),
        "missed_refusal": (not item.answerable and not predicted_abstain),
        "over_refusal": (item.answerable and predicted_abstain),
        "citation_id_valid_rate": (len(valid_ids) / len(cited_ids) if cited_ids else None),
        "cited_evidence_count": len(valid_ids),
        "off_gold_citation_rate": (
            (len(citation_ranges) - on_gold_count) / len(citation_ranges)
            if citation_ranges
            else None
        ),
        "retrieval_gold_evidence_coverage": retrieval_coverage,
        "context_gold_evidence_coverage": context_coverage,
        "gold_evidence_citation_coverage": citation_coverage,
        "answer_point_citation_coverage": (
            len(covered_claim_groups) / len(claim_groups) if claim_groups else None
        ),
    }
    labels: list[str] = []
    if item.answerable:
        if retrieval_coverage < 1:
            labels.append("retrieval_miss")
        elif context_coverage < 1:
            labels.append("assembly_loss")
        elif predicted_abstain:
            labels.append("over_refusal")
        elif covered_claim_groups != claim_groups:
            labels.append("citation_omission")
        if citation_ranges and on_gold_count < len(citation_ranges):
            labels.append("off_gold_citation")
    elif not predicted_abstain:
        labels.append("missed_refusal")
    citations = [
        {
            "citation_id": citation_id,
            "source_chunk_id": evidence_by_id[citation_id].source_chunk_id,
            "article_id": evidence_by_id[citation_id].article_id,
            "article_version": evidence_by_id[citation_id].article_version,
            "paragraph_range": list(evidence_by_id[citation_id].selected_paragraph_range),
        }
        for citation_id in sorted(valid_ids)
    ]
    return metrics, labels, citations


def _summarize(item_metrics: Sequence[dict[str, object]]) -> dict[str, object]:
    def number(item: dict[str, object], name: str) -> float:
        value = item[name]
        if not isinstance(value, (int, float)):
            raise TypeError(f"{name} 必须是数值")
        return float(value)

    count = len(item_metrics)
    answerable = [item for item in item_metrics if item["answerable"] is True]
    unanswerable = [item for item in item_metrics if item["answerable"] is False]
    correct_refusal_count = sum(bool(item["correct_refusal"]) for item in item_metrics)
    missed_refusal_count = sum(bool(item["missed_refusal"]) for item in item_metrics)
    over_refusal_count = sum(bool(item["over_refusal"]) for item in item_metrics)
    summary: dict[str, object] = {
        "question_count": count,
        "completed_question_count": count,
        "answerable_question_count": len(answerable),
        "unanswerable_question_count": len(unanswerable),
        "correct_refusal_count": correct_refusal_count,
        "missed_refusal_count": missed_refusal_count,
        "over_refusal_count": over_refusal_count,
        "correct_refusal_rate": (
            correct_refusal_count / len(unanswerable) if unanswerable else None
        ),
        "missed_refusal_rate": (
            missed_refusal_count / len(unanswerable) if unanswerable else None
        ),
        "over_refusal_rate": over_refusal_count / len(answerable) if answerable else None,
    }
    for name in (
        "citation_id_valid_rate",
        "off_gold_citation_rate",
        "retrieval_gold_evidence_coverage",
        "context_gold_evidence_coverage",
        "gold_evidence_citation_coverage",
        "answer_point_citation_coverage",
    ):
        population = (
            answerable
            if name
            in {
                "retrieval_gold_evidence_coverage",
                "context_gold_evidence_coverage",
                "gold_evidence_citation_coverage",
                "answer_point_citation_coverage",
                "off_gold_citation_rate",
            }
            else list(item_metrics)
        )
        values = [number(item, name) for item in population if item[name] is not None]
        summary[name] = statistics.fmean(values) if values else None
    summary["total_external_request_count"] = sum(
        int(number(item, "request_count")) for item in item_metrics
    )
    summary["latency_mean_ms"] = statistics.fmean(
        number(item, "latency_ms") for item in item_metrics
    )
    for field in ("difficulty", "origin"):
        grouped: dict[str, dict[str, object]] = {}
        for value in sorted({str(item[field]) for item in item_metrics}):
            members = [item for item in item_metrics if item[field] == value]
            grouped[value] = {
                "question_count": len(members),
                "correct_refusal_count": sum(bool(item["correct_refusal"]) for item in members),
                "over_refusal_count": sum(bool(item["over_refusal"]) for item in members),
            }
        summary[f"by_{field}"] = grouped
    return summary


def run_prepared_answer_batch(
    session: Session,
    batch: PreparedAnswerBatch,
    *,
    generator: Generator | None,
    provider_id: str,
    model_id: str,
) -> dict[str, object]:
    run_id = stable_id("answer-eval", batch.dataset_id, str(time.time_ns()), length=32)
    code_commit, worktree_dirty = _git_state()
    run = AnswerEvaluationRunRow(
        id=run_id,
        dataset_id=batch.dataset_id,
        build_id=batch.build_id,
        dataset_content_hash=batch.dataset_content_hash,
        corpus_snapshot=batch.corpus_snapshot,
        split=batch.split,
        retrieval_mode=batch.mode.value,
        k=batch.k,
        provider_id=provider_id,
        model_id=model_id,
        supplement_policy=SupplementPolicy.OFF.value,
        status="running",
        metrics_json="{}",
        code_commit=code_commit,
        worktree_dirty=worktree_dirty,
    )
    session.add(run)
    session.flush()
    completed_metrics: list[dict[str, object]] = []
    provider_failed = False
    error_count = 0
    for item in batch.items:
        started = time.perf_counter()
        result = None
        error_type = None
        labels: list[str] = []
        citations: list[dict[str, object]] = []
        metrics: dict[str, object] = {}
        try:
            result = answer_with_context(
                item.query, item.context, generator, supplement=SupplementPolicy.OFF
            )
            generation_latency_ms = (time.perf_counter() - started) * 1000
            latency_ms = (
                item.retrieval_latency_ms + item.assembly_latency_ms + generation_latency_ms
            )
            metrics, labels, citations = _score_item(item, result)
            metrics["request_count"] = result.request_count
            metrics["retrieval_latency_ms"] = item.retrieval_latency_ms
            metrics["assembly_latency_ms"] = item.assembly_latency_ms
            metrics["generation_latency_ms"] = generation_latency_ms
            metrics["latency_ms"] = latency_ms
            completed_metrics.append(metrics)
            status = "completed"
        except GeneratorProviderError as exc:
            generation_latency_ms = (time.perf_counter() - started) * 1000
            latency_ms = (
                item.retrieval_latency_ms + item.assembly_latency_ms + generation_latency_ms
            )
            error_type = type(exc).__name__
            labels = ["generation_error"]
            status = "error"
            provider_failed = True
            error_count += 1
        except AnswerGenerationError as exc:
            generation_latency_ms = (time.perf_counter() - started) * 1000
            latency_ms = (
                item.retrieval_latency_ms + item.assembly_latency_ms + generation_latency_ms
            )
            error_type = type(exc).__name__
            labels = ["generation_error"]
            status = "error"
            error_count += 1
        usage = (
            result.generation_metadata.get("usage", {})
            if result is not None and isinstance(result.generation_metadata.get("usage", {}), dict)
            else {}
        )
        session.add(
            AnswerEvaluationItemRow(
                id=stable_id("answer-eval-item", run_id, item.question_id, length=40),
                run_id=run_id,
                question_id=item.question_id,
                ordinal=item.ordinal,
                answerable=item.answerable,
                status=status,
                answer_json=(
                    json.dumps(result.answer.model_dump(mode="json"), ensure_ascii=False)
                    if result is not None
                    else None
                ),
                citations_json=json.dumps(citations, ensure_ascii=False),
                metrics_json=json.dumps(metrics, ensure_ascii=False),
                failure_labels_json=json.dumps(labels),
                request_count=result.request_count if result is not None else 0,
                latency_ms=latency_ms,
                usage_json=json.dumps(usage),
                error_type=error_type,
            )
        )
        session.flush()
        if provider_failed:
            break

    metrics = _summarize(completed_metrics) if completed_metrics else {
        "question_count": len(batch.items),
        "completed_question_count": 0,
    }
    metrics["planned_question_count"] = len(batch.items)
    metrics["selected_split"] = batch.split
    processed_count = len(completed_metrics) + error_count
    metrics["processed_question_count"] = processed_count
    run.metrics_json = json.dumps(metrics, ensure_ascii=False)
    run.status = (
        ("completed_with_errors" if completed_metrics else "failed")
        if provider_failed
        else "completed_with_errors"
        if error_count
        else "completed"
    )
    run.error_type = "GeneratorProviderError" if provider_failed else None
    run.completed_at = datetime.now(UTC)
    session.flush()
    return {"run_id": run_id, "status": run.status, "metrics": metrics}
