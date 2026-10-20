from __future__ import annotations

import json
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from xifeng_rag.storage import (
    ArticleRow,
    EvaluationEvidenceRow,
    EvaluationQuestionRow,
    EvaluationRunRow,
    TopicChunkRow,
)

SUMMARY_METRICS = (
    "precision_at_k",
    "recall_at_k",
    "mrr",
    "ndcg_at_k",
    "evidence_coverage_at_k",
    "latency_mean_ms",
    "latency_p95_ms",
)


def _json_object(value: str) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("评测结果格式无效")
    return parsed


def _json_list(value: str) -> list[dict[str, Any]]:
    parsed = json.loads(value)
    if not isinstance(parsed, list) or not all(isinstance(item, dict) for item in parsed):
        raise ValueError("评测逐题结果格式无效")
    return parsed


def _run_summary(row: EvaluationRunRow) -> dict[str, object]:
    metrics = _json_object(row.metrics_json)
    return {
        "id": row.id,
        "dataset_id": row.dataset_id,
        "dataset_content_hash": row.dataset_content_hash,
        "corpus_snapshot": row.corpus_snapshot,
        "build_id": row.build_id,
        "profile_id": row.profile_id,
        "split": metrics.get("selected_split", "all"),
        "k": row.k,
        "retrieval_mode": row.retrieval_mode,
        "reranker_enabled": row.reranker_enabled,
        "reranker_model_id": row.reranker_model_id,
        "code_commit": row.code_commit,
        "worktree_dirty": row.worktree_dirty,
        "created_at": row.created_at.isoformat(),
        "metrics": metrics,
    }


def list_retrieval_runs(session: Session, dataset_id: str) -> list[dict[str, object]]:
    rows = session.scalars(
        select(EvaluationRunRow)
        .where(EvaluationRunRow.dataset_id == dataset_id)
        .order_by(EvaluationRunRow.created_at.desc())
    ).all()
    return [_run_summary(row) for row in rows]


def retrieval_run_detail(session: Session, run_id: str) -> dict[str, object]:
    row = session.get(EvaluationRunRow, run_id)
    if row is None:
        raise ValueError("评测运行不存在")
    details = _json_list(row.details_json)
    question_ids = [str(item.get("question_id", "")) for item in details]
    questions = {
        item.id: item
        for item in session.scalars(
            select(EvaluationQuestionRow).where(EvaluationQuestionRow.id.in_(question_ids))
        ).all()
    }
    evidence_rows = session.scalars(
        select(EvaluationEvidenceRow)
        .where(EvaluationEvidenceRow.question_id.in_(question_ids))
        .order_by(EvaluationEvidenceRow.question_id, EvaluationEvidenceRow.ordinal)
    ).all()
    evidence: dict[str, list[dict[str, object]]] = {}
    for item in evidence_rows:
        evidence.setdefault(item.question_id, []).append(
            {
                "claim_group": item.claim_group,
                "claim": item.claim_text,
                "start_paragraph": item.start_paragraph,
                "end_paragraph": item.end_paragraph,
            }
        )

    candidate_ids = {
        str(candidate.get("chunk_id", ""))
        for detail in details
        for candidate in detail.get("retrieval_candidates", [])
        if isinstance(candidate, dict)
    }
    chunks = {
        item.id: item
        for item in session.scalars(
            select(TopicChunkRow).where(TopicChunkRow.id.in_(candidate_ids))
        ).all()
    }
    article_ids = {item.article_id for item in chunks.values()}
    articles = {
        item.id: item.title
        for item in session.scalars(select(ArticleRow).where(ArticleRow.id.in_(article_ids))).all()
    }

    rendered_details: list[dict[str, object]] = []
    for detail in details:
        question_id = str(detail.get("question_id", ""))
        question = questions.get(question_id)
        candidates: list[dict[str, object]] = []
        for rank, raw in enumerate(detail.get("retrieval_candidates", []), start=1):
            if not isinstance(raw, dict):
                continue
            chunk = chunks.get(str(raw.get("chunk_id", "")))
            candidates.append(
                {
                    **raw,
                    "rank": rank,
                    "article_title": articles.get(chunk.article_id, "") if chunk else "",
                    "topic_title": chunk.topic_title if chunk else "",
                    "paragraph_range": (
                        [chunk.start_paragraph, chunk.end_paragraph] if chunk else None
                    ),
                    "original_text": chunk.original_text if chunk else None,
                    "relevant": str(raw.get("chunk_id", ""))
                    in set(detail.get("mapped_relevant_chunk_ids", [])),
                }
            )
        rendered_details.append(
            {
                **detail,
                "query": question.query if question else "",
                "answerable": question.answerable if question else None,
                "gold_evidence": evidence.get(question_id, []),
                "retrieval_candidates": candidates,
            }
        )
    return {**_run_summary(row), "questions": rendered_details}


def compare_retrieval_runs(
    session: Session, baseline_run_id: str, current_run_id: str
) -> dict[str, object]:
    baseline = session.get(EvaluationRunRow, baseline_run_id)
    current = session.get(EvaluationRunRow, current_run_id)
    if baseline is None or current is None:
        raise ValueError("基线或当前评测运行不存在")
    baseline_metrics = _json_object(baseline.metrics_json)
    current_metrics = _json_object(current.metrics_json)
    reasons: list[str] = []
    checks = (
        (baseline.dataset_content_hash, current.dataset_content_hash, "题集内容哈希不同"),
        (baseline.corpus_snapshot, current.corpus_snapshot, "语料快照不同"),
        (
            baseline_metrics.get("selected_split"),
            current_metrics.get("selected_split"),
            "split 不同",
        ),
        (baseline.k, current.k, "K 不同"),
    )
    for left, right, message in checks:
        if left != right:
            reasons.append(message)
    compatible = not reasons
    metrics: dict[str, dict[str, float | None]] = {}
    for name in SUMMARY_METRICS:
        left = baseline_metrics.get(name)
        right = current_metrics.get(name)
        left_number = float(left) if isinstance(left, (int, float)) else None
        right_number = float(right) if isinstance(right, (int, float)) else None
        metrics[name] = {
            "baseline": left_number,
            "current": right_number,
            "delta": (
                right_number - left_number
                if compatible and left_number is not None and right_number is not None
                else None
            ),
        }
    baseline_details = {
        str(item.get("question_id", "")): item for item in _json_list(baseline.details_json)
    }
    current_details = {
        str(item.get("question_id", "")): item for item in _json_list(current.details_json)
    }
    questions: list[dict[str, object]] = []
    for question_id in sorted(set(baseline_details) | set(current_details)):
        left = baseline_details.get(question_id)
        right = current_details.get(question_id)
        values: dict[str, dict[str, float | None]] = {}
        for name in ("recall_at_k", "mrr", "evidence_coverage_at_k", "latency_ms"):
            left_value = left.get(name) if left else None
            right_value = right.get(name) if right else None
            left_number = float(left_value) if isinstance(left_value, (int, float)) else None
            right_number = float(right_value) if isinstance(right_value, (int, float)) else None
            values[name] = {
                "baseline": left_number,
                "current": right_number,
                "delta": (
                    right_number - left_number
                    if compatible and left_number is not None and right_number is not None
                    else None
                ),
            }
        coverage_delta = values["evidence_coverage_at_k"]["delta"]
        mrr_delta = values["mrr"]["delta"]
        status = "incompatible"
        if compatible and left is not None and right is not None:
            if (coverage_delta or 0.0) > 0 or (
                coverage_delta == 0 and (mrr_delta or 0.0) > 0
            ):
                status = "improved"
            elif (coverage_delta or 0.0) < 0 or (
                coverage_delta == 0 and (mrr_delta or 0.0) < 0
            ):
                status = "regressed"
            else:
                status = "unchanged"
        questions.append({"question_id": question_id, "status": status, "metrics": values})
    return {
        "compatible": compatible,
        "reasons": reasons,
        "baseline": _run_summary(baseline),
        "current": _run_summary(current),
        "metrics": metrics,
        "questions": questions,
    }
