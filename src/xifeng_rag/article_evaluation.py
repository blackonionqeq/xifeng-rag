from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from xifeng_rag.article_search import DOCUMENTS_FILE, search_articles
from xifeng_rag.corpus import stable_id
from xifeng_rag.domain import BuildStatus, RetrievalMode
from xifeng_rag.indexing import EmbeddingProvider
from xifeng_rag.storage import ArticleIndexBuildRow


class ArticleDiscoveryQuestion(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    expected_article_id: str = Field(min_length=1)
    expected_title: str | None = None
    review_status: str


class ArticleDiscoveryDataset(BaseModel):
    model_config = ConfigDict(extra="ignore")

    schema_version: Literal[1]
    queries: list[ArticleDiscoveryQuestion] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_question_ids(self) -> ArticleDiscoveryDataset:
        ids = [item.id for item in self.queries]
        if len(ids) != len(set(ids)):
            raise ValueError("文章反查评测题 ID 不能重复")
        if not any(item.review_status == "confirmed" for item in self.queries):
            raise ValueError("文章反查评测集没有人工确认题")
        return self

    @property
    def confirmed_questions(self) -> list[ArticleDiscoveryQuestion]:
        return [item for item in self.queries if item.review_status == "confirmed"]

    @property
    def content_hash(self) -> str:
        semantic = {
            "schema_version": self.schema_version,
            "queries": [
                {
                    "id": item.id,
                    "query": item.query,
                    "expected_article_id": item.expected_article_id,
                    "review_status": item.review_status,
                }
                for item in self.queries
            ],
        }
        encoded = json.dumps(
            semantic, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class ArticleDiscoveryCandidateResult(BaseModel):
    rank: int
    article_id: str
    title: str
    sources: list[str]
    rrf_score: float | None
    bm25_score: float | None
    bm25_rank: int | None
    dense_score: float | None
    dense_rank: int | None


class ArticleDiscoveryQuestionResult(BaseModel):
    question_id: str
    query: str
    expected_article_id: str
    expected_title: str | None
    status: Literal["evaluated", "expected_article_not_in_build"]
    expected_rank: int | None
    reciprocal_rank: float | None
    hit_at_1: bool | None
    hit_at_3: bool | None
    hit_at_5: bool | None
    latency_ms: float | None
    candidates: list[ArticleDiscoveryCandidateResult]


class ArticleDiscoveryModeMetrics(BaseModel):
    evaluated_question_count: int
    incompatible_question_count: int
    hit_at_1: float | None
    hit_at_3: float | None
    hit_at_5: float | None
    mrr: float | None
    latency_mean_ms: float | None
    latency_p95_ms: float | None


class ArticleDiscoveryModeResult(BaseModel):
    mode: RetrievalMode
    metrics: ArticleDiscoveryModeMetrics
    questions: list[ArticleDiscoveryQuestionResult]


class ArticleDiscoveryEvaluationResult(BaseModel):
    schema_version: Literal[1] = 1
    run_id: str
    task: Literal["article_discovery"] = "article_discovery"
    created_at: datetime
    dataset_path: str
    dataset_content_hash: str
    confirmed_question_count: int
    skipped_question_count: int
    build_id: str
    corpus_snapshot: str
    profile_id: str
    code_commit: str | None
    worktree_dirty: bool | None
    evaluation_k: int
    bm25_candidate_k: int
    dense_candidate_k: int
    rrf_k: int
    modes: list[ArticleDiscoveryModeResult]


def load_article_discovery_dataset(path: Path) -> ArticleDiscoveryDataset:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"文章反查评测集不存在：{resolved}")
    return ArticleDiscoveryDataset.model_validate_json(resolved.read_text(encoding="utf-8"))


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


def _resolve_build(session: Session, build_id: str | None) -> ArticleIndexBuildRow:
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
        raise ValueError("没有可用于评测的文章级索引")
    if not Path(build.path).is_dir():
        raise FileNotFoundError(f"文章索引目录不存在：{build.path}")
    return build


def _indexed_article_ids(build: ArticleIndexBuildRow) -> set[str]:
    document_path = Path(build.path) / DOCUMENTS_FILE
    payload = json.loads(document_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("文章索引文档清单格式错误")
    article_ids: set[str] = set()
    for item in payload:
        if not isinstance(item, dict):
            continue
        article_id = item.get("article_id")
        if isinstance(article_id, str):
            article_ids.add(article_id)
    if len(article_ids) != len(payload):
        raise ValueError("文章索引文档清单缺少唯一 article_id")
    return article_ids


def _mode_metrics(
    questions: list[ArticleDiscoveryQuestionResult],
) -> ArticleDiscoveryModeMetrics:
    evaluated = [item for item in questions if item.status == "evaluated"]
    latencies = [item.latency_ms for item in evaluated if item.latency_ms is not None]

    def rate(field: str) -> float | None:
        if not evaluated:
            return None
        return sum(bool(getattr(item, field)) for item in evaluated) / len(evaluated)

    return ArticleDiscoveryModeMetrics(
        evaluated_question_count=len(evaluated),
        incompatible_question_count=len(questions) - len(evaluated),
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
    )


def run_article_discovery_evaluation(
    session: Session,
    dataset: ArticleDiscoveryDataset,
    dataset_path: Path,
    embedding_provider: EmbeddingProvider,
    *,
    build_id: str | None = None,
    evaluation_k: int = 5,
    bm25_candidate_k: int = 5,
    dense_candidate_k: int = 5,
    rrf_k: int = 60,
) -> ArticleDiscoveryEvaluationResult:
    if evaluation_k < 5:
        raise ValueError("文章反查评测需要 evaluation_k 至少为 5")
    build = _resolve_build(session, build_id)
    indexed_article_ids = _indexed_article_ids(build)
    confirmed = dataset.confirmed_questions
    mode_results: list[ArticleDiscoveryModeResult] = []
    for mode in (RetrievalMode.BM25, RetrievalMode.DENSE, RetrievalMode.HYBRID):
        question_results: list[ArticleDiscoveryQuestionResult] = []
        for question in confirmed:
            if question.expected_article_id not in indexed_article_ids:
                question_results.append(
                    ArticleDiscoveryQuestionResult(
                        question_id=question.id,
                        query=question.query,
                        expected_article_id=question.expected_article_id,
                        expected_title=question.expected_title,
                        status="expected_article_not_in_build",
                        expected_rank=None,
                        reciprocal_rank=None,
                        hit_at_1=None,
                        hit_at_3=None,
                        hit_at_5=None,
                        latency_ms=None,
                        candidates=[],
                    )
                )
                continue
            started = time.perf_counter()
            results = search_articles(
                session,
                question.query,
                evaluation_k,
                mode=mode,
                embedding_provider=(
                    embedding_provider if mode != RetrievalMode.BM25 else None
                ),
                build_id=build.id,
                bm25_candidate_k=bm25_candidate_k,
                dense_candidate_k=dense_candidate_k,
                rrf_k=rrf_k,
            )
            latency_ms = (time.perf_counter() - started) * 1000
            expected_rank = next(
                (
                    rank
                    for rank, result in enumerate(results, start=1)
                    if result.article_id == question.expected_article_id
                ),
                None,
            )
            question_results.append(
                ArticleDiscoveryQuestionResult(
                    question_id=question.id,
                    query=question.query,
                    expected_article_id=question.expected_article_id,
                    expected_title=question.expected_title,
                    status="evaluated",
                    expected_rank=expected_rank,
                    reciprocal_rank=1.0 / expected_rank if expected_rank is not None else 0.0,
                    hit_at_1=expected_rank == 1,
                    hit_at_3=expected_rank is not None and expected_rank <= 3,
                    hit_at_5=expected_rank is not None and expected_rank <= 5,
                    latency_ms=latency_ms,
                    candidates=[
                        ArticleDiscoveryCandidateResult(
                            rank=rank,
                            article_id=result.article_id,
                            title=result.title,
                            sources=result.sources,
                            rrf_score=result.rrf_score,
                            bm25_score=result.bm25_score,
                            bm25_rank=result.bm25_rank,
                            dense_score=result.dense_score,
                            dense_rank=result.dense_rank,
                        )
                        for rank, result in enumerate(results, start=1)
                    ],
                )
            )
        mode_results.append(
            ArticleDiscoveryModeResult(
                mode=mode,
                metrics=_mode_metrics(question_results),
                questions=question_results,
            )
        )
    created_at = datetime.now(UTC)
    code_commit, worktree_dirty = _git_state()
    run_id = stable_id(
        "article-discovery-evaluation",
        dataset.content_hash,
        build.id,
        created_at.isoformat(),
        length=32,
    )
    return ArticleDiscoveryEvaluationResult(
        run_id=run_id,
        created_at=created_at,
        dataset_path=str(dataset_path.resolve()),
        dataset_content_hash=dataset.content_hash,
        confirmed_question_count=len(confirmed),
        skipped_question_count=len(dataset.queries) - len(confirmed),
        build_id=build.id,
        corpus_snapshot=build.corpus_snapshot,
        profile_id=build.profile_id,
        code_commit=code_commit,
        worktree_dirty=worktree_dirty,
        evaluation_k=evaluation_k,
        bm25_candidate_k=bm25_candidate_k,
        dense_candidate_k=dense_candidate_k,
        rrf_k=rrf_k,
        modes=mode_results,
    )


def save_article_discovery_evaluation(
    result: ArticleDiscoveryEvaluationResult, output_dir: Path
) -> Path:
    resolved = output_dir.resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    final_path = resolved / f"{result.run_id}.json"
    payload = result.model_dump_json(indent=2)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=resolved,
            prefix=".article-discovery-",
            suffix=".tmp",
            delete=False,
        ) as stream:
            stream.write(payload)
            temporary_path = Path(stream.name)
        os.replace(temporary_path, final_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return final_path
