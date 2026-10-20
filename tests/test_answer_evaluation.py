from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select
from typer.testing import CliRunner

from xifeng_rag.answer_evaluation import (
    PreparedAnswerBatch,
    PreparedAnswerItem,
    prepare_answer_batch,
    run_prepared_answer_batch,
)
from xifeng_rag.answering import prepare_answer_context
from xifeng_rag.cli import app
from xifeng_rag.domain import (
    ContextAssemblyConfigSnapshot,
    RetrievalCandidate,
    RetrievalMode,
    RetrievalScores,
)
from xifeng_rag.generation import (
    AnswerGenerationError,
    AnswerGenerationResult,
    AnswerStatus,
    GeneratedAnswer,
    GeneratorProviderError,
    GroundedSection,
    StatementType,
)
from xifeng_rag.storage import (
    AnswerEvaluationItemRow,
    AnswerEvaluationRunRow,
    ArticleRow,
    ArticleVersionRow,
    Database,
    EvaluationDatasetRow,
    EvaluationEvidenceRow,
    EvaluationQuestionRow,
    IndexBuildRow,
)

runner = CliRunner()


def _frozen_fixture(database: Database, *, include_answerable: bool = True) -> None:
    now = datetime(2026, 10, 20, tzinfo=UTC)
    with database.session() as session:
        session.add(ArticleRow(id="article", source_path="fixture.md", title="自造文章"))
        session.add(
            ArticleVersionRow(
                id="version",
                article_id="article",
                source_hash="a" * 64,
                canonical_hash="b" * 64,
                cleaner_version="test",
                canonical_text="自造正文",
                status="indexed",
            )
        )
        session.add(
            IndexBuildRow(
                id="build",
                profile_id="profile",
                corpus_snapshot="c" * 64,
                path="unused",
                status="active",
                active=True,
            )
        )
        session.add(
            EvaluationDatasetRow(
                id="dataset",
                name="正式集",
                version="v1",
                purpose="benchmark",
                status="frozen",
                index_corpus_snapshot="c" * 64,
                content_hash="d" * 64,
                frozen_at=now,
            )
        )
        ordinal = 1
        if include_answerable:
            session.add(
                EvaluationQuestionRow(
                    id="answerable",
                    dataset_id="dataset",
                    ordinal=ordinal,
                    query="可回答问题",
                    split="dev",
                    difficulty="medium",
                    origin="new_dev",
                    scope="single_article",
                    article_id="article",
                    version_id="version",
                    answerable=True,
                    reference_answer="参考答案",
                    review_status="reviewed",
                )
            )
            session.flush()
            session.add(
                EvaluationEvidenceRow(
                    id="gold",
                    question_id="answerable",
                    ordinal=1,
                    article_id="article",
                    version_id="version",
                    start_paragraph=2,
                    end_paragraph=3,
                    claim_group="point-1",
                    claim_text="答案要点",
                )
            )
            ordinal += 1
        session.add(
            EvaluationQuestionRow(
                id="unanswerable",
                dataset_id="dataset",
                ordinal=ordinal,
                query="不可回答问题",
                split="dev",
                difficulty="hard",
                origin="new_dev",
                scope="corpus",
                article_id="article",
                version_id="version",
                answerable=False,
                reference_answer="应拒答",
                review_status="reviewed",
            )
        )


def _candidate(private_text: str = "不能持久化的自造证据正文") -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id="chunk",
        article_id="article",
        article_version="version",
        article_title="自造文章",
        original_text=private_text,
        paragraph_range=(2, 3),
        retrieval_scores=RetrievalScores(bm25=1.0, bm25_rank=1),
    )


class FakeGenerator:
    def generate(self, question, context, supplement):
        del context, supplement
        if question == "不可回答问题":
            answer = GeneratedAnswer(
                status=AnswerStatus.INSUFFICIENT_EVIDENCE,
                limitations=["现有证据不支持问题所要求的信息。"],
            )
        else:
            answer = GeneratedAnswer(
                status=AnswerStatus.ANSWERED,
                grounded_sections=[
                    GroundedSection(
                        text="基于证据的回答。",
                        statement_type=StatementType.SOURCE_STATEMENT,
                        evidence_ids=["E1"],
                    )
                ],
            )
        return AnswerGenerationResult(
            answer=answer,
            request_count=1,
            provider_id="fake",
            model_id="fake-model",
            generation_metadata={"usage": {"total_tokens": 12}},
        )


def test_answer_evaluation_scores_refusal_citations_and_preserves_privacy(
    database: Database,
) -> None:
    _frozen_fixture(database)
    private_text = "不能持久化的自造证据正文"
    with database.session() as session:
        batch = prepare_answer_batch(
            session,
            "dataset",
            split="dev",
            mode=RetrievalMode.BM25,
            k=5,
            build_id=None,
            retriever=lambda session, query: [_candidate(private_text)],
            assembly_config=ContextAssemblyConfigSnapshot(
                max_evidence_items=5,
                max_total_chars=10000,
                include_ai_metadata=False,
            ),
        )
        outcome = run_prepared_answer_batch(
            session,
            batch,
            generator=FakeGenerator(),
            provider_id="fake",
            model_id="fake-model",
        )

    assert outcome["status"] == "completed"
    metrics = outcome["metrics"]
    assert isinstance(metrics, dict)
    assert metrics["correct_refusal_rate"] == 1.0
    assert metrics["over_refusal_rate"] == 0.0
    assert metrics["gold_evidence_citation_coverage"] == 1.0
    assert metrics["answer_point_citation_coverage"] == 1.0
    assert metrics["total_external_request_count"] == 2
    with database.session() as session:
        run = session.scalar(select(AnswerEvaluationRunRow))
        items = list(
            session.scalars(
                select(AnswerEvaluationItemRow).order_by(AnswerEvaluationItemRow.ordinal)
            ).all()
        )
    assert run is not None and len(items) == 2
    persisted = run.metrics_json + "".join(
        (item.answer_json or "")
        + item.citations_json
        + item.metrics_json
        + item.usage_json
        for item in items
    )
    assert private_text not in persisted
    answerable_metrics = json.loads(items[0].metrics_json)
    assert answerable_metrics["citation_id_valid_rate"] == 1.0
    assert json.loads(items[0].failure_labels_json) == []
    assert json.loads(items[0].citations_json)[0]["paragraph_range"] == [2, 3]


def test_no_evidence_refuses_locally_without_generator(database: Database) -> None:
    _frozen_fixture(database, include_answerable=False)

    class ForbiddenGenerator:
        def generate(self, *args, **kwargs):
            raise AssertionError("无证据时不得调用 Generator")

    with database.session() as session:
        batch = prepare_answer_batch(
            session,
            "dataset",
            split="dev",
            mode=RetrievalMode.BM25,
            k=5,
            build_id=None,
            retriever=lambda session, query: [],
            assembly_config=ContextAssemblyConfigSnapshot(
                max_evidence_items=5,
                max_total_chars=10000,
                include_ai_metadata=False,
            ),
        )
        outcome = run_prepared_answer_batch(
            session,
            batch,
            generator=ForbiddenGenerator(),
            provider_id="fake",
            model_id="fake-model",
        )
    assert outcome["status"] == "completed"
    metrics = outcome["metrics"]
    assert isinstance(metrics, dict)
    assert metrics["correct_refusal_rate"] == 1.0
    assert metrics["total_external_request_count"] == 0


def test_provider_failure_stops_remaining_external_calls(database: Database) -> None:
    _frozen_fixture(database)

    class FailingGenerator:
        calls = 0

        def generate(self, *args, **kwargs):
            self.calls += 1
            raise GeneratorProviderError("provider unavailable")

    generator = FailingGenerator()
    with database.session() as session:
        batch = prepare_answer_batch(
            session,
            "dataset",
            split="dev",
            mode=RetrievalMode.BM25,
            k=5,
            build_id=None,
            retriever=lambda session, query: [_candidate()],
            assembly_config=ContextAssemblyConfigSnapshot(
                max_evidence_items=5,
                max_total_chars=10000,
                include_ai_metadata=False,
            ),
        )
        outcome = run_prepared_answer_batch(
            session,
            batch,
            generator=generator,
            provider_id="fake",
            model_id="fake-model",
        )
    assert generator.calls == 1
    assert outcome["status"] == "failed"
    metrics = outcome["metrics"]
    assert isinstance(metrics, dict)
    assert metrics["processed_question_count"] == 1
    with database.session() as session:
        items = list(session.scalars(select(AnswerEvaluationItemRow)).all())
    assert len(items) == 1
    assert json.loads(items[0].failure_labels_json) == ["generation_error"]


def test_single_generation_error_is_recorded_and_next_question_continues(
    database: Database,
) -> None:
    _frozen_fixture(database)

    class RecoveringGenerator(FakeGenerator):
        calls = 0

        def generate(self, question, context, supplement):
            self.calls += 1
            if self.calls == 1:
                raise AnswerGenerationError("invalid structured output")
            return super().generate(question, context, supplement)

    generator = RecoveringGenerator()
    with database.session() as session:
        batch = prepare_answer_batch(
            session,
            "dataset",
            split="dev",
            mode=RetrievalMode.BM25,
            k=5,
            build_id=None,
            retriever=lambda session, query: [_candidate()],
            assembly_config=ContextAssemblyConfigSnapshot(
                max_evidence_items=5,
                max_total_chars=10000,
                include_ai_metadata=False,
            ),
        )
        outcome = run_prepared_answer_batch(
            session,
            batch,
            generator=generator,
            provider_id="fake",
            model_id="fake-model",
        )
    assert generator.calls == 2
    assert outcome["status"] == "completed_with_errors"
    with database.session() as session:
        items = list(
            session.scalars(
                select(AnswerEvaluationItemRow).order_by(AnswerEvaluationItemRow.ordinal)
            ).all()
        )
    assert [item.status for item in items] == ["error", "completed"]


def test_answer_run_decline_does_not_create_generator_or_call_network(
    tmp_path: Path, monkeypatch
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        f"project_root: {tmp_path.as_posix()}\nstorage:\n  data_dir: data\n",
        encoding="utf-8",
    )
    context = prepare_answer_context(
        [_candidate()],
        ContextAssemblyConfigSnapshot(
            max_evidence_items=5,
            max_total_chars=10000,
            include_ai_metadata=False,
        ),
    )
    batch = PreparedAnswerBatch(
        dataset_id="dataset",
        dataset_name="正式集",
        dataset_version="v1",
        dataset_content_hash="d" * 64,
        corpus_snapshot="c" * 64,
        build_id="build",
        split="dev",
        mode=RetrievalMode.BM25,
        k=5,
        items=(
            PreparedAnswerItem(
                question_id="question",
                ordinal=1,
                query="问题",
                answerable=True,
                difficulty="medium",
                origin="new_dev",
                context=context,
                retrieved_ranges=(("version", 2, 3),),
                gold_ranges=(),
                retrieval_latency_ms=1.0,
                assembly_latency_ms=1.0,
            ),
        ),
    )
    monkeypatch.setattr("xifeng_rag.cli.prepare_answer_batch", lambda *args, **kwargs: batch)

    class ForbiddenClient:
        def __init__(self, *args, **kwargs):
            raise AssertionError("取消后不得创建外部客户端")

    monkeypatch.setattr("xifeng_rag.cli.DeepSeekClient", ForbiddenClient)
    result = runner.invoke(
        app,
        [
            "--config",
            str(config),
            "eval",
            "answer-run",
            "dataset",
            "--split",
            "dev",
            "--mode",
            "bm25",
        ],
        input="n\n",
    )
    assert result.exit_code == 0
    assert "最大外部请求次数：2" in result.stderr
    assert "已取消；未调用外部 Generator" in result.stdout
