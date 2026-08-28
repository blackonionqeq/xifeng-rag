from __future__ import annotations

import hashlib
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from sqlalchemy import func, inspect, select

from xifeng_rag.config import CodexCliSettings
from xifeng_rag.domain import RetrievalCandidate, RetrievalScores
from xifeng_rag.evaluation import run_frozen_evaluation
from xifeng_rag.evaluation_curation import (
    AnswerPoint,
    DatasetPurpose,
    EvidenceRange,
    QuestionDraft,
    ReviewedAnswer,
    apply_ai_proposal,
    create_dataset,
    freeze_dataset,
    generate_ai_proposal,
    question_document,
    save_question,
    save_reviewed_answer,
)
from xifeng_rag.evaluation_ui import (
    EvaluationCurationApplication,
    create_evaluation_server,
)
from xifeng_rag.storage import (
    ApprovedManifestRow,
    ArticleRow,
    ArticleVersionRow,
    ChunkProposalRow,
    Database,
    EvaluationAIProposalRow,
    EvaluationDatasetRow,
    EvaluationEvidenceRow,
    EvaluationQuestionRow,
    EvaluationRunRow,
    EvaluationSnapshotArticleRow,
    IndexBuildRow,
    ParagraphRow,
    TopicChunkRow,
)
from xifeng_rag.structured_generation import GenerationResult


def _indexed_fixture(database: Database, tmp_path: Path) -> str:
    version_id = "version_eval_test"
    with database.session() as session:
        session.add(ArticleRow(id="article_eval_test", source_path="eval.md", title="自造评测文章"))
        session.add(
            ArticleVersionRow(
                id=version_id,
                article_id="article_eval_test",
                source_hash="a" * 64,
                canonical_hash="b" * 64,
                cleaner_version="test",
                canonical_text="第一单元\n第二单元\n第三单元\n第四单元",
                status="indexed",
            )
        )
        session.add_all(
            [
                ParagraphRow(
                    version_id=version_id,
                    number=number,
                    text=f"自造原子单元 {number}",
                    source_start=number,
                    source_end=number + 1,
                )
                for number in range(1, 5)
            ]
        )
        session.flush()
        session.add(
            ChunkProposalRow(
                id="proposal_eval_test",
                version_id=version_id,
                status="approved",
                provider_id="fixture",
                model_id="fixture",
                prompt_version="fixture",
                schema_version="fixture",
            )
        )
        session.flush()
        session.add(
            ApprovedManifestRow(
                id="manifest_eval_test",
                proposal_id="proposal_eval_test",
                version_id=version_id,
                topics_json="[]",
            )
        )
        session.flush()
        session.add_all(
            [
                TopicChunkRow(
                    id=f"chunk_eval_{index}",
                    manifest_id="manifest_eval_test",
                    version_id=version_id,
                    article_id="article_eval_test",
                    start_paragraph=start,
                    end_paragraph=end,
                    topic_title=f"话题 {index}",
                    context_summary=f"概括 {index}",
                    keywords_json="[]",
                    original_text=f"自造块 {index}",
                )
                for index, (start, end) in enumerate(((1, 2), (3, 4)), start=1)
            ]
        )
        session.add(
            IndexBuildRow(
                id="index_eval_test",
                profile_id="fixture-profile",
                corpus_snapshot=hashlib.sha256(b"chunk_eval_1\nchunk_eval_2").hexdigest(),
                path=str(tmp_path / "index"),
                status="active",
                active=True,
            )
        )
    return version_id


def _draft_question(database: Database, tmp_path: Path) -> tuple[str, str]:
    version_id = _indexed_fixture(database, tmp_path)
    with database.session() as session:
        dataset = create_dataset(session, "试运行", "v0.1", DatasetPurpose.PILOT)
        question = save_question(
            session,
            dataset.id,
            QuestionDraft(query="作者怎样解释第二单元？", version_id=version_id),
        )
        return dataset.id, question.id


def _review_value(reviewed: bool = True) -> ReviewedAnswer:
    return ReviewedAnswer(
        reference_answer="作者给出了一个自造解释。",
        answer_points=[
            AnswerPoint(
                claim_group="point-1",
                claim="作者解释了第二单元",
                evidence_ranges=[EvidenceRange(start_paragraph=2, end_paragraph=3)],
                uncertainty="",
            )
        ],
        reviewed=reviewed,
    )


def test_create_review_and_freeze_stable_evidence(database: Database, tmp_path: Path) -> None:
    dataset_id, question_id = _draft_question(database, tmp_path)
    with database.session() as session:
        save_reviewed_answer(session, question_id, _review_value())
        outcome = freeze_dataset(session, dataset_id)
        assert outcome.article_count == 1
        assert outcome.question_count == 1
        assert len(outcome.content_hash) == 64

    with database.session() as session:
        dataset = session.get(EvaluationDatasetRow, dataset_id)
        assert dataset is not None and dataset.status == "frozen"
        assert session.scalar(select(func.count()).select_from(EvaluationEvidenceRow)) == 1
        assert (
            session.scalar(select(func.count()).select_from(EvaluationSnapshotArticleRow)) == 1
        )
        with pytest.raises(ValueError, match="冻结评测集不可修改"):
            save_reviewed_answer(session, question_id, _review_value())


class FakeCodex:
    def __init__(self) -> None:
        self.prompt = ""

    def preflight(self) -> None:
        pass

    def generate(self, prompt: str, schema: dict[str, object]) -> GenerationResult:
        self.prompt = prompt
        assert schema["type"] == "object"
        return GenerationResult(
            content=json.dumps(
                {
                    "candidate_answer": "候选答案",
                    "answer_points": [
                        {
                            "claim_group": "point-1",
                            "claim": "候选要点",
                            "evidence_ranges": [
                                {"start_paragraph": 2, "end_paragraph": 2}
                            ],
                            "uncertainty": "",
                        }
                    ],
                },
                ensure_ascii=False,
            )
        )


def test_ai_candidate_is_structured_and_requires_human_review(
    database: Database, tmp_path: Path
) -> None:
    _, question_id = _draft_question(database, tmp_path)
    generator = FakeCodex()
    with database.session() as session:
        proposal = generate_ai_proposal(session, question_id, generator, "gpt-test")
        proposal_id = proposal.id
    assert "作者怎样解释第二单元" in generator.prompt
    assert "自造原子单元 4" in generator.prompt
    assert "BM25" not in generator.prompt

    with database.session() as session:
        question = session.get(EvaluationQuestionRow, question_id)
        assert question is not None and question.reference_answer is None
        apply_ai_proposal(session, proposal_id, accept=True)
    with database.session() as session:
        question = session.get(EvaluationQuestionRow, question_id)
        proposal = session.get(EvaluationAIProposalRow, proposal_id)
        assert question is not None and question.reference_answer == "候选答案"
        assert question.review_status == "draft"
        assert proposal is not None and proposal.status == "accepted"


def test_disclosure_is_complete_single_use_and_starts_async_candidate(
    database: Database, tmp_path: Path
) -> None:
    _, question_id = _draft_question(database, tmp_path)
    fake = FakeCodex()
    application = EvaluationCurationApplication(
        database,
        CodexCliSettings(default_model="gpt-5.6-terra", max_retries=1),
        token="secret",
        generator_factory=lambda: fake,
    )
    disclosure = application.disclosure(question_id)
    assert disclosure["article_version"] == "version_eval_test"
    character_count = disclosure["character_count"]
    assert isinstance(character_count, int) and character_count > 0
    assert disclosure["unit_count"] == 4
    assert disclosure["max_requests"] == 1
    assert disclosure["model"] == "gpt-5.6-terra"
    confirmation = str(disclosure["confirmation"])
    application.confirm_ai(confirmation)
    for _ in range(100):
        if application.job()["status"] != "running":
            break
        time.sleep(0.01)
    assert application.job()["status"] == "succeeded"
    with pytest.raises(ValueError, match="无效、已使用或已过期"):
        application.confirm_ai(confirmation)


def test_fresh_database_has_evaluation_curation_schema(database: Database) -> None:
    table_names = set(inspect(database.engine).get_table_names())
    assert {
        "evaluation_datasets",
        "evaluation_snapshot_articles",
        "evaluation_questions",
        "evaluation_evidence",
        "evaluation_ai_proposals",
    } <= table_names
    run_columns = {item["name"] for item in inspect(database.engine).get_columns("evaluation_runs")}
    assert {
        "dataset_content_hash",
        "corpus_snapshot",
        "profile_id",
        "k",
        "retrieval_mode",
        "bm25_candidate_k",
        "dense_candidate_k",
        "rrf_k",
    } <= run_columns


def test_question_document_does_not_duplicate_canonical_article(
    database: Database, tmp_path: Path
) -> None:
    _, question_id = _draft_question(database, tmp_path)
    with database.session() as session:
        document = question_document(session, question_id)
    article = document["article"]
    paragraphs = document["paragraphs"]
    assert isinstance(article, dict) and "canonical_text" not in article
    assert isinstance(paragraphs, list) and isinstance(paragraphs[0], dict)
    assert paragraphs[0]["text"] == "自造原子单元 1"


def test_frozen_evaluation_maps_ranges_and_checks_snapshot(
    database: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_id, question_id = _draft_question(database, tmp_path)
    with database.session() as session:
        save_reviewed_answer(session, question_id, _review_value())
        freeze_dataset(session, dataset_id)

    def fake_search(session, query: str, top_k: int, build_id: str | None = None, **kwargs):
        del session, query, top_k, build_id, kwargs
        return [
            RetrievalCandidate(
                chunk_id="chunk_eval_1",
                article_id="article_eval_test",
                article_version="version_eval_test",
                article_title="自造评测文章",
                original_text="不写入运行详情的自造原文",
                paragraph_range=(1, 2),
                retrieval_scores=RetrievalScores(bm25=1.0),
            )
        ]

    monkeypatch.setattr("xifeng_rag.evaluation.search", fake_search)
    with database.session() as session:
        outcome = run_frozen_evaluation(session, dataset_id, k=1)
        metrics = outcome["metrics"]
        assert isinstance(metrics, dict)
        assert metrics["recall_at_k"] == 0.5
        assert metrics["evidence_coverage_at_k"] == 0.5
    with database.session() as session:
        run = session.scalar(select(EvaluationRunRow))
        assert run is not None and run.dataset_content_hash
        assert "自造原文" not in run.details_json
        build = session.get(IndexBuildRow, "index_eval_test")
        assert build is not None
        build.corpus_snapshot = "d" * 64
    with database.session() as session, pytest.raises(ValueError, match="语料快照"):
        run_frozen_evaluation(session, dataset_id, k=1)


def test_evaluation_http_requires_token_and_does_not_send_without_confirmation(
    database: Database, tmp_path: Path
) -> None:
    _, question_id = _draft_question(database, tmp_path)
    fake = FakeCodex()
    application = EvaluationCurationApplication(
        database,
        CodexCliSettings(default_model="gpt-test"),
        token="test-evaluation-token",
        generator_factory=lambda: fake,
    )
    server, url = create_evaluation_server(application)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = url.split("/?", maxsplit=1)[0]
    try:
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(f"{base}/api/state", timeout=2)
        assert exc_info.value.code == 403

        disclosure_request = urllib.request.Request(
            f"{base}/api/questions/{question_id}/ai-disclosure",
            data=b"{}",
            headers={
                "Content-Type": "application/json",
                "X-Evaluation-Token": "test-evaluation-token",
            },
            method="POST",
        )
        with urllib.request.urlopen(disclosure_request, timeout=2) as response:
            disclosure = json.loads(response.read())
        assert response.headers["Content-Security-Policy"].startswith("default-src 'none'")
        assert fake.prompt == ""

        confirmation_request = urllib.request.Request(
            f"{base}/api/ai-confirm",
            data=json.dumps({"confirmation": disclosure["confirmation"]}).encode(),
            headers={
                "Content-Type": "application/json",
                "X-Evaluation-Token": "test-evaluation-token",
            },
            method="POST",
        )
        with urllib.request.urlopen(confirmation_request, timeout=2) as response:
            assert json.loads(response.read())["status"] == "running"
        for _ in range(100):
            if application.job()["status"] != "running":
                break
            time.sleep(0.01)
        assert application.job()["status"] == "succeeded"
        assert "自造原子单元" in fake.prompt
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_evaluation_page_has_no_remote_resources() -> None:
    static = Path(__file__).parents[1] / "src" / "xifeng_rag" / "evaluation_static"
    combined = "\n".join(
        (static / name).read_text(encoding="utf-8")
        for name in ("index.html", "app.js", "style.css")
    )
    assert "http://" not in combined
    assert "https://" not in combined
    assert "AI 辅助判断" in combined
    assert "window.confirm" in combined
    assert "/ai-disclosure" in combined
    assert "/api/ai-confirm" in combined
    assert '.review-pane button' in combined
