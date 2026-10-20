from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from typer.testing import CliRunner

from xifeng_rag.article_search import ArticleSearchResult
from xifeng_rag.cli import app
from xifeng_rag.demo_ui import DemoApplication, QuestionHistoryStore, create_demo_server
from xifeng_rag.domain import (
    ContextAssemblyConfigSnapshot,
    RetrievalCandidate,
    RetrievalScores,
)
from xifeng_rag.generation import (
    AnswerGenerationResult,
    AnswerStatus,
    GeneratedAnswer,
    GroundedSection,
    StatementType,
)

runner = CliRunner()


def _article_result() -> ArticleSearchResult:
    return ArticleSearchResult(
        article_id="article-1",
        version_id="version-1",
        title="自造文章",
        published_at="2026-10-20",
        article_summary="模型生成的安全概括。",
        matched_topics=["示例主题"],
        matched_subtopics=["局部案例"],
        sources=["overview:bm25"],
        rrf_score=0.1,
        bm25_score=1.0,
        bm25_rank=1,
        dense_score=None,
        dense_rank=None,
        strategy="disagreement_cascade",
        activated_layers=("overview",),
    )


def _candidate() -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id="chunk-1",
        article_id="article-1",
        article_version="version-1",
        article_title="自造文章",
        original_text="只用于演示测试的自造原文证据。",
        paragraph_range=(2, 3),
        retrieval_scores=RetrievalScores(bm25=1.0, bm25_rank=1),
    )


class FakeGenerator:
    def generate(self, question, context, supplement):
        assert question == "测试问题"
        assert context.evidence_items[0].citation_id == "E1"
        assert supplement.value == "off"
        return AnswerGenerationResult(
            answer=GeneratedAnswer(
                status=AnswerStatus.ANSWERED,
                grounded_sections=[
                    GroundedSection(
                        text="根据自造资料可以回答。",
                        statement_type=StatementType.SOURCE_STATEMENT,
                        evidence_ids=["E1"],
                    )
                ],
            ),
            request_count=1,
            provider_id="fake",
            model_id="fake-model",
            generation_metadata={"usage": {"total_tokens": 10}},
        )


def _application(
    *,
    candidates: list[RetrievalCandidate] | None = None,
    clock=lambda: 100.0,
    history_path: Path | None = None,
) -> tuple[DemoApplication, list[str]]:
    generator_calls: list[str] = []

    def generator_factory():
        generator_calls.append("created")
        return FakeGenerator()

    application = DemoApplication(
        article_searcher=lambda query, top_k: ([_article_result()], ["overview"]),
        evidence_searcher=lambda question, top_k: (
            [_candidate()] if candidates is None else candidates
        ),
        generator_factory=generator_factory,
        assembly_config=ContextAssemblyConfigSnapshot(
            max_evidence_items=5,
            max_total_chars=10000,
            include_ai_metadata=False,
        ),
        provider_name="Fake",
        provider_url="https://example.invalid",
        model_id="fake-model",
        question_history=QuestionHistoryStore(history_path) if history_path is not None else None,
        token="test-token",
        clock=clock,
    )
    return application, generator_calls


def test_article_discovery_returns_safe_presentation_fields() -> None:
    application, _ = _application()

    payload = application.search_articles({"query": "模糊主题", "top_k": 5})

    assert payload["strategy"] == "disagreement_cascade"
    assert payload["activated_layers"] == ["overview"]
    results = payload["results"]
    assert isinstance(results, list)
    assert results[0]["title"] == "自造文章"
    assert results[0]["article_summary"] == "模型生成的安全概括。"


def test_prepare_is_local_and_confirm_returns_cited_original_text() -> None:
    application, generator_calls = _application()

    disclosure = application.prepare_answer({"question": "测试问题"})

    assert generator_calls == []
    assert disclosure["evidence_count"] == 1
    assert disclosure["original_text_chars"] == len(_candidate().original_text)
    assert disclosure["max_external_requests"] == 2
    evidence = disclosure["evidence"]
    assert isinstance(evidence, list)
    assert evidence[0]["paragraph_range"] == [2, 3]

    result = application.answer({"prepared_id": disclosure["prepared_id"]})

    assert generator_calls == ["created"]
    answer = result["answer"]
    sources = result["sources"]
    assert isinstance(answer, dict) and answer["status"] == "answered"
    assert isinstance(sources, list)
    assert result["request_count"] == 1
    assert sources[0]["original_text"] == _candidate().original_text
    with pytest.raises(ValueError, match="已使用"):
        application.answer({"prepared_id": disclosure["prepared_id"]})


def test_no_evidence_refuses_without_creating_generator() -> None:
    application, generator_calls = _application(candidates=[])
    disclosure = application.prepare_answer({"question": "测试问题"})

    assert disclosure["will_call_external_generator"] is False
    assert disclosure["max_external_requests"] == 0
    result = application.answer({"prepared_id": disclosure["prepared_id"]})

    assert generator_calls == []
    answer = result["answer"]
    assert isinstance(answer, dict) and answer["status"] == "insufficient_evidence"
    assert result["request_count"] == 0


def test_prepared_result_expires() -> None:
    now = [100.0]
    application, generator_calls = _application(clock=lambda: now[0])
    disclosure = application.prepare_answer({"question": "测试问题"})
    now[0] += 601

    with pytest.raises(ValueError, match="已过期"):
        application.answer({"prepared_id": disclosure["prepared_id"]})
    assert generator_calls == []


def test_question_history_persists_deduplicates_and_can_be_cleared(tmp_path: Path) -> None:
    history_path = tmp_path / "demo-question-history.json"
    application, _ = _application(history_path=history_path)

    application.prepare_answer({"question": "第一个问题"})
    application.prepare_answer({"question": "第二个问题"})
    application.prepare_answer({"question": "第一个问题"})

    history = application.list_questions()["questions"]
    assert isinstance(history, list)
    assert [item["question"] for item in history] == ["第一个问题", "第二个问题"]
    assert json.loads(history_path.read_text(encoding="utf-8")) == history

    remaining = application.delete_question({"question": "第一个问题"})["questions"]
    assert isinstance(remaining, list)
    assert [item["question"] for item in remaining] == ["第二个问题"]
    assert application.clear_questions() == {"questions": []}
    assert json.loads(history_path.read_text(encoding="utf-8")) == []


def test_question_history_keeps_only_latest_fifty(tmp_path: Path) -> None:
    store = QuestionHistoryStore(tmp_path / "history.json")

    for index in range(51):
        store.record(f"问题 {index}")

    history = store.list()
    assert len(history) == 50
    assert history[0]["question"] == "问题 50"
    assert history[-1]["question"] == "问题 1"


def test_demo_http_server_serves_static_and_protects_api() -> None:
    application, _ = _application(candidates=[])
    server, url = create_demo_server(application)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = url.split("/?", 1)[0]
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            page = response.read().decode()
            assert "西风文章研究台" in page
            assert "候选文章" in page
            assert "DeepSeek 生成" in page
            assert "检查证据范围" in page

        unauthorized = urllib.request.Request(
            f"{base_url}/api/answers/prepare",
            data=json.dumps({"question": "测试问题"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(unauthorized, timeout=5)
        assert caught.value.code == 403

        authorized = urllib.request.Request(
            f"{base_url}/api/answers/prepare",
            data=json.dumps({"question": "测试问题"}).encode(),
            headers={"Content-Type": "application/json", "X-Demo-Token": "test-token"},
            method="POST",
        )
        with urllib.request.urlopen(authorized, timeout=5) as response:
            payload = json.load(response)
        assert payload["evidence_count"] == 0

        history_request = urllib.request.Request(
            f"{base_url}/api/questions/history",
            data=b"{}",
            headers={"Content-Type": "application/json", "X-Demo-Token": "test-token"},
            method="POST",
        )
        with urllib.request.urlopen(history_request, timeout=5) as response:
            assert json.load(response) == {"questions": []}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_demo_command_wires_local_services(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        f"project_root: {tmp_path.as_posix()}\nstorage:\n  data_dir: data\n",
        encoding="utf-8",
    )
    suite = tmp_path / "data" / "article_indexes" / "latest-layered-suite.json"
    suite.parent.mkdir(parents=True)
    suite.write_text("{}", encoding="utf-8")

    class FakeProvider:
        closed = False

        def close(self):
            self.closed = True

    provider = FakeProvider()
    captured: dict[str, object] = {}
    monkeypatch.setattr("xifeng_rag.cli._embedding_provider", lambda settings: provider)
    monkeypatch.setattr("xifeng_rag.cli.load_layered_suite", lambda path: {})

    def fake_run(application, *, open_browser):
        captured["application"] = application
        captured["open_browser"] = open_browser

    monkeypatch.setattr("xifeng_rag.cli.run_demo_server", fake_run)
    result = runner.invoke(
        app,
        ["--config", str(config), "demo", "--no-open"],
    )

    assert result.exit_code == 0
    application = captured["application"]
    assert isinstance(application, DemoApplication)
    assert application.question_history is not None
    assert application.question_history.path == tmp_path / "data" / "demo-question-history.json"
    assert captured["open_browser"] is False
    assert provider.closed is True
