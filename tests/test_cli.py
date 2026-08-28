from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import func, select
from typer.testing import CliRunner

from xifeng_rag.cli import _latest_pending_version_ids, app
from xifeng_rag.domain import RetrievalCandidate, RetrievalScores, VersionStatus
from xifeng_rag.storage import ArticleRow, ArticleVersionRow, ChunkProposalRow, Database
from xifeng_rag.structured_generation import GenerationResult

runner = CliRunner()


def test_corpus_scan_reports_resolved_path_and_count_without_body(
    tmp_path: Path, monkeypatch
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    private_body = "这是不能出现在普通输出中的夹具正文"
    (corpus / "one.md").write_text(f"# 标题\n\n{private_body}", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        f"project_root: {tmp_path.as_posix()}\n"
        "corpus:\n"
        "  source_dir: corpus\n"
        "storage:\n"
        "  data_dir: data\n",
        encoding="utf-8",
    )

    def forbid_network(*args, **kwargs):
        raise AssertionError("corpus scan must not perform network requests")

    monkeypatch.setattr("urllib.request.urlopen", forbid_network)
    result = runner.invoke(app, ["--config", str(config), "corpus", "scan"])
    assert result.exit_code == 0
    assert str(corpus.resolve()) in result.output
    assert "发现文章：1" in result.output
    assert "排除 0" in result.output
    assert private_body not in result.output


def test_corpus_manage_starts_local_dashboard(tmp_path: Path, monkeypatch) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    config = tmp_path / "config.yaml"
    config.write_text(
        f"project_root: {tmp_path.as_posix()}\n"
        "corpus:\n"
        "  source_dir: corpus\n"
        "storage:\n"
        "  data_dir: data\n",
        encoding="utf-8",
    )
    called = {}

    def fake_dashboard(*args, **kwargs):
        called["args"] = args
        called["kwargs"] = kwargs

    monkeypatch.setattr("xifeng_rag.cli.run_corpus_dashboard", fake_dashboard)
    result = runner.invoke(
        app, ["--config", str(config), "corpus", "manage", "--no-open"]
    )
    assert result.exit_code == 0
    assert called["args"][1] == corpus.resolve()
    assert called["kwargs"] == {"open_browser": False}


def test_eval_curate_starts_local_workspace(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        f"project_root: {tmp_path.as_posix()}\nstorage:\n  data_dir: data\n",
        encoding="utf-8",
    )
    called = {}

    def fake_server(*args, **kwargs):
        called["args"] = args
        called["kwargs"] = kwargs

    monkeypatch.setattr("xifeng_rag.cli.run_evaluation_server", fake_server)
    result = runner.invoke(
        app, ["--config", str(config), "eval", "curate", "--no-open"]
    )
    assert result.exit_code == 0
    assert called["args"][1].default_model == "gpt-5.6-terra"
    assert called["kwargs"] == {"open_browser": False}


def test_search_without_index_gives_next_command(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        f"project_root: {tmp_path.as_posix()}\nstorage:\n  data_dir: data\n",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["--config", str(config), "search", "问题"])
    assert result.exit_code == 1
    assert "xifeng index build" in result.output


def test_search_reranker_requires_hybrid_mode(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        f"project_root: {tmp_path.as_posix()}\nstorage:\n  data_dir: data\n",
        encoding="utf-8",
    )
    result = runner.invoke(
        app, ["--config", str(config), "search", "问题", "--mode", "bm25", "--rerank"]
    )
    assert result.exit_code == 1
    assert "只支持 --mode hybrid" in result.output


def test_rerank_candidate_depth_option_is_validated_by_cli(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        f"project_root: {tmp_path.as_posix()}\nstorage:\n  data_dir: data\n",
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        [
            "--config",
            str(config),
            "search",
            "问题",
            "--rerank-candidate-k",
            "0",
        ],
    )
    assert result.exit_code != 0
    assert "0" in result.output
    result = runner.invoke(
        app,
        [
            "--config",
            str(config),
            "search",
            "问题",
            "--rerank-candidate-k",
            "10",
        ],
    )
    assert result.exit_code == 1
    assert "只能与 --rerank 一起使用" in result.output


def test_search_assemble_json_uses_local_ranked_candidates(
    tmp_path: Path, monkeypatch
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        f"project_root: {tmp_path.as_posix()}\nstorage:\n  data_dir: data\n",
        encoding="utf-8",
    )
    private_body = "只用于本地组装测试的自造证据"

    def fake_search(*args, **kwargs):
        del args, kwargs
        return [
            RetrievalCandidate(
                chunk_id="chunk-1",
                article_id="article-1",
                article_version="version-1",
                article_title="自造文章",
                original_text=private_body,
                paragraph_range=(1, 2),
                retrieval_scores=RetrievalScores(bm25=1.0, bm25_rank=1),
                source=["bm25"],
                topic_title="AI 话题",
                context_summary="AI 概括",
            )
        ]

    def forbid_network(*args, **kwargs):
        raise AssertionError("context assembly must not perform network requests")

    monkeypatch.setattr("xifeng_rag.cli.search", fake_search)
    monkeypatch.setattr("urllib.request.urlopen", forbid_network)
    result = runner.invoke(
        app,
        [
            "--config",
            str(config),
            "search",
            "测试问题",
            "--mode",
            "bm25",
            "--assemble",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "ready"
    assert payload["evidence_items"][0]["citation_id"] == "E1"
    assert payload["evidence_items"][0]["original_text"] == private_body
    assert "AI 话题" not in payload["rendered_context"]


def test_ask_without_evidence_does_not_require_key_or_network(
    tmp_path: Path, monkeypatch
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        f"project_root: {tmp_path.as_posix()}\nstorage:\n  data_dir: data\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr("xifeng_rag.cli.search", lambda *args, **kwargs: [])

    def forbid_network(*args, **kwargs):
        raise AssertionError("无证据时不得调用外部 Generator")

    monkeypatch.setattr("urllib.request.urlopen", forbid_network)
    result = runner.invoke(
        app,
        ["--config", str(config), "ask", "无法回答的问题", "--mode", "bm25"],
    )

    assert result.exit_code == 0
    assert "现有资料不足以回答" in result.stdout
    assert "本次检索未找到" in result.stdout
    assert "即将把" not in result.stderr


def test_ask_discloses_scope_and_returns_validated_json(
    tmp_path: Path, monkeypatch
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        f"project_root: {tmp_path.as_posix()}\n"
        "storage:\n  data_dir: data\n"
        "generator:\n  model: test-model\n",
        encoding="utf-8",
    )
    private_body = "只发送给 Generator 的自造证据"

    def fake_search(*args, **kwargs):
        del args, kwargs
        return [
            RetrievalCandidate(
                chunk_id="chunk-1",
                article_id="article-1",
                article_version="version-1",
                article_title="自造文章",
                original_text=private_body,
                paragraph_range=(1, 1),
                retrieval_scores=RetrievalScores(bm25=1.0, bm25_rank=1),
                source=["bm25"],
            )
        ]

    class FakeDeepSeekClient:
        def __init__(self, settings, api_key):
            assert settings.model == "test-model"
            assert api_key == "test-key"

        def generate(self, prompt, schema):
            assert private_body in prompt
            assert schema["type"] == "object"
            return GenerationResult(
                content=json.dumps(
                    {
                        "status": "answered",
                        "grounded_sections": [
                            {
                                "text": "资料给出了可验证结论。",
                                "statement_type": "source_statement",
                                "evidence_ids": ["E1"],
                            }
                        ],
                        "supplementary_analysis": None,
                        "limitations": [],
                    },
                    ensure_ascii=False,
                )
            )

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr("xifeng_rag.cli.search", fake_search)
    monkeypatch.setattr("xifeng_rag.cli.DeepSeekClient", FakeDeepSeekClient)
    result = runner.invoke(
        app,
        [
            "--config",
            str(config),
            "ask",
            "测试问题",
            "--mode",
            "bm25",
            "--supplement",
            "off",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["answer"]["status"] == "answered"
    assert payload["request_count"] == 1
    assert payload["provider_id"] == "deepseek"
    assert payload["model_id"] == "test-model"
    assert payload["sources"][0]["citation_id"] == "E1"
    assert "test-model" in result.stderr
    assert "1 块" in result.stderr
    assert "最大外部请求次数：2" in result.stderr
    assert private_body not in result.stderr


def test_chunk_proposal_selection_uses_only_latest_active_version(
    database: Database,
) -> None:
    created = datetime(2026, 8, 25, tzinfo=UTC)
    with database.session() as session:
        session.add(ArticleRow(id="article-1", source_path="a.md", title="示例"))
        session.add_all(
            [
                ArticleVersionRow(
                    id="old-version",
                    article_id="article-1",
                    source_hash="a" * 64,
                    canonical_hash="b" * 64,
                    cleaner_version="v1",
                    canonical_text="旧版",
                    status=VersionStatus.PENDING_CHUNKING.value,
                    active=True,
                    created_at=created,
                ),
                ArticleVersionRow(
                    id="new-version",
                    article_id="article-1",
                    source_hash="a" * 64,
                    canonical_hash="c" * 64,
                    cleaner_version="v2",
                    canonical_text="新版",
                    status=VersionStatus.PENDING_CHUNKING.value,
                    active=True,
                    created_at=created + timedelta(seconds=1),
                ),
            ]
        )
    with database.session() as session:
        assert _latest_pending_version_ids(session) == ["new-version"]
        newest = session.get(ArticleVersionRow, "new-version")
        assert newest is not None
        newest.status = VersionStatus.PENDING_REVIEW.value
    with database.session() as session:
        assert _latest_pending_version_ids(session) == ["new-version"]
        newest = session.get(ArticleVersionRow, "new-version")
        assert newest is not None
        newest.status = VersionStatus.APPROVED.value
    with database.session() as session:
        assert _latest_pending_version_ids(session) == []
        assert len(list(session.scalars(select(ArticleVersionRow)).all())) == 2


def test_external_chunk_retry_cancelled_before_key_or_network(tmp_path: Path, monkeypatch) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    private_body = "这是只用于测试且不能出现在命令输出中的正文。"
    (corpus / "one.md").write_text(f"# 标题\n\n{private_body}", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        f"project_root: {tmp_path.as_posix()}\n"
        "corpus:\n"
        "  source_dir: corpus\n"
        "storage:\n"
        "  data_dir: data\n"
        "chunking:\n"
        "  external_profiles:\n"
        "    remote:\n"
        "      base_url: https://example.invalid/v1\n"
        "      api_key_env: MISSING_TEST_API_KEY\n"
        "      default_model: large-model\n",
        encoding="utf-8",
    )
    scan = runner.invoke(app, ["--config", str(config), "corpus", "scan"])
    assert scan.exit_code == 0
    database = Database(tmp_path / "data" / "xifeng.sqlite3")
    with database.session() as session:
        article = session.scalar(select(ArticleRow))
        assert article is not None
        article_id = article.id

    def forbid_network(*args, **kwargs):
        raise AssertionError("取消确认后不得发送网络请求")

    monkeypatch.setattr("urllib.request.urlopen", forbid_network)
    result = runner.invoke(
        app,
        ["--config", str(config), "chunks", "retry", article_id, "--provider", "remote"],
        input="n\n",
    )
    assert result.exit_code == 0
    assert "完整清洗后正文" in result.output
    assert "已取消" in result.output
    assert private_body not in result.output
    with database.session() as session:
        assert session.scalar(select(func.count()).select_from(ChunkProposalRow)) == 0


def test_codex_chunk_retry_cancelled_before_starting_cli(tmp_path: Path, monkeypatch) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    private_body = "这是不能出现在 Codex 命令或普通输出中的正文。"
    (corpus / "one.md").write_text(f"# 标题\n\n{private_body}", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        f"project_root: {tmp_path.as_posix()}\n"
        "corpus:\n"
        "  source_dir: corpus\n"
        "storage:\n"
        "  data_dir: data\n",
        encoding="utf-8",
    )
    assert runner.invoke(app, ["--config", str(config), "corpus", "scan"]).exit_code == 0
    database = Database(tmp_path / "data" / "xifeng.sqlite3")
    with database.session() as session:
        article = session.scalar(select(ArticleRow))
        assert article is not None
        article_id = article.id

    class ForbiddenCodexClient:
        def __init__(self, *args, **kwargs):
            raise AssertionError("取消确认后不得启动 Codex CLI")

    monkeypatch.setattr("xifeng_rag.cli.CodexCliClient", ForbiddenCodexClient)
    result = runner.invoke(
        app,
        ["--config", str(config), "chunks", "retry", article_id, "--provider", "codex-cli"],
        input="n\n",
    )
    assert result.exit_code == 0
    assert "OpenAI Codex CLI" in result.output
    assert "训练使用取决于" in result.output
    assert "已取消" in result.output
    assert private_body not in result.output
    with database.session() as session:
        assert session.scalar(select(func.count()).select_from(ChunkProposalRow)) == 0


def test_chunks_review_starts_local_ui_without_approving(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        f"project_root: {tmp_path.as_posix()}\nstorage:\n  data_dir: data\n",
        encoding="utf-8",
    )
    database = Database(tmp_path / "data" / "xifeng.sqlite3")
    database.initialize()
    with database.session() as session:
        session.add(ArticleRow(id="article", source_path="test.md", title="测试文章"))
        session.add(
            ArticleVersionRow(
                id="version",
                article_id="article",
                source_hash="a" * 64,
                canonical_hash="b" * 64,
                cleaner_version="test",
                canonical_text="测试正文",
                status=VersionStatus.PENDING_REVIEW.value,
            )
        )
        session.add(
            ChunkProposalRow(
                id="proposal",
                version_id="version",
                status="pending_review",
                model_id="test-model",
                prompt_version="test-prompt",
                schema_version="test-schema",
                topics_json='{"topics": []}',
            )
        )

    called = {}

    def fake_server(*args, **kwargs):
        called["args"] = args
        called["kwargs"] = kwargs

    monkeypatch.setattr("xifeng_rag.cli.run_review_server", fake_server)
    result = runner.invoke(
        app,
        [
            "--config",
            str(config),
            "chunks",
            "review",
            "version",
            "--proposal",
            "proposal",
            "--no-open",
        ],
    )
    assert result.exit_code == 0
    assert called["args"][1:3] == ("version", "proposal")
    assert called["kwargs"] == {"open_browser": False}
    with database.session() as session:
        assert session.scalar(select(func.count()).select_from(ChunkProposalRow)) == 1
