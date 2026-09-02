from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import pytest
from sqlalchemy import func, select

from xifeng_rag.article_search import ArticleBuildOutcome
from xifeng_rag.corpus import scan_corpus
from xifeng_rag.corpus_ui import (
    CorpusDashboardApplication,
    create_corpus_dashboard_server,
)
from xifeng_rag.domain import ProposalStatus, SubtopicSetStatus, VersionStatus
from xifeng_rag.storage import (
    ApprovedManifestRow,
    ArticleRow,
    ArticleVersionRow,
    ChunkProposalRow,
    Database,
    ParagraphRow,
    SubtopicSetRow,
    TopicChunkRow,
)


def _dashboard_fixture(database: Database, tmp_path: Path) -> tuple[Path, str, str]:
    corpus = tmp_path / "saved_articles"
    corpus.mkdir()
    (corpus / "article.md").write_text("# 自造文章\n\n第一段。\n\n第二段。", encoding="utf-8")
    (tmp_path / "index.json").write_text(
        json.dumps(
            {
                "sn:dashboard": {
                    "md_file": "saved_articles/article.md",
                    "captured_at": "2026-08-20 10:00:00",
                    "chinese_char_count": 20,
                }
            }
        ),
        encoding="utf-8",
    )
    with database.session() as session:
        scan_corpus(session, corpus)
        first = session.scalar(
            select(ArticleVersionRow).where(ArticleVersionRow.article_id == "sn:dashboard")
        )
        assert first is not None
        first.status = VersionStatus.INDEXED.value
        proposal = ChunkProposalRow(
            id="proposal_indexed",
            version_id=first.id,
            status=ProposalStatus.APPROVED.value,
            provider_id="test",
            model_id="test",
            prompt_version="test",
            schema_version="test",
            topics_json='{"topics": []}',
        )
        session.add(proposal)
        session.flush()
        manifest = ApprovedManifestRow(
            id="manifest_indexed",
            proposal_id=proposal.id,
            version_id=first.id,
            topics_json=json.dumps(
                {
                    "topics": [
                        {
                            "start_paragraph": 1,
                            "end_paragraph": 2,
                            "title": "核心论述",
                            "summary": "自造概括",
                            "keywords": ["自造"],
                            "include_in_index": True,
                        },
                        {
                            "start_paragraph": 3,
                            "end_paragraph": 3,
                            "title": "保留段落",
                            "summary": "不进入索引但仍属于批准清单",
                            "keywords": ["保留"],
                            "include_in_index": False,
                        },
                    ]
                },
                ensure_ascii=False,
            ),
        )
        session.add(manifest)
        session.flush()
        session.add(
            TopicChunkRow(
                id="chunk_indexed",
                manifest_id=manifest.id,
                version_id=first.id,
                article_id="sn:dashboard",
                start_paragraph=1,
                end_paragraph=3,
                topic_title="全文",
                context_summary="自造概括",
                keywords_json='["自造"]',
                original_text="自造正文",
            )
        )
        session.add(
            ChunkProposalRow(
                id="proposal_old_pending",
                version_id=first.id,
                status=ProposalStatus.PENDING_REVIEW.value,
                provider_id="test",
                model_id="test",
                prompt_version="test",
                schema_version="test",
                topics_json='{"topics": []}',
            )
        )
        latest = ArticleVersionRow(
            id="version_latest",
            article_id="sn:dashboard",
            source_hash="c" * 64,
            canonical_hash="d" * 64,
            cleaner_version="test",
            canonical_text="新版本一。\n\n新版本二。",
            status=VersionStatus.PENDING_REVIEW.value,
            created_at=first.created_at + timedelta(seconds=1),
        )
        session.add(latest)
        session.flush()
        session.add_all(
            [
                ParagraphRow(
                    version_id=latest.id,
                    number=number,
                    text=f"新版本 {number}",
                    source_start=number,
                    source_end=number + 1,
                )
                for number in range(1, 3)
            ]
        )
        session.add(
            ChunkProposalRow(
                id="proposal_pending",
                version_id=latest.id,
                status=ProposalStatus.PENDING_REVIEW.value,
                provider_id="codex-cli",
                model_id="test-model",
                prompt_version="test-prompt",
                schema_version="test-schema",
                topics_json=json.dumps(
                    {
                        "topics": [
                            {
                                "start_paragraph": 1,
                                "end_paragraph": 2,
                                "title": "新版本",
                                "summary": "自造摘要",
                                "keywords": ["自造"],
                                "include_in_index": True,
                            }
                        ]
                    }
                ),
            )
        )
        indexed_version_id = first.id
    return corpus, indexed_version_id, "version_latest"


def test_dashboard_aggregates_latest_and_indexed_versions(
    database: Database, tmp_path: Path
) -> None:
    corpus, indexed_version_id, latest_version_id = _dashboard_fixture(database, tmp_path)
    application = CorpusDashboardApplication(
        database, corpus, (), tmp_path / "reviews", tmp_path / "indexes"
    )
    payload = application.articles()
    assert payload["stats"] == {
        "total": 1,
        "active": 1,
        "indexed": 1,
        "pending": 1,
        "needs_cleanup_review": 0,
        "missing": 0,
        "coverage_percent": 100.0,
        "approved_overviews": 0,
        "pending_overviews": 0,
    }
    article = cast(list[dict[str, Any]], payload["articles"])[0]
    assert article["latest_version_id"] == latest_version_id
    assert article["indexed_version_id"] == indexed_version_id
    assert article["has_pending_update"] is True
    assert article["indexed_chunk_count"] == 1
    assert article["original_view"] == {
        "version_id": indexed_version_id,
        "character_count": len("# 自造文章\n\n第一段。\n\n第二段。"),
    }
    assert article["chunk_view"] == {
        "version_id": indexed_version_id,
        "manifest_id": "manifest_indexed",
        "topic_count": 2,
    }
    assert article["captured_at"] == "2026-08-20T10:00:00"
    assert article["published_at"] is None
    assert article["overview"] is None
    assert article["pending_proposals"][0]["proposal_id"] == "proposal_pending"


def test_dashboard_original_text_prefers_indexed_then_falls_back_to_latest(
    database: Database, tmp_path: Path
) -> None:
    corpus, indexed_version_id, latest_version_id = _dashboard_fixture(database, tmp_path)
    application = CorpusDashboardApplication(
        database, corpus, (), tmp_path / "reviews", tmp_path / "indexes"
    )

    indexed_payload = application.original_text_details("sn:dashboard")

    assert indexed_payload["version_id"] == indexed_version_id
    assert indexed_payload["canonical_text"] == "# 自造文章\n\n第一段。\n\n第二段。"
    assert indexed_payload["character_count"] == len(indexed_payload["canonical_text"])

    with database.session() as session:
        indexed = session.get(ArticleVersionRow, indexed_version_id)
        assert indexed is not None
        indexed.status = VersionStatus.APPROVED.value

    latest_payload = application.original_text_details("sn:dashboard")

    assert latest_payload["version_id"] == latest_version_id
    assert latest_payload["canonical_text"] == "新版本一。\n\n新版本二。"


def test_dashboard_chunk_details_are_read_only_and_use_only_approved_subtopics(
    database: Database, tmp_path: Path
) -> None:
    corpus, indexed_version_id, _ = _dashboard_fixture(database, tmp_path)
    with database.session() as session:
        session.add_all(
            [
                SubtopicSetRow(
                    id="subtopics-approved",
                    version_id=indexed_version_id,
                    manifest_id="manifest_indexed",
                    status=SubtopicSetStatus.APPROVED.value,
                    provider_id="test",
                    model_id="test",
                    prompt_version="test",
                    schema_version="test",
                    subtopics_json=json.dumps(
                        {
                            "topics": [
                                {
                                    "parent_start_unit": 1,
                                    "parent_end_unit": 2,
                                    "parent_title": "核心论述",
                                    "subtopics": [
                                        {
                                            "title": "已批准案例",
                                            "start_unit": 2,
                                            "end_unit": 2,
                                            "summary": "可见的自造细话题",
                                        }
                                    ],
                                }
                            ]
                        },
                        ensure_ascii=False,
                    ),
                ),
                SubtopicSetRow(
                    id="subtopics-pending",
                    version_id=indexed_version_id,
                    manifest_id="manifest_indexed",
                    status=SubtopicSetStatus.PENDING_REVIEW.value,
                    provider_id="test",
                    model_id="test",
                    prompt_version="test",
                    schema_version="test",
                    subtopics_json=json.dumps(
                        {
                            "topics": [
                                {
                                    "parent_start_unit": 1,
                                    "parent_end_unit": 2,
                                    "parent_title": "核心论述",
                                    "subtopics": [
                                        {
                                            "title": "待审不可见",
                                            "start_unit": 1,
                                            "end_unit": 1,
                                            "summary": "不应返回",
                                        }
                                    ],
                                }
                            ]
                        },
                        ensure_ascii=False,
                    ),
                ),
            ]
        )
    application = CorpusDashboardApplication(
        database, corpus, (), tmp_path / "reviews", tmp_path / "indexes"
    )

    payload = application.chunk_details("sn:dashboard")

    assert payload["version_id"] == indexed_version_id
    assert payload["topic_count"] == 2
    assert payload["subtopic_count"] == 1
    topics = cast(list[dict[str, Any]], payload["topics"])
    assert topics[0]["subtopics"] == [
        {
            "title": "已批准案例",
            "start_unit": 2,
            "end_unit": 2,
            "summary": "可见的自造细话题",
        }
    ]
    assert topics[1]["include_in_index"] is False
    assert topics[1]["subtopics"] == []
    assert "待审不可见" not in json.dumps(payload, ensure_ascii=False)


def test_dashboard_exposes_approved_chunks_before_index_build(
    database: Database, tmp_path: Path
) -> None:
    corpus, indexed_version_id, _ = _dashboard_fixture(database, tmp_path)
    with database.session() as session:
        version = session.get(ArticleVersionRow, indexed_version_id)
        assert version is not None
        version.status = VersionStatus.APPROVED.value
    application = CorpusDashboardApplication(
        database, corpus, (), tmp_path / "reviews", tmp_path / "indexes"
    )

    payload = application.articles()
    article = cast(list[dict[str, Any]], payload["articles"])[0]

    assert article["is_indexed"] is False
    assert article["chunk_view"] == {
        "version_id": indexed_version_id,
        "manifest_id": "manifest_indexed",
        "topic_count": 2,
    }


def test_dashboard_chunk_details_http_requires_token_and_returns_current_manifest(
    database: Database, tmp_path: Path
) -> None:
    corpus, indexed_version_id, _ = _dashboard_fixture(database, tmp_path)
    application = CorpusDashboardApplication(
        database,
        corpus,
        (),
        tmp_path / "reviews",
        tmp_path / "indexes",
        token="dashboard-secret",
    )
    server, url = create_corpus_dashboard_server(application)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = url.split("/?", maxsplit=1)[0]
    endpoint = f"{base}/api/chunks?article_id=sn%3Adashboard"
    original_endpoint = f"{base}/api/original-text?article_id=sn%3Adashboard"
    try:
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(endpoint, timeout=2)
        assert exc_info.value.code == 403

        request = urllib.request.Request(endpoint, headers={"X-Corpus-Token": "dashboard-secret"})
        with urllib.request.urlopen(request, timeout=2) as response:
            payload = json.loads(response.read())
        assert payload["version_id"] == indexed_version_id
        assert payload["topic_count"] == 2

        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(original_endpoint, timeout=2)
        assert exc_info.value.code == 403

        request = urllib.request.Request(
            original_endpoint, headers={"X-Corpus-Token": "dashboard-secret"}
        )
        with urllib.request.urlopen(request, timeout=2) as response:
            original_payload = json.loads(response.read())
        assert original_payload["version_id"] == indexed_version_id
        assert original_payload["canonical_text"] == "# 自造文章\n\n第一段。\n\n第二段。"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_dashboard_starts_only_current_pending_review(
    database: Database, tmp_path: Path, monkeypatch
) -> None:
    corpus, indexed_version_id, latest_version_id = _dashboard_fixture(database, tmp_path)
    captured = {}

    class FakeReviewApplication:
        def __init__(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

    class FakeReviewServer:
        def serve_forever(self):
            return None

        def shutdown(self):
            return None

        def server_close(self):
            return None

    monkeypatch.setattr("xifeng_rag.corpus_ui.ReviewApplication", FakeReviewApplication)
    monkeypatch.setattr(
        "xifeng_rag.corpus_ui.create_review_server",
        lambda application: (FakeReviewServer(), "http://127.0.0.1:12345/?token=test"),
    )

    def article_builder(session, path):
        del session
        return ArticleBuildOutcome("article-index-test", 1, path, 1)

    application = CorpusDashboardApplication(
        database,
        corpus,
        (),
        tmp_path / "reviews",
        tmp_path / "indexes",
        article_index_dir=tmp_path / "article-indexes",
        article_index_builder=article_builder,
    )
    with pytest.raises(ValueError, match="文章版本已不处于待审核状态"):
        application.start_review(
            {"version_id": indexed_version_id, "proposal_id": "proposal_old_pending"}
        )
    result = application.start_review(
        {"version_id": latest_version_id, "proposal_id": "proposal_pending"}
    )
    assert str(result["url"]).startswith("http://127.0.0.1:")
    assert captured["kwargs"]["article_index_dir"] == tmp_path / "article-indexes"
    assert captured["kwargs"]["article_index_builder"] is article_builder
    application.shutdown_children()


def test_dashboard_http_requires_token_scans_and_shuts_down(
    database: Database, tmp_path: Path
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "new.md").write_text("# 新文章\n\n自造正文。", encoding="utf-8")
    application = CorpusDashboardApplication(
        database,
        corpus,
        (),
        tmp_path / "reviews",
        tmp_path / "indexes",
        token="dashboard-secret",
    )
    server, url = create_corpus_dashboard_server(application)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = url.split("/?", maxsplit=1)[0]
    try:
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(f"{base}/api/articles", timeout=2)
        assert exc_info.value.code == 403

        scan_request = urllib.request.Request(
            f"{base}/api/scan",
            data=b"{}",
            headers={
                "Content-Type": "application/json",
                "X-Corpus-Token": "dashboard-secret",
            },
            method="POST",
        )
        with urllib.request.urlopen(scan_request, timeout=3) as response:
            scanned = json.loads(response.read())
        assert scanned["added"] == 1
        assert scanned["discovered"] == 1
        assert scanned["skipped_modified"] == 0

        with database.session() as session:
            initial_versions = session.scalar(select(func.count()).select_from(ArticleVersionRow))
        assert initial_versions is not None
        (corpus / "new.md").write_text("# 修改后的标题\n\n修改后的自造正文。", encoding="utf-8")
        with urllib.request.urlopen(scan_request, timeout=3) as response:
            skipped = json.loads(response.read())
        assert skipped["modified"] == 0
        assert skipped["skipped_modified"] == 1
        with database.session() as session:
            assert (
                session.scalar(select(func.count()).select_from(ArticleVersionRow))
                == initial_versions
            )
            article = session.scalar(select(ArticleRow))
            assert article is not None
            assert article.title == "新文章"

        include_modified_request = urllib.request.Request(
            f"{base}/api/scan",
            data=json.dumps({"include_modified": True}).encode(),
            headers={
                "Content-Type": "application/json",
                "X-Corpus-Token": "dashboard-secret",
            },
            method="POST",
        )
        with urllib.request.urlopen(include_modified_request, timeout=3) as response:
            included = json.loads(response.read())
        assert included["modified"] == 1
        assert included["skipped_modified"] == 0
        with database.session() as session:
            assert (
                session.scalar(select(func.count()).select_from(ArticleVersionRow))
                == initial_versions + 1
            )

        invalid_scan_request = urllib.request.Request(
            f"{base}/api/scan",
            data=json.dumps({"include_modified": "yes"}).encode(),
            headers={
                "Content-Type": "application/json",
                "X-Corpus-Token": "dashboard-secret",
            },
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(invalid_scan_request, timeout=2)
        assert exc_info.value.code == 400

        articles_request = urllib.request.Request(
            f"{base}/api/articles", headers={"X-Corpus-Token": "dashboard-secret"}
        )
        with urllib.request.urlopen(articles_request, timeout=2) as response:
            payload = json.loads(response.read())
            assert response.headers["Content-Security-Policy"].startswith("default-src 'none'")
        assert payload["stats"]["active"] == 1

        before = None
        with database.session() as session:
            before = session.scalar(select(func.count()).select_from(ArticleRow))
        shutdown_request = urllib.request.Request(
            f"{base}/api/shutdown",
            data=b"{}",
            headers={
                "Content-Type": "application/json",
                "X-Corpus-Token": "dashboard-secret",
            },
            method="POST",
        )
        with urllib.request.urlopen(shutdown_request, timeout=2) as response:
            shutdown = json.loads(response.read())
        assert "可以安全关闭" in shutdown["message"]
        thread.join(timeout=2)
        assert not thread.is_alive()
        with database.session() as session:
            assert session.scalar(select(func.count()).select_from(ArticleRow)) == before
    finally:
        if thread.is_alive():
            server.shutdown()
        application.shutdown_children()
        server.server_close()
        thread.join(timeout=2)


def test_dashboard_page_has_no_remote_resources() -> None:
    static = Path(__file__).parents[1] / "src" / "xifeng_rag" / "corpus_static"
    combined = "\n".join(
        (static / name).read_text(encoding="utf-8")
        for name in ("index.html", "app.js", "style.css")
    )
    assert "http://" not in combined
    assert "https://" not in combined
    assert "扫描语料更新" in combined
    assert "同时导入已修改文章" in combined
    assert "include_modified" in combined
    assert "/api/scan" in combined
    assert "/api/reviews" in combined
    assert "/api/chunks" in combined
    assert "/api/original-text" in combined
    assert "X-Corpus-Token" in combined
    assert 'id="subtopic-dialog"' in combined
    assert 'id="chunk-dialog"' in combined
    assert 'id="original-dialog"' in combined
    assert "查看分块" in combined
    assert "查看原文" in combined
    assert "暂无已批准细话题" in combined
    assert "请核对外发范围并确认" in combined
    assert "正在生成细话题" in combined
    assert "window.confirm" not in combined
