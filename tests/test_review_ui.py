from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from sqlalchemy import func, select

from xifeng_rag.article_overviews import JOINT_PROMPT_VERSION
from xifeng_rag.article_search import (
    ArticleBuildOutcome,
    ArticleIndexBuildError,
    selected_article_documents,
)
from xifeng_rag.domain import OverviewStatus, ProposalStatus, VersionStatus
from xifeng_rag.indexing import build_index
from xifeng_rag.review_ui import (
    ApprovalIndexError,
    ReviewApplication,
    create_review_server,
    load_review_document,
    save_review_draft,
)
from xifeng_rag.storage import (
    ApprovedManifestRow,
    ArticleIndexBuildRow,
    ArticleOverviewRow,
    ArticleRow,
    ArticleVersionRow,
    ChunkProposalRow,
    Database,
    IndexBuildRow,
    ParagraphRow,
    TopicChunkRow,
)


def _review_fixture(database: Database, *, joint: bool = False) -> tuple[str, str]:
    version_id = "version_test"
    proposal_id = "proposal_test"
    topics = {
        "topics": [
            {
                "start_paragraph": 1,
                "end_paragraph": 2,
                "title": "开头",
                "summary": "开头概括",
                "keywords": ["开头"],
                "include_in_index": True,
            },
            {
                "start_paragraph": 3,
                "end_paragraph": 4,
                "title": "结尾",
                "summary": "结尾概括",
                "keywords": ["结尾"],
                "include_in_index": True,
            },
        ]
    }
    with database.session() as session:
        session.add(ArticleRow(id="article_test", source_path="test.md", title="自造短文"))
        session.add(
            ArticleVersionRow(
                id=version_id,
                article_id="article_test",
                source_hash="a" * 64,
                canonical_hash="b" * 64,
                cleaner_version="test",
                canonical_text="一\n\n二\n\n三\n\n四",
                status=VersionStatus.PENDING_REVIEW.value,
            )
        )
        session.add_all(
            [
                ParagraphRow(
                    version_id=version_id,
                    number=number,
                    text=f"自造单元 {number}",
                    source_start=number,
                    source_end=number + 1,
                )
                for number in range(1, 5)
            ]
        )
        session.add(
            ChunkProposalRow(
                id=proposal_id,
                version_id=version_id,
                status=ProposalStatus.PENDING_REVIEW.value,
                provider_id="test-provider",
                model_id="test-model",
                prompt_version="test-prompt",
                schema_version="test-schema",
                topics_json=json.dumps(topics, ensure_ascii=False),
            )
        )
        if joint:
            session.flush()
            session.add(
                ArticleOverviewRow(
                    id="overview_test",
                    version_id=version_id,
                    proposal_id=proposal_id,
                    manifest_id=None,
                    article_summary="模型生成的文章整体概括。",
                    status=OverviewStatus.PENDING_REVIEW.value,
                    provider_id="test-provider",
                    model_id="test-model",
                    prompt_version=JOINT_PROMPT_VERSION,
                )
            )
    return version_id, proposal_id


def test_load_and_save_review_draft_without_approval(
    database: Database, tmp_path: Path
) -> None:
    version_id, proposal_id = _review_fixture(database)
    document = load_review_document(database, version_id, proposal_id)
    assert document.article_title == "自造短文"
    assert [item["number"] for item in document.paragraphs] == [1, 2, 3, 4]

    original_export = tmp_path / "reviews" / f"{version_id}.{proposal_id}.review.json"
    original_export.parent.mkdir()
    original_export.write_text("模型原始提案", encoding="utf-8")
    topics = [dict(item) for item in document.topics]
    topics[0]["end_paragraph"] = 1
    topics[1]["start_paragraph"] = 2
    topics[1]["include_in_index"] = False
    destination = save_review_draft(database, document, topics, original_export.parent)

    assert destination.name == f"{version_id}.{proposal_id}.draft.json"
    assert original_export.read_text(encoding="utf-8") == "模型原始提案"
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["topics"][0]["end_paragraph"] == 1
    assert payload["topics"][1]["include_in_index"] is False
    with database.session() as session:
        proposal = session.get(ChunkProposalRow, proposal_id)
        assert proposal is not None
        assert proposal.status == ProposalStatus.PENDING_REVIEW.value
        assert session.scalar(select(func.count()).select_from(ApprovedManifestRow)) == 0
        assert session.scalar(select(func.count()).select_from(TopicChunkRow)) == 0


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (lambda topics: topics[0].update(end_paragraph=3), "第 2 块应从段落 4 开始"),
        (lambda topics: topics[1].update(start_paragraph=4), "第 2 块应从段落 3 开始"),
        (
            lambda topics: [topic.update(include_in_index=False) for topic in topics],
            "至少需要保留一个",
        ),
    ],
)
def test_save_rejects_invalid_review(
    database: Database, tmp_path: Path, mutate, expected: str
) -> None:
    version_id, proposal_id = _review_fixture(database)
    document = load_review_document(database, version_id, proposal_id)
    topics = [dict(item) for item in document.topics]
    mutate(topics)
    with pytest.raises(ValueError, match=expected):
        save_review_draft(database, document, topics, tmp_path)
    assert list(tmp_path.glob("*.draft.json")) == []


def test_review_http_requires_token_and_saves_draft(
    database: Database, tmp_path: Path
) -> None:
    version_id, proposal_id = _review_fixture(database)
    application = ReviewApplication(
        database,
        version_id,
        proposal_id,
        tmp_path / "reviews",
        tmp_path / "indexes",
        token="test-secret",
    )
    server, url = create_review_server(application)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = url.split("/?", maxsplit=1)[0]
    try:
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(f"{base}/api/review", timeout=2)
        assert exc_info.value.code == 403

        request = urllib.request.Request(
            f"{base}/api/review", headers={"X-Review-Token": "test-secret"}
        )
        with urllib.request.urlopen(request, timeout=2) as response:
            payload = json.loads(response.read())
            assert response.headers["Content-Security-Policy"].startswith("default-src 'none'")
        assert payload["article_title"] == "自造短文"

        premature_shutdown = urllib.request.Request(
            f"{base}/api/shutdown",
            data=b"{}",
            headers={
                "Content-Type": "application/json",
                "X-Review-Token": "test-secret",
            },
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(premature_shutdown, timeout=2)
        assert exc_info.value.code == 409

        save_request = urllib.request.Request(
            f"{base}/api/draft",
            data=json.dumps({"topics": payload["topics"]}).encode(),
            headers={
                "Content-Type": "application/json",
                "X-Review-Token": "test-secret",
            },
            method="POST",
        )
        with urllib.request.urlopen(save_request, timeout=2) as response:
            result = json.loads(response.read())
        assert result["message"] == "草案已保存"
        assert Path(result["path"]).is_file()

        approval_request = urllib.request.Request(
            f"{base}/api/approve-and-index",
            data=json.dumps({"confirmed": True}).encode(),
            headers={
                "Content-Type": "application/json",
                "X-Review-Token": "test-secret",
            },
            method="POST",
        )
        with urllib.request.urlopen(approval_request, timeout=5) as response:
            approval = json.loads(response.read())
        assert approval["message"] == "清单已批准，活动索引已更新"
        assert approval["chunk_count"] == 2
        assert approval["included_count"] == 2
        assert approval["excluded_count"] == 0
        with database.session() as session:
            assert session.scalar(select(func.count()).select_from(ApprovedManifestRow)) == 1
            assert session.scalar(select(func.count()).select_from(TopicChunkRow)) == 2
            active = session.scalar(select(IndexBuildRow).where(IndexBuildRow.active.is_(True)))
            assert active is not None
            assert active.id == approval["index_id"]

        shutdown_request = urllib.request.Request(
            f"{base}/api/shutdown",
            data=b"{}",
            headers={
                "Content-Type": "application/json",
                "X-Review-Token": "test-secret",
            },
            method="POST",
        )
        with urllib.request.urlopen(shutdown_request, timeout=2) as response:
            shutdown = json.loads(response.read())
        assert shutdown["message"] == "审核服务已关闭，可以安全关闭此标签页"
        thread.join(timeout=2)
        assert not thread.is_alive()
        with database.session() as session:
            assert session.scalar(select(func.count()).select_from(ApprovedManifestRow)) == 1
            assert session.scalar(select(func.count()).select_from(TopicChunkRow)) == 2
            active = session.scalar(select(IndexBuildRow).where(IndexBuildRow.active.is_(True)))
            assert active is not None and active.id == approval["index_id"]
    finally:
        if thread.is_alive():
            server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_approval_requires_confirmation_and_saved_draft(
    database: Database, tmp_path: Path
) -> None:
    version_id, proposal_id = _review_fixture(database)
    application = ReviewApplication(
        database, version_id, proposal_id, tmp_path / "reviews", tmp_path / "indexes"
    )
    with pytest.raises(ValueError, match="明确确认"):
        application.approve_and_build({"confirmed": False})
    with pytest.raises(ValueError, match="先在当前审核会话中成功保存草案"):
        application.approve_and_build({"confirmed": True})
    with database.session() as session:
        assert session.scalar(select(func.count()).select_from(ApprovedManifestRow)) == 0


def test_approval_uses_injected_index_builder(database: Database, tmp_path: Path) -> None:
    version_id, proposal_id = _review_fixture(database)
    calls: list[Path] = []

    def custom_builder(session, index_dir):
        calls.append(index_dir)
        return build_index(session, index_dir)

    application = ReviewApplication(
        database,
        version_id,
        proposal_id,
        tmp_path / "reviews",
        tmp_path / "indexes",
        index_builder=custom_builder,
    )
    application.save({"topics": application.document.topics})

    result = application.approve_and_build({"confirmed": True})

    assert calls == [tmp_path / "indexes"]
    assert result["index_id"]


def test_review_rejects_pending_proposal_for_already_indexed_version(
    database: Database,
) -> None:
    version_id, proposal_id = _review_fixture(database)
    with database.session() as session:
        version = session.get(ArticleVersionRow, version_id)
        assert version is not None
        version.status = VersionStatus.INDEXED.value
    with pytest.raises(ValueError, match="文章版本状态不可审核"):
        load_review_document(database, version_id, proposal_id)


def test_index_failure_keeps_old_index_and_retry_reuses_manifest(
    database: Database, tmp_path: Path, monkeypatch
) -> None:
    version_id, proposal_id = _review_fixture(database)
    application = ReviewApplication(
        database, version_id, proposal_id, tmp_path / "reviews", tmp_path / "indexes"
    )
    application.save({"topics": application.document.topics})
    with database.session() as session:
        session.add(
            IndexBuildRow(
                id="old-index",
                profile_id="test-profile",
                corpus_snapshot="old",
                path=str(tmp_path / "old-index"),
                status="active",
                active=True,
            )
        )

    from xifeng_rag import review_ui

    real_build_index = review_ui.build_index

    def fail_build(*args, **kwargs):
        del args, kwargs
        raise review_ui.IndexBuildError(
            "failed-index", "failed-snapshot", tmp_path / "failed-index", "TestFailure", 1
        )

    monkeypatch.setattr(review_ui, "build_index", fail_build)
    with pytest.raises(ApprovalIndexError) as exc_info:
        application.approve_and_build({"confirmed": True})
    manifest_id = exc_info.value.manifest_id
    with database.session() as session:
        assert session.scalar(select(func.count()).select_from(ApprovedManifestRow)) == 1
        old = session.get(IndexBuildRow, "old-index")
        assert old is not None and old.active

    monkeypatch.setattr(review_ui, "build_index", real_build_index)
    result = application.approve_and_build({"confirmed": True})
    assert result["manifest_id"] == manifest_id
    with database.session() as session:
        assert session.scalar(select(func.count()).select_from(ApprovedManifestRow)) == 1
        active = session.scalar(select(IndexBuildRow).where(IndexBuildRow.active.is_(True)))
        assert active is not None
        assert active.id == result["index_id"]


def test_review_page_has_no_remote_resources() -> None:
    static = Path(__file__).parents[1] / "src" / "xifeng_rag" / "review_static"
    combined = "\n".join(
        (static / name).read_text(encoding="utf-8")
        for name in ("index.html", "app.js", "style.css")
    )
    assert "http://" not in combined
    assert "https://" not in combined
    assert "定位此块" in combined
    assert "scrollIntoView" in combined
    assert "locatedTopicIndex" in combined
    assert "批准并更新索引" in combined
    assert "window.confirm" in combined
    assert "/api/approve-and-index" in combined
    assert "关闭审核服务" in combined
    assert "completion-panel" in combined
    assert "/api/shutdown" in combined
    assert "文章整体概括" in combined
    assert "article_summary" in combined


def test_joint_review_saves_and_approves_overview_with_both_indexes(
    database: Database, tmp_path: Path
) -> None:
    version_id, proposal_id = _review_fixture(database, joint=True)
    document = load_review_document(database, version_id, proposal_id)
    assert document.article_summary == "模型生成的文章整体概括。"
    assert document.overview_id == "overview_test"

    article_calls: list[Path] = []

    def article_builder(session, index_dir):
        article_calls.append(index_dir)
        documents = selected_article_documents(session)
        assert len(documents) == 1
        assert documents[0].article_summary == "人工修订后的整体概括。"
        return ArticleBuildOutcome("article-index-test", 1, index_dir, 3)

    application = ReviewApplication(
        database,
        version_id,
        proposal_id,
        tmp_path / "reviews",
        tmp_path / "indexes",
        article_index_dir=tmp_path / "article-indexes",
        article_index_builder=article_builder,
    )
    application.save(
        {
            "article_summary": "人工修订后的整体概括。",
            "topics": application.document.topics,
        }
    )
    result = application.approve_and_build({"confirmed": True})

    assert result["message"] == "分块与概括已批准，两个活动索引已更新"
    assert result["overview_id"] == "overview_test"
    assert result["article_index_id"] == "article-index-test"
    assert article_calls == [tmp_path / "article-indexes"]
    with database.session() as session:
        overview = session.get(ArticleOverviewRow, "overview_test")
        assert overview is not None
        assert overview.status == OverviewStatus.APPROVED.value
        assert overview.article_summary == "人工修订后的整体概括。"
        assert overview.manifest_id == result["manifest_id"]


def test_joint_review_rejects_invalid_article_summary(
    database: Database, tmp_path: Path
) -> None:
    version_id, proposal_id = _review_fixture(database, joint=True)
    document = load_review_document(database, version_id, proposal_id)
    with pytest.raises(ValueError, match="article_summary"):
        save_review_draft(
            database,
            document,
            document.topics,
            tmp_path,
            article_summary_value="超" * 201,
        )
    assert list(tmp_path.glob("*.draft.json")) == []


def test_joint_article_index_failure_keeps_old_build_and_retries(
    database: Database, tmp_path: Path
) -> None:
    version_id, proposal_id = _review_fixture(database, joint=True)

    def fail_article_builder(session, index_dir):
        del session
        raise ArticleIndexBuildError(
            "failed-article-index",
            "article-profile",
            "snapshot",
            index_dir / "failed-article-index",
            "TestFailure",
            2,
        )

    application = ReviewApplication(
        database,
        version_id,
        proposal_id,
        tmp_path / "reviews",
        tmp_path / "indexes",
        article_index_dir=tmp_path / "article-indexes",
        article_index_builder=fail_article_builder,
    )
    application.save(
        {
            "article_summary": application.document.article_summary,
            "topics": application.document.topics,
        }
    )
    with database.session() as session:
        session.add(
            ArticleIndexBuildRow(
                id="old-article-index",
                profile_id="old-profile",
                corpus_snapshot="old",
                path=str(tmp_path / "old-article-index"),
                status="active",
                active=True,
            )
        )

    with pytest.raises(ApprovalIndexError) as exc_info:
        application.approve_and_build({"confirmed": True})
    assert exc_info.value.stage == "article"
    manifest_id = exc_info.value.manifest_id
    with database.session() as session:
        old = session.get(ArticleIndexBuildRow, "old-article-index")
        assert old is not None and old.active
        overview = session.get(ArticleOverviewRow, "overview_test")
        assert overview is not None
        assert overview.status == OverviewStatus.APPROVED.value
        assert overview.manifest_id == manifest_id

    def succeed_article_builder(session, index_dir):
        assert len(selected_article_documents(session)) == 1
        return ArticleBuildOutcome("new-article-index", 1, index_dir, 1)

    application.article_index_builder = succeed_article_builder
    result = application.approve_and_build({"confirmed": True})
    assert result["manifest_id"] == manifest_id
    assert result["article_index_id"] == "new-article-index"
    with database.session() as session:
        assert session.scalar(select(func.count()).select_from(ApprovedManifestRow)) == 1
        assert session.scalar(select(func.count()).select_from(ArticleOverviewRow)) == 1
