from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import select

from xifeng_rag.chunking import GenerationResult, approve_manifest, export_review, propose_chunks
from xifeng_rag.corpus import scan_corpus
from xifeng_rag.indexing import IndexBuildError, build_index, record_build_failure, search
from xifeng_rag.storage import ArticleVersionRow, Database, IndexBuildRow


class TwoTopicGenerator:
    def generate(self, prompt: str, schema: dict[str, object]) -> GenerationResult:
        return GenerationResult(
            content=json.dumps(
                {
                    "topics": [
                        {
                            "end_unit": 2,
                            "title": "苹果",
                            "summary": "苹果种植",
                            "keywords": ["苹果", "果园"],
                            "content_role": "substantive",
                        },
                        {
                            "end_unit": 4,
                            "title": "航天",
                            "summary": "火箭发射",
                            "keywords": ["火箭", "卫星"],
                            "content_role": "substantive",
                        },
                    ]
                },
                ensure_ascii=False,
            )
        )


def _approve_fixture(database: Database, tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "fixture.md").write_text(
        "# 示例\n\n果园种植苹果。\n\n红苹果成熟丰收。\n\n火箭发射卫星。",
        encoding="utf-8",
    )
    with database.session() as session:
        scan_corpus(session, corpus)
    with database.session() as session:
        version = session.scalar(select(ArticleVersionRow))
        assert version is not None
        propose_chunks(session, version, TwoTopicGenerator(), "fake", 0)
        version_id = version.id
    with database.session() as session:
        review, _ = export_review(session, version_id, tmp_path / "reviews")
        approve_manifest(session, review)


def test_build_publish_and_search_returns_original_citation(
    database: Database, tmp_path: Path
) -> None:
    _approve_fixture(database, tmp_path)
    index_root = tmp_path / "indexes"
    with database.session() as session:
        outcome = build_index(session, index_root)
        assert outcome.chunk_count == 2
        assert outcome.path.is_dir()
    with database.session() as session:
        candidates = search(session, "卫星火箭", 2)
        assert candidates[0].topic_title == "航天"
        assert "火箭发射卫星" in candidates[0].original_text
        assert candidates[0].paragraph_range == (3, 4)
        active = session.scalar(select(IndexBuildRow).where(IndexBuildRow.active.is_(True)))
        assert active is not None


def test_build_rejects_empty_index(database: Database, tmp_path: Path) -> None:
    with database.session() as session:
        try:
            build_index(session, tmp_path / "indexes")
        except ValueError as exc:
            assert "拒绝建立空索引" in str(exc)
        else:
            raise AssertionError("empty index must fail")


def test_bm25_build_contains_only_topics_included_by_review(
    database: Database, tmp_path: Path
) -> None:
    corpus = tmp_path / "role-corpus"
    corpus.mkdir()
    (corpus / "fixture.md").write_text(
        "# 示例\n这是一段足够长的有效正文，包含作者的观点、解释、案例以及完整结论。"
        "它应当作为实质内容进入索引。\n阅读付费提示。",
        encoding="utf-8",
    )
    response = json.dumps(
        {
            "topics": [
                {
                    "end_unit": 3,
                    "title": "正文",
                    "summary": "有效内容",
                    "keywords": ["正文"],
                },
                {
                    "end_unit": 4,
                    "title": "提示",
                    "summary": "阅读提示",
                    "keywords": ["提示"],
                },
            ]
        },
        ensure_ascii=False,
    )
    with database.session() as session:
        scan_corpus(session, corpus)
    with database.session() as session:
        version = session.scalar(select(ArticleVersionRow))
        assert version is not None

        class RoleGenerator:
            def generate(self, prompt: str, schema: dict[str, object]) -> GenerationResult:
                return GenerationResult(content=response)

        propose_chunks(
            session,
            version,
            RoleGenerator(),
            "role-fake",
            0,
        )
        version_id = version.id
    with database.session() as session:
        review, _ = export_review(session, version_id, tmp_path / "role-reviews")
        payload = json.loads(review.read_text(encoding="utf-8"))
        payload["topics"][1]["include_in_index"] = False
        review.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        approve_manifest(session, review)
        outcome = build_index(session, tmp_path / "role-indexes")
        assert outcome.chunk_count == 1


def test_failed_build_is_not_published_and_diagnostic_can_be_recorded(
    database: Database, tmp_path: Path, monkeypatch
) -> None:
    _approve_fixture(database, tmp_path)
    index_root = tmp_path / "indexes"

    def fail_index(*args, **kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr("bm25s.BM25.index", fail_index)
    try:
        with database.session() as session:
            build_index(session, index_root)
    except IndexBuildError as exc:
        with database.session() as session:
            record_build_failure(session, exc)
    else:
        raise AssertionError("synthetic build must fail")

    with database.session() as session:
        row = session.scalar(select(IndexBuildRow))
        assert row is not None
        assert row.status == "failed"
        assert row.error_type == "RuntimeError"
        assert not row.active
        assert not Path(row.path).exists()
