from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import func, select

from xifeng_rag.corpus import (
    CLEANER_VERSION,
    analyze_trailing_catalog,
    clean_markdown,
    discover_markdown,
    load_external_article_ids,
    load_html_published_at,
    scan_corpus,
)
from xifeng_rag.domain import VersionStatus
from xifeng_rag.storage import ArticleRow, ArticleVersionRow, Database, ParagraphRow


def test_clean_markdown_keeps_anchor_and_tracks_url() -> None:
    result = clean_markdown("# 标题\n\n正文引用[资料名称](https://example.test/a)。")
    assert "资料名称" in result.canonical_text
    assert "https://" not in result.canonical_text
    assert result.links[0].url == "https://example.test/a"
    assert result.paragraphs[1].source_start > result.paragraphs[0].source_start


def test_semantic_units_follow_lines_and_sentence_endings_with_exact_offsets() -> None:
    markdown = '# 标题\r\n第一行很短\r\n第二句。第三句！" 尾句\r\n'
    result = clean_markdown(markdown)
    assert [item.text for item in result.paragraphs] == [
        "# 标题",
        "第一行很短",
        "第二句。",
        '第三句！"',
        "尾句",
    ]
    for item in result.paragraphs:
        assert markdown[item.source_start : item.source_end] == item.text


def test_sentence_units_do_not_split_urls_decimals_or_english_periods() -> None:
    markdown = (
        "价格是3.14，参考[资料](https://example.test/a?x=1;y=2)。"
        "English sentence stays. Next clause;最后一句？"
    )
    result = clean_markdown(markdown)
    assert [item.text for item in result.paragraphs] == [
        "价格是3.14，参考资料。",
        "English sentence stays. Next clause;",
        "最后一句？",
    ]
    assert result.links[0].url == "https://example.test/a?x=1;y=2"
    source = markdown[result.paragraphs[0].source_start : result.paragraphs[0].source_end]
    assert source.endswith("。")


def test_sentence_units_protect_link_anchors_and_bare_url_query_punctuation() -> None:
    markdown = (
        "参考[为什么？不要拆！](https://example.test/a?x=1;y=2)。下一句；"
        "裸链接https://example.test/search?x=1;y=2!flag。结束。"
    )
    result = clean_markdown(markdown)
    assert [item.text for item in result.paragraphs] == [
        "参考为什么？不要拆！。",
        "下一句；",
        "裸链接https://example.test/search?x=1;y=2!flag。",
        "结束。",
    ]
    assert result.links[0].anchor == "为什么？不要拆！"


def test_clean_markdown_removes_high_confidence_trailing_catalog() -> None:
    links = "\n\n".join(f"[文章{i}](https://example.test/{i})" for i in range(6))
    markdown = "\n\n".join(["# 标题", "正文一", "正文二", "付费文章目录", links])
    result = clean_markdown(markdown)
    assert "正文二" in result.canonical_text
    assert "付费文章目录" not in result.canonical_text
    assert result.removed_ranges
    assert not result.needs_review


def test_clean_markdown_flags_ambiguous_trailing_section() -> None:
    markdown = "\n\n".join(
        [
            "# 标题",
            "正文一",
            "正文二",
            "备用号文章",
            "[甲](https://example.test/1)",
            "[乙](https://example.test/2)",
        ]
    )
    result = clean_markdown(markdown)
    assert result.needs_review
    assert not result.removed_ranges


def test_clean_markdown_backtracks_to_earliest_multi_account_catalog_intro() -> None:
    first_links = "\n\n".join(
        f"[大号文章{i}](https://example.test/main/{i})" for i in range(6)
    )
    second_links = "\n\n".join(
        f"[小号文章{i}](https://example.test/side/{i})" for i in range(5)
    )
    intro = "所以读者问，怎么知道是否全部的付费文都看过了。给你整理出来了。"
    markdown = "\n\n".join(
        [
            "# 标题",
            "正文有完整结论。",
            intro,
            "**这是大号****全部的付费文章：**",
            first_links,
            "**这是小号****所有的付费文章：****",
            second_links,
        ]
    )

    analysis = analyze_trailing_catalog(markdown)
    result = clean_markdown(markdown)

    assert analysis.status == "remove"
    assert analysis.source_start == markdown.index(intro)
    assert result.removed_ranges == [(markdown.index(intro), len(markdown))]
    assert "正文有完整结论" in result.canonical_text
    assert "付费文" not in result.canonical_text


def test_clean_markdown_detects_catalog_heading_after_packed_link_line() -> None:
    packed = "".join(
        f"[文章{i}](https://example.test/packed/{i})" for i in range(6)
    )
    marker = "**这是小号****所有的付费文章：**"
    trailing = "\n\n".join(
        f"[后续{i}](https://example.test/trailing/{i})" for i in range(5)
    )
    markdown = "\n\n".join(["# 标题", "正文。", packed + marker, trailing])

    result = clean_markdown(markdown)

    assert result.removed_ranges == [(markdown.index(packed), len(markdown))]
    assert result.canonical_text == "# 标题\n\n正文。"


def test_clean_markdown_counts_empty_anchor_as_catalog_structure() -> None:
    links = ["[](https://example.test/empty)"] + [
        f"[文章{i}](https://example.test/{i})" for i in range(4)
    ]
    marker = "这里是所有的付费文章链接："
    markdown = "\n\n".join(["# 标题", "正文。", marker, *links])

    result = clean_markdown(markdown)

    assert result.removed_ranges == [(markdown.index(marker), len(markdown))]
    assert result.canonical_text == "# 标题\n\n正文。"


def test_clean_markdown_does_not_remove_normal_tail_reference() -> None:
    markdown = "\n\n".join(
        [
            "# 标题",
            "正文讨论付费服务本身，但不是文章目录。",
            "可参考[一份资料](https://example.test/reference)。",
        ]
    )

    result = clean_markdown(markdown)

    assert result.removed_ranges == []
    assert not result.needs_review
    assert "一份资料" in result.canonical_text


def test_clean_markdown_requires_review_when_body_resumes_after_catalog() -> None:
    links = "\n\n".join(
        f"[文章{i}](https://example.test/{i})" for i in range(6)
    )
    resumed = "这是重新恢复的正文论述。" * 20
    markdown = "\n\n".join(
        ["# 标题", "正文。", "这里是所有的付费文章链接：", links, resumed]
    )

    result = clean_markdown(markdown)

    assert result.needs_review
    assert result.removed_ranges == []
    assert result.canonical_text.count("这是重新恢复的正文论述。") == 20


def test_discover_rejects_missing_or_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="不存在"):
        discover_markdown(tmp_path / "missing")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="没有顶层 Markdown"):
        discover_markdown(empty)


def test_load_external_article_ids_from_source_parent(tmp_path: Path) -> None:
    corpus = tmp_path / "saved_articles"
    corpus.mkdir()
    (tmp_path / "index.json").write_text(
        '{"sn:abc123": {"md_file": "saved_articles/example.md"}, "_updated": "now"}',
        encoding="utf-8",
    )
    assert load_external_article_ids(corpus) == {"example.md": "sn:abc123"}


def test_load_html_published_at_prefers_original_epoch(tmp_path: Path) -> None:
    source = tmp_path / "example.html"
    source.write_text(
        "create_time: '2026-06-04 11:26', ori_create_time: '1780543678' * 1",
        encoding="utf-8",
    )
    published = load_html_published_at(source)
    assert published is not None
    assert published.isoformat() == "2026-06-04T11:27:58"


def test_load_html_published_at_falls_back_to_readable_time(tmp_path: Path) -> None:
    source = tmp_path / "example.html"
    source.write_text("create_time: '2026-06-04 11:27'", encoding="utf-8")
    published = load_html_published_at(source)
    assert published is not None
    assert published.isoformat() == "2026-06-04T11:27:00"
    assert load_html_published_at(tmp_path / "missing.html") is None


def test_scan_updates_source_metadata_without_creating_new_version(
    database: Database, tmp_path: Path
) -> None:
    corpus = tmp_path / "saved_articles"
    corpus.mkdir()
    source = corpus / "example.md"
    source.write_text("# 示例\n\n自造正文。", encoding="utf-8")
    (corpus / "example.html").write_text(
        "ori_create_time: '1780543678' * 1", encoding="utf-8"
    )
    catalog = tmp_path / "index.json"
    metadata = {
        "sn:metadata": {
            "md_file": "saved_articles/example.md",
            "captured_at": "2026-08-20 12:30:00",
            "chinese_char_count": 123,
        }
    }
    catalog.write_text(json.dumps(metadata), encoding="utf-8")

    with database.session() as session:
        assert scan_corpus(session, corpus).added == 1
    with database.session() as session:
        article = session.get(ArticleRow, "sn:metadata")
        assert article is not None
        assert article.captured_at is not None
        assert article.captured_at.isoformat() == "2026-08-20T12:30:00"
        assert article.published_at is not None
        assert article.published_at.isoformat() == "2026-06-04T11:27:58"
        assert article.source_modified_at is not None
        assert article.source_size_bytes == source.stat().st_size
        assert article.source_character_count == 123
        version_count = session.scalar(select(func.count()).select_from(ArticleVersionRow))
        paragraph_count = session.scalar(select(func.count()).select_from(ParagraphRow))

    metadata["sn:metadata"]["captured_at"] = "2026-08-21 09:00:00"
    metadata["sn:metadata"]["chinese_char_count"] = 125
    catalog.write_text(json.dumps(metadata), encoding="utf-8")
    with database.session() as session:
        summary = scan_corpus(session, corpus)
        assert summary.metadata_updated == 1
        assert summary.unchanged == 0
    with database.session() as session:
        article = session.get(ArticleRow, "sn:metadata")
        assert article is not None
        assert article.captured_at is not None
        assert article.captured_at.isoformat() == "2026-08-21T09:00:00"
        assert article.source_character_count == 125
        assert session.scalar(select(func.count()).select_from(ArticleVersionRow)) == version_count
        assert session.scalar(select(func.count()).select_from(ParagraphRow)) == paragraph_count

    with database.session() as session:
        summary = scan_corpus(session, corpus)
        assert summary.metadata_updated == 0
        assert summary.unchanged == 1


def test_scan_classifies_added_unchanged_modified_and_missing(
    database: Database, tmp_path: Path
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    first = corpus / "first.md"
    first.write_text("# 第一篇\n\n第一段。\n\n第二段。", encoding="utf-8")
    with database.session() as session:
        summary = scan_corpus(session, corpus)
        assert summary.added == 1
    with database.session() as session:
        summary = scan_corpus(session, corpus)
        assert summary.unchanged == 1
        assert session.scalar(select(func.count()).select_from(ParagraphRow)) == 3

    first.write_text("# 第一篇\n\n第一段修改。\n\n第二段。", encoding="utf-8")
    second = corpus / "second.md"
    second.write_text("# 第二篇\n\n正文。", encoding="utf-8")
    with database.session() as session:
        summary = scan_corpus(session, corpus)
        assert summary.modified == 1
        assert summary.added == 1

    first.unlink()
    with database.session() as session:
        summary = scan_corpus(session, corpus)
        assert summary.missing == 1
        article = session.scalar(select(ArticleRow).where(ArticleRow.source_path == "first.md"))
        assert article is not None and not article.active
        statuses = set(
            session.scalars(
                select(ArticleVersionRow.status).where(ArticleVersionRow.article_id == article.id)
            ).all()
        )
        assert VersionStatus.INACTIVE.value in statuses


def test_scan_excludes_by_stable_id_and_can_restore(database: Database, tmp_path: Path) -> None:
    corpus = tmp_path / "saved_articles"
    corpus.mkdir()
    first = corpus / "first.md"
    second = corpus / "second.md"
    first.write_text("# 第一篇\n\n完整正文。", encoding="utf-8")
    second.write_text("# 第二篇\n\n完整正文。", encoding="utf-8")
    (tmp_path / "index.json").write_text(
        json.dumps(
            {
                "sn:first": {"md_file": "saved_articles/first.md"},
                "sn:second": {"md_file": "saved_articles/second.md"},
            }
        ),
        encoding="utf-8",
    )

    with database.session() as session:
        assert scan_corpus(session, corpus).added == 2
    with database.session() as session:
        summary = scan_corpus(session, corpus, ["sn:first", "sn:not-found"])
        assert summary.discovered == 2
        assert summary.excluded == 1
        assert summary.missing == 0
        assert summary.unmatched_excluded_ids == ("sn:not-found",)
        article = session.get(ArticleRow, "sn:first")
        assert article is not None and not article.active
        version = session.scalar(
            select(ArticleVersionRow).where(ArticleVersionRow.article_id == "sn:first")
        )
        assert version is not None and not version.active
        assert version.status == VersionStatus.PENDING_CHUNKING.value

    with database.session() as session:
        summary = scan_corpus(session, corpus)
        assert summary.excluded == 0
        assert summary.unchanged == 2
        article = session.get(ArticleRow, "sn:first")
        assert article is not None and article.active
        version = session.scalar(
            select(ArticleVersionRow).where(ArticleVersionRow.article_id == "sn:first")
        )
        assert version is not None and version.active


def test_scan_creates_new_version_when_cleaner_version_changes(
    database: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("# 标题\n正文。", encoding="utf-8")

    monkeypatch.setattr("xifeng_rag.corpus.CLEANER_VERSION", "test-cleaner-v1")
    with database.session() as session:
        assert scan_corpus(session, corpus).added == 1

    monkeypatch.setattr("xifeng_rag.corpus.CLEANER_VERSION", CLEANER_VERSION)
    with database.session() as session:
        summary = scan_corpus(session, corpus)
        assert summary.modified == 1
        versions = list(
            session.scalars(select(ArticleVersionRow).order_by(ArticleVersionRow.created_at)).all()
        )
        assert len(versions) == 2
        assert versions[0].source_hash == versions[1].source_hash
        assert versions[0].cleaner_version != versions[1].cleaner_version
        assert not versions[0].active
        assert versions[0].status == VersionStatus.INACTIVE.value
        assert versions[1].active


def test_scan_reactivates_unapproved_current_cleaner_version_after_index_switch(
    database: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("# 标题\n正文。", encoding="utf-8")

    monkeypatch.setattr("xifeng_rag.corpus.CLEANER_VERSION", "test-cleaner-v1")
    with database.session() as session:
        scan_corpus(session, corpus)
    monkeypatch.setattr("xifeng_rag.corpus.CLEANER_VERSION", CLEANER_VERSION)
    with database.session() as session:
        scan_corpus(session, corpus)
        versions = list(
            session.scalars(select(ArticleVersionRow).order_by(ArticleVersionRow.created_at)).all()
        )
        versions[0].active = True
        versions[0].status = VersionStatus.INDEXED.value
        versions[1].active = False
        versions[1].status = VersionStatus.INACTIVE.value

    with database.session() as session:
        summary = scan_corpus(session, corpus)
        assert summary.modified == 1
        assert summary.unchanged == 0
        versions = list(
            session.scalars(select(ArticleVersionRow).order_by(ArticleVersionRow.created_at)).all()
        )
        assert versions[0].active
        assert versions[0].status == VersionStatus.INDEXED.value
        assert versions[1].active
        assert versions[1].status == VersionStatus.PENDING_CHUNKING.value
