from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import select

from xifeng_rag.article_search import ArticleBuildOutcome
from xifeng_rag.corpus_ui import CorpusDashboardApplication
from xifeng_rag.domain import (
    ArticleSubtopics,
    ProposalStatus,
    SubtopicSetStatus,
    TopicProposal,
    VersionStatus,
)
from xifeng_rag.storage import (
    ApprovedManifestRow,
    ArticleRow,
    ArticleVersionRow,
    ChunkProposalRow,
    Database,
    ParagraphRow,
    SubtopicSetRow,
)
from xifeng_rag.structured_generation import GenerationResult
from xifeng_rag.subtopic_review_ui import SubtopicReviewApplication
from xifeng_rag.subtopics import (
    approve_subtopics,
    generate_for_existing_article,
    generation_prompt,
)


class FakeGenerator:
    def __init__(self, content: str) -> None:
        self.content = content
        self.prompts: list[str] = []

    def generate(self, prompt: str, schema: dict[str, object]) -> GenerationResult:
        assert schema["required"] == ["topics"]
        self.prompts.append(prompt)
        return GenerationResult(content=self.content, metadata={"client": "fake"})


def _approved_article(database: Database) -> tuple[str, str]:
    topics = {
        "topics": [
            {
                "start_paragraph": 1,
                "end_paragraph": 2,
                "title": "第一块",
                "summary": "第一块概括",
                "keywords": ["第一"],
                "include_in_index": True,
            },
            {
                "start_paragraph": 3,
                "end_paragraph": 4,
                "title": "排除块",
                "summary": "排除块概括",
                "keywords": ["排除"],
                "include_in_index": False,
            },
        ]
    }
    with database.session() as session:
        session.add(ArticleRow(id="article-subtopics", source_path="test.md", title="测试文章"))
        version = ArticleVersionRow(
            id="version-subtopics",
            article_id="article-subtopics",
            source_hash="a" * 64,
            canonical_hash="b" * 64,
            cleaner_version="test",
            canonical_text="一。\n\n二。\n\n三。\n\n四。",
            status=VersionStatus.INDEXED.value,
        )
        session.add(version)
        session.flush()
        session.add_all(
            [
                ParagraphRow(
                    version_id=version.id,
                    number=number,
                    text=f"正文 {number}",
                    source_start=number,
                    source_end=number + 1,
                )
                for number in range(1, 5)
            ]
        )
        proposal = ChunkProposalRow(
            id="chunk-proposal-subtopics",
            version_id=version.id,
            status=ProposalStatus.APPROVED.value,
            provider_id="test",
            model_id="test",
            prompt_version="test",
            schema_version="test",
            topics_json=json.dumps(topics, ensure_ascii=False),
        )
        session.add(proposal)
        session.flush()
        manifest = ApprovedManifestRow(
            id="manifest-subtopics",
            proposal_id=proposal.id,
            version_id=version.id,
            topics_json=json.dumps(topics, ensure_ascii=False),
        )
        session.add(manifest)
    return "version-subtopics", "manifest-subtopics"


def test_existing_article_generation_uses_fixed_included_parent_and_approves(
    database: Database,
) -> None:
    version_id, manifest_id = _approved_article(database)
    response = json.dumps(
        {
            "topics": [
                {
                    "parent_start_unit": 1,
                    "parent_end_unit": 2,
                    "parent_title": "模型回显可被规范化",
                    "subtopics": [
                        {
                            "title": "具体案例",
                            "start_unit": 2,
                            "end_unit": 2,
                            "summary": "一个容易被记住的具体案例。",
                        }
                    ],
                }
            ]
        },
        ensure_ascii=False,
    )
    fake = FakeGenerator(response)
    with database.session() as session:
        row = generate_for_existing_article(
            session,
            version_id,
            fake,
            provider_id="codex-cli",
            model_id="fake-model",
        )
        assert row.status == SubtopicSetStatus.PENDING_REVIEW.value
        assert "1–2：第一块" in fake.prompts[0]
        assert "排除块" not in fake.prompts[0]
        assert "[3]" not in fake.prompts[0]
        approved = approve_subtopics(
            session,
            row.id,
            ArticleSubtopics.model_validate_json(row.subtopics_json).model_dump(),
        )
        assert approved.status == SubtopicSetStatus.APPROVED.value
        assert approved.manifest_id == manifest_id
        parsed = ArticleSubtopics.model_validate_json(approved.subtopics_json)
        assert parsed.topics[0].parent_title == "第一块"


def test_existing_article_generation_rejects_subtopic_outside_parent(
    database: Database,
) -> None:
    version_id, _ = _approved_article(database)
    invalid = json.dumps(
        {
            "topics": [
                {
                    "parent_start_unit": 1,
                    "parent_end_unit": 2,
                    "parent_title": "第一块",
                    "subtopics": [
                        {
                            "title": "越界",
                            "start_unit": 1,
                            "end_unit": 3,
                            "summary": "越过父范围。",
                        }
                    ],
                }
            ]
        },
        ensure_ascii=False,
    )
    with database.session() as session:
        with pytest.raises(ValueError, match="父话题范围"):
            generate_for_existing_article(
                session,
                version_id,
                FakeGenerator(invalid),
                provider_id="codex-cli",
                model_id="fake-model",
            )
        assert session.scalar(select(SubtopicSetRow)) is None


def test_generation_prompt_does_not_require_full_coverage(database: Database) -> None:
    version_id, _ = _approved_article(database)
    with database.session() as session:
        paragraphs = list(
            session.scalars(
                select(ParagraphRow)
                .where(ParagraphRow.version_id == version_id)
                .order_by(ParagraphRow.number)
            ).all()
        )
    topics = TopicProposal.model_validate(
        {
            "topics": [
                {
                    "start_paragraph": 1,
                    "end_paragraph": 2,
                    "title": "第一块",
                    "summary": "概括",
                    "keywords": ["词"],
                }
            ]
        }
    )
    prompt = generation_prompt("标题", topics, paragraphs)
    assert "不要求覆盖父话题" in prompt
    assert "最多返回 4 个细话题" not in prompt
    assert "父话题和整篇文章都不设细话题业务数量上限" in prompt
    assert "相邻、范围重叠或表达同一机制的候选\n必须合并" in prompt
    assert "细话题范围不得少于 10 个单元" in prompt
    assert "不得修改" in prompt


def test_subtopic_schema_keeps_twenty_item_safety_limit() -> None:
    subtopics = [
        {
            "title": f"细话题 {number}",
            "start_unit": 1,
            "end_unit": 1,
            "summary": "简短概括。",
        }
        for number in range(21)
    ]
    with pytest.raises(ValueError, match="at most 20 items"):
        ArticleSubtopics.model_validate(
            {
                "topics": [
                    {
                        "parent_start_unit": 1,
                        "parent_end_unit": 1,
                        "parent_title": "父话题",
                        "subtopics": subtopics,
                    }
                ]
            }
        )


def test_dashboard_discloses_scope_and_requires_confirmation_before_generation(
    database: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    version_id, _ = _approved_article(database)
    response = json.dumps(
        {
            "topics": [
                {
                    "parent_start_unit": 1,
                    "parent_end_unit": 2,
                    "parent_title": "第一块",
                    "subtopics": [],
                }
            ]
        },
        ensure_ascii=False,
    )
    fake = FakeGenerator(response)

    class FakeServer:
        def serve_forever(self) -> None:
            return None

        def shutdown(self) -> None:
            return None

        def server_close(self) -> None:
            return None

    monkeypatch.setattr(
        "xifeng_rag.corpus_ui.create_subtopic_review_server",
        lambda application: (FakeServer(), "http://127.0.0.1:12345/?token=test"),
    )
    application = CorpusDashboardApplication(
        database,
        tmp_path,
        (),
        tmp_path / "reviews",
        tmp_path / "indexes",
        subtopic_generator_factory=lambda model: fake,
        subtopic_model="fake-codex",
    )
    disclosure = application.subtopic_disclosure({"version_id": version_id})
    assert disclosure["article_title"] == "测试文章"
    character_count = disclosure["character_count"]
    assert isinstance(character_count, int)
    assert character_count > 0
    assert disclosure["unit_count"] == 4
    assert disclosure["provider"] == "codex-cli"
    assert disclosure["model"] == "fake-codex"
    assert disclosure["max_requests"] == 1

    with pytest.raises(ValueError, match="明确确认"):
        application.generate_or_review_subtopics({"version_id": version_id})
    result = application.generate_or_review_subtopics({"version_id": version_id, "confirmed": True})
    assert result["set_id"]
    assert len(fake.prompts) == 1
    application.shutdown_children()


def test_existing_article_review_requires_save_then_approval_and_index_build(
    database: Database, tmp_path: Path
) -> None:
    version_id, _ = _approved_article(database)
    response = json.dumps(
        {
            "topics": [
                {
                    "parent_start_unit": 1,
                    "parent_end_unit": 2,
                    "parent_title": "第一块",
                    "subtopics": [
                        {
                            "title": "具体案例",
                            "start_unit": 2,
                            "end_unit": 2,
                            "summary": "一个容易被记住的具体案例。",
                        }
                    ],
                }
            ]
        },
        ensure_ascii=False,
    )
    with database.session() as session:
        pending = generate_for_existing_article(
            session,
            version_id,
            FakeGenerator(response),
            provider_id="codex-cli",
            model_id="fake-model",
        )
        set_id = pending.id

    builds: list[Path] = []

    def build_article_index(session, path: Path) -> ArticleBuildOutcome:
        del session
        builds.append(path)
        return ArticleBuildOutcome("article-index-subtopics", 1, path, 1)

    application = SubtopicReviewApplication(
        database,
        set_id,
        tmp_path / "reviews",
        article_index_dir=tmp_path / "article-indexes",
        article_index_builder=build_article_index,
    )
    document = application.document()
    assert document["article_title"] == "测试文章"
    with pytest.raises(ValueError, match="先保存"):
        application.approve_and_build({"confirmed": True})
    application.save({"topics": document["topics"]})
    result = application.approve_and_build({"confirmed": True})
    assert result["subtopic_count"] == 1
    assert builds == [tmp_path / "article-indexes"]
    application.ensure_shutdown_allowed()
    with database.session() as session:
        approved = session.get(SubtopicSetRow, set_id)
        assert approved is not None
        assert approved.status == SubtopicSetStatus.APPROVED.value
