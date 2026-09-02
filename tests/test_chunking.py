from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, cast

import pytest
from sqlalchemy import select

from xifeng_rag.chunking import (
    CodexCliClient,
    GenerationResult,
    OpenAICompatibleClient,
    approve_manifest,
    export_review,
    ollama_topic_schema,
    proposal_warnings,
    propose_chunks,
    resolve_cleanup,
    segmentation_prompt,
)
from xifeng_rag.config import CodexCliSettings, ExternalChunkingProfile
from xifeng_rag.corpus import scan_corpus, stable_id
from xifeng_rag.domain import OverviewStatus, ProposalStatus, TopicProposal, VersionStatus
from xifeng_rag.storage import (
    ApprovedManifestRow,
    ArticleOverviewRow,
    ArticleVersionRow,
    ChunkProposalRow,
    Database,
    ParagraphRow,
    SubtopicSetRow,
    TopicChunkRow,
)


class FakeGenerator:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.calls = 0
        self.prompts: list[str] = []

    def generate(self, prompt: str, schema: dict[str, object]) -> GenerationResult:
        self.calls += 1
        self.prompts.append(prompt)
        assert "带编号单元" in prompt
        assert schema["type"] == "object"
        return GenerationResult(content=next(self.responses), metadata={"eval_count": 8})


def _pending_version(database: Database, tmp_path: Path) -> ArticleVersionRow:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text(
        "# 标题\n\n第一部分包含较长且完整的论述内容。\n\n"
        "第二部分继续提供解释和具体案例。\n\n"
        "第三部分给出反思以及最终结论。\n\n提示。",
        encoding="utf-8",
    )
    with database.session() as session:
        scan_corpus(session, corpus)
    with database.session() as session:
        version = session.scalar(select(ArticleVersionRow))
        assert version is not None
        session.expunge(version)
        return version


def _valid_response() -> str:
    return json.dumps(
        {
            "article_summary": "全文先说明起始内容，再讨论后续解释与结论。",
            "topics": [
                {
                    "end_unit": 2,
                    "title": "起始",
                    "summary": "开篇内容",
                    "keywords": ["开篇"],
                },
                {
                    "end_unit": 5,
                    "title": "后续",
                    "summary": "后续内容",
                    "keywords": ["后续"],
                },
            ],
        },
        ensure_ascii=False,
    )


def test_joint_generation_approves_optional_subtopics_separately(
    database: Database, tmp_path: Path
) -> None:
    detached = _pending_version(database, tmp_path)
    response = json.dumps(
        {
            "article_summary": "文章整体概括。",
            "topics": [
                {
                    "end_unit": 5,
                    "title": "完整论述",
                    "summary": "全文概括",
                    "keywords": ["主题"],
                    "subtopics": [
                        {
                            "title": "具体案例",
                            "start_unit": 2,
                            "end_unit": 3,
                            "summary": "文章中容易被单独记住的案例。",
                        },
                        {
                            "title": "越界条目",
                            "start_unit": 4,
                            "end_unit": 6,
                            "summary": "该条目应被宽松丢弃。",
                        },
                    ],
                }
            ],
        },
        ensure_ascii=False,
    )
    with database.session() as session:
        version = session.get(ArticleVersionRow, detached.id)
        assert version is not None
        outcome = propose_chunks(session, version, FakeGenerator([response]), "fake-model", 0)
        proposal = session.get(ChunkProposalRow, outcome.proposal_id)
        assert proposal is not None
        parsed = TopicProposal.model_validate_json(proposal.topics_json)
        assert [item.title for item in parsed.topics[0].subtopics] == ["具体案例"]
        diagnostics = json.loads(proposal.diagnostics_json)
        assert diagnostics[-1]["type"] == "invalid_subtopic_range"
        review, _ = export_review(session, detached.id, tmp_path / "reviews", outcome.proposal_id)
        manifest_id = approve_manifest(session, review)
        manifest = session.get(ApprovedManifestRow, manifest_id)
        assert manifest is not None and manifest.topics_json is not None
        approved_topics = TopicProposal.model_validate_json(manifest.topics_json)
        assert approved_topics.topics[0].subtopics == []
        supplement = session.scalar(
            select(SubtopicSetRow).where(SubtopicSetRow.manifest_id == manifest_id)
        )
        assert supplement is not None
        assert "具体案例" in supplement.subtopics_json


def test_ollama_schema_uses_supported_conservative_subset() -> None:
    schema = ollama_topic_schema()
    schema_text = json.dumps(schema)
    for unsupported in ("minItems", "maxItems", "minLength", "maxLength", "$ref", "$defs"):
        assert unsupported not in schema_text
    assert schema["required"] == ["article_summary", "topics"]
    assert "article_summary" in schema["properties"]  # type: ignore[operator]
    item_schema = schema["properties"]["topics"]["items"]  # type: ignore[index]
    assert "content_role" not in item_schema["properties"]  # type: ignore[index]
    assert "content_role" not in item_schema["required"]  # type: ignore[index]


def test_semantic_prompt_allows_one_topic_without_size_warnings(
    database: Database, tmp_path: Path
) -> None:
    detached = _pending_version(database, tmp_path)
    response = json.dumps(
        {
            "article_summary": "全文围绕同一主题完整展开。",
            "topics": [
                {
                    "end_unit": 5,
                    "title": "完整论述",
                    "summary": "全文围绕同一主题展开",
                    "keywords": ["主题"],
                }
            ],
        },
        ensure_ascii=False,
    )
    with database.session() as session:
        version = session.get(ArticleVersionRow, detached.id)
        assert version is not None
        fake = FakeGenerator([response])
        outcome = propose_chunks(session, version, fake, "fake-model", 0)
        assert outcome.status == ProposalStatus.PENDING_REVIEW
        assert outcome.warnings == []

        paragraphs = list(
            session.scalars(
                select(ParagraphRow)
                .where(ParagraphRow.version_id == detached.id)
                .order_by(ParagraphRow.number)
            ).all()
        )
        prompt = segmentation_prompt("标题", paragraphs)
        assert "不预设块数" in prompt
        assert "只返回每块的 end_unit" in prompt
        assert "先在内部识别全文主旨和顶层提纲" in prompt
        assert "显式篇章结构的优先级高于局部语义变化" in prompt
        assert "案例、类比、数据、原因、推导、细节和过渡" in prompt
        assert "当前只做单层顶层分块" in prompt
        assert "文末文章目录" in prompt
        assert "是否进入索引由用户审核决定" in prompt
        assert "不要输出内容角色" in prompt
        assert "4–6" not in prompt
        row = session.get(ChunkProposalRow, outcome.proposal_id)
        assert row is not None
        normalized = TopicProposal.model_validate_json(row.topics_json)
        assert proposal_warnings(normalized, paragraphs) == []
        overview = session.scalar(
            select(ArticleOverviewRow).where(ArticleOverviewRow.proposal_id == outcome.proposal_id)
        )
        assert overview is not None
        assert overview.article_summary == "全文围绕同一主题完整展开。"
        assert overview.status == OverviewStatus.PENDING_REVIEW.value
        assert overview.manifest_id is None


def test_propose_blocks_residual_trailing_catalog_before_generator_call(
    database: Database, tmp_path: Path
) -> None:
    detached = _pending_version(database, tmp_path)
    with database.session() as session:
        version = session.get(ArticleVersionRow, detached.id)
        assert version is not None
        paragraphs = list(
            session.scalars(
                select(ParagraphRow)
                .where(ParagraphRow.version_id == version.id)
                .order_by(ParagraphRow.number)
            ).all()
        )
        cursor = paragraphs[-1].source_end + 2
        marker = ParagraphRow(
            version_id=version.id,
            number=6,
            text="这是所有的付费文章链接：",
            source_start=cursor,
            source_end=cursor + 12,
        )
        session.add(marker)
        cursor = marker.source_end + 2
        links = []
        for number in range(7, 12):
            unit = ParagraphRow(
                version_id=version.id,
                number=number,
                text=f"自造目录文章{number}",
                source_start=cursor,
                source_end=cursor + 10,
            )
            session.add(unit)
            links.append(
                {
                    "anchor": unit.text,
                    "url": f"https://example.test/{number}",
                    "source_start": cursor,
                    "source_end": cursor + 10,
                }
            )
            cursor += 12
        version.links_json = json.dumps(links, ensure_ascii=False)

    fake = FakeGenerator([])
    with database.session() as session:
        version = session.get(ArticleVersionRow, detached.id)
        assert version is not None
        with pytest.raises(ValueError, match="残留付费文章目录"):
            propose_chunks(session, version, fake, "fake-model", 0)
        assert fake.calls == 0


def test_propose_export_edit_and_approve(database: Database, tmp_path: Path) -> None:
    detached = _pending_version(database, tmp_path)
    fake = FakeGenerator([_valid_response()])
    with database.session() as session:
        version = session.get(ArticleVersionRow, detached.id)
        assert version is not None
        outcome = propose_chunks(session, version, fake, "fake-model", 0)
        assert outcome.proposal_id
    with database.session() as session:
        json_path, markdown_path = export_review(session, detached.id, tmp_path / "reviews")
        assert markdown_path.is_file()
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert payload["article_summary"] == "全文先说明起始内容，再讨论后续解释与结论。"
        assert "文章整体概括（AI）" in markdown_path.read_text(encoding="utf-8")
        payload["article_summary"] = "人工修订后的整体概括。"
        payload["topics"][0]["title"] = "人工修订标题"
        json_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        proposal = session.get(ChunkProposalRow, payload["proposal_id"])
        assert proposal is not None
        proposal.prompt_version = "legacy-prompt-version"
        manifest_id = approve_manifest(session, json_path)
        assert manifest_id
    with database.session() as session:
        version = session.get(ArticleVersionRow, detached.id)
        assert version is not None and version.status == VersionStatus.APPROVED.value
        chunks = list(session.scalars(select(TopicChunkRow)).all())
        assert len(chunks) == 2
        assert chunks[0].id == stable_id(
            "chunk",
            detached.id,
            "legacy-prompt-version",
            manifest_id,
            "1:2",
            length=40,
        )
        assert chunks[0].topic_title == "人工修订标题"
        assert "第一部分" in chunks[0].original_text
        manifest = session.get(ApprovedManifestRow, manifest_id)
        assert manifest is not None and manifest.topics_json is not None
        assert json.loads(manifest.topics_json)["topics"][0]["title"] == "人工修订标题"
        overview = session.scalar(
            select(ArticleOverviewRow).where(
                ArticleOverviewRow.proposal_id == payload["proposal_id"]
            )
        )
        assert overview is not None
        assert overview.article_summary == "人工修订后的整体概括。"
        assert overview.status == OverviewStatus.APPROVED.value
        assert overview.manifest_id == manifest_id


def test_invalid_coverage_retries_then_fails(database: Database, tmp_path: Path) -> None:
    detached = _pending_version(database, tmp_path)
    invalid = json.dumps(
        {
            "article_summary": "这是一份覆盖不完整的错误概括。",
            "topics": [
                {
                    "end_unit": 4,
                    "title": "缺口",
                    "summary": "缺少最后一段",
                    "keywords": ["错误"],
                    "content_role": "substantive",
                }
            ],
        }
    )
    fake = FakeGenerator([invalid, invalid])
    with database.session() as session:
        version = session.get(ArticleVersionRow, detached.id)
        assert version is not None
        outcome = propose_chunks(session, version, fake, "fake-model", 1)
        assert outcome.status == ProposalStatus.FAILED
        assert "2 次尝试" in outcome.warnings[0]
    with database.session() as session:
        row = session.scalar(select(ChunkProposalRow))
        assert row is not None and row.status == ProposalStatus.FAILED.value
        diagnostics = json.loads(row.diagnostics_json)
        assert diagnostics[0]["error_type"] == "TopicCoverageError"
        assert diagnostics[0]["issues"] == ["分块未覆盖到最后单元 5"]
        assert "单元覆盖不合法" in fake.prompts[1]


def test_missing_joint_overview_retries_and_uses_valid_summary(
    database: Database, tmp_path: Path
) -> None:
    detached = _pending_version(database, tmp_path)
    missing = json.loads(_valid_response())
    missing.pop("article_summary")
    fake = FakeGenerator([json.dumps(missing, ensure_ascii=False), _valid_response()])
    with database.session() as session:
        version = session.get(ArticleVersionRow, detached.id)
        assert version is not None
        outcome = propose_chunks(session, version, fake, "fake-model", 1)
        assert outcome.status == ProposalStatus.PENDING_REVIEW
        assert fake.calls == 2
        assert "补齐所有字段" in fake.prompts[1]
        overview = session.scalar(
            select(ArticleOverviewRow).where(ArticleOverviewRow.proposal_id == outcome.proposal_id)
        )
        assert overview is not None
        assert overview.article_summary == "全文先说明起始内容，再讨论后续解释与结论。"


@pytest.mark.parametrize(
    ("end_units", "expected_issue"),
    [
        ([3, 2], "结束单元 2 早于连续起点 4"),
        ([6], "结束单元 6 越界"),
    ],
)
def test_descending_or_out_of_range_end_units_are_rejected(
    database: Database,
    tmp_path: Path,
    end_units: list[int],
    expected_issue: str,
) -> None:
    detached = _pending_version(database, tmp_path)
    topics = [
        {
            "end_unit": end_unit,
            "title": f"话题{index}",
            "summary": "测试边界",
            "keywords": ["边界"],
            "content_role": "substantive",
        }
        for index, end_unit in enumerate(end_units, start=1)
    ]
    with database.session() as session:
        version = session.get(ArticleVersionRow, detached.id)
        assert version is not None
        outcome = propose_chunks(
            session,
            version,
            FakeGenerator(
                [
                    json.dumps(
                        {"article_summary": "目录角色测试文章概括。", "topics": topics},
                        ensure_ascii=False,
                    )
                ]
            ),
            "fake-model",
            0,
        )
        assert outcome.status == ProposalStatus.FAILED
        row = session.get(ChunkProposalRow, outcome.proposal_id)
        assert row is not None
        assert expected_issue in json.loads(row.diagnostics_json)[0]["issues"][0]


def test_generated_content_roles_are_ignored_and_topics_default_to_included(
    database: Database, tmp_path: Path
) -> None:
    detached = _pending_version(database, tmp_path)
    invalid = json.dumps(
        {
            "article_summary": "正文被划分为两个连续话题。",
            "topics": [
                {
                    "end_unit": end_unit,
                    "title": f"模板{index}",
                    "summary": "错误标记为模板",
                    "keywords": ["模板"],
                    "content_role": "boilerplate",
                }
                for index, end_unit in enumerate([1, 2, 3, 5], start=1)
            ],
        },
        ensure_ascii=False,
    )
    fake = FakeGenerator([invalid])
    with database.session() as session:
        version = session.get(ArticleVersionRow, detached.id)
        assert version is not None
        outcome = propose_chunks(session, version, fake, "fake-model", 0)
        assert outcome.status == ProposalStatus.PENDING_REVIEW
        row = session.get(ChunkProposalRow, outcome.proposal_id)
        assert row is not None
        proposal = TopicProposal.model_validate_json(row.topics_json)
        assert all(topic.include_in_index for topic in proposal.topics)
        assert json.loads(row.diagnostics_json) == []


def test_pending_review_version_can_add_current_prompt_proposal_without_overwrite(
    database: Database, tmp_path: Path
) -> None:
    detached = _pending_version(database, tmp_path)
    old_topics = json.dumps(
        {
            "article_summary": "缓存提案应保留同一次生成的整体概括。",
            "topics": [
                {
                    "start_paragraph": 1,
                    "end_paragraph": 5,
                    "title": "旧提案",
                    "summary": "旧版本提示词生成的提案",
                    "keywords": ["旧提案"],
                }
            ],
        },
        ensure_ascii=False,
    )
    with database.session() as session:
        version = session.get(ArticleVersionRow, detached.id)
        assert version is not None
        version.status = VersionStatus.PENDING_REVIEW.value
        session.add(
            ChunkProposalRow(
                id="old-proposal",
                version_id=version.id,
                status=ProposalStatus.PENDING_REVIEW.value,
                model_id="fake-model",
                prompt_version="topic-segmentation-v3-semantic-units-roles",
                schema_version="topic-proposal-v2-end-unit-role",
                topics_json=old_topics,
            )
        )

    fake = FakeGenerator([_valid_response()])
    with database.session() as session:
        version = session.get(ArticleVersionRow, detached.id)
        assert version is not None
        outcome = propose_chunks(session, version, fake, "fake-model", 0)
        assert outcome.status == ProposalStatus.PENDING_REVIEW
        assert outcome.proposal_id != "old-proposal"
        assert fake.calls == 1

    class ForbidGenerator:
        def generate(self, prompt: str, schema: dict[str, object]) -> GenerationResult:
            raise AssertionError("当前 prompt 提案应从缓存复用")

    with database.session() as session:
        version = session.get(ArticleVersionRow, detached.id)
        assert version is not None
        cached = propose_chunks(session, version, ForbidGenerator(), "fake-model", 0)
        assert cached.proposal_id == outcome.proposal_id
        rows = list(
            session.scalars(
                select(ChunkProposalRow).where(ChunkProposalRow.version_id == version.id)
            ).all()
        )
        assert {row.id for row in rows} == {"old-proposal", outcome.proposal_id}
        old = session.get(ChunkProposalRow, "old-proposal")
        assert old is not None and old.topics_json == old_topics


def test_retry_metadata_and_custom_prompt_are_isolated(database: Database, tmp_path: Path) -> None:
    detached = _pending_version(database, tmp_path)
    custom = "本次完全自定义分块要求，只输出约定 JSON。"
    with database.session() as session:
        version = session.get(ArticleVersionRow, detached.id)
        assert version is not None
        first_generator = FakeGenerator([_valid_response()])
        first = propose_chunks(
            session,
            version,
            first_generator,
            "external-model",
            0,
            provider_id="deepseek",
            custom_prompt=custom,
            retry_id="retry-one",
        )
        assert custom in first_generator.prompts[0]
        assert "根据本次收到的完整原文独立生成 article_summary" in first_generator.prompts[0]
        assert "不要根据 topics 中的块级概括机械拼接" in first_generator.prompts[0]
        row = session.get(ChunkProposalRow, first.proposal_id)
        assert row is not None
        assert row.provider_id == "deepseek"
        assert row.prompt_text == custom
        assert row.prompt_hash is not None
        assert row.retry_id == "retry-one"

    with database.session() as session:
        version = session.get(ArticleVersionRow, detached.id)
        assert version is not None
        default_generator = FakeGenerator([_valid_response()])
        second = propose_chunks(session, version, default_generator, "fake-model", 0)
        assert custom not in default_generator.prompts[0]
        assert second.proposal_id != first.proposal_id


def test_retry_id_always_creates_parallel_proposals_and_requires_explicit_export(
    database: Database, tmp_path: Path
) -> None:
    detached = _pending_version(database, tmp_path)
    proposal_ids: list[str] = []
    for retry_id in ("retry-one", "retry-two"):
        with database.session() as session:
            version = session.get(ArticleVersionRow, detached.id)
            assert version is not None
            outcome = propose_chunks(
                session,
                version,
                FakeGenerator([_valid_response()]),
                "same-model",
                0,
                retry_id=retry_id,
            )
            proposal_ids.append(outcome.proposal_id)
    assert proposal_ids[0] != proposal_ids[1]

    with database.session() as session:
        with pytest.raises(ValueError, match="多个待审核提案"):
            export_review(session, detached.id, tmp_path / "reviews")
        first_json, _ = export_review(session, detached.id, tmp_path / "reviews", proposal_ids[0])
        second_json, _ = export_review(session, detached.id, tmp_path / "reviews", proposal_ids[1])
        assert first_json != second_json
        assert proposal_ids[0] in first_json.name
        assert proposal_ids[1] in second_json.name


def test_openai_compatible_client_requests_json_without_logging_raw_response(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {"content": _valid_response()},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                }
            ).encode()

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    profile = ExternalChunkingProfile(
        base_url="https://example.invalid/v1/",
        api_key_env="IGNORED",
        default_model="large-model",
        timeout_seconds=12,
    )
    client = OpenAICompatibleClient(profile, "override-model", "secret-key")
    result = client.generate("private article payload", ollama_topic_schema())

    assert captured["url"] == "https://example.invalid/v1/chat/completions"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["model"] == "override-model"
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["messages"][1]["content"] == "private article payload"
    assert result.metadata == {
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        "finish_reason": "stop",
    }
    assert "private article payload" not in json.dumps(result.metadata)


def test_codex_cli_client_is_ephemeral_isolated_and_schema_constrained(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[dict[str, object]] = []

    def fake_run(arguments, **kwargs):
        calls.append({"arguments": list(arguments), **kwargs})
        if arguments[-1] == "--version":
            return subprocess.CompletedProcess(arguments, 0, "codex-cli 0.test\n", "")
        if arguments[-2:] == ["login", "status"]:
            return subprocess.CompletedProcess(arguments, 0, "", "Logged in using ChatGPT\n")
        output_index = arguments.index("--output-last-message") + 1
        output_path = Path(arguments[output_index])
        schema_index = arguments.index("--output-schema") + 1
        schema = json.loads(Path(arguments[schema_index]).read_text(encoding="utf-8"))
        assert schema["additionalProperties"] is False
        assert schema["properties"]["topics"]["items"]["additionalProperties"] is False
        output_path.write_text(_valid_response(), encoding="utf-8")
        return subprocess.CompletedProcess(arguments, 0, "ignored stdout", "private progress")

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr("shutil.which", lambda executable: None)
    settings = CodexCliSettings(
        executable="custom-codex",
        default_model="configured-model",
        timeout_seconds=30,
        max_retries=0,
    )
    client = CodexCliClient(settings, "selected-model")
    client.preflight()
    result = client.generate("private article payload", ollama_topic_schema())

    execution = calls[-1]
    arguments = execution["arguments"]
    assert isinstance(arguments, list)
    assert arguments[:2] == ["custom-codex", "exec"]
    assert "--ephemeral" in arguments
    assert arguments[arguments.index("--sandbox") : arguments.index("--sandbox") + 2] == [
        "--sandbox",
        "read-only",
    ]
    assert "--ignore-user-config" in arguments
    assert "--ignore-rules" in arguments
    assert "--skip-git-repo-check" in arguments
    assert "private article payload" not in arguments
    assert "private article payload" in str(execution["input"])
    assert execution["cwd"] == Path(arguments[arguments.index("--cd") + 1])
    assert not Path(arguments[arguments.index("--cd") + 1]).exists()
    assert result.content == _valid_response()
    assert result.metadata == {
        "client": "codex-cli",
        "cli_version": "codex-cli 0.test",
        "auth_type": "chatgpt",
    }
    assert "private article payload" not in json.dumps(result.metadata)


def test_strict_output_schema_recurses_into_defs() -> None:
    from xifeng_rag.evaluation_curation import candidate_schema
    from xifeng_rag.structured_generation import strict_output_schema

    schema = strict_output_schema(candidate_schema())
    definitions = cast(dict[str, dict[str, Any]], schema["$defs"])

    assert schema["additionalProperties"] is False
    assert definitions["AnswerPoint"]["additionalProperties"] is False
    assert definitions["EvidenceRange"]["additionalProperties"] is False


def test_end_units_are_normalized_to_continuous_ranges(database: Database, tmp_path: Path) -> None:
    detached = _pending_version(database, tmp_path)
    response = json.dumps(
        {
            "article_summary": "失效文章测试概括。",
            "topics": [
                {
                    "end_unit": 2,
                    "title": "起始",
                    "summary": "开篇内容",
                    "keywords": ["开篇"],
                    "content_role": "substantive",
                },
                {
                    "end_unit": 5,
                    "title": "后续",
                    "summary": "后续内容",
                    "keywords": ["后续"],
                    "content_role": "substantive",
                },
            ],
        },
        ensure_ascii=False,
    )
    with database.session() as session:
        version = session.get(ArticleVersionRow, detached.id)
        assert version is not None
        outcome = propose_chunks(session, version, FakeGenerator([response]), "fake-model", 0)
        assert outcome.status == ProposalStatus.PENDING_REVIEW
        row = session.get(ChunkProposalRow, outcome.proposal_id)
        assert row is not None
        proposal = TopicProposal.model_validate_json(row.topics_json)
        assert proposal.topics[1].start_paragraph == 3
        assert "normalized_topic_starts" not in json.loads(row.generation_json)


def test_cleanup_resolution_keep_all(database: Database, tmp_path: Path) -> None:
    detached = _pending_version(database, tmp_path)
    with database.session() as session:
        version = session.get(ArticleVersionRow, detached.id)
        assert version is not None
        version.status = VersionStatus.NEEDS_CLEANUP_REVIEW.value
    with database.session() as session:
        assert resolve_cleanup(session, detached.id, None) == detached.id
    with database.session() as session:
        version = session.get(ArticleVersionRow, detached.id)
        assert version is not None and version.status == VersionStatus.PENDING_CHUNKING.value


def test_inactive_article_proposal_cannot_be_approved(database: Database, tmp_path: Path) -> None:
    detached = _pending_version(database, tmp_path)
    with database.session() as session:
        version = session.get(ArticleVersionRow, detached.id)
        assert version is not None
        propose_chunks(session, version, FakeGenerator([_valid_response()]), "fake-model", 0)
    with database.session() as session:
        json_path, _ = export_review(session, detached.id, tmp_path / "reviews")
        version = session.get(ArticleVersionRow, detached.id)
        assert version is not None
        version.active = False
        with pytest.raises(ValueError, match="已停用"):
            approve_manifest(session, json_path)


def test_old_review_without_roles_defaults_to_included(database: Database, tmp_path: Path) -> None:
    detached = _pending_version(database, tmp_path)
    with database.session() as session:
        version = session.get(ArticleVersionRow, detached.id)
        assert version is not None
        propose_chunks(session, version, FakeGenerator([_valid_response()]), "fake-model", 0)
    with database.session() as session:
        review, _ = export_review(session, detached.id, tmp_path / "reviews")
        payload = json.loads(review.read_text(encoding="utf-8"))
        for topic in payload["topics"]:
            topic.pop("include_in_index")
        review.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        approve_manifest(session, review)
        assert len(list(session.scalars(select(TopicChunkRow)).all())) == 2


def test_legacy_review_boilerplate_maps_to_excluded(database: Database, tmp_path: Path) -> None:
    detached = _pending_version(database, tmp_path)
    with database.session() as session:
        version = session.get(ArticleVersionRow, detached.id)
        assert version is not None
        propose_chunks(session, version, FakeGenerator([_valid_response()]), "fake-model", 0)
    with database.session() as session:
        review, _ = export_review(session, detached.id, tmp_path / "reviews")
        payload = json.loads(review.read_text(encoding="utf-8"))
        for topic in payload["topics"]:
            topic.pop("include_in_index")
            topic["content_role"] = "substantive"
        payload["topics"][1]["content_role"] = "boilerplate"
        review.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        manifest_id = approve_manifest(session, review)
        assert len(list(session.scalars(select(TopicChunkRow)).all())) == 1
        manifest = session.get(ApprovedManifestRow, manifest_id)
        assert manifest is not None and manifest.topics_json is not None
        decisions = [
            topic["include_in_index"] for topic in json.loads(manifest.topics_json)["topics"]
        ]
        assert decisions == [True, False]


def test_excluded_topic_is_audited_but_not_materialized_as_topic_chunk(
    database: Database, tmp_path: Path
) -> None:
    detached = _pending_version(database, tmp_path)
    response = json.dumps(
        {
            "article_summary": "索引排除测试概括。",
            "topics": [
                {
                    "end_unit": 4,
                    "title": "正文",
                    "summary": "主要论述",
                    "keywords": ["正文"],
                },
                {
                    "end_unit": 5,
                    "title": "阅读提示",
                    "summary": "固定提示",
                    "keywords": ["提示"],
                },
            ],
        },
        ensure_ascii=False,
    )
    with database.session() as session:
        version = session.get(ArticleVersionRow, detached.id)
        assert version is not None
        propose_chunks(session, version, FakeGenerator([response]), "fake-model", 0)
    with database.session() as session:
        review, markdown = export_review(session, detached.id, tmp_path / "reviews")
        assert "进入索引：`true`" in markdown.read_text(encoding="utf-8")
        payload = json.loads(review.read_text(encoding="utf-8"))
        payload["topics"][1]["include_in_index"] = False
        review.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        manifest_id = approve_manifest(session, review)
        chunks = list(session.scalars(select(TopicChunkRow)).all())
        assert len(chunks) == 1
        manifest = session.get(ApprovedManifestRow, manifest_id)
        assert manifest is not None and manifest.topics_json is not None
        decisions = [
            topic["include_in_index"] for topic in json.loads(manifest.topics_json)["topics"]
        ]
        assert decisions == [True, False]


def test_all_excluded_review_cannot_be_approved(database: Database, tmp_path: Path) -> None:
    detached = _pending_version(database, tmp_path)
    with database.session() as session:
        version = session.get(ArticleVersionRow, detached.id)
        assert version is not None
        propose_chunks(session, version, FakeGenerator([_valid_response()]), "fake-model", 0)
    with database.session() as session:
        review, _ = export_review(session, detached.id, tmp_path / "reviews")
        payload = json.loads(review.read_text(encoding="utf-8"))
        for topic in payload["topics"]:
            topic["include_in_index"] = False
        review.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        with pytest.raises(ValueError, match="全部话题均设为不索引"):
            approve_manifest(session, review)
