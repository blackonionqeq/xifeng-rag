from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from xifeng_rag.article_overviews import (
    approve_joint_article_overview,
    create_joint_article_overview_draft,
    joint_article_overview_for_proposal,
)
from xifeng_rag.config import ExternalChunkingProfile, OllamaSettings
from xifeng_rag.corpus import is_explicit_catalog_marker, sha256_text, stable_id
from xifeng_rag.domain import (
    GeneratedTopicProposal,
    ProposalStatus,
    Topic,
    TopicProposal,
    VersionStatus,
    validate_topic_coverage,
)
from xifeng_rag.storage import (
    ApprovedManifestRow,
    ArticleRow,
    ArticleVersionRow,
    ChunkProposalRow,
    CleanupDecisionRow,
    ParagraphRow,
    TopicChunkRow,
)
from xifeng_rag.structured_generation import CodexCliClient as CodexCliClient
from xifeng_rag.structured_generation import GenerationResult, StructuredGenerator
from xifeng_rag.subtopics import create_joint_approved_subtopics

PROMPT_VERSION = "topic-segmentation-v8-joint-overview"
SCHEMA_VERSION = "topic-proposal-v4-joint-overview"

DEFAULT_SEGMENTATION_INSTRUCTION = """你为一篇付费长文章提出单层话题分块。
正文已被拆成细粒度语义候选单元。

要求：
1. 所有单元必须按原顺序恰好覆盖一次。你只返回每块的 end_unit；程序会推导起点。
2. 选择边界前，先在内部识别全文主旨和顶层提纲。优先查找 Markdown 标题、序号、
   “第一点/第二点/第三点”、作者声明的主题数量、前言和总结等篇章结构信号。
3. 显式篇章结构的优先级高于局部语义变化。前言、并列原则或阶段、总结可以是同级顶层
   话题；案例、类比、数据、原因、推导、细节和过渡默认归入所属顶层话题，不能仅因它们
   可以单独命名就提升为新的顶层话题。
4. 只有出现与现有提纲无关的独立主题时才增加顶层话题。文章没有明确结构信号时，再根据
   主题、论证对象、核心观点或叙事阶段的明显转换推断边界。
5. 输出能完整表达文章顶层结构的最少必要话题块。不预设块数或字符数；短而连贯的文章
   可以只有一个话题。阅读提示、付费说明或单句转折不要被机械拆成多个小话题。
6. 清洗后若仍意外出现文末文章目录、推广区或相关文章链接列表，它不属于正文顶层话题，
   必须与最后一个实质话题分开，不能把目录内容写进正文话题的标题、概括或关键词。
7. 当前只做单层顶层分块。每个顶层话题可以额外输出可选 subtopics，用来记录用户可能单独
   记住的具体案例、局部概念、机制或特殊表述。细话题不要求覆盖父话题；没有合适内容时返回
   空数组。不要把每句话都概括为细话题。
8. end_unit 必须严格递增，最后一个 end_unit 必须等于全文最后一个单元编号。
9. 每个细话题只返回简短 title、精确 start_unit/end_unit 和一句忠实 summary，范围必须位于
   所属顶层话题内。顶层和细话题都不复制原文，也不要把概括伪装为作者原话。
   是否进入索引由用户审核决定，不要输出内容角色或索引决定。"""

JOINT_OVERVIEW_INSTRUCTION = """同时根据本次收到的完整原文独立生成 article_summary：
1. 先说明文章核心判断，再覆盖主要论证路径和对文章反查有帮助的重要延伸主题。
2. 不要根据 topics 中的块级概括机械拼接；article_summary 与 topics 是对完整原文的并列输出。
3. 不补充外部事实，不复制或假装引用作者原话。
4. 使用一段简洁中文，优选 80 至 140 字，最长不超过 200 字，最多 2 至 3 句。
5. 只返回 JSON Schema 要求的 article_summary 和 topics。"""


class TopicCoverageError(ValueError):
    def __init__(self, issues: list[str]) -> None:
        super().__init__("；".join(issues))
        self.issues = issues


def ollama_topic_schema() -> dict[str, object]:
    """Return the conservative JSON Schema subset accepted by Ollama grammars."""
    return {
        "type": "object",
        "properties": {
            "article_summary": {"type": "string"},
            "topics": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "end_unit": {"type": "integer"},
                        "title": {"type": "string"},
                        "summary": {"type": "string"},
                        "keywords": {"type": "array", "items": {"type": "string"}},
                        "subtopics": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string"},
                                    "start_unit": {"type": "integer"},
                                    "end_unit": {"type": "integer"},
                                    "summary": {"type": "string"},
                                },
                                "required": [
                                    "title",
                                    "start_unit",
                                    "end_unit",
                                    "summary",
                                ],
                            },
                        },
                    },
                    "required": [
                        "end_unit",
                        "title",
                        "summary",
                        "keywords",
                        "subtopics",
                    ],
                },
            },
        },
        "required": ["article_summary", "topics"],
    }


class OllamaClient:
    def __init__(self, settings: OllamaSettings) -> None:
        self.settings = settings

    def healthy(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.settings.host}/api/tags", timeout=3) as response:
                return response.status == 200
        except (OSError, urllib.error.URLError):
            return False

    def installed_models(self) -> list[str]:
        with urllib.request.urlopen(f"{self.settings.host}/api/tags", timeout=10) as response:
            payload = json.load(response)
        return [item["name"] for item in payload.get("models", [])]

    def start_server(self) -> subprocess.Popen[bytes] | None:
        if self.healthy():
            return None
        executable = self.settings.executable.resolve()
        if not executable.is_file():
            raise FileNotFoundError(f"Ollama executable 不存在：{executable}")
        env = os.environ.copy()
        env["OLLAMA_MODELS"] = str(self.settings.model_dir.resolve())
        env["OLLAMA_HOST"] = "127.0.0.1:11434"
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        process = subprocess.Popen(
            [str(executable), "serve"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
        for _ in range(30):
            if self.healthy():
                return process
            time.sleep(0.25)
        process.terminate()
        raise RuntimeError("Ollama 服务未能在 7.5 秒内启动")

    def generate(self, prompt: str, schema: dict[str, object]) -> GenerationResult:
        payload = {
            "model": self.settings.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": schema,
            "options": {
                "temperature": 0,
                "num_ctx": self.settings.context_window,
                "num_predict": 4096,
            },
        }
        request = urllib.request.Request(
            f"{self.settings.host}/api/chat",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.settings.timeout_seconds) as response:
            result = json.load(response)
        message = result.get("message", {})
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("Ollama 响应缺少 message.content")
        metadata_keys = {
            "total_duration",
            "load_duration",
            "prompt_eval_count",
            "prompt_eval_duration",
            "eval_count",
            "eval_duration",
            "done_reason",
        }
        return GenerationResult(
            content=content,
            metadata={key: result[key] for key in metadata_keys if key in result},
        )


class OpenAICompatibleClient:
    def __init__(
        self,
        profile: ExternalChunkingProfile,
        model: str,
        api_key: str,
    ) -> None:
        self.profile = profile
        self.model = model
        self.api_key = api_key

    def generate(self, prompt: str, schema: dict[str, object]) -> GenerationResult:
        schema_instruction = (
            "仅返回一个符合以下 JSON Schema 的 JSON 对象，不要输出 Markdown：\n"
            + json.dumps(schema, ensure_ascii=False)
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": schema_instruction},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": 4096,
            "temperature": 0,
            "stream": False,
        }
        request = urllib.request.Request(
            f"{self.profile.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.profile.timeout_seconds) as response:
            result = json.load(response)
        choices = result.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("外部模型响应缺少 choices")
        first = choices[0]
        if not isinstance(first, dict):
            raise ValueError("外部模型响应 choices 格式错误")
        message = first.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise ValueError("外部模型响应缺少 message.content")
        metadata: dict[str, object] = {}
        if isinstance(result.get("usage"), dict):
            metadata["usage"] = result["usage"]
        if isinstance(first.get("finish_reason"), str):
            metadata["finish_reason"] = first["finish_reason"]
        return GenerationResult(content=content, metadata=metadata)


@dataclass(frozen=True)
class ProposalOutcome:
    proposal_id: str
    version_id: str
    warnings: list[str]
    status: ProposalStatus


def _paragraphs(session: Session, version_id: str) -> list[ParagraphRow]:
    return list(
        session.scalars(
            select(ParagraphRow)
            .where(ParagraphRow.version_id == version_id)
            .order_by(ParagraphRow.number)
        ).all()
    )


def _residual_catalog_start(paragraphs: list[ParagraphRow], links_json: str) -> int | None:
    try:
        links = json.loads(links_json)
    except json.JSONDecodeError as exc:
        raise ValueError("文章版本的链接元数据损坏") from exc
    if not isinstance(links, list):
        raise ValueError("文章版本的链接元数据格式错误")
    positions: list[tuple[int, int]] = []
    for item in links:
        if not isinstance(item, dict):
            continue
        source_start = item.get("source_start")
        source_end = item.get("source_end")
        if isinstance(source_start, int) and isinstance(source_end, int):
            positions.append((source_start, source_end))
    for index in range(len(paragraphs)):
        paragraph = paragraphs[index]
        if not is_explicit_catalog_marker(paragraph.text):
            continue
        downstream = [item for item in positions if item[0] >= paragraph.source_start]
        linked_units = sum(
            any(start < unit.source_end and end > unit.source_start for start, end in downstream)
            for unit in paragraphs[index:]
        )
        if len(downstream) >= 5 and linked_units >= 3:
            return paragraph.number
    return None


def segmentation_prompt(
    title: str,
    paragraphs: list[ParagraphRow],
    instruction: str | None = None,
) -> str:
    numbered = "\n\n".join(f"[{item.number}] {item.text}" for item in paragraphs)
    selected_instruction = instruction or DEFAULT_SEGMENTATION_INSTRUCTION
    return f"""{selected_instruction}

{JOINT_OVERVIEW_INSTRUCTION}

文章标题：{title}

带编号单元：
{numbered}
"""


def proposal_warnings(proposal: TopicProposal, paragraphs: list[ParagraphRow]) -> list[str]:
    del proposal, paragraphs
    return []


def _safe_generation_diagnostic(attempt: int, exc: Exception) -> dict[str, object]:
    diagnostic: dict[str, object] = {
        "attempt": attempt,
        "error_type": type(exc).__name__,
    }
    if isinstance(exc, TopicCoverageError):
        diagnostic["issues"] = exc.issues
    elif isinstance(exc, ValidationError):
        diagnostic["issues"] = [
            {
                "location": ".".join(str(part) for part in error["loc"]),
                "type": error["type"],
            }
            for error in exc.errors(include_url=False, include_input=False)
        ]
    return diagnostic


def _retry_instruction(exc: Exception) -> str:
    if isinstance(exc, TopicCoverageError):
        details = "；".join(exc.issues)
        return f"上一次输出的单元覆盖不合法：{details}。请修正边界并重新返回完整 JSON。"
    if isinstance(exc, ValidationError):
        return "上一次输出不符合 JSON Schema。请补齐所有字段并仅返回合法的完整 JSON。"
    return "上一次输出无法处理。请严格按照要求和 JSON Schema 重新返回完整 JSON。"


def _normalize_generated_topics(
    generated: GeneratedTopicProposal, unit_count: int
) -> tuple[TopicProposal, list[dict[str, object]]]:
    expected_start = 1
    normalized_topics: list[Topic] = []
    subtopic_diagnostics: list[dict[str, object]] = []
    for index, topic in enumerate(generated.topics, start=1):
        if topic.end_unit < expected_start:
            raise TopicCoverageError(
                [f"第 {index} 块结束单元 {topic.end_unit} 早于连续起点 {expected_start}"]
            )
        if topic.end_unit > unit_count:
            raise TopicCoverageError([f"第 {index} 块结束单元 {topic.end_unit} 越界"])
        valid_subtopics = []
        for subtopic_index, subtopic in enumerate(topic.subtopics, start=1):
            if subtopic.start_unit < expected_start or subtopic.end_unit > topic.end_unit:
                subtopic_diagnostics.append(
                    {
                        "type": "invalid_subtopic_range",
                        "topic_index": index,
                        "subtopic_index": subtopic_index,
                    }
                )
                continue
            valid_subtopics.append(subtopic)
        normalized_topics.append(
            Topic(
                start_paragraph=expected_start,
                end_paragraph=topic.end_unit,
                title=topic.title,
                summary=topic.summary,
                keywords=topic.keywords,
                subtopics=valid_subtopics,
                include_in_index=True,
            )
        )
        expected_start = topic.end_unit + 1
    if expected_start != unit_count + 1:
        raise TopicCoverageError([f"分块未覆盖到最后单元 {unit_count}"])
    return TopicProposal(topics=normalized_topics), subtopic_diagnostics


def propose_chunks(
    session: Session,
    version: ArticleVersionRow,
    generator: StructuredGenerator,
    model_id: str,
    max_retries: int,
    *,
    provider_id: str = "ollama",
    custom_prompt: str | None = None,
    retry_id: str | None = None,
) -> ProposalOutcome:
    allowed_statuses = {
        VersionStatus.PENDING_CHUNKING.value,
        VersionStatus.PENDING_REVIEW.value,
    }
    if version.status not in allowed_statuses:
        raise ValueError(f"文章版本状态不能生成提案：{version.status}")
    article = session.get(ArticleRow, version.article_id)
    if article is None:
        raise ValueError("文章不存在")
    paragraphs = _paragraphs(session, version.id)
    residual_catalog = _residual_catalog_start(paragraphs, version.links_json)
    if residual_catalog is not None:
        raise ValueError(
            f"规范正文尾部从单元 {residual_catalog} 起疑似残留付费文章目录；"
            "请先重新扫描语料或完成清洗审核"
        )
    selected_instruction = custom_prompt or DEFAULT_SEGMENTATION_INSTRUCTION
    prompt_hash = sha256_text(selected_instruction)
    prompt_version = PROMPT_VERSION if custom_prompt is None else f"custom-{prompt_hash[:12]}"
    identity_parts = [
        "proposal",
        version.id,
        provider_id,
        model_id,
        prompt_hash,
        SCHEMA_VERSION,
    ]
    if retry_id is not None:
        identity_parts.append(retry_id)
    proposal_id = stable_id(*identity_parts, length=32)
    existing = session.get(ChunkProposalRow, proposal_id)
    if existing is not None and existing.status == ProposalStatus.PENDING_REVIEW.value:
        proposal = TopicProposal.model_validate_json(existing.topics_json)
        return ProposalOutcome(
            proposal_id,
            version.id,
            proposal_warnings(proposal, paragraphs),
            ProposalStatus.PENDING_REVIEW,
        )

    row = existing or ChunkProposalRow(
        id=proposal_id,
        version_id=version.id,
        status=ProposalStatus.GENERATING.value,
        provider_id=provider_id,
        model_id=model_id,
        prompt_version=prompt_version,
        prompt_text=custom_prompt,
        prompt_hash=prompt_hash,
        retry_id=retry_id,
        schema_version=SCHEMA_VERSION,
    )
    session.add(row)
    prompt = segmentation_prompt(article.title, paragraphs, selected_instruction)
    diagnostics: list[dict[str, object]] = []
    last_error: Exception | None = None
    attempt_prompt = prompt
    for attempt in range(max_retries + 1):
        try:
            result = generator.generate(attempt_prompt, ollama_topic_schema())
            generated = GeneratedTopicProposal.model_validate_json(result.content)
            proposal, subtopic_diagnostics = _normalize_generated_topics(generated, len(paragraphs))
            diagnostics.extend(subtopic_diagnostics)
            coverage_errors = validate_topic_coverage(proposal, len(paragraphs))
            if coverage_errors:
                raise TopicCoverageError(coverage_errors)
            row.status = ProposalStatus.PENDING_REVIEW.value
            row.topics_json = proposal.model_dump_json()
            generation_metadata = dict(result.metadata)
            row.generation_json = json.dumps(generation_metadata, ensure_ascii=False)
            row.diagnostics_json = json.dumps(diagnostics, ensure_ascii=False)
            session.flush()
            create_joint_article_overview_draft(
                session,
                version_id=version.id,
                proposal_id=proposal_id,
                article_summary=generated.article_summary,
                provider_id=provider_id,
                model_id=model_id,
                generation_metadata=generation_metadata,
            )
            version.status = VersionStatus.PENDING_REVIEW.value
            return ProposalOutcome(
                proposal_id,
                version.id,
                proposal_warnings(proposal, paragraphs),
                ProposalStatus.PENDING_REVIEW,
            )
        except Exception as exc:  # safe diagnostic boundary for model failures
            last_error = exc
            diagnostics.append(_safe_generation_diagnostic(attempt + 1, exc))
            attempt_prompt = f"{prompt}\n\n纠错要求：{_retry_instruction(exc)}"
    row.status = ProposalStatus.FAILED.value
    row.diagnostics_json = json.dumps(diagnostics, ensure_ascii=False)
    return ProposalOutcome(
        proposal_id,
        version.id,
        [f"分块提案在 {max_retries + 1} 次尝试后失败：{type(last_error).__name__}"],
        ProposalStatus.FAILED,
    )


def export_review(
    session: Session,
    version_id: str,
    output_dir: Path,
    proposal_id: str | None = None,
) -> tuple[Path, Path]:
    version = session.get(ArticleVersionRow, version_id)
    if version is None:
        raise ValueError(f"文章版本不存在：{version_id}")
    article = session.get(ArticleRow, version.article_id)
    statement = select(ChunkProposalRow).where(
        ChunkProposalRow.version_id == version_id,
        ChunkProposalRow.status == ProposalStatus.PENDING_REVIEW.value,
    )
    if proposal_id is not None:
        statement = statement.where(ChunkProposalRow.id == proposal_id)
    proposals = list(session.scalars(statement.order_by(ChunkProposalRow.created_at.desc())).all())
    if proposal_id is None and len(proposals) > 1:
        raise ValueError("存在多个待审核提案，请使用 --proposal 明确指定")
    proposal_row = proposals[0] if proposals else None
    if proposal_row is None or article is None:
        raise ValueError("没有可导出的待审核提案")
    proposal = TopicProposal.model_validate_json(proposal_row.topics_json)
    overview = joint_article_overview_for_proposal(session, proposal_row.id)
    paragraphs = _paragraphs(session, version_id)
    by_number = {item.number: item for item in paragraphs}
    payload = {
        "proposal_id": proposal_row.id,
        "article_id": article.id,
        "article_version": version.id,
        "canonical_content_hash": version.canonical_hash,
        "provider_id": proposal_row.provider_id,
        "model_id": proposal_row.model_id,
        "prompt_version": proposal_row.prompt_version,
        "schema_version": proposal_row.schema_version,
        "topics": [topic.model_dump() for topic in proposal.topics],
    }
    if overview is not None:
        payload["article_summary"] = overview.article_summary
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{version.id}.{proposal_row.id}.review"
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [f"# {article.title} 分块审核", "", f"版本：`{version.id}`", ""]
    if overview is not None:
        lines.extend(["## 文章整体概括（AI）", "", overview.article_summary, ""])
    for index, topic in enumerate(proposal.topics, start=1):
        selected = [
            by_number[number].text
            for number in range(topic.start_paragraph, topic.end_paragraph + 1)
        ]
        size = sum(map(len, selected))
        first = selected[0][:160].replace("\n", " ")
        last = selected[-1][-160:].replace("\n", " ")
        lines.extend(
            [
                f"## {index}. {topic.title}",
                "",
                f"- 单元：{topic.start_paragraph}–{topic.end_paragraph}；约 {size} 字符",
                f"- 进入索引：`{str(topic.include_in_index).lower()}`",
                f"- 概括（AI）：{topic.summary}",
                f"- 关键词（AI）：{'、'.join(topic.keywords)}",
                f"- 开头预览（原文）：{first}",
                f"- 结尾预览（原文）：{last}",
                "",
            ]
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def approve_manifest(session: Session, manifest_path: Path) -> str:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    proposal_id = payload.get("proposal_id")
    version_id = payload.get("article_version")
    row = session.get(ChunkProposalRow, proposal_id)
    version = session.get(ArticleVersionRow, version_id)
    if row is None or version is None or row.version_id != version_id:
        raise ValueError("审核文件与提案或文章版本不匹配")
    article = session.get(ArticleRow, version.article_id)
    if not version.active or article is None or not article.active:
        raise ValueError("文章或文章版本已停用，不能批准提案")
    if version.status != VersionStatus.PENDING_REVIEW.value:
        raise ValueError(f"文章版本状态不可批准：{version.status}")
    if payload.get("canonical_content_hash") != version.canonical_hash:
        raise ValueError("审核文件对应的规范正文哈希已变化")
    if row.status != ProposalStatus.PENDING_REVIEW.value:
        raise ValueError(f"提案状态不可批准：{row.status}")
    proposal = TopicProposal(topics=payload.get("topics", []))
    paragraphs = _paragraphs(session, version_id)
    errors = validate_topic_coverage(proposal, len(paragraphs))
    if errors:
        raise ValueError("；".join(errors))
    if not any(topic.include_in_index for topic in proposal.topics):
        raise ValueError("全部话题均设为不索引，不能批准空正文索引")
    overview = joint_article_overview_for_proposal(session, proposal_id)
    joint_summary: str | None = None
    if overview is not None:
        summary_value = payload.get("article_summary")
        if not isinstance(summary_value, str):
            raise ValueError("联合分块审核文件缺少文章整体概括")
        joint_summary = summary_value
    core_proposal = TopicProposal(
        topics=[topic.model_copy(update={"subtopics": []}, deep=True) for topic in proposal.topics]
    )
    manifest_id = stable_id(
        "manifest", proposal_id, sha256_text(core_proposal.model_dump_json()), length=32
    )
    existing_manifest = session.get(ApprovedManifestRow, manifest_id)
    if existing_manifest is not None:
        create_joint_approved_subtopics(
            session,
            version_id=version_id,
            manifest_id=manifest_id,
            source_proposal_id=proposal_id,
            provider_id=row.provider_id,
            model_id=row.model_id,
            proposal=proposal,
        )
        if joint_summary is not None:
            approve_joint_article_overview(
                session,
                proposal_id=proposal_id,
                manifest_id=manifest_id,
                article_summary=joint_summary,
            )
        return manifest_id
    manifest = ApprovedManifestRow(
        id=manifest_id,
        proposal_id=proposal_id,
        version_id=version_id,
        manifest_version=1,
        topics_json=core_proposal.model_dump_json(),
    )
    session.add(manifest)
    session.flush()
    by_number = {item.number: item for item in paragraphs}
    for topic in core_proposal.topics:
        if not topic.include_in_index:
            continue
        original_text = "\n\n".join(
            by_number[number].text
            for number in range(topic.start_paragraph, topic.end_paragraph + 1)
        )
        chunk_id = stable_id(
            "chunk",
            version_id,
            row.prompt_version,
            manifest_id,
            f"{topic.start_paragraph}:{topic.end_paragraph}",
            length=40,
        )
        session.add(
            TopicChunkRow(
                id=chunk_id,
                manifest_id=manifest_id,
                version_id=version_id,
                article_id=version.article_id,
                start_paragraph=topic.start_paragraph,
                end_paragraph=topic.end_paragraph,
                topic_title=topic.title,
                context_summary=topic.summary,
                keywords_json=json.dumps(topic.keywords, ensure_ascii=False),
                original_text=original_text,
            )
        )
    row.status = ProposalStatus.APPROVED.value
    version.status = VersionStatus.APPROVED.value
    create_joint_approved_subtopics(
        session,
        version_id=version_id,
        manifest_id=manifest_id,
        source_proposal_id=proposal_id,
        provider_id=row.provider_id,
        model_id=row.model_id,
        proposal=proposal,
    )
    if joint_summary is not None:
        approve_joint_article_overview(
            session,
            proposal_id=proposal_id,
            manifest_id=manifest_id,
            article_summary=joint_summary,
        )
    return manifest_id


def resolve_cleanup(session: Session, version_id: str, truncate_from_paragraph: int | None) -> str:
    version = session.get(ArticleVersionRow, version_id)
    if version is None or version.status != VersionStatus.NEEDS_CLEANUP_REVIEW.value:
        raise ValueError("文章版本不存在或不处于清洗审核状态")
    paragraphs = _paragraphs(session, version_id)
    if truncate_from_paragraph is None:
        version.status = VersionStatus.PENDING_CHUNKING.value
        session.add(CleanupDecisionRow(version_id=version_id, action="keep_all"))
        return version_id
    if truncate_from_paragraph <= 1 or truncate_from_paragraph > len(paragraphs):
        raise ValueError(f"截断单元必须在 2 到 {len(paragraphs)} 之间")
    kept = paragraphs[: truncate_from_paragraph - 1]
    canonical_text = "\n\n".join(item.text for item in kept)
    canonical_hash = sha256_text(canonical_text)
    new_id = stable_id("version", version.article_id, canonical_hash, version.cleaner_version)
    if session.get(ArticleVersionRow, new_id) is not None:
        raise ValueError(f"人工裁决后的文章版本已存在：{new_id}")
    new_version = ArticleVersionRow(
        id=new_id,
        article_id=version.article_id,
        source_hash=version.source_hash,
        canonical_hash=canonical_hash,
        cleaner_version=version.cleaner_version,
        canonical_text=canonical_text,
        status=VersionStatus.PENDING_CHUNKING.value,
        removed_ranges_json=json.dumps([[kept[-1].source_end, paragraphs[-1].source_end]]),
        warnings_json="[]",
        links_json=version.links_json,
        active=True,
    )
    session.add(new_version)
    session.flush()
    session.add_all(
        [
            ParagraphRow(
                version_id=new_id,
                number=item.number,
                text=item.text,
                source_start=item.source_start,
                source_end=item.source_end,
            )
            for item in kept
        ]
    )
    version.active = False
    version.status = VersionStatus.INACTIVE.value
    session.add(
        CleanupDecisionRow(
            version_id=version_id,
            action="truncate",
            truncate_from_paragraph=truncate_from_paragraph,
        )
    )
    return new_id


def proposal_generated_at() -> str:
    return datetime.now(UTC).isoformat()
