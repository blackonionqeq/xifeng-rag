from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from xifeng_rag.config import GeneratorSettings
from xifeng_rag.domain import AssemblyResult, AssemblyStatus, EvidenceItem
from xifeng_rag.structured_generation import GenerationResult, StructuredGenerator

GENERATOR_PROMPT_VERSION = "deepseek-grounded-answer-v2"
_CITATION_PATTERN = re.compile(r"(?<![A-Za-z0-9_])E[1-9][0-9]*(?![A-Za-z0-9_])")


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    PARTIALLY_ANSWERED = "partially_answered"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class StatementType(StrEnum):
    SOURCE_STATEMENT = "source_statement"
    INFERENCE = "inference"


class SupplementPolicy(StrEnum):
    AUTO = "auto"
    ON = "on"
    OFF = "off"


class GroundedSection(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    text: str = Field(min_length=1, max_length=8000)
    statement_type: StatementType
    evidence_ids: list[str] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def validate_section(self) -> GroundedSection:
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("同一结论段不能重复引用证据 ID")
        if _CITATION_PATTERN.search(self.text):
            raise ValueError("结论正文不能自行渲染证据 ID")
        return self


class GeneratedAnswer(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    status: AnswerStatus
    grounded_sections: list[GroundedSection] = Field(default_factory=list, max_length=20)
    supplementary_analysis: str | None = Field(default=None, min_length=1, max_length=8000)
    limitations: list[str] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def validate_status_shape(self) -> GeneratedAnswer:
        if any(not limitation.strip() for limitation in self.limitations):
            raise ValueError("资料限制说明不能为空")
        if self.status == AnswerStatus.ANSWERED and not self.grounded_sections:
            raise ValueError("answered 必须包含基于语料的结论")
        if self.status == AnswerStatus.PARTIALLY_ANSWERED:
            if not self.grounded_sections:
                raise ValueError("partially_answered 必须包含可回答部分")
            if not self.limitations:
                raise ValueError("partially_answered 必须说明未回答部分")
        if self.status == AnswerStatus.INSUFFICIENT_EVIDENCE:
            if self.grounded_sections:
                raise ValueError("insufficient_evidence 不能包含语料结论")
            if not self.limitations:
                raise ValueError("insufficient_evidence 必须说明证据不足")
        if self.supplementary_analysis and _CITATION_PATTERN.search(
            self.supplementary_analysis
        ):
            raise ValueError("模型补充分析不能引用本地语料证据 ID")
        return self


class AnswerGenerationResult(BaseModel):
    answer: GeneratedAnswer
    request_count: int = Field(ge=0, le=2)
    provider_id: str | None = None
    model_id: str | None = None
    prompt_version: str = GENERATOR_PROMPT_VERSION
    generation_metadata: dict[str, object] = Field(default_factory=dict)


class Generator(Protocol):
    def generate(
        self,
        question: str,
        context: AssemblyResult,
        supplement: SupplementPolicy,
    ) -> AnswerGenerationResult: ...


class AnswerGenerationError(RuntimeError):
    pass


def _safe_validation_detail(exc: Exception) -> str:
    if isinstance(exc, json.JSONDecodeError):
        return "输出不是合法 JSON"
    if isinstance(exc, ValidationError):
        errors = exc.errors(include_url=False, include_context=False, include_input=False)
        return json.dumps(errors, ensure_ascii=False, separators=(",", ":"))
    return str(exc)


def _validate_evidence_and_policy(
    answer: GeneratedAnswer,
    context: AssemblyResult,
    supplement: SupplementPolicy,
) -> None:
    allowed_ids = {item.citation_id for item in context.evidence_items}
    cited_ids = {
        evidence_id
        for section in answer.grounded_sections
        for evidence_id in section.evidence_ids
    }
    unknown = sorted(cited_ids - allowed_ids)
    if unknown:
        raise ValueError(f"回答引用了不存在的证据 ID：{', '.join(unknown)}")
    if supplement == SupplementPolicy.OFF and answer.supplementary_analysis is not None:
        raise ValueError("supplement=off 时不能生成模型补充分析")
    if supplement == SupplementPolicy.ON and answer.supplementary_analysis is None:
        raise ValueError("supplement=on 时必须生成模型补充分析")


def _parse_answer(
    content: str,
    context: AssemblyResult,
    supplement: SupplementPolicy,
) -> tuple[GeneratedAnswer, dict[str, object]]:
    payload = json.loads(content)
    normalization_metadata: dict[str, object] = {}
    try:
        answer = GeneratedAnswer.model_validate(payload)
    except ValidationError as original_error:
        normalized_payload = _partial_answer_payload(payload)
        if normalized_payload is None:
            raise original_error
        answer = GeneratedAnswer.model_validate(normalized_payload)
        normalization_metadata = {
            "status_normalized": True,
            "original_status": AnswerStatus.INSUFFICIENT_EVIDENCE.value,
            "normalized_status": AnswerStatus.PARTIALLY_ANSWERED.value,
        }
    _validate_evidence_and_policy(answer, context, supplement)
    return answer, normalization_metadata


def _partial_answer_payload(payload: object) -> dict[str, object] | None:
    if not isinstance(payload, dict):
        return None
    sections = payload.get("grounded_sections")
    limitations = payload.get("limitations")
    normalizable = (
        payload.get("status") == AnswerStatus.INSUFFICIENT_EVIDENCE.value
        and isinstance(sections, list)
        and bool(sections)
        and isinstance(limitations, list)
        and bool(limitations)
    )
    if not normalizable:
        return None
    normalized: dict[str, object] = dict(payload)
    normalized["status"] = AnswerStatus.PARTIALLY_ANSWERED.value
    return normalized


def _generation_prompt(
    question: str,
    context: AssemblyResult,
    supplement: SupplementPolicy,
) -> str:
    supplement_instruction = {
        SupplementPolicy.AUTO: (
            "只有问题明确要求评价、建议、预测或你的观点时，才填写 supplementary_analysis；"
            "否则必须为 null。"
        ),
        SupplementPolicy.ON: "必须填写 supplementary_analysis。",
        SupplementPolicy.OFF: "supplementary_analysis 必须为 null。",
    }[supplement]
    return f"""你是本地文章研究台的回答 Generator。请严格遵守以下规则：

1. grounded_sections 只能依据给定 evidence。
   资料明确表达用 source_statement；归纳或判断用 inference。
2. 每个 grounded section 必须绑定实际支持它的 evidence_ids，不得在 text 中自行写引用标记。
3. 资料只支持部分问题时使用 partially_answered；不支持核心问题时使用 insufficient_evidence。
4. 不得使用自身知识填补 grounded_sections 中的事实缺口。
5. {supplement_instruction}
6. 模型补充只能给评价框架、风险、反例或建议，不能引入具体数据、事件、引文或他人观点，
   也不能引用 evidence ID。
7. limitations 只说明资料范围和无法确认的部分。

当前问题：
{question}

本地证据 JSON：
{context.rendered_context}
"""


def _repair_prompt(
    original_prompt: str,
    invalid_content: str,
    validation_detail: str,
) -> str:
    return f"""上一次输出未通过本地结构校验。请只修复 JSON 结构、状态关系和引用约束，
不得新增证据、事实或改变回答依据。

校验错误：{validation_detail}

原始任务：
{original_prompt}

上一次输出：
{invalid_content}
"""


class EvidenceBoundGenerator:
    def __init__(
        self,
        client: StructuredGenerator,
        *,
        provider_id: str | None = None,
        model_id: str | None = None,
    ) -> None:
        self.client = client
        self.provider_id = provider_id
        self.model_id = model_id

    def generate(
        self,
        question: str,
        context: AssemblyResult,
        supplement: SupplementPolicy,
    ) -> AnswerGenerationResult:
        if not question.strip():
            raise ValueError("问题不能为空")
        if context.status == AssemblyStatus.NO_EVIDENCE:
            return AnswerGenerationResult(
                answer=GeneratedAnswer(
                    status=AnswerStatus.INSUFFICIENT_EVIDENCE,
                    limitations=["本次检索未找到可用于回答的证据。"],
                ),
                request_count=0,
            )
        if context.status != AssemblyStatus.READY or not context.evidence_items:
            raise ValueError("上下文状态与证据内容不一致")

        schema = GeneratedAnswer.model_json_schema()
        original_prompt = _generation_prompt(question.strip(), context, supplement)
        prompt = original_prompt
        last_error = "未知校验错误"
        metadata: dict[str, object] = {}
        for request_count in (1, 2):
            result = self.client.generate(prompt, schema)
            metadata = dict(result.metadata)
            try:
                answer, normalization_metadata = _parse_answer(
                    result.content, context, supplement
                )
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                last_error = _safe_validation_detail(exc)
                if request_count == 2:
                    break
                prompt = _repair_prompt(original_prompt, result.content, last_error)
                continue
            metadata.update(normalization_metadata)
            return AnswerGenerationResult(
                answer=answer,
                request_count=request_count,
                provider_id=self.provider_id,
                model_id=self.model_id,
                generation_metadata=metadata,
            )
        raise AnswerGenerationError(f"Generator 输出连续两次未通过校验：{last_error}")


class DeepSeekClient:
    def __init__(self, settings: GeneratorSettings, api_key: str) -> None:
        self.settings = settings
        self.api_key = api_key

    def generate(self, prompt: str, schema: dict[str, object]) -> GenerationResult:
        schema_instruction = (
            "只返回符合以下 JSON Schema 的 JSON 对象，不要输出 Markdown 或额外文字：\n"
            + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
        )
        payload = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": schema_instruction},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": self.settings.max_output_tokens,
            "temperature": self.settings.temperature,
            "stream": False,
        }
        result = self._request_json(
            "POST",
            "/chat/completions",
            payload,
        )
        choices = result.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise AnswerGenerationError("DeepSeek 响应缺少有效 choices")
        first = choices[0]
        message = first.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise AnswerGenerationError("DeepSeek 响应缺少 message.content")
        metadata: dict[str, object] = {}
        if isinstance(result.get("usage"), dict):
            metadata["usage"] = result["usage"]
        if isinstance(first.get("finish_reason"), str):
            metadata["finish_reason"] = first["finish_reason"]
        return GenerationResult(content=content, metadata=metadata)

    def list_models(self) -> list[str]:
        result = self._request_json("GET", "/models")
        models = result.get("data")
        if not isinstance(models, list):
            raise AnswerGenerationError("DeepSeek 模型列表响应格式错误")
        return sorted(
            item["id"]
            for item in models
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        )

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, object] | None = None,
    ) -> dict[str, object]:
        data = (
            json.dumps(payload, ensure_ascii=False).encode("utf-8")
            if payload is not None
            else None
        )
        request = urllib.request.Request(
            f"{self.settings.base_url.rstrip('/')}{path}",
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method=method,
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.settings.timeout_seconds
            ) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            messages = {
                400: "DeepSeek 拒绝了请求格式（HTTP 400）",
                401: "DeepSeek API 密钥无效（HTTP 401）",
                402: "DeepSeek 账户余额不足（HTTP 402）",
                422: "DeepSeek 请求参数无效（HTTP 422）",
                429: "DeepSeek 请求达到速率限制（HTTP 429）",
                500: "DeepSeek 服务内部错误（HTTP 500）",
                503: "DeepSeek 服务暂不可用（HTTP 503）",
            }
            raise AnswerGenerationError(
                messages.get(exc.code, f"DeepSeek API 请求失败（HTTP {exc.code}）")
            ) from exc
        except (TimeoutError, urllib.error.URLError) as exc:
            raise AnswerGenerationError("DeepSeek API 连接失败或超时") from exc
        if not isinstance(result, dict):
            raise AnswerGenerationError("DeepSeek 响应不是 JSON 对象")
        return result


def cited_evidence(answer: GeneratedAnswer, context: AssemblyResult) -> list[EvidenceItem]:
    cited_ids = {
        evidence_id
        for section in answer.grounded_sections
        for evidence_id in section.evidence_ids
    }
    return [item for item in context.evidence_items if item.citation_id in cited_ids]


def render_answer(answer: GeneratedAnswer, context: AssemblyResult) -> str:
    lines: list[str] = []
    if answer.status == AnswerStatus.INSUFFICIENT_EVIDENCE:
        lines.extend(["## 根据现有资料", "", "现有资料不足以回答这个问题。"])
    elif answer.grounded_sections:
        lines.extend(["## 根据现有资料", ""])
        for section in answer.grounded_sections:
            prefix = (
                "（基于资料的推断）"
                if section.statement_type == StatementType.INFERENCE
                else ""
            )
            citations = "".join(f"[{item}]" for item in section.evidence_ids)
            lines.append(f"{prefix}{section.text} {citations}".strip())
            lines.append("")
        if lines[-1] == "":
            lines.pop()

    if answer.supplementary_analysis:
        lines.extend(
            [
                "",
                "## 模型补充分析",
                "",
                "> 以下内容属于模型分析，不完全来自本地语料，也不代表作者观点。",
                "",
                answer.supplementary_analysis,
            ]
        )
    if answer.limitations:
        lines.extend(["", "## 资料限制", ""])
        lines.extend(f"- {item}" for item in answer.limitations)

    sources = cited_evidence(answer, context)
    if sources:
        lines.extend(["", "## 引用来源", ""])
        for item in sources:
            start, end = item.selected_paragraph_range
            excerpt = item.original_text[:240].replace("\n", " ")
            if len(item.original_text) > 240:
                excerpt += "…"
            lines.append(
                f"- [{item.citation_id}] 《{item.article_title}》 {item.article_version} "
                f"¶{start}–{end}：{excerpt}"
            )
    return "\n".join(lines).strip()
