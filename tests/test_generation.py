from __future__ import annotations

import json
import urllib.error
from email.message import Message

import pytest

from xifeng_rag.config import GeneratorSettings
from xifeng_rag.context_assembly import assemble_context
from xifeng_rag.domain import (
    ContextAssemblyConfigSnapshot,
    RetrievalCandidate,
    RetrievalScores,
)
from xifeng_rag.generation import (
    AnswerGenerationError,
    AnswerStatus,
    DeepSeekClient,
    EvidenceBoundGenerator,
    StatementType,
    SupplementPolicy,
    render_answer,
)
from xifeng_rag.structured_generation import GenerationResult


def context(*, empty: bool = False):
    candidates = []
    if not empty:
        candidates = [
            RetrievalCandidate(
                chunk_id="chunk-1",
                article_id="article-1",
                article_version="version-1",
                article_title="自造文章",
                original_text="作者明确重视长期可持续性。",
                paragraph_range=(2, 3),
                retrieval_scores=RetrievalScores(rrf=0.1, rrf_rank=1),
                source=["bm25", "dense"],
            )
        ]
    return assemble_context(
        candidates,
        ContextAssemblyConfigSnapshot(
            max_evidence_items=5,
            max_total_chars=50_000,
            include_ai_metadata=False,
        ),
    )


def answer_payload(
    *,
    evidence_ids: list[str] | None = None,
    supplement: str | None = None,
    status: str = "answered",
    limitations: list[str] | None = None,
) -> str:
    sections = []
    if status != "insufficient_evidence":
        sections = [
            {
                "text": "作者更看重长期可持续性。",
                "statement_type": "inference",
                "evidence_ids": evidence_ids or ["E1"],
            }
        ]
    return json.dumps(
        {
            "status": status,
            "grounded_sections": sections,
            "supplementary_analysis": supplement,
            "limitations": limitations or [],
        },
        ensure_ascii=False,
    )


class FakeClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.prompts: list[str] = []

    def generate(self, prompt: str, schema: dict[str, object]) -> GenerationResult:
        assert schema["type"] == "object"
        self.prompts.append(prompt)
        return GenerationResult(content=self.responses.pop(0), metadata={"test": True})


class ForbiddenClient:
    def generate(self, prompt: str, schema: dict[str, object]) -> GenerationResult:
        del prompt, schema
        raise AssertionError("无证据时不得调用外部 Generator")


def test_no_evidence_returns_locally_without_external_request() -> None:
    result = EvidenceBoundGenerator(ForbiddenClient()).generate(
        "问题", context(empty=True), SupplementPolicy.AUTO
    )

    assert result.request_count == 0
    assert result.answer.status == AnswerStatus.INSUFFICIENT_EVIDENCE
    assert "未找到" in result.answer.limitations[0]


def test_valid_grounded_answer_is_rendered_with_inference_and_source() -> None:
    result = EvidenceBoundGenerator(FakeClient([answer_payload()])).generate(
        "作者看重什么？", context(), SupplementPolicy.AUTO
    )
    rendered = render_answer(result.answer, context())

    assert result.request_count == 1
    assert result.answer.grounded_sections[0].statement_type == StatementType.INFERENCE
    assert "（基于资料的推断）" in rendered
    assert "[E1]" in rendered
    assert "《自造文章》 version-1 ¶2–3" in rendered
    assert "作者明确重视长期可持续性" in rendered


def test_unknown_citation_is_repaired_once_with_same_evidence() -> None:
    private_body = context().evidence_items[0].original_text
    client = FakeClient(
        [
            answer_payload(evidence_ids=["E9"]),
            answer_payload(evidence_ids=["E1"]),
        ]
    )

    result = EvidenceBoundGenerator(client).generate(
        "问题", context(), SupplementPolicy.OFF
    )

    assert result.request_count == 2
    assert "不存在的证据 ID" in client.prompts[1]
    assert private_body in client.prompts[0]
    assert private_body in client.prompts[1]


def test_two_invalid_outputs_fail_without_exposing_raw_output() -> None:
    private_output = "不能进入错误信息的外部原始响应"
    client = FakeClient([private_output, private_output])

    with pytest.raises(AnswerGenerationError) as raised:
        EvidenceBoundGenerator(client).generate("问题", context(), SupplementPolicy.AUTO)

    assert len(client.prompts) == 2
    assert private_output not in str(raised.value)


@pytest.mark.parametrize(
    ("policy", "supplement", "should_pass"),
    [
        (SupplementPolicy.OFF, None, True),
        (SupplementPolicy.OFF, "模型分析", False),
        (SupplementPolicy.ON, None, False),
        (SupplementPolicy.ON, "可以考虑长期风险。", True),
        (SupplementPolicy.AUTO, None, True),
        (SupplementPolicy.AUTO, "可以考虑长期风险。", True),
    ],
)
def test_supplement_policy_is_enforced(
    policy: SupplementPolicy, supplement: str | None, should_pass: bool
) -> None:
    client = FakeClient([answer_payload(supplement=supplement)] * 2)
    if should_pass:
        result = EvidenceBoundGenerator(client).generate("问题", context(), policy)
        assert result.answer.supplementary_analysis == supplement
    else:
        with pytest.raises(AnswerGenerationError):
            EvidenceBoundGenerator(client).generate("问题", context(), policy)


def test_supplement_cannot_claim_an_evidence_id() -> None:
    client = FakeClient([answer_payload(supplement="参考 E1 的模型分析")] * 2)

    with pytest.raises(AnswerGenerationError, match="模型补充分析不能引用"):
        EvidenceBoundGenerator(client).generate("问题", context(), SupplementPolicy.ON)


def test_partial_and_insufficient_answers_require_limitations() -> None:
    partial = FakeClient(
        [answer_payload(status="partially_answered", limitations=["缺少后续材料"])]
    )
    partial_result = EvidenceBoundGenerator(partial).generate(
        "问题", context(), SupplementPolicy.OFF
    )
    assert partial_result.answer.status == AnswerStatus.PARTIALLY_ANSWERED

    insufficient = FakeClient(
        [
            answer_payload(
                status="insufficient_evidence",
                limitations=["材料与问题无关"],
            )
        ]
    )
    insufficient_result = EvidenceBoundGenerator(insufficient).generate(
        "问题", context(), SupplementPolicy.OFF
    )
    assert insufficient_result.answer.status == AnswerStatus.INSUFFICIENT_EVIDENCE


def test_deepseek_http_errors_are_classified_without_response_body(monkeypatch) -> None:
    private_body = b"private service response"

    def fail(*args, **kwargs):
        del args, kwargs
        raise urllib.error.HTTPError(
            "https://api.deepseek.com/chat/completions",
            401,
            "Unauthorized",
            Message(),
            None,
        )

    monkeypatch.setattr("urllib.request.urlopen", fail)
    client = DeepSeekClient(GeneratorSettings(), "secret-key")

    with pytest.raises(AnswerGenerationError) as raised:
        client.generate("prompt", {"type": "object"})

    assert "HTTP 401" in str(raised.value)
    assert private_body.decode() not in str(raised.value)
    assert "secret-key" not in str(raised.value)


def test_deepseek_model_list_is_sorted(monkeypatch) -> None:
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            del args

        def read(self):
            return b'{"data":[{"id":"model-b"},{"id":"model-a"}]}'

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response())
    client = DeepSeekClient(GeneratorSettings(), "secret-key")

    assert client.list_models() == ["model-a", "model-b"]
