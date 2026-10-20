from __future__ import annotations

from collections.abc import Sequence

from xifeng_rag.context_assembly import assemble_context
from xifeng_rag.domain import (
    AssemblyResult,
    AssemblyStatus,
    ContextAssemblyConfigSnapshot,
    RetrievalCandidate,
)
from xifeng_rag.generation import (
    AnswerGenerationResult,
    AnswerStatus,
    GeneratedAnswer,
    Generator,
    SupplementPolicy,
)


def prepare_answer_context(
    candidates: Sequence[RetrievalCandidate],
    config: ContextAssemblyConfigSnapshot,
) -> AssemblyResult:
    """Build the exact evidence context used by both interactive and evaluated answers."""
    return assemble_context(candidates, config)


def answer_with_context(
    question: str,
    context: AssemblyResult,
    generator: Generator | None,
    *,
    supplement: SupplementPolicy = SupplementPolicy.OFF,
) -> AnswerGenerationResult:
    """Run the production answer policy, including its local no-evidence refusal."""
    if context.status == AssemblyStatus.NO_EVIDENCE:
        return AnswerGenerationResult(
            answer=GeneratedAnswer(
                status=AnswerStatus.INSUFFICIENT_EVIDENCE,
                limitations=["本次检索未找到可用于回答的证据。"],
            ),
            request_count=0,
        )
    if generator is None:
        raise ValueError("有证据时必须提供 Generator")
    return generator.generate(question, context, supplement)
