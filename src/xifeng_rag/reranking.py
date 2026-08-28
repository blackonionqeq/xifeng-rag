from __future__ import annotations

import math
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from xifeng_rag.domain import RetrievalCandidate

INPUT_TEMPLATE_VERSION = "article-title-original-text-v1"


@dataclass(frozen=True)
class RerankScore:
    score: float
    truncated: bool


class Reranker(Protocol):
    @property
    def metadata(self) -> Mapping[str, object]: ...

    def rerank(
        self, query: str, candidates: list[RetrievalCandidate]
    ) -> list[RerankScore]: ...

    def close(self) -> None: ...


def _process_memory_bytes() -> int | None:
    if sys.platform != "win32":
        try:
            import resource

            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            multiplier = 1 if sys.platform == "darwin" else 1024
            return int(usage * multiplier)
        except (ImportError, OSError, ValueError):
            return None
    try:
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        process = ctypes.windll.kernel32.GetCurrentProcess()  # type: ignore[attr-defined]
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(  # type: ignore[attr-defined]
            process, ctypes.byref(counters), counters.cb
        )
        return int(counters.WorkingSetSize) if ok else None
    except (AttributeError, OSError, TypeError, ValueError):
        return None


class SentenceTransformersReranker:
    def __init__(
        self,
        model_id: str,
        *,
        revision: str | None = None,
        cache_dir: Path | None = None,
        device: str = "auto",
        batch_size: int = 8,
        max_pair_tokens: int = 1024,
    ) -> None:
        from sentence_transformers import CrossEncoder

        started = time.perf_counter()
        memory_before = _process_memory_bytes()
        resolved_device = None if device == "auto" else device
        effective_cache_dir = None if Path(model_id).is_dir() else cache_dir
        self._model: Any = CrossEncoder(
            model_id,
            revision=revision,
            cache_folder=(
                str(effective_cache_dir) if effective_cache_dir is not None else None
            ),
            device=resolved_device,
            max_length=max_pair_tokens,
        )
        self._batch_size = batch_size
        self._max_pair_tokens = max_pair_tokens
        self._scored_candidate_count = 0
        self._truncated_candidate_count = 0
        loaded_revision = revision
        try:
            loaded_revision = loaded_revision or self._model.model.config._commit_hash
        except AttributeError:
            loaded_revision = loaded_revision or "unresolved"
        try:
            resolved_model_device = str(self._model.model.device)
        except AttributeError:
            resolved_model_device = str(resolved_device or "unresolved")
        memory_after = _process_memory_bytes()
        memory_delta = (
            memory_after - memory_before
            if memory_before is not None and memory_after is not None
            else None
        )
        self._metadata: dict[str, object] = {
            "provider": "sentence-transformers",
            "model_id": model_id,
            "model_revision": loaded_revision,
            "device": resolved_model_device,
            "batch_size": batch_size,
            "max_pair_tokens": max_pair_tokens,
            "candidate_k": None,
            "input_template_version": INPUT_TEMPLATE_VERSION,
            "load_duration_ms": round((time.perf_counter() - started) * 1000, 3),
            "process_memory_delta_bytes": memory_delta,
            "cuda_peak_memory_bytes": None,
        }
        self._reset_cuda_peak_memory()

    @property
    def metadata(self) -> dict[str, object]:
        metadata = dict(self._metadata)
        metadata["cuda_peak_memory_bytes"] = self._cuda_peak_memory()
        metadata["scored_candidate_count"] = self._scored_candidate_count
        metadata["truncated_candidate_count"] = self._truncated_candidate_count
        return metadata

    @staticmethod
    def _passage(candidate: RetrievalCandidate) -> str:
        return f"文章：《{candidate.article_title}》\n{candidate.original_text}"

    def _is_truncated(self, query: str, passage: str) -> bool:
        encoded = self._model.tokenizer(
            query,
            passage,
            add_special_tokens=True,
            truncation=False,
        )
        input_ids = encoded.get("input_ids", [])
        return len(input_ids) > self._max_pair_tokens

    def rerank(
        self, query: str, candidates: list[RetrievalCandidate]
    ) -> list[RerankScore]:
        if not candidates:
            return []
        passages = [self._passage(candidate) for candidate in candidates]
        truncated = [self._is_truncated(query, passage) for passage in passages]
        scores = self._model.predict(
            [(query, passage) for passage in passages],
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        array = np.asarray(scores, dtype=np.float32).reshape(-1)
        if array.shape != (len(candidates),):
            raise ValueError(
                "Reranker 输出数量错误："
                f"期望 {len(candidates)}，实际 {array.shape[0]}"
            )
        if not np.isfinite(array).all():
            raise ValueError("Reranker 输出包含非有限值")
        self._scored_candidate_count += len(candidates)
        self._truncated_candidate_count += sum(truncated)
        return [
            RerankScore(score=float(score), truncated=was_truncated)
            for score, was_truncated in zip(array, truncated, strict=True)
        ]

    def _reset_cuda_peak_memory(self) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except (ImportError, RuntimeError):
            pass

    @staticmethod
    def _cuda_peak_memory() -> int | None:
        try:
            import torch

            return int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        except (ImportError, RuntimeError):
            return None

    def close(self) -> None:
        self._model = None  # type: ignore[assignment]
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass


def rerank_candidates(
    query: str,
    candidates: list[RetrievalCandidate],
    reranker: Reranker,
    top_k: int,
) -> list[RetrievalCandidate]:
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")
    if not candidates:
        return []
    results = reranker.rerank(query, candidates)
    if len(results) != len(candidates):
        raise ValueError(
            f"Reranker 输出数量错误：期望 {len(candidates)}，实际 {len(results)}"
        )
    scored: list[RetrievalCandidate] = []
    for candidate, result in zip(candidates, results, strict=True):
        if not math.isfinite(result.score):
            raise ValueError("Reranker 输出包含非有限值")
        scores = candidate.retrieval_scores.model_copy(
            update={
                "reranker": result.score,
                "reranker_truncated": result.truncated,
            }
        )
        scored.append(candidate.model_copy(update={"retrieval_scores": scores}))
    scored.sort(
        key=lambda item: (
            -float(item.retrieval_scores.reranker or 0.0),
            item.retrieval_scores.rrf_rank or 10**9,
            item.chunk_id,
        )
    )
    ranked: list[RetrievalCandidate] = []
    for rank, candidate in enumerate(scored, start=1):
        scores = candidate.retrieval_scores.model_copy(update={"reranker_rank": rank})
        ranked.append(candidate.model_copy(update={"retrieval_scores": scores}))
    return ranked[:top_k]
