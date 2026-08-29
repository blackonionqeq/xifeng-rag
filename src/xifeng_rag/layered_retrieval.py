from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from xifeng_rag.corpus import stable_id

CHILD_MANIFEST_VERSION = "child-manifest-v1"


@dataclass(frozen=True)
class AtomicUnit:
    number: int
    text: str


@dataclass(frozen=True)
class ParentChunk:
    id: str
    article_id: str
    version_id: str
    start_paragraph: int
    end_paragraph: int


@dataclass(frozen=True)
class ChildChunk:
    id: str
    parent_chunk_id: str
    article_id: str
    article_version: str
    start_paragraph: int
    end_paragraph: int
    original_text: str

    def manifest_entry(self) -> ChildManifestEntry:
        return ChildManifestEntry(
            id=self.id,
            parent_chunk_id=self.parent_chunk_id,
            article_id=self.article_id,
            article_version=self.article_version,
            start_paragraph=self.start_paragraph,
            end_paragraph=self.end_paragraph,
        )


@dataclass(frozen=True)
class ChildManifestEntry:
    id: str
    parent_chunk_id: str
    article_id: str
    article_version: str
    start_paragraph: int
    end_paragraph: int


def _render(units: Sequence[AtomicUnit]) -> str:
    return "\n\n".join(unit.text for unit in units)


def split_parent_chunk(
    parent: ParentChunk,
    units: Sequence[AtomicUnit],
    *,
    splitter_version: str = "atomic-greedy-v1",
    target_chars: int = 400,
    max_chars: int = 600,
    tail_merge_chars: int = 200,
) -> list[ChildChunk]:
    if target_chars <= 0 or max_chars <= 0 or target_chars > max_chars:
        raise ValueError("子块目标长度与上限配置非法")
    if tail_merge_chars < 0 or tail_merge_chars > target_chars:
        raise ValueError("尾块合并阈值配置非法")
    ordered = sorted(units, key=lambda item: item.number)
    expected = list(range(parent.start_paragraph, parent.end_paragraph + 1))
    if [unit.number for unit in ordered] != expected:
        raise ValueError(f"父块原子单元范围不完整：{parent.id}")

    groups: list[list[AtomicUnit]] = []
    current: list[AtomicUnit] = []
    for unit in ordered:
        if not current:
            current = [unit]
            if len(unit.text) > max_chars:
                groups.append(current)
                current = []
            continue
        current_text = _render(current)
        if len(current_text) >= target_chars:
            groups.append(current)
            current = [unit]
            if len(unit.text) > max_chars:
                groups.append(current)
                current = []
            continue
        proposed = _render([*current, unit])
        if len(proposed) > max_chars:
            groups.append(current)
            current = [unit]
            if len(unit.text) > max_chars:
                groups.append(current)
                current = []
        else:
            current.append(unit)
    if current:
        groups.append(current)
    if len(groups) >= 2 and len(_render(groups[-1])) < tail_merge_chars:
        merged = [*groups[-2], *groups[-1]]
        if len(_render(merged)) <= max_chars:
            groups[-2:] = [merged]

    return [
        ChildChunk(
            id=stable_id(
                "child",
                parent.id,
                splitter_version,
                f"{group[0].number}:{group[-1].number}",
                length=40,
            ),
            parent_chunk_id=parent.id,
            article_id=parent.article_id,
            article_version=parent.version_id,
            start_paragraph=group[0].number,
            end_paragraph=group[-1].number,
            original_text=_render(group),
        )
        for group in groups
    ]


def write_child_manifest(path: Path, children: Sequence[ChildChunk]) -> None:
    payload = {
        "version": CHILD_MANIFEST_VERSION,
        "children": [asdict(child.manifest_entry()) for child in children],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_child_manifest(path: Path) -> list[ChildManifestEntry]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != CHILD_MANIFEST_VERSION:
        raise ValueError("子块清单版本不受支持")
    entries = [ChildManifestEntry(**item) for item in payload.get("children", [])]
    if not entries or len({item.id for item in entries}) != len(entries):
        raise ValueError("子块清单为空或包含重复 ID")
    return entries


def validate_candidate_limit(
    children: Sequence[ChildChunk | ChildManifestEntry],
    *,
    parent_candidate_k: int,
    child_candidate_k: int,
    candidate_limit: int,
) -> int:
    counts: dict[str, int] = {}
    for child in children:
        counts[child.parent_chunk_id] = counts.get(child.parent_chunk_id, 0) + 1
    worst_case = sum(sorted(counts.values(), reverse=True)[:parent_candidate_k])
    worst_case += min(child_candidate_k, len(children))
    if worst_case > candidate_limit:
        raise ValueError(
            f"分层候选最坏数量 {worst_case} 超过上限 {candidate_limit}"
        )
    return worst_case
