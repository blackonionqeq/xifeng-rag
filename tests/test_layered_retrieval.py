from __future__ import annotations

import pytest

from xifeng_rag.layered_retrieval import (
    AtomicUnit as Unit,
)
from xifeng_rag.layered_retrieval import (
    ParentChunk as Parent,
)
from xifeng_rag.layered_retrieval import (
    split_parent_chunk,
    validate_candidate_limit,
)


def parent(*, end_paragraph: int = 5) -> Parent:
    return Parent("parent-1", "article-1", "version-1", 1, end_paragraph)


def test_splitter_is_deterministic_and_merges_short_tail() -> None:
    units = [
        Unit(index, chr(64 + index) * size)
        for index, size in enumerate([210, 210, 210, 80, 80], 1)
    ]

    first = split_parent_chunk(parent(), units)
    second = split_parent_chunk(parent(), units)

    assert [(item.start_paragraph, item.end_paragraph) for item in first] == [(1, 2), (3, 5)]
    assert [item.id for item in first] == [item.id for item in second]
    assert all(item.parent_chunk_id == "parent-1" for item in first)


def test_splitter_keeps_a_single_overlong_unit_intact() -> None:
    parent_chunk = parent(end_paragraph=2)
    children = split_parent_chunk(parent_chunk, [Unit(1, "甲" * 700), Unit(2, "乙" * 20)])

    assert [(item.start_paragraph, item.end_paragraph) for item in children] == [(1, 1), (2, 2)]
    assert len(children[0].original_text) == 700


def test_splitter_rejects_incomplete_parent_range() -> None:
    with pytest.raises(ValueError, match="范围不完整"):
        split_parent_chunk(parent(), [Unit(1, "甲"), Unit(3, "乙")])


def test_candidate_limit_uses_worst_parent_expansion() -> None:
    units = [Unit(index, "甲" * 210) for index in range(1, 6)]
    first = split_parent_chunk(parent(), units, target_chars=200, max_chars=300)
    second = [
        item.__class__(
            id=f"second-{item.id}",
            parent_chunk_id="parent-2",
            article_id=item.article_id,
            article_version=item.article_version,
            start_paragraph=item.start_paragraph,
            end_paragraph=item.end_paragraph,
            original_text=item.original_text,
        )
        for item in first
    ]

    assert validate_candidate_limit(
        [*first, *second], parent_candidate_k=1, child_candidate_k=2, candidate_limit=7
    ) == 7
    with pytest.raises(ValueError, match="超过上限"):
        validate_candidate_limit(
            [*first, *second], parent_candidate_k=2, child_candidate_k=2, candidate_limit=7
        )
