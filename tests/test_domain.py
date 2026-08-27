from xifeng_rag.domain import Topic, TopicProposal, validate_topic_coverage


def topic(start: int, end: int) -> Topic:
    return Topic(
        start_paragraph=start,
        end_paragraph=end,
        title=f"话题 {start}",
        summary="忠实概括",
        keywords=["关键词"],
    )


def test_coverage_accepts_exact_contiguous_ranges() -> None:
    proposal = TopicProposal(topics=[topic(1, 2), topic(3, 4)])
    assert validate_topic_coverage(proposal, 4) == []


def test_coverage_reports_gap_overlap_and_tail() -> None:
    proposal = TopicProposal(topics=[topic(1, 2), topic(2, 3)])
    errors = validate_topic_coverage(proposal, 4)
    assert "应从段落 3 开始" in errors[0]
    assert "未覆盖到最后段落 4" in errors[1]


def test_legacy_content_role_maps_to_indexing_decision() -> None:
    base = {
        "start_paragraph": 1,
        "end_paragraph": 1,
        "title": "旧话题",
        "summary": "旧概括",
        "keywords": ["旧关键词"],
    }
    assert Topic.model_validate({**base, "content_role": "substantive"}).include_in_index
    assert not Topic.model_validate({**base, "content_role": "boilerplate"}).include_in_index
    explicit = Topic.model_validate(
        {**base, "content_role": "boilerplate", "include_in_index": True}
    )
    assert explicit.include_in_index
    assert "content_role" not in explicit.model_dump()
