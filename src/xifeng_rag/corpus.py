from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from xifeng_rag.domain import ParagraphData, VersionStatus
from xifeng_rag.storage import (
    ApprovedManifestRow,
    ArticleRow,
    ArticleVersionRow,
    ParagraphRow,
    utcnow,
)

CLEANER_VERSION = "deterministic-v4-trailing-catalog-backtrack"
LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(([^)\s]+)(?:\s+['\"][^'\"]*['\"])?\)")
DETECTION_LINK_PATTERN = re.compile(r"\[[^\]]*\]\([^)]+\)")
TRAILING_MARKER_PATTERN = re.compile(
    r"(付费文章.{0,12}(目录|合集)|文章目录|备用号|公众号|扫码|二维码|往期.{0,6}文章)",
    re.IGNORECASE,
)
EXPLICIT_CATALOG_PATTERN = re.compile(
    r"((全部|所有)(的)?付费(文章|文)(链接)?|"
    r"付费(文章|文)(链接)?(在哪|目录|合集)|"
    r"付费(文章|文).{0,16}(目录|合集))",
    re.IGNORECASE,
)
MARKDOWN_DECORATION_PATTERN = re.compile(r"[*_~`#]+")
HEADING_PATTERN = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", re.MULTILINE)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_id(prefix: str, *parts: str, length: int = 24) -> str:
    digest = sha256_text("\x1f".join(parts))[:length]
    return f"{prefix}_{digest}"


@dataclass(frozen=True)
class MarkdownLink:
    anchor: str
    url: str
    source_start: int
    source_end: int


@dataclass(frozen=True)
class CleanupResult:
    canonical_text: str
    paragraphs: list[ParagraphData]
    links: list[MarkdownLink]
    removed_ranges: list[tuple[int, int]]
    warnings: list[str]
    needs_review: bool


@dataclass(frozen=True)
class TrailingCatalogAnalysis:
    status: str
    source_start: int | None
    source_end: int | None
    line_number: int | None
    link_count: int
    linked_lines: int
    reason: str | None


@dataclass(frozen=True)
class ScanSummary:
    source_dir: Path
    discovered: int
    excluded: int
    unmatched_excluded_ids: tuple[str, ...]
    added: int
    modified: int
    metadata_updated: int
    unchanged: int
    missing: int
    needs_cleanup_review: int


@dataclass(frozen=True)
class _RawParagraph:
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class ExternalArticleMetadata:
    article_id: str
    captured_at: datetime | None = None
    published_at: datetime | None = None
    character_count: int | None = None


SENTENCE_ENDINGS = frozenset("。！？；?!;")
CLOSING_PUNCTUATION = frozenset("\"'”’」』）》】〕〗")
INDEPENDENT_LINK_PATTERN = re.compile(r"^\s*(?:[-*+]\s+)?\[[^\]]+\]\([^)]+\)\s*$")
MARKDOWN_HEADING_LINE = re.compile(r"^\s{0,3}#{1,6}\s+")
# ASCII ``?``, ``!`` and ``;`` are valid inside URL paths and queries, so they
# stay in the protected match. Chinese sentence terminators remain outside it.
BARE_URL_PATTERN = re.compile(r"https?://[^\s\])}>。！？；]+", re.IGNORECASE)
ORIGINAL_PUBLISH_EPOCH_PATTERN = re.compile(
    r"\bori_create_time\s*:\s*['\"]?(\d{10})['\"]?"
)
PUBLISH_TEXT_PATTERN = re.compile(
    r"(?<!ori_)\bcreate_time\s*:\s*['\"](\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}(?::\d{2})?)['\"]"
)
CHINA_STANDARD_TIME = timezone(timedelta(hours=8))


def _raw_lines(markdown: str) -> list[_RawParagraph]:
    lines: list[_RawParagraph] = []
    for match in re.finditer(r"[^\r\n]+", markdown):
        raw_text = match.group(0)
        leading = len(raw_text) - len(raw_text.lstrip())
        stripped = raw_text.strip()
        if stripped:
            start = match.start() + leading
            lines.append(_RawParagraph(stripped, start, start + len(stripped)))
    return lines


def _sentence_units(line: _RawParagraph) -> list[_RawParagraph]:
    if MARKDOWN_HEADING_LINE.match(line.text) or INDEPENDENT_LINK_PATTERN.fullmatch(line.text):
        return [line]
    units: list[_RawParagraph] = []
    protected_ranges = [(match.start(), match.end()) for match in LINK_PATTERN.finditer(line.text)]
    protected_ranges.extend(
        (match.start(), match.end()) for match in BARE_URL_PATTERN.finditer(line.text)
    )
    start = 0
    index = 0
    while index < len(line.text):
        inside_url = any(start <= index < end for start, end in protected_ranges)
        if line.text[index] not in SENTENCE_ENDINGS or inside_url:
            index += 1
            continue
        end = index + 1
        while end < len(line.text) and line.text[end] in CLOSING_PUNCTUATION:
            end += 1
        part = line.text[start:end]
        leading = len(part) - len(part.lstrip())
        trailing_end = len(part.rstrip())
        if leading < trailing_end:
            units.append(
                _RawParagraph(
                    part[leading:trailing_end],
                    line.start + start + leading,
                    line.start + start + trailing_end,
                )
            )
        start = end
        index = end
    remainder = line.text[start:]
    leading = len(remainder) - len(remainder.lstrip())
    trailing_end = len(remainder.rstrip())
    if leading < trailing_end:
        units.append(
            _RawParagraph(
                remainder[leading:trailing_end],
                line.start + start + leading,
                line.start + start + trailing_end,
            )
        )
    return units


def _replace_links(text: str, source_offset: int) -> tuple[str, list[MarkdownLink]]:
    links: list[MarkdownLink] = []

    def replace(match: re.Match[str]) -> str:
        links.append(
            MarkdownLink(
                anchor=match.group(1).strip(),
                url=match.group(2),
                source_start=source_offset + match.start(),
                source_end=source_offset + match.end(),
            )
        )
        return match.group(1).strip()

    return LINK_PATTERN.sub(replace, text), links


def _catalog_detection_text(text: str) -> str:
    value = MARKDOWN_DECORATION_PATTERN.sub("", text)
    return re.sub(r"\s+", " ", value).strip()


def is_explicit_catalog_marker(text: str) -> bool:
    return bool(EXPLICIT_CATALOG_PATTERN.search(_catalog_detection_text(text)))


def _catalog_link_stats(lines: list[_RawParagraph]) -> tuple[int, int, int]:
    counts = [len(DETECTION_LINK_PATTERN.findall(item.text)) for item in lines]
    return sum(counts), sum(count > 0 for count in counts), max(counts, default=0)


def _has_resumed_body(lines: list[_RawParagraph]) -> bool:
    for item in lines[1:]:
        link_count = len(DETECTION_LINK_PATTERN.findall(item.text))
        residual = DETECTION_LINK_PATTERN.sub(
            "", _catalog_detection_text(item.text)
        ).strip(" ：:，,。；;")
        if not residual or is_explicit_catalog_marker(residual):
            continue
        if TRAILING_MARKER_PATTERN.search(residual):
            continue
        sentence_endings = sum(residual.count(mark) for mark in "。！？；?!;")
        if link_count == 0 and (len(residual) >= 120 or sentence_endings >= 4):
            return True
        if link_count > 0 and len(residual) >= 160:
            return True
    return False


def _analyze_trailing_catalog_lines(raw: list[_RawParagraph]) -> TrailingCatalogAnalysis:
    if not raw:
        return TrailingCatalogAnalysis("none", None, None, None, 0, 0, None)
    ambiguous: tuple[int, int, int, str] | None = None
    high_confidence: list[tuple[int, int, int]] = []

    for index in range(len(raw)):
        text = _catalog_detection_text(raw[index].text)
        explicit = bool(EXPLICIT_CATALOG_PATTERN.search(text))
        weak = bool(TRAILING_MARKER_PATTERN.search(text))
        if not explicit and not weak:
            continue
        suffix = raw[index:]
        link_count, linked_lines, max_line_links = _catalog_link_stats(suffix)
        link_ratio = linked_lines / len(suffix)
        dense_links = (
            link_count >= 5
            and linked_lines >= 3
            and (link_ratio >= 0.45 or (max_line_links >= 5 and link_ratio >= 0.25))
        )
        resumed_body = _has_resumed_body(suffix)
        if explicit and dense_links and not resumed_body:
            high_confidence.append((index, link_count, linked_lines))
            continue
        if (explicit and link_count >= 2) or (weak and link_count >= 2 and link_ratio >= 0.25):
            reason = "catalog-signals-conflict" if resumed_body else "catalog-signals-incomplete"
            if ambiguous is None or index < ambiguous[0]:
                ambiguous = (index, link_count, linked_lines, reason)

    if high_confidence:
        index, link_count, linked_lines = min(high_confidence, key=lambda item: item[0])
        return TrailingCatalogAnalysis(
            "remove",
            raw[index].start,
            raw[-1].end,
            index + 1,
            link_count,
            linked_lines,
            "explicit-marker-and-dense-tail-links",
        )

    tail = raw[-min(20, len(raw)) :]
    tail_link_count, tail_linked_lines, _ = _catalog_link_stats(tail)
    if ambiguous is None and tail_link_count >= 5 and tail_linked_lines >= 3:
        first_linked = next(
            index
            for index, item in enumerate(raw[-min(20, len(raw)) :], start=len(raw) - len(tail))
            if DETECTION_LINK_PATTERN.search(item.text)
        )
        ambiguous = (first_linked, tail_link_count, tail_linked_lines, "dense-tail-links")

    if ambiguous is not None:
        index, link_count, linked_lines, reason = ambiguous
        return TrailingCatalogAnalysis(
            "review",
            raw[index].start,
            raw[-1].end,
            index + 1,
            link_count,
            linked_lines,
            reason,
        )
    return TrailingCatalogAnalysis("none", None, None, None, 0, 0, None)


def analyze_trailing_catalog(markdown: str) -> TrailingCatalogAnalysis:
    return _analyze_trailing_catalog_lines(_raw_lines(markdown))


def clean_markdown(markdown: str) -> CleanupResult:
    raw = _raw_lines(markdown)
    if not raw:
        raise ValueError("Markdown 不包含可索引正文")

    analysis = _analyze_trailing_catalog_lines(raw)
    cutoff = (
        next(
            index
            for index, item in enumerate(raw)
            if analysis.source_start is not None and item.start == analysis.source_start
        )
        if analysis.status == "remove"
        else None
    )
    needs_review = analysis.status == "review"
    warnings: list[str] = []
    if needs_review:
        warnings.append(
            f"尾部单元 {analysis.line_number} 疑似目录或推广区，需要人工确认"
        )

    kept_lines = raw if cutoff is None else raw[:cutoff]
    kept = [unit for line in kept_lines for unit in _sentence_units(line)]
    removed_ranges = [] if cutoff is None else [(raw[cutoff].start, raw[-1].end)]
    paragraphs: list[ParagraphData] = []
    links: list[MarkdownLink] = []
    canonical_parts: list[str] = []
    for block in kept:
        normalized, block_links = _replace_links(block.text, block.start)
        normalized = re.sub(r"[ \t]+", " ", normalized).strip()
        if not normalized:
            continue
        links.extend(block_links)
        canonical_parts.append(normalized)
        paragraphs.append(
            ParagraphData(
                number=len(paragraphs) + 1,
                text=normalized,
                source_start=block.start,
                source_end=block.end,
            )
        )
    if not paragraphs:
        raise ValueError("清洗后没有可索引正文")
    return CleanupResult(
        canonical_text="\n\n".join(canonical_parts),
        paragraphs=paragraphs,
        links=links,
        removed_ranges=removed_ranges,
        warnings=warnings,
        needs_review=needs_review,
    )


def extract_title(markdown: str, fallback: str) -> str:
    match = HEADING_PATTERN.search(markdown)
    return match.group(1).strip() if match else fallback


def discover_markdown(source_dir: Path) -> list[Path]:
    resolved = source_dir.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"语料目录不存在：{resolved}")
    files = sorted(path for path in resolved.glob("*.md") if path.is_file())
    if not files:
        raise FileNotFoundError(f"语料目录没有顶层 Markdown：{resolved}")
    return files


def _optional_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        return parsed.astimezone(UTC).replace(tzinfo=None)
    return parsed


def load_html_published_at(path: Path) -> datetime | None:
    """Read publication metadata from the matching HTML without indexing its body."""
    if not path.is_file():
        return None
    html = path.read_text(encoding="utf-8-sig", errors="replace")
    epoch_match = ORIGINAL_PUBLISH_EPOCH_PATTERN.search(html)
    if epoch_match is not None:
        return (
            datetime.fromtimestamp(int(epoch_match.group(1)), UTC)
            .astimezone(CHINA_STANDARD_TIME)
            .replace(tzinfo=None)
        )
    text_match = PUBLISH_TEXT_PATTERN.search(html)
    if text_match is None:
        return None
    return datetime.fromisoformat(text_match.group(1))


def load_external_article_metadata(
    source_dir: Path,
) -> dict[str, ExternalArticleMetadata]:
    candidates = [source_dir / "index.json", source_dir.parent / "index.json"]
    index_path = next((path for path in candidates if path.is_file()), None)
    if index_path is None:
        return {}
    payload = json.loads(index_path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"外部 index.json 顶层必须是对象：{index_path}")
    mapping: dict[str, ExternalArticleMetadata] = {}
    for key, metadata in payload.items():
        if not isinstance(key, str) or not key.startswith("sn:") or not isinstance(metadata, dict):
            continue
        md_file = metadata.get("md_file")
        if not isinstance(md_file, str) or not md_file.strip():
            continue
        filename = Path(md_file.replace("\\", "/")).name
        existing = mapping.get(filename)
        if existing is not None and existing.article_id != key:
            raise ValueError(f"index.json 对同一 Markdown 给出多个文章 ID：{filename}")
        character_count = metadata.get("chinese_char_count")
        mapping[filename] = ExternalArticleMetadata(
            article_id=key,
            captured_at=_optional_datetime(metadata.get("captured_at")),
            published_at=load_html_published_at(source_dir / Path(filename).with_suffix(".html")),
            character_count=(
                character_count
                if isinstance(character_count, int) and not isinstance(character_count, bool)
                else None
            ),
        )
    return mapping


def load_external_article_ids(source_dir: Path) -> dict[str, str]:
    return {
        filename: metadata.article_id
        for filename, metadata in load_external_article_metadata(source_dir).items()
    }


def scan_corpus(
    session: Session,
    source_dir: Path,
    excluded_article_ids: list[str] | tuple[str, ...] = (),
) -> ScanSummary:
    files = discover_markdown(source_dir)
    external_metadata = load_external_article_metadata(source_dir)
    known = {row.source_path: row for row in session.scalars(select(ArticleRow)).all()}
    seen: set[str] = set()
    excluded_ids = frozenset(excluded_article_ids)
    matched_excluded_ids: set[str] = set()
    added = modified = metadata_updated = unchanged = excluded = needs_review = 0

    for path in files:
        source_path = path.relative_to(source_dir).as_posix()
        seen.add(source_path)
        known_article = known.get(source_path)
        metadata = external_metadata.get(path.name)
        article_id = (
            metadata.article_id
            if metadata is not None
            else (
            known_article.id
            if known_article is not None
            else stable_id("article", source_path.casefold())
            )
        )
        matching_exclusions = excluded_ids.intersection(
            {article_id, known_article.id if known_article is not None else article_id}
        )
        if matching_exclusions:
            matched_excluded_ids.update(matching_exclusions)
            excluded += 1
            if known_article is not None:
                known_article.active = False
                session.execute(
                    update(ArticleVersionRow)
                    .where(ArticleVersionRow.article_id == known_article.id)
                    .values(active=False)
                )
            continue
        source_stat = path.stat()
        source_modified_at = datetime.fromtimestamp(source_stat.st_mtime, UTC).replace(tzinfo=None)
        markdown = path.read_text(encoding="utf-8-sig")
        source_hash = sha256_text(markdown)
        cleaned = clean_markdown(markdown)
        canonical_hash = sha256_text(cleaned.canonical_text)
        article = known.get(source_path)
        metadata_changed = False
        if article is None:
            article = ArticleRow(
                id=article_id,
                source_path=source_path,
                title=extract_title(markdown, path.stem),
                active=True,
                captured_at=metadata.captured_at if metadata is not None else None,
                published_at=metadata.published_at if metadata is not None else None,
                source_modified_at=source_modified_at,
                source_size_bytes=source_stat.st_size,
                source_character_count=(metadata.character_count if metadata is not None else None),
                metadata_synced_at=utcnow(),
            )
            session.add(article)
            added += 1
        else:
            metadata_changed = (
                article.captured_at != (metadata.captured_at if metadata is not None else None)
                or article.published_at
                != (metadata.published_at if metadata is not None else None)
                or article.source_modified_at != source_modified_at
                or article.source_size_bytes != source_stat.st_size
                or article.source_character_count
                != (metadata.character_count if metadata is not None else None)
            )
            article.active = True
            article.title = extract_title(markdown, path.stem)
            article.captured_at = metadata.captured_at if metadata is not None else None
            article.published_at = metadata.published_at if metadata is not None else None
            article.source_modified_at = source_modified_at
            article.source_size_bytes = source_stat.st_size
            article.source_character_count = (
                metadata.character_count if metadata is not None else None
            )
            article.metadata_synced_at = utcnow()

        active_version = session.scalar(
            select(ArticleVersionRow)
            .where(ArticleVersionRow.article_id == article.id, ArticleVersionRow.active.is_(True))
            .order_by(ArticleVersionRow.created_at.desc())
        )
        if (
            active_version is not None
            and active_version.source_hash == source_hash
            and active_version.cleaner_version == CLEANER_VERSION
        ):
            if known_article is not None and metadata_changed:
                metadata_updated += 1
            else:
                unchanged += 1
            continue
        if (
            active_version is not None
            and active_version.canonical_hash == canonical_hash
            and active_version.cleaner_version == CLEANER_VERSION
        ):
            active_version.source_hash = source_hash
            if known_article is not None and metadata_changed:
                metadata_updated += 1
            else:
                unchanged += 1
            continue
        if active_version is not None:
            modified += 1
        version_id = stable_id("version", article.id, canonical_hash, CLEANER_VERSION)
        replaceable_statuses = {
            VersionStatus.NEEDS_CLEANUP_REVIEW.value,
            VersionStatus.PENDING_CHUNKING.value,
            VersionStatus.PENDING_REVIEW.value,
        }
        if (
            active_version is not None
            and active_version.id != version_id
            and active_version.status in replaceable_statuses
        ):
            active_version.active = False
            active_version.status = VersionStatus.INACTIVE.value
        existing_version = session.get(ArticleVersionRow, version_id)
        if existing_version is not None:
            existing_version.active = True
            existing_version.source_hash = source_hash
            if existing_version.status == VersionStatus.INACTIVE.value:
                has_manifest = session.scalar(
                    select(ApprovedManifestRow.id).where(
                        ApprovedManifestRow.version_id == existing_version.id
                    )
                )
                existing_version.status = (
                    VersionStatus.APPROVED.value
                    if has_manifest is not None
                    else (
                        VersionStatus.NEEDS_CLEANUP_REVIEW.value
                        if cleaned.needs_review
                        else VersionStatus.PENDING_CHUNKING.value
                    )
                )
            if active_version is None:
                unchanged += 1
            continue
        status = (
            VersionStatus.NEEDS_CLEANUP_REVIEW
            if cleaned.needs_review
            else VersionStatus.PENDING_CHUNKING
        )
        if cleaned.needs_review:
            needs_review += 1
        version = ArticleVersionRow(
            id=version_id,
            article_id=article.id,
            source_hash=source_hash,
            canonical_hash=canonical_hash,
            cleaner_version=CLEANER_VERSION,
            canonical_text=cleaned.canonical_text,
            status=status.value,
            removed_ranges_json=json.dumps(cleaned.removed_ranges),
            warnings_json=json.dumps(cleaned.warnings, ensure_ascii=False),
            links_json=json.dumps([link.__dict__ for link in cleaned.links], ensure_ascii=False),
            active=True,
        )
        session.add(version)
        session.flush()
        session.add_all(
            [
                ParagraphRow(
                    version_id=version_id,
                    number=item.number,
                    text=item.text,
                    source_start=item.source_start,
                    source_end=item.source_end,
                )
                for item in cleaned.paragraphs
            ]
        )

    missing_paths = set(known) - seen
    if missing_paths:
        session.execute(
            update(ArticleRow).where(ArticleRow.source_path.in_(missing_paths)).values(active=False)
        )
        missing_ids = [known[path].id for path in missing_paths]
        session.execute(
            update(ArticleVersionRow)
            .where(ArticleVersionRow.article_id.in_(missing_ids))
            .values(active=False, status=VersionStatus.INACTIVE.value)
        )
    return ScanSummary(
        source_dir=source_dir.resolve(),
        discovered=len(files),
        excluded=excluded,
        unmatched_excluded_ids=tuple(sorted(excluded_ids - matched_excluded_ids)),
        added=added,
        modified=modified,
        metadata_updated=metadata_updated,
        unchanged=unchanged,
        missing=len(missing_paths),
        needs_cleanup_review=needs_review,
    )
