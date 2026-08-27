from __future__ import annotations

import json
import os
import secrets
import threading
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit

from pydantic import ValidationError
from sqlalchemy import select

from xifeng_rag.chunking import approve_manifest
from xifeng_rag.domain import (
    ProposalStatus,
    TopicProposal,
    VersionStatus,
    validate_topic_coverage,
)
from xifeng_rag.indexing import IndexBuildError, build_index, record_build_failure
from xifeng_rag.storage import (
    ApprovedManifestRow,
    ArticleRow,
    ArticleVersionRow,
    ChunkProposalRow,
    Database,
    ParagraphRow,
)

MAX_REQUEST_BYTES = 256 * 1024
STATIC_DIR = Path(__file__).with_name("review_static")


@dataclass(frozen=True)
class ReviewIdentity:
    proposal_id: str
    article_id: str
    article_version: str
    canonical_content_hash: str
    provider_id: str
    model_id: str
    prompt_version: str
    schema_version: str


@dataclass(frozen=True)
class ReviewDocument:
    identity: ReviewIdentity
    article_title: str
    paragraphs: list[dict[str, object]]
    topics: list[dict[str, object]]

    def as_dict(self) -> dict[str, object]:
        return {
            **self.identity.__dict__,
            "article_title": self.article_title,
            "paragraphs": self.paragraphs,
            "topics": self.topics,
        }


class ApprovalIndexError(RuntimeError):
    def __init__(self, manifest_id: str, message: str) -> None:
        super().__init__(message)
        self.manifest_id = manifest_id


def load_review_document(
    database: Database, version_id: str, proposal_id: str
) -> ReviewDocument:
    with database.session() as session:
        version = session.get(ArticleVersionRow, version_id)
        proposal = session.get(ChunkProposalRow, proposal_id)
        if version is None:
            raise ValueError(f"文章版本不存在：{version_id}")
        if proposal is None or proposal.version_id != version_id:
            raise ValueError("提案不存在或与文章版本不匹配")
        if proposal.status != ProposalStatus.PENDING_REVIEW.value:
            raise ValueError(f"提案状态不可审核：{proposal.status}")
        if version.status != VersionStatus.PENDING_REVIEW.value:
            raise ValueError(f"文章版本状态不可审核：{version.status}")
        article = session.get(ArticleRow, version.article_id)
        if article is None:
            raise ValueError("提案对应的文章不存在")
        paragraphs = list(
            session.scalars(
                select(ParagraphRow)
                .where(ParagraphRow.version_id == version_id)
                .order_by(ParagraphRow.number)
            ).all()
        )
        normalized = TopicProposal.model_validate_json(proposal.topics_json)
        coverage_errors = validate_topic_coverage(normalized, len(paragraphs))
        if coverage_errors:
            raise ValueError("提案覆盖校验失败：" + "；".join(coverage_errors))
        return ReviewDocument(
            identity=ReviewIdentity(
                proposal_id=proposal.id,
                article_id=article.id,
                article_version=version.id,
                canonical_content_hash=version.canonical_hash,
                provider_id=proposal.provider_id,
                model_id=proposal.model_id,
                prompt_version=proposal.prompt_version,
                schema_version=proposal.schema_version,
            ),
            article_title=article.title,
            paragraphs=[{"number": item.number, "text": item.text} for item in paragraphs],
            topics=[topic.model_dump() for topic in normalized.topics],
        )


def save_review_draft(
    database: Database,
    document: ReviewDocument,
    topics_value: object,
    output_dir: Path,
) -> Path:
    try:
        proposal = TopicProposal.model_validate({"topics": topics_value})
    except ValidationError as exc:
        raise ValueError(_validation_message(exc)) from exc
    coverage_errors = validate_topic_coverage(proposal, len(document.paragraphs))
    if coverage_errors:
        raise ValueError("；".join(coverage_errors))
    if not any(topic.include_in_index for topic in proposal.topics):
        raise ValueError("至少需要保留一个进入索引的话题块")

    with database.session() as session:
        version = session.get(ArticleVersionRow, document.identity.article_version)
        row = session.get(ChunkProposalRow, document.identity.proposal_id)
        if version is None or row is None or row.version_id != version.id:
            raise ValueError("文章版本或提案已不存在")
        if row.status != ProposalStatus.PENDING_REVIEW.value:
            raise ValueError(f"提案状态已变化，不能保存草案：{row.status}")
        if version.canonical_hash != document.identity.canonical_content_hash:
            raise ValueError("规范正文哈希已变化，不能保存旧审核草案")

    payload = {
        **document.identity.__dict__,
        "topics": [topic.model_dump() for topic in proposal.topics],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = _safe_draft_filename(
        document.identity.article_version, document.identity.proposal_id
    )
    destination = output_dir / filename
    temporary = output_dir / f".{filename}.{secrets.token_hex(8)}.tmp"
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination.resolve()


def _safe_draft_filename(version_id: str, proposal_id: str) -> str:
    for value in (version_id, proposal_id):
        contains_separator = any(separator in value for separator in ("/", "\\"))
        if not value or Path(value).name != value or contains_separator:
            raise ValueError("文章版本或提案 ID 不能用于草案文件名")
    return f"{version_id}.{proposal_id}.draft.json"


def _validation_message(exc: ValidationError) -> str:
    messages = []
    for error in exc.errors(include_url=False, include_input=False):
        location = ".".join(map(str, error["loc"]))
        messages.append(f"{location}: {error['msg']}")
    return "；".join(messages)


class ReviewApplication:
    def __init__(
        self,
        database: Database,
        version_id: str,
        proposal_id: str,
        output_dir: Path,
        index_dir: Path,
        token: str | None = None,
    ) -> None:
        self.database = database
        self.document = load_review_document(database, version_id, proposal_id)
        self.output_dir = output_dir
        self.index_dir = index_dir
        self.token = token or secrets.token_urlsafe(32)
        self.draft_path: Path | None = None
        self._approval_lock = threading.Lock()
        self._completed = False

    def save(self, body: object) -> Path:
        if not isinstance(body, dict) or "topics" not in body:
            raise ValueError("请求必须包含 topics")
        destination = save_review_draft(
            self.database, self.document, body["topics"], self.output_dir
        )
        self.draft_path = destination
        return destination

    def approve_and_build(self, body: object) -> dict[str, object]:
        if not isinstance(body, dict) or body.get("confirmed") is not True:
            raise ValueError("必须明确确认批准不可变清单并更新索引")
        with self._approval_lock:
            if self.draft_path is None or not self.draft_path.is_file():
                raise ValueError("请先在当前审核会话中成功保存草案")
            with self.database.session() as session:
                existing = session.scalar(
                    select(ApprovedManifestRow).where(
                        ApprovedManifestRow.proposal_id
                        == self.document.identity.proposal_id
                    )
                )
                manifest_id = (
                    existing.id
                    if existing is not None
                    else approve_manifest(session, self.draft_path)
                )
            try:
                with self.database.session() as session:
                    outcome = build_index(session, self.index_dir)
            except IndexBuildError as exc:
                with self.database.session() as session:
                    record_build_failure(session, exc)
                raise ApprovalIndexError(manifest_id, str(exc)) from exc
            except Exception as exc:
                message = f"索引更新失败：{type(exc).__name__}"
                raise ApprovalIndexError(manifest_id, message) from exc
            draft = TopicProposal.model_validate_json(
                self.draft_path.read_text(encoding="utf-8")
            )
            included_count = sum(topic.include_in_index for topic in draft.topics)
            self._completed = True
            return {
                "message": "清单已批准，活动索引已更新",
                "manifest_id": manifest_id,
                "index_id": outcome.build_id,
                "chunk_count": outcome.chunk_count,
                "included_count": included_count,
                "excluded_count": len(draft.topics) - included_count,
                "duration_ms": outcome.duration_ms,
            }

    def ensure_shutdown_allowed(self) -> None:
        if not self._completed:
            raise ValueError("只有批准并成功更新索引后才能关闭审核服务")


class ReviewHTTPServer(ThreadingHTTPServer):
    daemon_threads = True


def create_review_server(
    application: ReviewApplication,
) -> tuple[ReviewHTTPServer, str]:
    handler = _handler_for(application)
    server = ReviewHTTPServer(("127.0.0.1", 0), handler)
    host, port = server.server_address[:2]
    query = urlencode({"token": application.token})
    return server, f"http://{host}:{port}/?{query}"


def run_review_server(
    database: Database,
    version_id: str,
    proposal_id: str,
    output_dir: Path,
    index_dir: Path,
    *,
    open_browser: bool = True,
) -> None:
    application = ReviewApplication(
        database, version_id, proposal_id, output_dir, index_dir
    )
    server, url = create_review_server(application)
    print(f"本地审核地址：{url}")
    print("按 Ctrl+C 停止审核服务。")
    if open_browser and not webbrowser.open(url):
        print("未能自动打开浏览器，请复制上面的地址。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n审核服务已停止。")
    finally:
        server.server_close()


def _handler_for(application: ReviewApplication) -> type[BaseHTTPRequestHandler]:
    class ReviewHandler(BaseHTTPRequestHandler):
        server_version = "XifengReview/0"

        def do_GET(self) -> None:
            parsed = urlsplit(self.path)
            if parsed.path == "/":
                if not self._query_token_valid(parsed.query):
                    self._json_error(HTTPStatus.FORBIDDEN, "审核令牌无效")
                    return
                self._static_response("index.html", "text/html; charset=utf-8")
                return
            if parsed.path == "/api/review":
                if not self._token_valid(parsed.query):
                    self._json_error(HTTPStatus.FORBIDDEN, "审核令牌无效")
                    return
                self._json_response(HTTPStatus.OK, application.document.as_dict())
                return
            static_files = {
                "/static/app.js": ("app.js", "text/javascript; charset=utf-8"),
                "/static/style.css": ("style.css", "text/css; charset=utf-8"),
            }
            selected = static_files.get(parsed.path)
            if selected is not None:
                self._static_response(*selected)
                return
            self._json_error(HTTPStatus.NOT_FOUND, "页面不存在")

        def do_POST(self) -> None:
            parsed = urlsplit(self.path)
            if parsed.path not in {
                "/api/draft",
                "/api/approve-and-index",
                "/api/shutdown",
            }:
                self._json_error(HTTPStatus.NOT_FOUND, "接口不存在")
                return
            if not self._token_valid(parsed.query):
                self._json_error(HTTPStatus.FORBIDDEN, "审核令牌无效")
                return
            if self.headers.get_content_type() != "application/json":
                self._json_error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "只接受 application/json")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            if length <= 0 or length > MAX_REQUEST_BYTES:
                self._json_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "请求体为空或过大")
                return
            try:
                body = json.loads(self.rfile.read(length))
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._json_error(HTTPStatus.BAD_REQUEST, "请求不是有效 JSON")
                return
            if parsed.path == "/api/shutdown":
                try:
                    application.ensure_shutdown_allowed()
                except ValueError as exc:
                    self._json_error(HTTPStatus.CONFLICT, str(exc))
                    return
                self._json_response(
                    HTTPStatus.OK, {"message": "审核服务已关闭，可以安全关闭此标签页"}
                )
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return
            if parsed.path == "/api/approve-and-index":
                try:
                    result = application.approve_and_build(body)
                except ApprovalIndexError as exc:
                    self._json_response(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        {
                            "error": str(exc),
                            "approval_status": "approved",
                            "manifest_id": exc.manifest_id,
                            "retryable": True,
                        },
                    )
                    return
                except ValueError as exc:
                    self._json_error(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc))
                    return
                self._json_response(HTTPStatus.OK, result)
                return
            try:
                destination = application.save(body)
            except ValueError as exc:
                self._json_error(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc))
                return
            except OSError:
                self._json_error(HTTPStatus.INTERNAL_SERVER_ERROR, "草案写入失败")
                return
            self._json_response(
                HTTPStatus.OK, {"path": str(destination), "message": "草案已保存"}
            )

        def _query_token_valid(self, query: str) -> bool:
            values = parse_qs(query).get("token", [])
            return len(values) == 1 and secrets.compare_digest(values[0], application.token)

        def _token_valid(self, query: str) -> bool:
            supplied = self.headers.get("X-Review-Token", "")
            if supplied:
                return secrets.compare_digest(supplied, application.token)
            return self._query_token_valid(query)

        def _static_response(self, filename: str, content_type: str) -> None:
            try:
                content = (STATIC_DIR / filename).read_bytes()
            except OSError:
                self._json_error(HTTPStatus.INTERNAL_SERVER_ERROR, "界面资源不可用")
                return
            self.send_response(HTTPStatus.OK)
            self._security_headers(content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def _json_response(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self._security_headers("application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def _json_error(self, status: HTTPStatus, message: str) -> None:
            self._json_response(status, {"error": message})

        def _security_headers(self, content_type: str) -> None:
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'none'; script-src 'self'; style-src 'self'; "
                "connect-src 'self'; base-uri 'none'; frame-ancestors 'none'",
            )

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    return ReviewHandler
