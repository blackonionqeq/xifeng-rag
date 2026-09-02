from __future__ import annotations

import json
import secrets
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit

from sqlalchemy import select

from xifeng_rag.domain import ArticleSubtopics, SubtopicSetStatus
from xifeng_rag.review_ui import ArticleIndexBuilder
from xifeng_rag.storage import (
    ArticleRow,
    ArticleVersionRow,
    Database,
    ParagraphRow,
    SubtopicSetRow,
)
from xifeng_rag.subtopics import approve_subtopics

MAX_REQUEST_BYTES = 256 * 1024
STATIC_DIR = Path(__file__).with_name("subtopic_review_static")


class SubtopicReviewApplication:
    def __init__(
        self,
        database: Database,
        set_id: str,
        output_dir: Path,
        *,
        article_index_dir: Path | None = None,
        article_index_builder: ArticleIndexBuilder | None = None,
        token: str | None = None,
    ) -> None:
        self.database = database
        self.set_id = set_id
        self.output_dir = output_dir
        self.article_index_dir = article_index_dir
        self.article_index_builder = article_index_builder
        self.token = token or secrets.token_urlsafe(32)
        self.draft_path: Path | None = None
        self._lock = threading.Lock()
        self._approved = False
        self._indexed = False

    def document(self) -> dict[str, object]:
        with self.database.session() as session:
            row = session.get(SubtopicSetRow, self.set_id)
            if row is None:
                raise ValueError("细话题草案不存在")
            version = session.get(ArticleVersionRow, row.version_id)
            article = session.get(ArticleRow, version.article_id) if version else None
            if version is None or article is None:
                raise ValueError("细话题草案对应的文章不存在")
            paragraphs = list(
                session.scalars(
                    select(ParagraphRow)
                    .where(ParagraphRow.version_id == version.id)
                    .order_by(ParagraphRow.number)
                ).all()
            )
            parsed = ArticleSubtopics.model_validate_json(row.subtopics_json)
            return {
                "set_id": row.id,
                "article_title": article.title,
                "article_version": version.id,
                "provider_id": row.provider_id,
                "model_id": row.model_id,
                "status": row.status,
                "paragraphs": [{"number": item.number, "text": item.text} for item in paragraphs],
                "topics": parsed.model_dump()["topics"],
            }

    def save(self, body: object) -> Path:
        if not isinstance(body, dict) or "topics" not in body:
            raise ValueError("请求必须包含 topics")
        parsed = ArticleSubtopics.model_validate({"topics": body["topics"]})
        self.output_dir.mkdir(parents=True, exist_ok=True)
        destination = self.output_dir / f"{self.set_id}.subtopics.draft.json"
        destination.write_text(
            json.dumps(
                {"set_id": self.set_id, **parsed.model_dump()},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        self.draft_path = destination
        return destination

    def approve_and_build(self, body: object) -> dict[str, object]:
        if not isinstance(body, dict) or body.get("confirmed") is not True:
            raise ValueError("必须明确确认批准细话题并更新文章反查索引")
        with self._lock:
            if not self._approved:
                if self.draft_path is None or not self.draft_path.is_file():
                    raise ValueError("请先保存当前细话题草案")
                payload = json.loads(self.draft_path.read_text(encoding="utf-8"))
                with self.database.session() as session:
                    row = approve_subtopics(session, self.set_id, {"topics": payload["topics"]})
                    count = sum(
                        len(parent.subtopics)
                        for parent in ArticleSubtopics.model_validate_json(
                            row.subtopics_json
                        ).topics
                    )
                self._approved = True
            else:
                with self.database.session() as session:
                    row = session.get(SubtopicSetRow, self.set_id)
                    if row is None or row.status != SubtopicSetStatus.APPROVED.value:
                        raise ValueError("细话题批准状态异常")
                    count = sum(
                        len(parent.subtopics)
                        for parent in ArticleSubtopics.model_validate_json(
                            row.subtopics_json
                        ).topics
                    )
            if self.article_index_builder is None or self.article_index_dir is None:
                raise ValueError("文章反查索引构建器未配置")
            with self.database.session() as session:
                outcome = self.article_index_builder(session, self.article_index_dir)
            self._indexed = True
            return {
                "message": "细话题已批准并更新文章反查索引",
                "set_id": self.set_id,
                "subtopic_count": count,
                "article_index_id": outcome.build_id,
            }

    def ensure_shutdown_allowed(self) -> None:
        if not self._indexed:
            raise ValueError("细话题尚未批准并成功更新文章反查索引")


class SubtopicReviewHTTPServer(ThreadingHTTPServer):
    daemon_threads = True


def create_subtopic_review_server(
    application: SubtopicReviewApplication,
) -> tuple[SubtopicReviewHTTPServer, str]:
    server = SubtopicReviewHTTPServer(("127.0.0.1", 0), _handler_for(application))
    host, port = server.server_address[:2]
    return server, f"http://{host}:{port}/?{urlencode({'token': application.token})}"


def _handler_for(
    application: SubtopicReviewApplication,
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "XifengSubtopicReview/0"

        def do_GET(self) -> None:
            parsed = urlsplit(self.path)
            if parsed.path == "/":
                if not self._token_valid(parsed.query):
                    self._json_error(HTTPStatus.FORBIDDEN, "审核令牌无效")
                    return
                self._static("index.html", "text/html; charset=utf-8")
                return
            if parsed.path == "/api/review":
                if not self._token_valid(parsed.query):
                    self._json_error(HTTPStatus.FORBIDDEN, "审核令牌无效")
                    return
                try:
                    payload = application.document()
                except ValueError as exc:
                    self._json_error(HTTPStatus.CONFLICT, str(exc))
                    return
                self._json_response(HTTPStatus.OK, payload)
                return
            static = {
                "/static/app.js": ("app.js", "text/javascript; charset=utf-8"),
                "/static/style.css": ("style.css", "text/css; charset=utf-8"),
            }.get(parsed.path)
            if static:
                self._static(*static)
                return
            self._json_error(HTTPStatus.NOT_FOUND, "页面不存在")

        def do_POST(self) -> None:
            parsed = urlsplit(self.path)
            if parsed.path not in {"/api/draft", "/api/approve", "/api/shutdown"}:
                self._json_error(HTTPStatus.NOT_FOUND, "接口不存在")
                return
            if not self._token_valid(parsed.query):
                self._json_error(HTTPStatus.FORBIDDEN, "审核令牌无效")
                return
            try:
                body = self._read_json()
                if parsed.path == "/api/draft":
                    path = application.save(body)
                    self._json_response(HTTPStatus.OK, {"message": "草案已保存", "path": str(path)})
                elif parsed.path == "/api/approve":
                    self._json_response(HTTPStatus.OK, application.approve_and_build(body))
                else:
                    application.ensure_shutdown_allowed()
                    self._json_response(HTTPStatus.OK, {"message": "审核服务已关闭"})
                    threading.Thread(target=self.server.shutdown, daemon=True).start()
            except ValueError as exc:
                self._json_error(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc))
            except Exception as exc:
                self._json_error(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    f"细话题操作失败：{type(exc).__name__}",
                )

        def _read_json(self) -> object:
            if self.headers.get_content_type() != "application/json":
                raise ValueError("只接受 application/json")
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_REQUEST_BYTES:
                raise ValueError("请求体为空或过大")
            return json.loads(self.rfile.read(length))

        def _token_valid(self, query: str) -> bool:
            supplied = self.headers.get("X-Review-Token", "")
            if supplied:
                return secrets.compare_digest(supplied, application.token)
            values = parse_qs(query).get("token", [])
            return len(values) == 1 and secrets.compare_digest(values[0], application.token)

        def _static(self, filename: str, content_type: str) -> None:
            content = (STATIC_DIR / filename).read_bytes()
            self.send_response(HTTPStatus.OK)
            self._headers(content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def _json_response(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self._headers("application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def _json_error(self, status: HTTPStatus, message: str) -> None:
            self._json_response(status, {"error": message})

        def _headers(self, content_type: str) -> None:
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

    return Handler
