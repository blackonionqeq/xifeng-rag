from __future__ import annotations

import json
import secrets
import threading
import webbrowser
from collections import defaultdict
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit

from sqlalchemy import func, select

from xifeng_rag.corpus import discover_markdown, scan_corpus
from xifeng_rag.domain import ProposalStatus, VersionStatus
from xifeng_rag.review_ui import (
    IndexBuilder,
    ReviewApplication,
    ReviewHTTPServer,
    create_review_server,
)
from xifeng_rag.storage import (
    ArticleRow,
    ArticleVersionRow,
    ChunkProposalRow,
    Database,
    TopicChunkRow,
)

MAX_REQUEST_BYTES = 64 * 1024
STATIC_DIR = Path(__file__).with_name("corpus_static")


class ScanInProgressError(RuntimeError):
    pass


def _isoformat(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


class CorpusDashboardApplication:
    def __init__(
        self,
        database: Database,
        source_dir: Path,
        excluded_article_ids: list[str] | tuple[str, ...],
        review_dir: Path,
        index_dir: Path,
        index_builder: IndexBuilder | None = None,
        token: str | None = None,
    ) -> None:
        self.database = database
        self.source_dir = source_dir.resolve()
        self.excluded_article_ids = tuple(excluded_article_ids)
        self.review_dir = review_dir
        self.index_dir = index_dir
        self.index_builder = index_builder
        self.token = token or secrets.token_urlsafe(32)
        self._scan_lock = threading.Lock()
        self._child_lock = threading.Lock()
        self._review_servers: list[tuple[ReviewHTTPServer, threading.Thread]] = []

    def articles(self) -> dict[str, object]:
        with self.database.session() as session:
            article_rows = list(session.scalars(select(ArticleRow)).all())
            article_ids = [row.id for row in article_rows]
            versions = (
                list(
                    session.scalars(
                        select(ArticleVersionRow).where(
                            ArticleVersionRow.article_id.in_(article_ids)
                        )
                    ).all()
                )
                if article_ids
                else []
            )
            versions_by_article: dict[str, list[ArticleVersionRow]] = defaultdict(list)
            for version in versions:
                versions_by_article[version.article_id].append(version)

            latest_versions: dict[str, ArticleVersionRow] = {}
            indexed_versions: dict[str, ArticleVersionRow] = {}
            for article_id, candidates in versions_by_article.items():
                latest_versions[article_id] = max(
                    candidates, key=lambda item: (item.created_at, item.id)
                )
                indexed = [
                    item
                    for item in candidates
                    if item.status == VersionStatus.INDEXED.value and item.active
                ]
                if indexed:
                    indexed_versions[article_id] = max(
                        indexed, key=lambda item: (item.created_at, item.id)
                    )

            latest_ids = [version.id for version in latest_versions.values()]
            proposal_rows = (
                list(
                    session.scalars(
                        select(ChunkProposalRow)
                        .where(ChunkProposalRow.version_id.in_(latest_ids))
                        .order_by(ChunkProposalRow.created_at.desc())
                    ).all()
                )
                if latest_ids
                else []
            )
            proposals_by_version: dict[str, list[ChunkProposalRow]] = defaultdict(list)
            for proposal in proposal_rows:
                proposals_by_version[proposal.version_id].append(proposal)

            indexed_ids = [version.id for version in indexed_versions.values()]
            chunk_counts = {
                version_id: count
                for version_id, count in session.execute(
                    select(TopicChunkRow.version_id, func.count(TopicChunkRow.id))
                    .where(TopicChunkRow.version_id.in_(indexed_ids))
                    .group_by(TopicChunkRow.version_id)
                ).all()
            }

        items: list[dict[str, object]] = []
        for article in article_rows:
            latest = latest_versions.get(article.id)
            indexed = indexed_versions.get(article.id)
            proposals = proposals_by_version.get(latest.id, []) if latest is not None else []
            pending_proposals = (
                [
                    proposal
                    for proposal in proposals
                    if proposal.status == ProposalStatus.PENDING_REVIEW.value
                ]
                if article.active
                and latest is not None
                and latest.status == VersionStatus.PENDING_REVIEW.value
                else []
            )
            indexed_chunk_count = chunk_counts.get(indexed.id, 0) if indexed is not None else 0
            items.append(
                {
                    "article_id": article.id,
                    "title": article.title,
                    "source_path": article.source_path,
                    "source_active": article.active,
                    "captured_at": _isoformat(article.captured_at),
                    "source_modified_at": _isoformat(article.source_modified_at),
                    "source_size_bytes": article.source_size_bytes,
                    "source_character_count": article.source_character_count,
                    "first_imported_at": _isoformat(article.created_at),
                    "latest_version_id": latest.id if latest is not None else None,
                    "latest_status": latest.status if latest is not None else "discovered",
                    "indexed_version_id": indexed.id if indexed is not None else None,
                    "indexed_chunk_count": indexed_chunk_count,
                    "proposal_count": len(proposals),
                    "pending_proposals": [
                        {
                            "proposal_id": proposal.id,
                            "provider_id": proposal.provider_id,
                            "model_id": proposal.model_id,
                            "prompt_version": proposal.prompt_version,
                            "created_at": _isoformat(proposal.created_at),
                        }
                        for proposal in pending_proposals
                    ],
                    "has_pending_update": (
                        indexed is not None and latest is not None and indexed.id != latest.id
                    ),
                    "is_indexed": indexed is not None and indexed_chunk_count > 0,
                }
            )

        active_count = sum(bool(item["source_active"]) for item in items)
        indexed_count = sum(
            bool(item["source_active"] and item["is_indexed"]) for item in items
        )
        pending_count = sum(
            bool(
                item["source_active"]
                and (
                    not item["is_indexed"]
                    or item["has_pending_update"]
                    or item["latest_status"] != VersionStatus.INDEXED.value
                )
            )
            for item in items
        )
        return {
            "articles": items,
            "stats": {
                "total": len(items),
                "active": active_count,
                "indexed": indexed_count,
                "pending": pending_count,
                "needs_cleanup_review": sum(
                    item["latest_status"] == VersionStatus.NEEDS_CLEANUP_REVIEW.value
                    for item in items
                    if item["source_active"]
                ),
                "missing": sum(not bool(item["source_active"]) for item in items),
                "coverage_percent": (
                    round(indexed_count * 100 / active_count, 1) if active_count else 0.0
                ),
            },
        }

    def scan(self) -> dict[str, object]:
        if not self._scan_lock.acquire(blocking=False):
            raise ScanInProgressError("已有语料扫描正在进行")
        try:
            discovered = len(discover_markdown(self.source_dir))
            with self.database.session() as session:
                summary = scan_corpus(
                    session, self.source_dir, self.excluded_article_ids
                )
            return {
                "source_dir": str(summary.source_dir),
                "discovered": discovered,
                "excluded": summary.excluded,
                "added": summary.added,
                "modified": summary.modified,
                "metadata_updated": summary.metadata_updated,
                "unchanged": summary.unchanged,
                "missing": summary.missing,
                "needs_cleanup_review": summary.needs_cleanup_review,
                "unmatched_excluded_ids": list(summary.unmatched_excluded_ids),
            }
        finally:
            self._scan_lock.release()

    def start_review(self, body: object) -> dict[str, object]:
        if not isinstance(body, dict):
            raise ValueError("请求必须是对象")
        version_id = body.get("version_id")
        proposal_id = body.get("proposal_id")
        if not isinstance(version_id, str) or not isinstance(proposal_id, str):
            raise ValueError("必须提供文章版本和待审提案")
        with self.database.session() as session:
            version = session.get(ArticleVersionRow, version_id)
            proposal = session.get(ChunkProposalRow, proposal_id)
            if version is None or proposal is None or proposal.version_id != version.id:
                raise ValueError("提案不存在或与文章版本不匹配")
            article = session.get(ArticleRow, version.article_id)
            if article is None or not article.active:
                raise ValueError("文章来源已缺失，不能启动审核")
            if version.status != VersionStatus.PENDING_REVIEW.value:
                raise ValueError("文章版本已不处于待审核状态")
            if proposal.status != ProposalStatus.PENDING_REVIEW.value:
                raise ValueError("提案已不处于待审状态")
            latest = session.scalar(
                select(ArticleVersionRow)
                .where(ArticleVersionRow.article_id == version.article_id)
                .order_by(ArticleVersionRow.created_at.desc(), ArticleVersionRow.id.desc())
            )
            if latest is None or latest.id != version.id:
                raise ValueError("只能审核文章的最新来源版本")

        application = ReviewApplication(
            self.database,
            version_id,
            proposal_id,
            self.review_dir,
            self.index_dir,
            self.index_builder,
        )
        server, url = create_review_server(application)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        with self._child_lock:
            self._review_servers.append((server, thread))
        return {"url": url, "message": "审核页面已启动"}

    def shutdown_children(self) -> None:
        with self._child_lock:
            children = list(self._review_servers)
            self._review_servers.clear()
        for server, thread in children:
            if thread.is_alive():
                server.shutdown()
            server.server_close()
            thread.join(timeout=2)


class CorpusDashboardHTTPServer(ThreadingHTTPServer):
    daemon_threads = True


def create_corpus_dashboard_server(
    application: CorpusDashboardApplication,
) -> tuple[CorpusDashboardHTTPServer, str]:
    server = CorpusDashboardHTTPServer(("127.0.0.1", 0), _handler_for(application))
    host, port = server.server_address[:2]
    return server, f"http://{host}:{port}/?{urlencode({'token': application.token})}"


def run_corpus_dashboard(
    database: Database,
    source_dir: Path,
    excluded_article_ids: list[str] | tuple[str, ...],
    review_dir: Path,
    index_dir: Path,
    index_builder: IndexBuilder | None = None,
    *,
    open_browser: bool = True,
) -> None:
    application = CorpusDashboardApplication(
        database, source_dir, excluded_article_ids, review_dir, index_dir, index_builder
    )
    server, url = create_corpus_dashboard_server(application)
    print(f"本地语料管理地址：{url}")
    print("按 Ctrl+C 停止管理服务。")
    if open_browser and not webbrowser.open(url):
        print("未能自动打开浏览器，请复制上面的地址。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n语料管理服务已停止。")
    finally:
        application.shutdown_children()
        server.server_close()


def _handler_for(
    application: CorpusDashboardApplication,
) -> type[BaseHTTPRequestHandler]:
    class CorpusDashboardHandler(BaseHTTPRequestHandler):
        server_version = "XifengCorpusDashboard/0"

        def do_GET(self) -> None:
            parsed = urlsplit(self.path)
            if parsed.path == "/":
                if not self._query_token_valid(parsed.query):
                    self._json_error(HTTPStatus.FORBIDDEN, "管理令牌无效")
                    return
                self._static_response("index.html", "text/html; charset=utf-8")
                return
            if parsed.path == "/api/articles":
                if not self._token_valid(parsed.query):
                    self._json_error(HTTPStatus.FORBIDDEN, "管理令牌无效")
                    return
                self._json_response(HTTPStatus.OK, application.articles())
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
            if parsed.path not in {"/api/scan", "/api/reviews", "/api/shutdown"}:
                self._json_error(HTTPStatus.NOT_FOUND, "接口不存在")
                return
            if not self._token_valid(parsed.query):
                self._json_error(HTTPStatus.FORBIDDEN, "管理令牌无效")
                return
            try:
                body = self._read_json()
            except ValueError as exc:
                self._json_error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            if parsed.path == "/api/scan":
                try:
                    result = application.scan()
                except ScanInProgressError as exc:
                    self._json_error(HTTPStatus.CONFLICT, str(exc))
                    return
                except Exception as exc:
                    self._json_error(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        f"语料扫描失败：{type(exc).__name__}: {exc}",
                    )
                    return
                self._json_response(HTTPStatus.OK, result)
                return
            if parsed.path == "/api/reviews":
                try:
                    result = application.start_review(body)
                except ValueError as exc:
                    self._json_error(HTTPStatus.CONFLICT, str(exc))
                    return
                self._json_response(HTTPStatus.OK, result)
                return
            self._json_response(
                HTTPStatus.OK, {"message": "语料管理服务已关闭，可以安全关闭此标签页"}
            )
            threading.Thread(target=self.server.shutdown, daemon=True).start()

        def _read_json(self) -> object:
            if self.headers.get_content_type() != "application/json":
                raise ValueError("只接受 application/json")
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise ValueError("Content-Length 无效") from exc
            if length <= 0 or length > MAX_REQUEST_BYTES:
                raise ValueError("请求体为空或过大")
            try:
                return json.loads(self.rfile.read(length))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise ValueError("请求不是有效 JSON") from exc

        def _query_token_valid(self, query: str) -> bool:
            values = parse_qs(query).get("token", [])
            return len(values) == 1 and secrets.compare_digest(
                values[0], application.token
            )

        def _token_valid(self, query: str) -> bool:
            supplied = self.headers.get("X-Corpus-Token", "")
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

    return CorpusDashboardHandler
