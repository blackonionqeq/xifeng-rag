from __future__ import annotations

import json
import secrets
import threading
import time
import webbrowser
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit

from xifeng_rag.answering import answer_with_context, prepare_answer_context
from xifeng_rag.article_search import ArticleSearchResult
from xifeng_rag.domain import (
    AssemblyResult,
    ContextAssemblyConfigSnapshot,
    RetrievalCandidate,
)
from xifeng_rag.generation import Generator, SupplementPolicy, cited_evidence

MAX_REQUEST_BYTES = 32 * 1024
MAX_PREPARED_ITEMS = 20
PREPARED_TTL_SECONDS = 600
MAX_QUESTION_HISTORY = 50
STATIC_DIR = Path(__file__).with_name("demo_static")

ArticleSearcher = Callable[[str, int], tuple[list[ArticleSearchResult], list[str]]]
EvidenceSearcher = Callable[[str, int], list[RetrievalCandidate]]
GeneratorFactory = Callable[[], Generator]


@dataclass(frozen=True)
class PreparedQuestion:
    question: str
    context: AssemblyResult
    expires_at: float


class QuestionHistoryStore:
    def __init__(self, path: Path, *, limit: int = MAX_QUESTION_HISTORY) -> None:
        self.path = path
        self.limit = limit
        self._lock = threading.Lock()

    def list(self) -> list[dict[str, str]]:
        with self._lock:
            return self._load_locked()

    def record(self, question: str) -> list[dict[str, str]]:
        with self._lock:
            items = [item for item in self._load_locked() if item["question"] != question]
            items.insert(
                0,
                {
                    "question": question,
                    "last_used_at": datetime.now(UTC).isoformat(),
                },
            )
            items = items[: self.limit]
            self._write_locked(items)
            return items

    def delete(self, question: str) -> list[dict[str, str]]:
        with self._lock:
            items = [item for item in self._load_locked() if item["question"] != question]
            self._write_locked(items)
            return items

    def clear(self) -> list[dict[str, str]]:
        with self._lock:
            self._write_locked([])
            return []

    def _load_locked(self) -> list[dict[str, str]]:
        if not self.path.is_file():
            return []
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("最近问题文件格式无效")
        items: list[dict[str, str]] = []
        for raw in payload:
            if not isinstance(raw, dict):
                raise ValueError("最近问题文件格式无效")
            question = raw.get("question")
            last_used_at = raw.get("last_used_at")
            if not isinstance(question, str) or not isinstance(last_used_at, str):
                raise ValueError("最近问题文件格式无效")
            items.append({"question": question, "last_used_at": last_used_at})
        return items[: self.limit]

    def _write_locked(self, items: list[dict[str, str]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temporary.write_text(
            json.dumps(items, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)


class DemoApplication:
    def __init__(
        self,
        *,
        article_searcher: ArticleSearcher,
        evidence_searcher: EvidenceSearcher,
        generator_factory: GeneratorFactory,
        assembly_config: ContextAssemblyConfigSnapshot,
        provider_name: str,
        provider_url: str,
        model_id: str,
        question_history: QuestionHistoryStore | None = None,
        token: str | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.article_searcher = article_searcher
        self.evidence_searcher = evidence_searcher
        self.generator_factory = generator_factory
        self.assembly_config = assembly_config
        self.provider_name = provider_name
        self.provider_url = provider_url
        self.model_id = model_id
        self.question_history = question_history
        self.token = token or secrets.token_urlsafe(32)
        self.clock = clock
        self._prepared: dict[str, PreparedQuestion] = {}
        self._prepared_lock = threading.Lock()
        self._generation_lock = threading.Lock()

    @staticmethod
    def _query(value: object, *, label: str) -> str:
        query = str(value or "").strip()
        if not query:
            raise ValueError(f"{label}不能为空")
        if len(query) > 2000:
            raise ValueError(f"{label}不能超过 2000 字符")
        return query

    def search_articles(self, body: dict[str, object]) -> dict[str, object]:
        query = self._query(body.get("query"), label="主题描述")
        raw_top_k = body.get("top_k", 5)
        if not isinstance(raw_top_k, (str, int)):
            raise ValueError("top_k 必须是整数")
        top_k = int(raw_top_k)
        if not 1 <= top_k <= 10:
            raise ValueError("top_k 必须在 1 到 10 之间")
        results, activated_layers = self.article_searcher(query, top_k)
        return {
            "query": query,
            "strategy": "disagreement_cascade",
            "activated_layers": activated_layers,
            "results": [asdict(item) for item in results],
        }

    def prepare_answer(self, body: dict[str, object]) -> dict[str, object]:
        question = self._query(body.get("question"), label="问题")
        candidates = self.evidence_searcher(question, 5)
        context = prepare_answer_context(candidates, self.assembly_config)
        if self.question_history is not None:
            self.question_history.record(question)
        now = self.clock()
        prepared_id = secrets.token_urlsafe(32)
        prepared = PreparedQuestion(
            question=question,
            context=context,
            expires_at=now + PREPARED_TTL_SECONDS,
        )
        with self._prepared_lock:
            self._purge_locked(now)
            if len(self._prepared) >= MAX_PREPARED_ITEMS:
                oldest = min(self._prepared, key=lambda key: self._prepared[key].expires_at)
                del self._prepared[oldest]
            self._prepared[prepared_id] = prepared
        return {
            "prepared_id": prepared_id,
            "expires_in_seconds": PREPARED_TTL_SECONDS,
            "question": question,
            "context_status": context.status.value,
            "provider": self.provider_name,
            "provider_url": self.provider_url,
            "model": self.model_id,
            "evidence_count": context.usage.selected_count,
            "original_text_chars": context.usage.original_text_chars,
            "max_external_requests": 2 if context.evidence_items else 0,
            "will_call_external_generator": bool(context.evidence_items),
            "evidence": [self._evidence_preview(item) for item in context.evidence_items],
        }

    def list_questions(self) -> dict[str, object]:
        questions = self.question_history.list() if self.question_history is not None else []
        return {"questions": questions}

    def delete_question(self, body: dict[str, object]) -> dict[str, object]:
        question = self._query(body.get("question"), label="问题")
        questions = (
            self.question_history.delete(question) if self.question_history is not None else []
        )
        return {"questions": questions}

    def clear_questions(self) -> dict[str, object]:
        questions = self.question_history.clear() if self.question_history is not None else []
        return {"questions": questions}

    @staticmethod
    def _evidence_preview(item) -> dict[str, object]:
        preview = item.original_text[:240].replace("\n", " ")
        if len(item.original_text) > 240:
            preview += "…"
        return {
            "citation_id": item.citation_id,
            "article_title": item.article_title,
            "article_version": item.article_version,
            "paragraph_range": list(item.selected_paragraph_range),
            "preview": preview,
        }

    def answer(self, body: dict[str, object]) -> dict[str, object]:
        prepared_id = str(body.get("prepared_id", ""))
        if not prepared_id:
            raise ValueError("prepared_id 不能为空")
        now = self.clock()
        with self._prepared_lock:
            self._purge_locked(now)
            prepared = self._prepared.pop(prepared_id, None)
        if prepared is None:
            raise ValueError("准备结果无效、已使用或已过期，请重新检索")
        if not self._generation_lock.acquire(blocking=False):
            raise RuntimeError("已有回答正在生成，请稍后再试")
        try:
            started = time.perf_counter()
            generator = self.generator_factory() if prepared.context.evidence_items else None
            result = answer_with_context(
                prepared.question,
                prepared.context,
                generator,
                supplement=SupplementPolicy.OFF,
            )
            latency_ms = (time.perf_counter() - started) * 1000
            sources = [
                {
                    "citation_id": item.citation_id,
                    "article_id": item.article_id,
                    "article_title": item.article_title,
                    "article_version": item.article_version,
                    "paragraph_range": list(item.selected_paragraph_range),
                    "original_text": item.original_text,
                }
                for item in cited_evidence(result.answer, prepared.context)
            ]
            usage = result.generation_metadata.get("usage", {})
            return {
                "answer": result.answer.model_dump(mode="json"),
                "sources": sources,
                "request_count": result.request_count,
                "provider": result.provider_id,
                "model": result.model_id,
                "latency_ms": latency_ms,
                "usage": usage if isinstance(usage, dict) else {},
            }
        finally:
            self._generation_lock.release()

    def _purge_locked(self, now: float) -> None:
        self._prepared = {
            key: value for key, value in self._prepared.items() if value.expires_at > now
        }

    def can_shutdown(self) -> bool:
        return not self._generation_lock.locked()

    def clear(self) -> None:
        with self._prepared_lock:
            self._prepared.clear()


class DemoHTTPServer(ThreadingHTTPServer):
    daemon_threads = True


def _handler_for(application: DemoApplication) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "XifengDemo/0"

        def log_message(self, format: str, *args: object) -> None:
            del format, args

        def _headers(self, status: HTTPStatus, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'none'; script-src 'self'; style-src 'self'; "
                "connect-src 'self'; img-src 'self'; base-uri 'none'; frame-ancestors 'none'",
            )
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()

        def _json(self, value: object, status: HTTPStatus = HTTPStatus.OK) -> None:
            payload = json.dumps(value, ensure_ascii=False).encode()
            self._headers(status, "application/json; charset=utf-8")
            self.wfile.write(payload)

        def _authorized(self) -> bool:
            parsed = urlsplit(self.path)
            query_token = parse_qs(parsed.query).get("token", [None])[0]
            return secrets.compare_digest(
                self.headers.get("X-Demo-Token") or query_token or "", application.token
            )

        def _body(self) -> dict[str, object]:
            if self.headers.get_content_type() != "application/json":
                raise TypeError("请求必须使用 application/json")
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_REQUEST_BYTES:
                raise ValueError("请求体为空或过大")
            value = json.loads(self.rfile.read(length))
            if not isinstance(value, dict):
                raise ValueError("请求体必须是 JSON 对象")
            return value

        def _error(self, exc: Exception) -> None:
            if isinstance(exc, TypeError):
                status = HTTPStatus.UNSUPPORTED_MEDIA_TYPE
            elif isinstance(exc, RuntimeError):
                status = HTTPStatus.CONFLICT
            else:
                status = HTTPStatus.BAD_REQUEST
            self._json({"error": str(exc)}, status)

        def _static(self, name: str, content_type: str) -> None:
            payload = (STATIC_DIR / name).read_bytes()
            self._headers(HTTPStatus.OK, content_type)
            self.wfile.write(payload)

        def do_GET(self) -> None:
            parsed = urlsplit(self.path)
            if parsed.path in {"/", "/index.html"}:
                return self._static("index.html", "text/html; charset=utf-8")
            if parsed.path == "/app.js":
                return self._static("app.js", "text/javascript; charset=utf-8")
            if parsed.path == "/style.css":
                return self._static("style.css", "text/css; charset=utf-8")
            self._json({"error": "接口不存在"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            parsed = urlsplit(self.path)
            if not self._authorized():
                return self._json({"error": "无效令牌"}, HTTPStatus.FORBIDDEN)
            try:
                body = self._body()
                if parsed.path == "/api/articles/search":
                    return self._json(application.search_articles(body))
                if parsed.path == "/api/answers/prepare":
                    return self._json(application.prepare_answer(body))
                if parsed.path == "/api/answers/confirm":
                    return self._json(application.answer(body))
                if parsed.path == "/api/questions/history":
                    return self._json(application.list_questions())
                if parsed.path == "/api/questions/delete":
                    return self._json(application.delete_question(body))
                if parsed.path == "/api/questions/clear":
                    return self._json(application.clear_questions())
                if parsed.path == "/api/shutdown":
                    if not application.can_shutdown():
                        raise RuntimeError("回答生成期间不能关闭演示服务")
                    application.clear()
                    self._json({"message": "演示服务已关闭"})
                    threading.Thread(target=self.server.shutdown, daemon=True).start()
                    return
                self._json({"error": "接口不存在"}, HTTPStatus.NOT_FOUND)
            except Exception as exc:
                self._error(exc)

    return Handler


def create_demo_server(application: DemoApplication) -> tuple[DemoHTTPServer, str]:
    server = DemoHTTPServer(("127.0.0.1", 0), _handler_for(application))
    host, port = server.server_address[:2]
    return server, f"http://{host}:{port}/?{urlencode({'token': application.token})}"


def run_demo_server(application: DemoApplication, *, open_browser: bool = True) -> None:
    server, url = create_demo_server(application)
    print(f"西风研究台演示地址：{url}")
    print("按 Ctrl+C 停止演示服务。")
    if open_browser and not webbrowser.open(url):
        print("未能自动打开浏览器，请复制上面的本地地址。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n演示服务已停止。")
    finally:
        application.clear()
        server.server_close()
