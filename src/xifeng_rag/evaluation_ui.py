from __future__ import annotations

import hashlib
import json
import secrets
import threading
import time
import webbrowser
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit

from sqlalchemy import select

from xifeng_rag.config import CodexCliSettings
from xifeng_rag.evaluation_curation import (
    DatasetPurpose,
    QuestionDraft,
    ReviewedAnswer,
    apply_ai_proposal,
    create_dataset,
    freeze_dataset,
    generate_ai_proposal,
    indexed_articles,
    question_document,
    save_question,
    save_reviewed_answer,
)
from xifeng_rag.storage import (
    Database,
    EvaluationDatasetRow,
    EvaluationQuestionRow,
)
from xifeng_rag.structured_generation import CodexCliClient

MAX_REQUEST_BYTES = 512 * 1024
STATIC_DIR = Path(__file__).with_name("evaluation_static")


def _question_fingerprint(document: dict[str, object], model: str) -> str:
    question = document["question"]
    article = document["article"]
    assert isinstance(question, dict) and isinstance(article, dict)
    value = "\n".join(
        [str(question["id"]), str(question["query"]), str(article["version_id"]), model]
    )
    return hashlib.sha256(value.encode()).hexdigest()


class EvaluationCurationApplication:
    def __init__(
        self,
        database: Database,
        codex_settings: CodexCliSettings,
        token: str | None = None,
        generator_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.database = database
        self.codex_settings = codex_settings
        self.model = codex_settings.default_model
        self.token = token or secrets.token_urlsafe(32)
        self._generator_factory = generator_factory or (
            lambda: CodexCliClient(self.codex_settings, self.model)
        )
        self._confirmations: dict[str, tuple[str, str, float]] = {}
        self._confirmation_lock = threading.Lock()
        self._job_lock = threading.Lock()
        self._job: dict[str, object] = {"status": "idle"}

    def state(self) -> dict[str, object]:
        with self.database.session() as session:
            datasets = list(
                session.scalars(
                    select(EvaluationDatasetRow).order_by(EvaluationDatasetRow.created_at.desc())
                ).all()
            )
            questions = list(
                session.scalars(
                    select(EvaluationQuestionRow).order_by(
                        EvaluationQuestionRow.dataset_id, EvaluationQuestionRow.ordinal
                    )
                ).all()
            )
            counts: dict[str, dict[str, int]] = {}
            for question in questions:
                item = counts.setdefault(question.dataset_id, {"total": 0, "reviewed": 0})
                item["total"] += 1
                item["reviewed"] += int(question.review_status == "reviewed")
            return {
                "datasets": [
                    {
                        "id": item.id,
                        "name": item.name,
                        "version": item.version,
                        "purpose": item.purpose,
                        "status": item.status,
                        "content_hash": item.content_hash,
                        "frozen_at": item.frozen_at.isoformat() if item.frozen_at else None,
                        "question_count": counts.get(item.id, {}).get("total", 0),
                        "reviewed_count": counts.get(item.id, {}).get("reviewed", 0),
                    }
                    for item in datasets
                ],
                "questions": [
                    {
                        "id": item.id,
                        "dataset_id": item.dataset_id,
                        "ordinal": item.ordinal,
                        "query": item.query,
                        "split": item.split,
                        "version_id": item.version_id,
                        "review_status": item.review_status,
                    }
                    for item in questions
                ],
                "articles": indexed_articles(session),
                "ai_job": self.job(),
                "model": self.model,
            }

    def create_dataset(self, body: dict[str, object]) -> dict[str, object]:
        purpose = DatasetPurpose(str(body.get("purpose", "pilot")))
        with self.database.session() as session:
            row = create_dataset(
                session, str(body.get("name", "")), str(body.get("version", "")), purpose
            )
            return {"id": row.id, "status": row.status}

    def save_question(self, body: dict[str, object]) -> dict[str, object]:
        self._reject_mutation_while_running()
        dataset_id = str(body.get("dataset_id", ""))
        question_id = str(body["question_id"]) if body.get("question_id") else None
        value = QuestionDraft.model_validate(body)
        with self.database.session() as session:
            row = save_question(session, dataset_id, value, question_id)
            return {"id": row.id, "review_status": row.review_status}

    def document(self, question_id: str) -> dict[str, object]:
        with self.database.session() as session:
            return question_document(session, question_id)

    def save_review(self, question_id: str, body: dict[str, object]) -> dict[str, object]:
        self._reject_mutation_while_running()
        value = ReviewedAnswer.model_validate(body)
        with self.database.session() as session:
            row = save_reviewed_answer(session, question_id, value)
            return {"id": row.id, "review_status": row.review_status}

    def disclosure(self, question_id: str) -> dict[str, object]:
        with self.database.session() as session:
            document = question_document(session, question_id)
            dataset = document["dataset"]
            if not isinstance(dataset, dict) or dataset["status"] != "draft":
                raise ValueError("冻结评测集不能调用 AI")
        fingerprint = _question_fingerprint(document, self.model)
        confirmation = secrets.token_urlsafe(32)
        with self._confirmation_lock:
            self._confirmations = {
                key: value
                for key, value in self._confirmations.items()
                if value[2] > time.monotonic()
            }
            self._confirmations[confirmation] = (
                question_id,
                fingerprint,
                time.monotonic() + 600,
            )
        question = document["question"]
        article = document["article"]
        assert isinstance(question, dict) and isinstance(article, dict)
        return {
            "confirmation": confirmation,
            "question": question["query"],
            "article_title": article["title"],
            "article_id": article["id"],
            "article_version": article["version_id"],
            "canonical_hash": article["canonical_hash"],
            "character_count": article["character_count"],
            "unit_count": article["unit_count"],
            "provider": "codex-cli",
            "model": self.model,
            "max_requests": 1,
            "disclosure": "将外发问题、文章标题、完整清洗正文及编号原子单元",
        }

    def confirm_ai(self, confirmation: str) -> dict[str, object]:
        if not self._job_lock.acquire(blocking=False):
            raise RuntimeError("已有 AI 任务正在运行")
        try:
            with self._confirmation_lock:
                pending = self._confirmations.pop(confirmation, None)
            if pending is None:
                raise ValueError("确认凭据无效、已使用或已过期")
            question_id, fingerprint, expires_at = pending
            if expires_at <= time.monotonic():
                raise ValueError("确认凭据无效、已使用或已过期")
            with self.database.session() as session:
                document = question_document(session, question_id)
            if _question_fingerprint(document, self.model) != fingerprint:
                raise ValueError("问题或文章已变化，请重新确认")
            job_id = secrets.token_hex(16)
            self._job = {"id": job_id, "status": "running", "question_id": question_id}
            thread = threading.Thread(
                target=self._run_ai,
                args=(job_id, question_id),
                daemon=True,
                name="xifeng-evaluation-codex",
            )
            thread.start()
            return dict(self._job)
        except Exception:
            self._job_lock.release()
            raise

    def _run_ai(self, job_id: str, question_id: str) -> None:
        try:
            generator = self._generator_factory()
            preflight = getattr(generator, "preflight", None)
            if callable(preflight):
                preflight()
            with self.database.session() as session:
                proposal = generate_ai_proposal(session, question_id, generator, self.model)
                proposal_id = proposal.id
            self._job = {
                "id": job_id,
                "status": "succeeded",
                "question_id": question_id,
                "proposal_id": proposal_id,
            }
        except Exception as exc:
            self._job = {
                "id": job_id,
                "status": "failed",
                "question_id": question_id,
                "error_type": type(exc).__name__,
                "error_message": (
                    str(exc) if type(exc) is RuntimeError else "候选生成或校验失败"
                ),
            }
        finally:
            self._job_lock.release()

    def job(self) -> dict[str, object]:
        return dict(self._job)

    def review_proposal(self, proposal_id: str, accept: bool) -> dict[str, object]:
        self._reject_mutation_while_running()
        with self.database.session() as session:
            row = apply_ai_proposal(session, proposal_id, accept)
            return {"id": row.id, "status": row.status, "question_id": row.question_id}

    def freeze(self, dataset_id: str) -> dict[str, object]:
        self._reject_mutation_while_running()
        with self.database.session() as session:
            outcome = freeze_dataset(session, dataset_id)
            return {
                "dataset_id": outcome.dataset_id,
                "content_hash": outcome.content_hash,
                "corpus_snapshot": outcome.corpus_snapshot,
                "article_count": outcome.article_count,
                "question_count": outcome.question_count,
            }

    def _reject_mutation_while_running(self) -> None:
        if self._job_lock.locked():
            raise RuntimeError("AI 任务运行期间不能修改或关闭工作台")

    def can_shutdown(self) -> bool:
        return not self._job_lock.locked()


class EvaluationCurationHTTPServer(ThreadingHTTPServer):
    daemon_threads = True


def _handler_for(application: EvaluationCurationApplication) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
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
            body = json.dumps(value, ensure_ascii=False).encode()
            self._headers(status, "application/json; charset=utf-8")
            self.wfile.write(body)

        def _authorized(self) -> bool:
            parsed = urlsplit(self.path)
            query_token = parse_qs(parsed.query).get("token", [None])[0]
            return secrets.compare_digest(
                self.headers.get("X-Evaluation-Token") or query_token or "", application.token
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

        def do_GET(self) -> None:
            parsed = urlsplit(self.path)
            if parsed.path in {"/", "/index.html"}:
                return self._static("index.html", "text/html; charset=utf-8")
            if parsed.path == "/app.js":
                return self._static("app.js", "text/javascript; charset=utf-8")
            if parsed.path == "/style.css":
                return self._static("style.css", "text/css; charset=utf-8")
            if not self._authorized():
                return self._json({"error": "无效令牌"}, HTTPStatus.FORBIDDEN)
            try:
                if parsed.path == "/api/state":
                    return self._json(application.state())
                if parsed.path == "/api/ai-job":
                    return self._json(application.job())
                if parsed.path.startswith("/api/questions/"):
                    return self._json(application.document(parsed.path.rsplit("/", 1)[-1]))
                self._json({"error": "接口不存在"}, HTTPStatus.NOT_FOUND)
            except Exception as exc:
                self._error(exc)

        def _static(self, name: str, content_type: str) -> None:
            body = (STATIC_DIR / name).read_bytes()
            self._headers(HTTPStatus.OK, content_type)
            self.wfile.write(body)

        def do_POST(self) -> None:
            parsed = urlsplit(self.path)
            if not self._authorized():
                return self._json({"error": "无效令牌"}, HTTPStatus.FORBIDDEN)
            try:
                body = self._body()
                if parsed.path == "/api/datasets":
                    return self._json(application.create_dataset(body), HTTPStatus.CREATED)
                if parsed.path == "/api/questions":
                    return self._json(application.save_question(body))
                if parsed.path.endswith("/review") and parsed.path.startswith("/api/questions/"):
                    question_id = parsed.path.split("/")[3]
                    return self._json(application.save_review(question_id, body))
                if parsed.path.endswith("/ai-disclosure") and parsed.path.startswith(
                    "/api/questions/"
                ):
                    question_id = parsed.path.split("/")[3]
                    return self._json(application.disclosure(question_id))
                if parsed.path == "/api/ai-confirm":
                    return self._json(application.confirm_ai(str(body.get("confirmation", ""))))
                if parsed.path.endswith("/decision") and parsed.path.startswith("/api/proposals/"):
                    proposal_id = parsed.path.split("/")[3]
                    return self._json(
                        application.review_proposal(proposal_id, bool(body.get("accept")))
                    )
                if parsed.path.endswith("/freeze") and parsed.path.startswith("/api/datasets/"):
                    dataset_id = parsed.path.split("/")[3]
                    return self._json(application.freeze(dataset_id))
                if parsed.path == "/api/shutdown":
                    if not application.can_shutdown():
                        raise RuntimeError("AI 任务运行期间不能关闭工作台")
                    self._json({"message": "评测工作台已关闭"})
                    threading.Thread(target=self.server.shutdown, daemon=True).start()
                    return
                self._json({"error": "接口不存在"}, HTTPStatus.NOT_FOUND)
            except Exception as exc:
                self._error(exc)

    return Handler


def create_evaluation_server(
    application: EvaluationCurationApplication,
) -> tuple[EvaluationCurationHTTPServer, str]:
    server = EvaluationCurationHTTPServer(("127.0.0.1", 0), _handler_for(application))
    port = server.server_address[1]
    url = f"http://127.0.0.1:{port}/?{urlencode({'token': application.token})}"
    return server, url


def run_evaluation_server(
    database: Database,
    codex_settings: CodexCliSettings,
    open_browser: bool = True,
) -> None:
    application = EvaluationCurationApplication(database, codex_settings)
    server, url = create_evaluation_server(application)
    print(f"评测工作台：{url}")
    if open_browser and not webbrowser.open(url):
        print("未能自动打开浏览器，请复制上面的本地地址。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        if not application.can_shutdown():
            print("AI 任务仍在运行，暂不关闭。")
            server.serve_forever()
    finally:
        server.server_close()
