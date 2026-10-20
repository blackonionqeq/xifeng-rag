from __future__ import annotations

import json
import secrets
import threading
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from socketserver import BaseServer
from urllib.parse import parse_qs, urlencode, urlsplit

STATIC_DIR = Path(__file__).with_name("workbench_static")
MAX_REQUEST_BYTES = 8 * 1024


@dataclass(frozen=True)
class RunningService:
    server: BaseServer
    url: str
    cleanup: Callable[[], None] = lambda: None


@dataclass(frozen=True)
class ServiceDefinition:
    service_id: str
    title: str
    description: str
    command: str
    factory: Callable[[], RunningService]


@dataclass
class ServiceState:
    status: str = "stopped"
    running: RunningService | None = None
    error: str | None = None
    thread: threading.Thread | None = None
    stop_requested: bool = False


class WorkbenchApplication:
    def __init__(
        self,
        services: list[ServiceDefinition],
        commands: list[dict[str, str]],
        *,
        token: str | None = None,
    ) -> None:
        self.services = {item.service_id: item for item in services}
        self.commands = commands
        self.token = token or secrets.token_urlsafe(32)
        self._states = {item.service_id: ServiceState() for item in services}
        self._lock = threading.Lock()

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            services = [
                self._payload(definition, self._states[definition.service_id])
                for definition in self.services.values()
            ]
        return {"services": services, "commands": self.commands}

    @staticmethod
    def _payload(
        definition: ServiceDefinition, state: ServiceState
    ) -> dict[str, object]:
        running = state.running
        return {
            "service_id": definition.service_id,
            "title": definition.title,
            "description": definition.description,
            "command": definition.command,
            "status": state.status,
            "url": running.url if running is not None else None,
            "error": state.error,
        }

    def start(self, service_id: str) -> dict[str, object]:
        definition = self._definition(service_id)
        already_running = False
        with self._lock:
            state = self._states[service_id]
            if state.status == "running":
                already_running = True
            if state.status == "starting":
                raise RuntimeError("服务正在启动")
            if not already_running:
                state.status = "starting"
                state.error = None
                state.stop_requested = False
        if already_running:
            return self.snapshot_service(service_id)
        thread = threading.Thread(
            target=self._serve,
            args=(definition,),
            daemon=True,
            name=f"workbench-{service_id}",
        )
        with self._lock:
            self._states[service_id].thread = thread
        thread.start()
        return self.snapshot_service(service_id)

    def _serve(self, definition: ServiceDefinition) -> None:
        running: RunningService | None = None
        try:
            running = definition.factory()
            with self._lock:
                state = self._states[definition.service_id]
                should_serve = not state.stop_requested
                if should_serve:
                    state.running = running
                    state.status = "running"
            if should_serve:
                running.server.serve_forever()
        except Exception as exc:
            with self._lock:
                state = self._states[definition.service_id]
                state.error = f"{type(exc).__name__}: {exc}"
                state.status = "failed"
        finally:
            cleanup_error: str | None = None
            if running is not None:
                try:
                    running.cleanup()
                except Exception as exc:
                    cleanup_error = f"{type(exc).__name__}: {exc}"
                try:
                    running.server.server_close()
                except Exception as exc:
                    cleanup_error = cleanup_error or f"{type(exc).__name__}: {exc}"
            with self._lock:
                state = self._states[definition.service_id]
                state.running = None
                state.thread = None
                state.stop_requested = False
                if cleanup_error is not None:
                    state.error = cleanup_error
                    state.status = "failed"
                elif state.status != "failed":
                    state.status = "stopped"

    def stop(self, service_id: str) -> dict[str, object]:
        self._definition(service_id)
        with self._lock:
            state = self._states[service_id]
            if state.status == "starting":
                raise RuntimeError("服务仍在启动，请稍后停止")
            running = state.running
            thread = state.thread
        if running is not None:
            running.server.shutdown()
        if thread is not None:
            thread.join(timeout=10)
        return self.snapshot_service(service_id)

    def stop_all(self) -> None:
        with self._lock:
            active = [
                (state.running, state.thread)
                for state in self._states.values()
                if state.thread is not None
            ]
            for state in self._states.values():
                state.stop_requested = True
        for item, _ in active:
            if item is not None:
                item.server.shutdown()
        for _, thread in active:
            if thread is not None:
                thread.join(timeout=120)

    def snapshot_service(self, service_id: str) -> dict[str, object]:
        definition = self._definition(service_id)
        with self._lock:
            return self._payload(definition, self._states[service_id])

    def _definition(self, service_id: str) -> ServiceDefinition:
        try:
            return self.services[service_id]
        except KeyError as exc:
            raise ValueError("未知服务") from exc


class WorkbenchHTTPServer(ThreadingHTTPServer):
    daemon_threads = True


def _handler_for(application: WorkbenchApplication) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "XifengWorkbench/0"

        def log_message(self, format: str, *args: object) -> None:
            del format, args

        def _authorized(self) -> bool:
            parsed = urlsplit(self.path)
            query_token = parse_qs(parsed.query).get("token", [None])[0]
            supplied = self.headers.get("X-Workbench-Token") or query_token or ""
            return secrets.compare_digest(supplied, application.token)

        def _json(self, value: object, status: HTTPStatus = HTTPStatus.OK) -> None:
            payload = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(payload)

        def _static(self, name: str, content_type: str) -> None:
            payload = (STATIC_DIR / name).read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self'; "
                "connect-src 'self'; base-uri 'none'; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            path = urlsplit(self.path).path
            if path in {"/", "/index.html"}:
                return self._static("index.html", "text/html; charset=utf-8")
            if path == "/app.js":
                return self._static("app.js", "text/javascript; charset=utf-8")
            if path == "/style.css":
                return self._static("style.css", "text/css; charset=utf-8")
            if path == "/api/status" and self._authorized():
                return self._json(application.snapshot())
            return self._json({"error": "接口不存在或令牌无效"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            if not self._authorized():
                return self._json({"error": "无效令牌"}, HTTPStatus.FORBIDDEN)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_REQUEST_BYTES:
                    raise ValueError("请求体为空或过大")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("请求体必须是对象")
                path = urlsplit(self.path).path
                if path == "/api/services/start":
                    return self._json(application.start(str(body.get("service_id", ""))))
                if path == "/api/services/stop":
                    return self._json(application.stop(str(body.get("service_id", ""))))
                if path == "/api/shutdown":
                    application.stop_all()
                    self._json({"message": "工作台已关闭"})
                    threading.Thread(target=self.server.shutdown, daemon=True).start()
                    return
                self._json({"error": "接口不存在"}, HTTPStatus.NOT_FOUND)
            except RuntimeError as exc:
                self._json({"error": str(exc)}, HTTPStatus.CONFLICT)
            except Exception as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    return Handler


def create_workbench_server(
    application: WorkbenchApplication,
) -> tuple[WorkbenchHTTPServer, str]:
    server = WorkbenchHTTPServer(("127.0.0.1", 0), _handler_for(application))
    host, port = server.server_address[:2]
    return server, f"http://{host}:{port}/?{urlencode({'token': application.token})}"


def run_workbench(application: WorkbenchApplication, *, open_browser: bool = True) -> None:
    server, url = create_workbench_server(application)
    print(f"西风项目控制台：{url}")
    print("按 Ctrl+C 停止控制台及其启动的服务。")
    if open_browser and not webbrowser.open(url):
        print("未能自动打开浏览器，请复制上面的地址。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n项目控制台已停止。")
    finally:
        application.stop_all()
        server.server_close()
