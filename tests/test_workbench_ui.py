from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from typer.testing import CliRunner

from xifeng_rag.cli import app
from xifeng_rag.workbench_ui import (
    RunningService,
    ServiceDefinition,
    WorkbenchApplication,
    create_workbench_server,
)

runner = CliRunner()


class QuietHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        del format, args

    def do_GET(self) -> None:
        self.send_response(200)
        self.end_headers()


def _application(cleaned: list[str] | None = None) -> WorkbenchApplication:
    def factory() -> RunningService:
        server = ThreadingHTTPServer(("127.0.0.1", 0), QuietHandler)
        port = server.server_address[1]

        def cleanup() -> None:
            if cleaned is not None:
                cleaned.append("done")

        return RunningService(
            server,
            f"http://127.0.0.1:{port}/",
            cleanup,
        )

    return WorkbenchApplication(
        [ServiceDefinition("demo", "体验项目", "说明", "xifeng demo", factory)],
        [{"title": "环境检查", "command": "uv run xifeng doctor"}],
        token="test-token",
    )


def _wait_for_status(application: WorkbenchApplication, expected: str) -> dict[str, object]:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        service = application.snapshot_service("demo")
        if service["status"] == expected:
            return service
        time.sleep(0.01)
    raise AssertionError(f"服务未进入状态：{expected}")


def test_workbench_starts_reuses_and_stops_service() -> None:
    cleaned: list[str] = []
    application = _application(cleaned)

    assert application.start("demo")["status"] == "starting"
    running = _wait_for_status(application, "running")
    assert isinstance(running["url"], str)
    assert application.start("demo")["url"] == running["url"]

    application.stop("demo")
    _wait_for_status(application, "stopped")
    assert cleaned == ["done"]


def test_workbench_http_requires_token_and_lists_commands() -> None:
    application = _application()
    server, url = create_workbench_server(application)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = url.split("/?", 1)[0]
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            assert "西风项目控制台" in response.read().decode()
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(f"{base_url}/api/status", timeout=5)
        assert caught.value.code == 404
        status_url = f"{base_url}/api/status?token=test-token"
        with urllib.request.urlopen(status_url, timeout=5) as response:
            payload = json.load(response)
        assert payload["services"][0]["title"] == "体验项目"
        assert payload["commands"][0]["command"] == "uv run xifeng doctor"
    finally:
        application.stop_all()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_workbench_command_wires_three_services(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        f"project_root: {tmp_path.as_posix()}\nstorage:\n  data_dir: data\n",
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_run(application, *, open_browser):
        captured["application"] = application
        captured["open_browser"] = open_browser

    monkeypatch.setattr("xifeng_rag.cli.run_workbench", fake_run)
    result = runner.invoke(app, ["--config", str(config), "workbench", "--no-open"])

    assert result.exit_code == 0
    application = captured["application"]
    assert isinstance(application, WorkbenchApplication)
    assert list(application.services) == ["demo", "corpus", "evaluation"]
    assert application.services["evaluation"].title == "评测工作台"
    assert captured["open_browser"] is False

    evaluation: dict[str, object] = {}

    def fake_evaluation_application(*args, **kwargs):
        del args
        evaluation["runner"] = kwargs.get("evaluation_runner")
        return object()

    server = ThreadingHTTPServer(("127.0.0.1", 0), QuietHandler)
    monkeypatch.setattr(
        "xifeng_rag.cli.EvaluationCurationApplication", fake_evaluation_application
    )
    monkeypatch.setattr(
        "xifeng_rag.cli.create_evaluation_server",
        lambda _application: (server, "http://127.0.0.1:12345/"),
    )
    try:
        running = application.services["evaluation"].factory()
        assert callable(evaluation["runner"])
        assert running.url == "http://127.0.0.1:12345/"
    finally:
        server.server_close()
