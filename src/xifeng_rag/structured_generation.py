from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

from xifeng_rag.config import CodexCliSettings


class GenerationResult(BaseModel):
    content: str
    metadata: dict[str, object] = Field(default_factory=dict)


class StructuredGenerator(Protocol):
    def generate(self, prompt: str, schema: dict[str, object]) -> GenerationResult: ...


def strict_output_schema(schema: dict[str, object]) -> dict[str, object]:
    """Return a copy compatible with strict structured-output validators."""
    copied = json.loads(json.dumps(schema))

    def visit(node: object) -> None:
        if isinstance(node, list):
            for value in node:
                visit(value)
            return
        if not isinstance(node, dict):
            return
        if node.get("type") == "object":
            node["additionalProperties"] = False
        for value in node.values():
            visit(value)

    visit(copied)
    return copied


class CodexCliClient:
    def __init__(self, settings: CodexCliSettings, model: str) -> None:
        self.settings = settings
        self.model = model
        self._metadata: dict[str, object] = {"client": "codex-cli"}

    def _command_prefix(self) -> list[str]:
        executable = self.settings.executable
        resolved: str | None = None
        if os.name == "nt" and not Path(executable).suffix:
            resolved = shutil.which(f"{executable}.cmd")
        resolved = resolved or shutil.which(executable) or executable
        if Path(resolved).suffix.lower() == ".ps1":
            return [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-File",
                resolved,
            ]
        return [resolved]

    def _run_probe(self, arguments: list[str]) -> subprocess.CompletedProcess[str]:
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        return subprocess.run(
            [*self._command_prefix(), *arguments],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
            creationflags=creationflags,
        )

    def preflight(self) -> None:
        version = self._run_probe(["--version"])
        if version.returncode != 0:
            raise RuntimeError("Codex CLI 无法运行")
        login = self._run_probe(["login", "status"])
        if login.returncode != 0:
            raise RuntimeError("Codex CLI 尚未登录")
        version_text = version.stdout.strip().splitlines()
        login_text = f"{login.stdout}\n{login.stderr}".lower()
        if "chatgpt" in login_text:
            auth_type = "chatgpt"
        elif "api" in login_text:
            auth_type = "api-key"
        else:
            auth_type = "unknown"
        self._metadata = {
            "client": "codex-cli",
            "cli_version": version_text[0][:120] if version_text else "unknown",
            "auth_type": auth_type,
        }

    def generate(self, prompt: str, schema: dict[str, object]) -> GenerationResult:
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        with tempfile.TemporaryDirectory(prefix="xifeng-codex-structured-") as directory:
            workdir = Path(directory)
            schema_path = workdir / "output-schema.json"
            output_path = workdir / "structured-output.json"
            schema_path.write_text(
                json.dumps(strict_output_schema(schema), ensure_ascii=False), encoding="utf-8"
            )
            command = [
                *self._command_prefix(),
                "exec",
                "--ephemeral",
                "--sandbox",
                "read-only",
                "--ignore-user-config",
                "--ignore-rules",
                "--skip-git-repo-check",
                "--model",
                self.model,
                "--cd",
                str(workdir),
                "--output-schema",
                str(schema_path),
                "--output-last-message",
                str(output_path),
                "--color",
                "never",
                "-",
            ]
            codex_prompt = (
                "不要调用 shell、网络、MCP 或任何其他工具。只分析下面提供的内容，"
                "并返回符合输出 Schema 的 JSON 对象。\n\n" + prompt
            )
            completed = subprocess.run(
                command,
                input=codex_prompt,
                capture_output=True,
                text=True,
                timeout=self.settings.timeout_seconds,
                check=False,
                cwd=workdir,
                creationflags=creationflags,
            )
            if completed.returncode != 0:
                raise RuntimeError(f"Codex CLI 生成失败，退出码 {completed.returncode}")
            if not output_path.is_file():
                raise ValueError("Codex CLI 未生成结构化输出")
            content = output_path.read_text(encoding="utf-8").strip()
            if not content:
                raise ValueError("Codex CLI 结构化输出为空")
        return GenerationResult(content=content, metadata=dict(self._metadata))
