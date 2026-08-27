from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class CorpusSettings(BaseModel):
    source_dir: Path = Path("../test-reverse-wechat-official-accounts-paid-article/saved_articles")
    excluded_article_ids: list[str] = Field(default_factory=list)

    @field_validator("excluded_article_ids")
    @classmethod
    def validate_excluded_article_ids(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("excluded_article_ids 不能包含空值")
        if len(normalized) != len(set(normalized)):
            raise ValueError("excluded_article_ids 不能包含重复值")
        return normalized


class StorageSettings(BaseModel):
    data_dir: Path = Path("data")


class OllamaSettings(BaseModel):
    executable: Path = Path("D:/ai-tools/ollama/ollama.exe")
    model_dir: Path = Path("D:/ai-models/ollama")
    host: str = "http://127.0.0.1:11434"
    model: str = "qwen3:4b-instruct-2507-q4_K_M"
    context_window: int = 32768
    timeout_seconds: int = 900
    max_retries: int = 2


class ExternalChunkingProfile(BaseModel):
    base_url: str
    api_key_env: str = Field(min_length=1)
    default_model: str = Field(min_length=1)
    timeout_seconds: int = Field(default=900, gt=0)
    max_retries: int = Field(default=2, ge=0, le=5)


class CodexCliSettings(BaseModel):
    executable: str = Field(default="codex", min_length=1)
    default_model: str = Field(default="gpt-5.6-terra", min_length=1)
    timeout_seconds: int = Field(default=900, gt=0)
    max_retries: int = Field(default=1, ge=0, le=2)


class ChunkingSettings(BaseModel):
    codex_cli: CodexCliSettings = Field(default_factory=CodexCliSettings)
    external_profiles: dict[str, ExternalChunkingProfile] = Field(default_factory=dict)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="XIFENG_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="forbid",
    )

    project_root: Path = Field(default_factory=Path.cwd)
    corpus: CorpusSettings = Field(default_factory=CorpusSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)

    @property
    def source_dir(self) -> Path:
        return self._resolve(self.corpus.source_dir)

    @property
    def data_dir(self) -> Path:
        return self._resolve(self.storage.data_dir)

    @property
    def database_path(self) -> Path:
        return self.data_dir / "xifeng.sqlite3"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "indexes"

    @property
    def review_dir(self) -> Path:
        return self.data_dir / "reviews"

    def _resolve(self, path: Path) -> Path:
        value = path if path.is_absolute() else self.project_root / path
        return value.resolve()


def load_settings(config_path: Path | None = None) -> Settings:
    """Load YAML settings, then apply XIFENG_* environment overrides."""
    if config_path is None:
        local_config = Path.cwd() / "config.local.yaml"
        config_path = local_config if local_config.is_file() else None
    values: dict[str, object] = {}
    if config_path is not None:
        with config_path.open("r", encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream) or {}
        if not isinstance(loaded, dict):
            raise ValueError("配置文件顶层必须是对象")
        values = loaded
    raw_root = values.get("project_root", Path.cwd())
    if not isinstance(raw_root, (str, Path)):
        raise ValueError("project_root 必须是路径")
    configured_root = Path(raw_root)
    if not configured_root.is_absolute():
        root_base = config_path.parent if config_path is not None else Path.cwd()
        configured_root = (root_base / configured_root).resolve()
        values["project_root"] = configured_root
    environment_root = os.environ.get("XIFENG_PROJECT_ROOT")
    dotenv_root = Path(environment_root) if environment_root else configured_root
    if not dotenv_root.is_absolute():
        dotenv_root = Path.cwd() / dotenv_root
    load_dotenv(dotenv_root.resolve() / ".env", override=False)
    # BaseSettings gives init values priority. Construct the YAML model first, then
    # let a second settings instance apply environment values over its defaults.
    yaml_settings = Settings.model_validate(values)
    defaults = yaml_settings.model_dump()
    env_settings = Settings()
    env_fields = env_settings.model_fields_set
    for field in env_fields:
        defaults[field] = getattr(env_settings, field)
    return Settings.model_validate(defaults)
