from __future__ import annotations

import os
from pathlib import Path

import pytest

from xifeng_rag.config import load_settings


def test_yaml_paths_resolve_from_config_and_environment_overrides(
    tmp_path: Path, monkeypatch
) -> None:
    config = tmp_path / "settings.yaml"
    config.write_text(
        "project_root: .\ncorpus:\n  source_dir: corpus\nollama:\n  model: yaml-model\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("XIFENG_OLLAMA__MODEL", "env-model")
    settings = load_settings(config)
    assert settings.project_root == tmp_path
    assert settings.source_dir == (tmp_path / "corpus").resolve()
    assert settings.ollama.model == "env-model"


def test_excluded_article_ids_are_normalized_and_validated(tmp_path: Path) -> None:
    config = tmp_path / "settings.yaml"
    config.write_text(
        "corpus:\n  excluded_article_ids:\n    - ' sn:first '\n",
        encoding="utf-8",
    )
    assert load_settings(config).corpus.excluded_article_ids == ["sn:first"]

    config.write_text(
        "corpus:\n  excluded_article_ids:\n    - sn:first\n    - sn:first\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="重复"):
        load_settings(config)


def test_external_chunking_profiles_load_without_storing_a_key(tmp_path: Path) -> None:
    config = tmp_path / "settings.yaml"
    config.write_text(
        "chunking:\n"
        "  external_profiles:\n"
        "    deepseek:\n"
        "      base_url: https://api.deepseek.com\n"
        "      api_key_env: TEST_DEEPSEEK_API_KEY\n"
        "      default_model: deepseek-chat\n",
        encoding="utf-8",
    )
    profile = load_settings(config).chunking.external_profiles["deepseek"]
    assert profile.base_url == "https://api.deepseek.com"
    assert profile.api_key_env == "TEST_DEEPSEEK_API_KEY"
    assert not hasattr(profile, "api_key")


def test_codex_cli_chunking_settings_have_safe_defaults_and_allow_overrides(
    tmp_path: Path,
) -> None:
    empty_config = tmp_path / "empty.yaml"
    empty_config.write_text("{}\n", encoding="utf-8")
    defaults = load_settings(empty_config).chunking.codex_cli
    assert defaults.executable == "codex"
    assert defaults.default_model == "gpt-5.6-terra"
    assert defaults.max_retries == 1

    config = tmp_path / "settings.yaml"
    config.write_text(
        "chunking:\n"
        "  codex_cli:\n"
        "    executable: custom-codex\n"
        "    default_model: custom-model\n"
        "    timeout_seconds: 45\n"
        "    max_retries: 0\n",
        encoding="utf-8",
    )
    configured = load_settings(config).chunking.codex_cli
    assert configured.executable == "custom-codex"
    assert configured.default_model == "custom-model"
    assert configured.timeout_seconds == 45
    assert configured.max_retries == 0


def test_project_dotenv_loads_api_key_without_overriding_process_environment(
    tmp_path: Path, monkeypatch
) -> None:
    config = tmp_path / "settings.yaml"
    config.write_text("project_root: .\n", encoding="utf-8")
    (tmp_path / ".env").write_text("DEEPSEEK_API_KEY=dotenv-secret\n", encoding="utf-8")

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    load_settings(config)
    assert os.environ["DEEPSEEK_API_KEY"] == "dotenv-secret"

    monkeypatch.setenv("DEEPSEEK_API_KEY", "process-secret")
    load_settings(config)
    assert os.environ["DEEPSEEK_API_KEY"] == "process-secret"


def test_missing_project_dotenv_is_allowed(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "settings.yaml"
    config.write_text("project_root: .\n", encoding="utf-8")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    settings = load_settings(config)
    assert settings.project_root == tmp_path
    assert "DEEPSEEK_API_KEY" not in os.environ


def test_embedding_cache_path_and_retrieval_defaults(tmp_path: Path) -> None:
    config = tmp_path / "settings.yaml"
    config.write_text(
        "project_root: .\nembedding:\n  model_cache_dir: models/huggingface\n",
        encoding="utf-8",
    )
    settings = load_settings(config)
    assert settings.embedding_cache_dir == (tmp_path / "models/huggingface").resolve()
    assert settings.embedding.model_id == "BAAI/bge-base-zh-v1.5"
    assert settings.retrieval.bm25_candidate_k == 5
    assert settings.retrieval.dense_candidate_k == 5
    assert settings.retrieval.rrf_k == 60
    assert settings.reranker.model_id == "BAAI/bge-reranker-v2-m3"
    assert settings.reranker.max_pair_tokens == 1024
    assert settings.reranker.candidate_k == 5
    assert settings.context_assembly.max_evidence_items == 5
    assert settings.context_assembly.max_total_chars == 50_000
    assert not settings.context_assembly.include_ai_metadata
    assert settings.generator.provider == "deepseek"
    assert settings.generator.base_url == "https://api.deepseek.com"
    assert settings.generator.api_key_env == "DEEPSEEK_API_KEY"
    assert settings.generator.model == "deepseek-v4-flash"
    assert settings.generator.max_output_tokens == 4096


def test_context_assembly_yaml_and_environment_settings(
    tmp_path: Path, monkeypatch
) -> None:
    config = tmp_path / "settings.yaml"
    config.write_text(
        "context_assembly:\n"
        "  max_evidence_items: 3\n"
        "  max_total_chars: 40000\n"
        "  include_ai_metadata: false\n",
        encoding="utf-8",
    )
    settings = load_settings(config)
    assert settings.context_assembly.max_evidence_items == 3
    assert settings.context_assembly.max_total_chars == 40_000
    assert not settings.context_assembly.include_ai_metadata

    monkeypatch.setenv("XIFENG_CONTEXT_ASSEMBLY__INCLUDE_AI_METADATA", "true")
    environment_config = tmp_path / "environment.yaml"
    environment_config.write_text("{}\n", encoding="utf-8")
    environment_settings = load_settings(environment_config)
    assert environment_settings.context_assembly.include_ai_metadata


def test_reranker_cache_path_resolves_from_project_root(tmp_path: Path) -> None:
    config = tmp_path / "settings.yaml"
    config.write_text(
        "project_root: .\nreranker:\n  model_cache_dir: models/reranker\n",
        encoding="utf-8",
    )
    settings = load_settings(config)
    assert settings.reranker_cache_dir == (tmp_path / "models/reranker").resolve()
