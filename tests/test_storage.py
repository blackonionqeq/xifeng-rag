from __future__ import annotations

import sqlite3
from pathlib import Path

from sqlalchemy import inspect, text

from xifeng_rag.storage import Database


def test_manifest_topics_migration_upgrades_existing_database(tmp_path: Path) -> None:
    path = tmp_path / "legacy.sqlite3"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE alembic_version (
            version_num VARCHAR(32) NOT NULL PRIMARY KEY
        );
        INSERT INTO alembic_version(version_num) VALUES ('20260825_0001');
        CREATE TABLE approved_manifests (
            id VARCHAR(96) PRIMARY KEY,
            proposal_id VARCHAR(96) UNIQUE,
            version_id VARCHAR(96),
            manifest_version INTEGER,
            approved_at DATETIME
        );
        """
    )
    connection.close()

    database = Database(path)
    database.initialize()

    columns = {
        column["name"] for column in inspect(database.engine).get_columns("approved_manifests")
    }
    assert "topics_json" in columns


def test_fresh_database_has_chunk_retry_metadata(database: Database) -> None:
    columns = {column["name"] for column in inspect(database.engine).get_columns("chunk_proposals")}
    assert {"provider_id", "prompt_text", "prompt_hash", "retry_id"} <= columns


def test_fresh_database_has_article_overviews_and_publication_time(database: Database) -> None:
    inspector = inspect(database.engine)
    article_columns = {item["name"] for item in inspector.get_columns("articles")}
    overview_columns = {item["name"] for item in inspector.get_columns("article_overviews")}
    assert "published_at" in article_columns
    assert {
        "version_id",
        "proposal_id",
        "manifest_id",
        "article_summary",
        "status",
        "provider_id",
        "model_id",
        "prompt_version",
    } <= overview_columns
    assert inspector.has_table("article_index_builds")


def test_evaluation_curation_migration_preserves_existing_runs(tmp_path: Path) -> None:
    path = tmp_path / "evaluation-legacy.sqlite3"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE alembic_version (
            version_num VARCHAR(32) NOT NULL PRIMARY KEY
        );
        INSERT INTO alembic_version(version_num) VALUES ('20260826_0004');
        CREATE TABLE index_builds (
            id VARCHAR(96) PRIMARY KEY
        );
        INSERT INTO index_builds(id) VALUES ('existing-build');
        CREATE TABLE evaluation_runs (
            id VARCHAR(96) PRIMARY KEY,
            dataset_id TEXT NOT NULL,
            build_id VARCHAR(96) REFERENCES index_builds(id),
            metrics_json TEXT NOT NULL,
            details_json TEXT NOT NULL,
            created_at DATETIME NOT NULL
        );
        INSERT INTO evaluation_runs(
            id, dataset_id, build_id, metrics_json, details_json, created_at
        ) VALUES ('existing-run', 'legacy-set', 'existing-build', '{}', '[]', CURRENT_TIMESTAMP);
        """
    )
    connection.close()

    database = Database(path)
    database.initialize()

    columns = {item["name"] for item in inspect(database.engine).get_columns("evaluation_runs")}
    assert {"dataset_content_hash", "corpus_snapshot", "profile_id", "k"} <= columns
    assert {
        "reranker_enabled",
        "reranker_model_id",
        "reranker_candidate_k",
        "reranker_max_pair_tokens",
    } <= columns
    with database.engine.connect() as migrated:
        assert migrated.execute(text("SELECT COUNT(*) FROM evaluation_runs")).scalar_one() == 1
