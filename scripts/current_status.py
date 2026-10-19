"""Read the current local corpus, index, and evaluation status without loading article text."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

from xifeng_rag.config import load_settings


def scalar(connection: sqlite3.Connection, query: str) -> int:
    return int(connection.execute(query).fetchone()[0])


def current_status(database_path: Path) -> dict[str, object]:
    if not database_path.is_file():
        raise FileNotFoundError(f"数据库不存在：{database_path}；请先运行 uv run xifeng init")

    database_uri = database_path.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(database_uri, uri=True)) as connection:
        connection.execute("PRAGMA query_only=ON")
        articles = {
            "imported": scalar(connection, "SELECT COUNT(*) FROM articles"),
            "source_active": scalar(connection, "SELECT COUNT(*) FROM articles WHERE active=1"),
            "approved": scalar(
                connection,
                """SELECT COUNT(DISTINCT v.article_id)
                   FROM approved_manifests AS m
                   JOIN article_versions AS v ON v.id=m.version_id
                   JOIN articles AS a ON a.id=v.article_id
                   WHERE a.active=1 AND v.active=1""",
            ),
        }

        # Match indexing._selected_chunks: select the newest eligible active version
        # per active article, then sort stable chunk IDs to form the Build snapshot.
        rows = connection.execute(
            """SELECT t.id, t.article_id, t.version_id
               FROM topic_chunks AS t
               JOIN article_versions AS v ON v.id=t.version_id
               JOIN articles AS a ON a.id=t.article_id
               WHERE a.active=1 AND v.active=1 AND v.status IN ('approved', 'indexed')
               ORDER BY v.created_at DESC, t.start_paragraph"""
        )
        selected_versions: dict[str, str] = {}
        chunk_ids: list[str] = []
        for chunk_id, article_id, version_id in rows:
            if selected_versions.setdefault(article_id, version_id) == version_id:
                chunk_ids.append(chunk_id)
        snapshot = hashlib.sha256("\n".join(sorted(chunk_ids)).encode()).hexdigest()
        active_builds = connection.execute(
            "SELECT id, corpus_snapshot FROM index_builds WHERE active=1"
        ).fetchall()
        index = {
            "eligible_articles": len(selected_versions),
            "eligible_chunks": len(chunk_ids),
            "active_build_id": active_builds[0][0] if len(active_builds) == 1 else None,
            "active_build_matches_current_selection": (
                len(active_builds) == 1 and active_builds[0][1] == snapshot
            ),
        }

        evaluations = [
            {
                "purpose": purpose,
                "status": status,
                "datasets": dataset_count,
                "questions": question_count,
            }
            for purpose, status, dataset_count, question_count in connection.execute(
                """SELECT d.purpose, d.status, COUNT(DISTINCT d.id), COUNT(q.id)
                   FROM evaluation_datasets AS d
                   LEFT JOIN evaluation_questions AS q ON q.dataset_id=d.id
                   GROUP BY d.purpose, d.status
                   ORDER BY d.purpose, d.status"""
            )
        ]

    return {
        "database": str(database_path.resolve()),
        "database_modified_at": (
            datetime.fromtimestamp(database_path.stat().st_mtime).astimezone().isoformat()
        ),
        "articles": articles,
        "index": index,
        "evaluations": evaluations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, help="SQLite 路径；默认从项目配置读取")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出，便于自动处理")
    args = parser.parse_args()
    database_path = args.db if args.db is not None else load_settings().database_path
    report = current_status(database_path)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    articles = report["articles"]
    index = report["index"]
    print(f"数据库：{report['database']}")
    print(f"最近修改：{report['database_modified_at']}")
    print(
        f"文章：已导入 {articles['imported']}，来源有效 {articles['source_active']}，"
        f"已批准 {articles['approved']}"
    )
    print(f"当前可建索引：{index['eligible_articles']} 篇，{index['eligible_chunks']} 个话题块")
    if index["active_build_id"] is None:
        print("活动证据索引：无或记录异常")
    else:
        match = (
            "一致"
            if index["active_build_matches_current_selection"]
            else "不一致，可能尚未重建"
        )
        print(f"活动证据索引：{index['active_build_id']}；与当前可建范围{match}")
    if report["evaluations"]:
        for item in report["evaluations"]:
            print(
                f"评测：{item['purpose']}/{item['status']}，"
                f"{item['datasets']} 套，{item['questions']} 题"
            )
    else:
        print("评测：暂无题集")


if __name__ == "__main__":
    main()
