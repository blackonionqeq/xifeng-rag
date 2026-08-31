"""从 xifeng 数据库导出一篇文章清洗后的编号原子单元。

输出与 eval curate 界面、Codex 候选使用的编号基准一致（paragraphs 表的 number）。
仅使用标准库，可在项目虚拟环境或系统 Python 下直接运行。

用法：
    python export_units.py "文章标题" [--db data/xifeng.sqlite3] [--json]
    python export_units.py "sn:<文章ID>" [--json]
    python export_units.py --list [--db data/xifeng.sqlite3]

退出码：0 成功；1 未找到文章；2 匹配到多篇文章（候选清单已打印）。
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path


def _connect(db_path: Path) -> sqlite3.Connection:
    if not db_path.is_file():
        print(f"错误：数据库不存在：{db_path}", file=sys.stderr)
        print(
            "请先在项目根目录运行 `uv run xifeng corpus scan`，或用 --db 指定路径。",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return sqlite3.connect(db_path)


def _match_article(conn: sqlite3.Connection, query: str) -> tuple[str, str]:
    """按文章 ID 或标题匹配，返回 (article_id, title)。"""
    query = query.strip()
    if query.lower().endswith(".md"):
        query = query[:-3]
    rows = conn.execute(
        "SELECT id, title FROM articles WHERE active = 1 ORDER BY title"
    ).fetchall()
    if query.startswith("sn:"):
        for article_id, title in rows:
            if article_id == query:
                return article_id, title
        print(f"错误：未找到文章 ID {query}", file=sys.stderr)
        raise SystemExit(1)
    exact = [(article_id, title) for article_id, title in rows if title == query]
    if len(exact) == 1:
        return exact[0]
    partial = [
        (article_id, title) for article_id, title in rows if query in title or title in query
    ]
    if len(partial) == 1:
        return partial[0]
    if len(partial) > 1:
        print(
            f"错误：匹配到 {len(partial)} 篇文章，请用更完整的标题或 sn: 文章 ID：",
            file=sys.stderr,
        )
        for article_id, title in partial:
            print(f"  {article_id}  {title}", file=sys.stderr)
        raise SystemExit(2)
    print(f"错误：未找到标题含「{query}」的文章。用 --list 查看全部文章。", file=sys.stderr)
    raise SystemExit(1)


def _load_version(conn: sqlite3.Connection, article_id: str) -> tuple[str, int]:
    row = conn.execute(
        "SELECT id, status FROM article_versions WHERE article_id = ? AND active = 1",
        (article_id,),
    ).fetchone()
    if row is None:
        print("错误：该文章没有活跃版本。请先运行 `uv run xifeng corpus scan`。", file=sys.stderr)
        raise SystemExit(1)
    version_id, status = row
    unit_count = conn.execute(
        "SELECT COUNT(*) FROM paragraphs WHERE version_id = ?", (version_id,)
    ).fetchone()[0]
    return version_id, unit_count, status  # type: ignore[return-value]


def _load_paragraphs(conn: sqlite3.Connection, version_id: str) -> list[tuple[int, str]]:
    return conn.execute(
        "SELECT number, text FROM paragraphs WHERE version_id = ? ORDER BY number",
        (version_id,),
    ).fetchall()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("article", nargs="?", help="文章标题（支持部分匹配）或 sn: 文章 ID")
    parser.add_argument("--db", default="data/xifeng.sqlite3", help="SQLite 数据库路径")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出（含元数据）")
    parser.add_argument("--list", action="store_true", help="列出全部文章标题与 ID")
    args = parser.parse_args()

    db_path = Path(args.db)
    conn = _connect(db_path)
    try:
        if args.list:
            for article_id, title in conn.execute(
                "SELECT id, title FROM articles WHERE active = 1 ORDER BY title"
            ):
                print(f"{article_id}  {title}")
            return
        if not args.article:
            parser.error("请提供文章标题，或使用 --list 查看文章列表")
        article_id, title = _match_article(conn, args.article)
        version_id, unit_count, status = _load_version(conn, article_id)
        paragraphs = _load_paragraphs(conn, version_id)
        if args.json:
            print(
                json.dumps(
                    {
                        "article_id": article_id,
                        "title": title,
                        "version_id": version_id,
                        "version_status": status,
                        "unit_count": unit_count,
                        "paragraphs": [{"number": n, "text": t} for n, t in paragraphs],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(f"# 文章：{title}")
            print(f"# 文章ID：{article_id}")
            print(f"# 版本：{version_id}（状态 {status}，共 {unit_count} 个原子单元）")
            for number, text in paragraphs:
                print(f"[{number}] {text}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
