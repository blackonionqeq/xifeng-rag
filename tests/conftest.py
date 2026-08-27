from __future__ import annotations

from pathlib import Path

import pytest

from xifeng_rag.storage import Database


@pytest.fixture
def database(tmp_path: Path) -> Database:
    value = Database(tmp_path / "data" / "test.sqlite3")
    value.initialize()
    return value
