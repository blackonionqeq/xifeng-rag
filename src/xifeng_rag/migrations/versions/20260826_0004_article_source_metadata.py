"""Store article source metadata used by the corpus dashboard."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision: str = "20260826_0004"
down_revision: str | None = "20260826_0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _columns() -> set[str]:
    return {column["name"] for column in inspect(op.get_bind()).get_columns("articles")}


def upgrade() -> None:
    if not inspect(op.get_bind()).has_table("articles"):
        return
    columns = _columns()
    additions = {
        "captured_at": sa.DateTime(timezone=True),
        "source_modified_at": sa.DateTime(timezone=True),
        "source_size_bytes": sa.Integer(),
        "source_character_count": sa.Integer(),
        "metadata_synced_at": sa.DateTime(timezone=True),
    }
    for name, column_type in additions.items():
        if name not in columns:
            op.add_column("articles", sa.Column(name, column_type, nullable=True))


def downgrade() -> None:
    if not inspect(op.get_bind()).has_table("articles"):
        return
    columns = _columns()
    for name in (
        "metadata_synced_at",
        "source_character_count",
        "source_size_bytes",
        "source_modified_at",
        "captured_at",
    ):
        if name in columns:
            op.drop_column("articles", name)
