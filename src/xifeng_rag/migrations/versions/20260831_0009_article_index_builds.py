"""Add independent article-level index build records."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision: str = "20260831_0009"
down_revision: str | None = "20260831_0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    inspector = inspect(op.get_bind())
    if inspector.has_table("article_index_builds"):
        return
    op.create_table(
        "article_index_builds",
        sa.Column("id", sa.String(length=96), primary_key=True),
        sa.Column("profile_id", sa.String(length=96), nullable=False),
        sa.Column("corpus_snapshot", sa.String(length=64), nullable=False),
        sa.Column("path", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=40), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("error_type", sa.Text(), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    for name in ("profile_id", "status", "active"):
        op.create_index(
            f"ix_article_index_builds_{name}", "article_index_builds", [name]
        )


def downgrade() -> None:
    if inspect(op.get_bind()).has_table("article_index_builds"):
        op.drop_table("article_index_builds")
