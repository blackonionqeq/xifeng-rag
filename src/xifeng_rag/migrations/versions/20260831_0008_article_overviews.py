"""Add original publication time and reviewable article overviews."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision: str = "20260831_0008"
down_revision: str | None = "20260828_0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    inspector = inspect(op.get_bind())
    if inspector.has_table("articles"):
        columns = {item["name"] for item in inspector.get_columns("articles")}
        if "published_at" not in columns:
            op.add_column("articles", sa.Column("published_at", sa.DateTime(), nullable=True))
    if not inspector.has_table("article_overviews"):
        op.create_table(
            "article_overviews",
            sa.Column("id", sa.String(length=96), primary_key=True),
            sa.Column(
                "version_id",
                sa.String(length=96),
                sa.ForeignKey("article_versions.id"),
                nullable=False,
            ),
            sa.Column(
                "proposal_id",
                sa.String(length=96),
                sa.ForeignKey("chunk_proposals.id"),
                nullable=True,
            ),
            sa.Column(
                "manifest_id",
                sa.String(length=96),
                sa.ForeignKey("approved_manifests.id"),
                nullable=True,
            ),
            sa.Column("article_summary", sa.Text(), nullable=False),
            sa.Column("status", sa.String(length=40), nullable=False),
            sa.Column("provider_id", sa.Text(), nullable=False),
            sa.Column("model_id", sa.Text(), nullable=False),
            sa.Column("prompt_version", sa.String(length=40), nullable=False),
            sa.Column("generation_json", sa.Text(), nullable=False, server_default="{}"),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("approved_at", sa.DateTime(), nullable=True),
        )
        for name in ("version_id", "proposal_id", "manifest_id", "status"):
            op.create_index(f"ix_article_overviews_{name}", "article_overviews", [name])


def downgrade() -> None:
    inspector = inspect(op.get_bind())
    if inspector.has_table("article_overviews"):
        op.drop_table("article_overviews")
    if inspector.has_table("articles"):
        columns = {item["name"] for item in inspector.get_columns("articles")}
        if "published_at" in columns:
            op.drop_column("articles", "published_at")
