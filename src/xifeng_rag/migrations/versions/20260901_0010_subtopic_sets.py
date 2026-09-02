"""Add optional reviewed subtopic sets."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision: str = "20260901_0010"
down_revision: str | None = "20260831_0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    if inspect(op.get_bind()).has_table("subtopic_sets"):
        return
    op.create_table(
        "subtopic_sets",
        sa.Column("id", sa.String(length=96), primary_key=True),
        sa.Column(
            "version_id",
            sa.String(length=96),
            sa.ForeignKey("article_versions.id"),
            nullable=False,
        ),
        sa.Column(
            "manifest_id",
            sa.String(length=96),
            sa.ForeignKey("approved_manifests.id"),
            nullable=False,
        ),
        sa.Column(
            "source_proposal_id",
            sa.String(length=96),
            sa.ForeignKey("chunk_proposals.id"),
            nullable=True,
        ),
        sa.Column("status", sa.String(length=40), nullable=False),
        sa.Column("provider_id", sa.Text(), nullable=False),
        sa.Column("model_id", sa.Text(), nullable=False),
        sa.Column("prompt_version", sa.String(length=64), nullable=False),
        sa.Column("schema_version", sa.String(length=64), nullable=False),
        sa.Column("subtopics_json", sa.Text(), nullable=False, server_default='{"topics": []}'),
        sa.Column("diagnostics_json", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("generation_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True),
    )
    for name in ("version_id", "manifest_id", "source_proposal_id", "status"):
        op.create_index(f"ix_subtopic_sets_{name}", "subtopic_sets", [name])


def downgrade() -> None:
    if inspect(op.get_bind()).has_table("subtopic_sets"):
        op.drop_table("subtopic_sets")
