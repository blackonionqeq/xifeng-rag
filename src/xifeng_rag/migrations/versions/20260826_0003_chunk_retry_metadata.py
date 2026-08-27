"""Store provider and one-off prompt metadata for chunk retries."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision: str = "20260826_0003"
down_revision: str | None = "20260825_0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _columns() -> set[str]:
    return {column["name"] for column in inspect(op.get_bind()).get_columns("chunk_proposals")}


def upgrade() -> None:
    if not inspect(op.get_bind()).has_table("chunk_proposals"):
        return
    columns = _columns()
    if "provider_id" not in columns:
        op.add_column(
            "chunk_proposals",
            sa.Column("provider_id", sa.Text(), nullable=False, server_default="ollama"),
        )
    if "prompt_text" not in columns:
        op.add_column("chunk_proposals", sa.Column("prompt_text", sa.Text(), nullable=True))
    if "prompt_hash" not in columns:
        op.add_column("chunk_proposals", sa.Column("prompt_hash", sa.String(64), nullable=True))
    if "retry_id" not in columns:
        op.add_column("chunk_proposals", sa.Column("retry_id", sa.String(64), nullable=True))
        op.create_index(
            "ix_chunk_proposals_retry_id", "chunk_proposals", ["retry_id"], unique=False
        )


def downgrade() -> None:
    if not inspect(op.get_bind()).has_table("chunk_proposals"):
        return
    columns = _columns()
    if "retry_id" in columns:
        op.drop_index("ix_chunk_proposals_retry_id", table_name="chunk_proposals")
        op.drop_column("chunk_proposals", "retry_id")
    if "prompt_hash" in columns:
        op.drop_column("chunk_proposals", "prompt_hash")
    if "prompt_text" in columns:
        op.drop_column("chunk_proposals", "prompt_text")
    if "provider_id" in columns:
        op.drop_column("chunk_proposals", "provider_id")
