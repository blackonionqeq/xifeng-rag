"""Add difficulty and origin metadata to evaluation questions."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision: str = "20261019_0011"
down_revision: str | None = "20260901_0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    if not inspect(op.get_bind()).has_table("evaluation_questions"):
        return
    columns = {item["name"] for item in inspect(op.get_bind()).get_columns("evaluation_questions")}
    if "difficulty" not in columns:
        op.add_column(
            "evaluation_questions",
            sa.Column("difficulty", sa.String(length=20), nullable=False, server_default="medium"),
        )
    if "origin" not in columns:
        op.add_column(
            "evaluation_questions",
            sa.Column("origin", sa.String(length=20), nullable=False, server_default="new_dev"),
        )
        op.execute(
            "UPDATE evaluation_questions SET origin = 'legacy_seen' "
            "WHERE dataset_id IN (SELECT id FROM evaluation_datasets WHERE purpose = 'pilot')"
        )


def downgrade() -> None:
    if not inspect(op.get_bind()).has_table("evaluation_questions"):
        return
    columns = {item["name"] for item in inspect(op.get_bind()).get_columns("evaluation_questions")}
    if "origin" in columns:
        op.drop_column("evaluation_questions", "origin")
    if "difficulty" in columns:
        op.drop_column("evaluation_questions", "difficulty")
