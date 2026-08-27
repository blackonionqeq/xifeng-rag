"""Store the complete reviewed topic manifest."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision: str = "20260825_0002"
down_revision: str | None = "20260825_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _has_topics_json() -> bool:
    columns = inspect(op.get_bind()).get_columns("approved_manifests")
    return any(column["name"] == "topics_json" for column in columns)


def upgrade() -> None:
    # The initial migration creates metadata for a fresh database. The guard
    # keeps both fresh installs and databases created before this migration valid.
    if not _has_topics_json():
        op.add_column("approved_manifests", sa.Column("topics_json", sa.Text(), nullable=True))


def downgrade() -> None:
    if _has_topics_json():
        op.drop_column("approved_manifests", "topics_json")
