"""Record retrieval mode and fusion parameters on evaluation runs."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision: str = "20260827_0006"
down_revision: str | None = "20260827_0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    if not inspect(op.get_bind()).has_table("evaluation_runs"):
        return
    columns = {item["name"] for item in inspect(op.get_bind()).get_columns("evaluation_runs")}
    additions = {
        "retrieval_mode": sa.String(length=20),
        "bm25_candidate_k": sa.Integer(),
        "dense_candidate_k": sa.Integer(),
        "rrf_k": sa.Integer(),
    }
    for name, kind in additions.items():
        if name not in columns:
            op.add_column("evaluation_runs", sa.Column(name, kind, nullable=True))


def downgrade() -> None:
    if not inspect(op.get_bind()).has_table("evaluation_runs"):
        return
    columns = {item["name"] for item in inspect(op.get_bind()).get_columns("evaluation_runs")}
    for name in ("rrf_k", "dense_candidate_k", "bm25_candidate_k", "retrieval_mode"):
        if name in columns:
            op.drop_column("evaluation_runs", name)
