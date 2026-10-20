"""Add end-to-end answer evaluation runs and items."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision: str = "20261020_0012"
down_revision: str | None = "20261019_0011"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    inspector = inspect(op.get_bind())
    if not inspector.has_table("answer_evaluation_runs"):
        op.create_table(
            "answer_evaluation_runs",
            sa.Column("id", sa.String(length=96), primary_key=True),
            sa.Column("dataset_id", sa.String(length=96), sa.ForeignKey("evaluation_datasets.id")),
            sa.Column("build_id", sa.String(length=96), sa.ForeignKey("index_builds.id")),
            sa.Column("dataset_content_hash", sa.String(length=64), nullable=False),
            sa.Column("corpus_snapshot", sa.String(length=64), nullable=False),
            sa.Column("split", sa.String(length=20), nullable=False),
            sa.Column("retrieval_mode", sa.String(length=20), nullable=False),
            sa.Column("k", sa.Integer(), nullable=False),
            sa.Column("provider_id", sa.Text(), nullable=False),
            sa.Column("model_id", sa.Text(), nullable=False),
            sa.Column("supplement_policy", sa.String(length=20), nullable=False),
            sa.Column("status", sa.String(length=40), nullable=False),
            sa.Column("metrics_json", sa.Text(), nullable=False, server_default="{}"),
            sa.Column("error_type", sa.Text(), nullable=True),
            sa.Column("code_commit", sa.String(length=64), nullable=True),
            sa.Column("worktree_dirty", sa.Boolean(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        )
        for name in ("dataset_id", "build_id", "split", "status"):
            op.create_index(f"ix_answer_evaluation_runs_{name}", "answer_evaluation_runs", [name])
    inspector = inspect(op.get_bind())
    if not inspector.has_table("answer_evaluation_items"):
        op.create_table(
            "answer_evaluation_items",
            sa.Column("id", sa.String(length=120), primary_key=True),
            sa.Column("run_id", sa.String(length=96), sa.ForeignKey("answer_evaluation_runs.id")),
            sa.Column(
                "question_id",
                sa.String(length=96),
                sa.ForeignKey("evaluation_questions.id"),
            ),
            sa.Column("ordinal", sa.Integer(), nullable=False),
            sa.Column("answerable", sa.Boolean(), nullable=False),
            sa.Column("status", sa.String(length=40), nullable=False),
            sa.Column("answer_json", sa.Text(), nullable=True),
            sa.Column("citations_json", sa.Text(), nullable=False, server_default="[]"),
            sa.Column("metrics_json", sa.Text(), nullable=False, server_default="{}"),
            sa.Column("failure_labels_json", sa.Text(), nullable=False, server_default="[]"),
            sa.Column("request_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("latency_ms", sa.Float(), nullable=False, server_default="0"),
            sa.Column("usage_json", sa.Text(), nullable=False, server_default="{}"),
            sa.Column("error_type", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        )
        for name in ("run_id", "question_id", "status"):
            op.create_index(f"ix_answer_evaluation_items_{name}", "answer_evaluation_items", [name])


def downgrade() -> None:
    for table in ("answer_evaluation_items", "answer_evaluation_runs"):
        if inspect(op.get_bind()).has_table(table):
            op.drop_table(table)
