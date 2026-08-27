"""Add reviewable evaluation datasets and stable evidence ranges."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision: str = "20260827_0005"
down_revision: str | None = "20260826_0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _add_run_columns() -> None:
    inspector = inspect(op.get_bind())
    if not inspector.has_table("evaluation_runs"):
        return
    columns = {item["name"] for item in inspector.get_columns("evaluation_runs")}
    additions = {
        "dataset_content_hash": sa.String(length=64),
        "corpus_snapshot": sa.String(length=64),
        "profile_id": sa.String(length=96),
        "k": sa.Integer(),
        "code_commit": sa.String(length=64),
        "worktree_dirty": sa.Boolean(),
    }
    for name, kind in additions.items():
        if name not in columns:
            op.add_column("evaluation_runs", sa.Column(name, kind, nullable=True))


def upgrade() -> None:
    _add_run_columns()
    inspector = inspect(op.get_bind())
    if not inspector.has_table("evaluation_datasets"):
        op.create_table(
            "evaluation_datasets",
            sa.Column("id", sa.String(length=96), primary_key=True),
            sa.Column("name", sa.Text(), nullable=False),
            sa.Column("version", sa.String(length=40), nullable=False),
            sa.Column("purpose", sa.String(length=20), nullable=False),
            sa.Column("status", sa.String(length=20), nullable=False),
            sa.Column("index_corpus_snapshot", sa.String(length=64), nullable=True),
            sa.Column("content_hash", sa.String(length=64), nullable=True),
            sa.Column("frozen_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        )
        op.create_index("ix_evaluation_datasets_purpose", "evaluation_datasets", ["purpose"])
        op.create_index("ix_evaluation_datasets_status", "evaluation_datasets", ["status"])
        op.create_index(
            "ix_evaluation_datasets_content_hash", "evaluation_datasets", ["content_hash"]
        )
    if not inspector.has_table("evaluation_snapshot_articles"):
        op.create_table(
            "evaluation_snapshot_articles",
            sa.Column("id", sa.String(length=120), primary_key=True),
            sa.Column(
                "dataset_id", sa.String(length=96), sa.ForeignKey("evaluation_datasets.id")
            ),
            sa.Column("article_id", sa.String(length=80), sa.ForeignKey("articles.id")),
            sa.Column(
                "version_id", sa.String(length=96), sa.ForeignKey("article_versions.id")
            ),
            sa.Column("canonical_hash", sa.String(length=64), nullable=False),
            sa.Column(
                "manifest_id", sa.String(length=96), sa.ForeignKey("approved_manifests.id")
            ),
        )
        op.create_index(
            "ix_evaluation_snapshot_articles_dataset_id",
            "evaluation_snapshot_articles",
            ["dataset_id"],
        )
        op.create_index(
            "ix_evaluation_snapshot_articles_article_id",
            "evaluation_snapshot_articles",
            ["article_id"],
        )
        op.create_index(
            "ix_evaluation_snapshot_articles_version_id",
            "evaluation_snapshot_articles",
            ["version_id"],
        )
    if not inspector.has_table("evaluation_questions"):
        op.create_table(
            "evaluation_questions",
            sa.Column("id", sa.String(length=96), primary_key=True),
            sa.Column(
                "dataset_id", sa.String(length=96), sa.ForeignKey("evaluation_datasets.id")
            ),
            sa.Column("ordinal", sa.Integer(), nullable=False),
            sa.Column("query", sa.Text(), nullable=False),
            sa.Column("split", sa.String(length=20), nullable=False),
            sa.Column("scope", sa.String(length=20), nullable=False),
            sa.Column("article_id", sa.String(length=80), sa.ForeignKey("articles.id")),
            sa.Column(
                "version_id", sa.String(length=96), sa.ForeignKey("article_versions.id")
            ),
            sa.Column("answerable", sa.Boolean(), nullable=False),
            sa.Column("reference_answer", sa.Text(), nullable=True),
            sa.Column("review_status", sa.String(length=20), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        )
        for name in ("dataset_id", "split", "article_id", "version_id", "review_status"):
            op.create_index(f"ix_evaluation_questions_{name}", "evaluation_questions", [name])
    if not inspector.has_table("evaluation_evidence"):
        op.create_table(
            "evaluation_evidence",
            sa.Column("id", sa.String(length=120), primary_key=True),
            sa.Column(
                "question_id", sa.String(length=96), sa.ForeignKey("evaluation_questions.id")
            ),
            sa.Column("ordinal", sa.Integer(), nullable=False),
            sa.Column("article_id", sa.String(length=80), sa.ForeignKey("articles.id")),
            sa.Column(
                "version_id", sa.String(length=96), sa.ForeignKey("article_versions.id")
            ),
            sa.Column("start_paragraph", sa.Integer(), nullable=False),
            sa.Column("end_paragraph", sa.Integer(), nullable=False),
            sa.Column("claim_group", sa.String(length=80), nullable=False),
            sa.Column("claim_text", sa.Text(), nullable=False),
            sa.Column("uncertainty", sa.Text(), nullable=False, server_default=""),
        )
        for name in ("question_id", "article_id", "version_id"):
            op.create_index(f"ix_evaluation_evidence_{name}", "evaluation_evidence", [name])
    if not inspector.has_table("evaluation_ai_proposals"):
        op.create_table(
            "evaluation_ai_proposals",
            sa.Column("id", sa.String(length=96), primary_key=True),
            sa.Column(
                "question_id", sa.String(length=96), sa.ForeignKey("evaluation_questions.id")
            ),
            sa.Column("provider_id", sa.String(length=40), nullable=False),
            sa.Column("model_id", sa.Text(), nullable=False),
            sa.Column("prompt_version", sa.String(length=40), nullable=False),
            sa.Column("schema_version", sa.String(length=40), nullable=False),
            sa.Column("status", sa.String(length=20), nullable=False),
            sa.Column("proposal_json", sa.Text(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        )
        op.create_index(
            "ix_evaluation_ai_proposals_question_id",
            "evaluation_ai_proposals",
            ["question_id"],
        )
        op.create_index(
            "ix_evaluation_ai_proposals_status", "evaluation_ai_proposals", ["status"]
        )


def downgrade() -> None:
    for table in (
        "evaluation_ai_proposals",
        "evaluation_evidence",
        "evaluation_questions",
        "evaluation_snapshot_articles",
        "evaluation_datasets",
    ):
        if inspect(op.get_bind()).has_table(table):
            op.drop_table(table)
    if inspect(op.get_bind()).has_table("evaluation_runs"):
        columns = {item["name"] for item in inspect(op.get_bind()).get_columns("evaluation_runs")}
        for name in (
            "worktree_dirty",
            "code_commit",
            "k",
            "profile_id",
            "corpus_snapshot",
            "dataset_content_hash",
        ):
            if name in columns:
                op.drop_column("evaluation_runs", name)
