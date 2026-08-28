"""Record local reranker configuration and diagnostics on evaluation runs."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision: str = "20260828_0007"
down_revision: str | None = "20260827_0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    if not inspect(op.get_bind()).has_table("evaluation_runs"):
        return
    columns = {item["name"] for item in inspect(op.get_bind()).get_columns("evaluation_runs")}
    additions = {
        "reranker_enabled": sa.Boolean(),
        "reranker_provider": sa.Text(),
        "reranker_model_id": sa.Text(),
        "reranker_model_revision": sa.Text(),
        "reranker_device": sa.Text(),
        "reranker_batch_size": sa.Integer(),
        "reranker_max_pair_tokens": sa.Integer(),
        "reranker_candidate_k": sa.Integer(),
        "reranker_input_template_version": sa.Text(),
        "reranker_load_duration_ms": sa.Float(),
        "reranker_truncated_candidate_count": sa.Integer(),
        "reranker_process_memory_delta_bytes": sa.Integer(),
        "reranker_cuda_peak_memory_bytes": sa.Integer(),
    }
    for name, kind in additions.items():
        if name not in columns:
            op.add_column("evaluation_runs", sa.Column(name, kind, nullable=True))


def downgrade() -> None:
    if not inspect(op.get_bind()).has_table("evaluation_runs"):
        return
    columns = {item["name"] for item in inspect(op.get_bind()).get_columns("evaluation_runs")}
    names = (
        "reranker_cuda_peak_memory_bytes",
        "reranker_process_memory_delta_bytes",
        "reranker_truncated_candidate_count",
        "reranker_load_duration_ms",
        "reranker_input_template_version",
        "reranker_candidate_k",
        "reranker_max_pair_tokens",
        "reranker_batch_size",
        "reranker_device",
        "reranker_model_revision",
        "reranker_model_id",
        "reranker_provider",
        "reranker_enabled",
    )
    for name in names:
        if name in columns:
            op.drop_column("evaluation_runs", name)
