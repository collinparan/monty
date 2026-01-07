"""Add technician, region, prediction, and trained_model tables.

Revision ID: 0002
Revises: 0001
Create Date: 2024-01-06 00:00:01
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create regions table
    op.create_table(
        "regions",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("zip_codes", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("metrics", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_regions_id"), "regions", ["id"], unique=False)
    op.create_index(op.f("ix_regions_name"), "regions", ["name"], unique=True)

    # Create technicians table
    op.create_table(
        "technicians",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("external_id", sa.String(length=100), nullable=False),
        sa.Column("region", sa.String(length=50), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("tenure_days", sa.Integer(), nullable=True),
        sa.Column("metrics", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("skills", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("certifications", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("profile_text", sa.Text(), nullable=True),
        sa.Column("embedding", Vector(dim=1536), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_technicians_id"), "technicians", ["id"], unique=False)
    op.create_index(op.f("ix_technicians_external_id"), "technicians", ["external_id"], unique=True)
    op.create_index(op.f("ix_technicians_region"), "technicians", ["region"], unique=False)
    op.create_index(op.f("ix_technicians_status"), "technicians", ["status"], unique=False)

    # Create HNSW index for vector similarity search
    op.create_index(
        "ix_technicians_embedding_hnsw",
        "technicians",
        ["embedding"],
        unique=False,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )

    # Create trained_models table
    op.create_table(
        "trained_models",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("version", sa.String(length=50), nullable=False),
        sa.Column("model_type", sa.String(length=50), nullable=False),
        sa.Column("target", sa.String(length=50), nullable=False),
        sa.Column("file_path", sa.Text(), nullable=False),
        sa.Column("metrics", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("feature_importance", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("training_config", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=True),
        sa.Column("primary_score", sa.Float(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_trained_models_id"), "trained_models", ["id"], unique=False)
    op.create_index(
        op.f("ix_trained_models_model_type"), "trained_models", ["model_type"], unique=False
    )

    # Create predictions table
    op.create_table(
        "predictions",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("technician_id", sa.UUID(), nullable=False),
        sa.Column("model_id", sa.UUID(), nullable=False),
        sa.Column("prediction_type", sa.String(length=50), nullable=False),
        sa.Column("probability", sa.Float(), nullable=False),
        sa.Column("predicted_class", sa.String(length=50), nullable=True),
        sa.Column("explanation", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["model_id"], ["trained_models.id"]),
        sa.ForeignKeyConstraint(["technician_id"], ["technicians.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_predictions_id"), "predictions", ["id"], unique=False)
    op.create_index(
        op.f("ix_predictions_technician_id"), "predictions", ["technician_id"], unique=False
    )
    op.create_index(op.f("ix_predictions_model_id"), "predictions", ["model_id"], unique=False)
    op.create_index(
        op.f("ix_predictions_prediction_type"), "predictions", ["prediction_type"], unique=False
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_predictions_prediction_type"), table_name="predictions")
    op.drop_index(op.f("ix_predictions_model_id"), table_name="predictions")
    op.drop_index(op.f("ix_predictions_technician_id"), table_name="predictions")
    op.drop_index(op.f("ix_predictions_id"), table_name="predictions")
    op.drop_table("predictions")

    op.drop_index(op.f("ix_trained_models_model_type"), table_name="trained_models")
    op.drop_index(op.f("ix_trained_models_id"), table_name="trained_models")
    op.drop_table("trained_models")

    op.drop_index("ix_technicians_embedding_hnsw", table_name="technicians")
    op.drop_index(op.f("ix_technicians_status"), table_name="technicians")
    op.drop_index(op.f("ix_technicians_region"), table_name="technicians")
    op.drop_index(op.f("ix_technicians_external_id"), table_name="technicians")
    op.drop_index(op.f("ix_technicians_id"), table_name="technicians")
    op.drop_table("technicians")

    op.drop_index(op.f("ix_regions_name"), table_name="regions")
    op.drop_index(op.f("ix_regions_id"), table_name="regions")
    op.drop_table("regions")
