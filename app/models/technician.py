"""Technician model for storing technician data and predictions."""

from __future__ import annotations

from typing import Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel


class Technician(BaseModel):
    """Model representing a 1099 technician."""

    __tablename__ = "technicians"

    # External identifier from Snowflake
    external_id: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)

    # Basic info
    region: Mapped[str] = mapped_column(String(50), index=True, nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), index=True, nullable=False, default="ACTIVE"
    )  # ACTIVE, CHURNED, INACTIVE
    tenure_days: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Performance metrics (JSONB for flexibility)
    metrics: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True, default=dict)
    # Expected structure:
    # {
    #     "avg_jobs_per_week": float,
    #     "avg_rating": float,
    #     "completion_rate": float,
    #     "jobs_last_30d": int,
    #     "revenue_last_30d": float,
    # }

    # Skills and certifications
    skills: Mapped[Optional[list]] = mapped_column(JSONB, nullable=True, default=list)
    certifications: Mapped[Optional[list]] = mapped_column(JSONB, nullable=True, default=list)

    # Profile text for embedding generation
    profile_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Vector embedding for semantic search (1536 dimensions for OpenAI)
    embedding: Mapped[Optional[list]] = mapped_column(Vector(1536), nullable=True)

    # Indexes
    __table_args__ = (
        Index(
            "ix_technicians_embedding_hnsw",
            embedding,
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    def __repr__(self) -> str:
        return f"<Technician(id={self.id}, external_id={self.external_id}, region={self.region})>"
