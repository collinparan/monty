"""Region model for storing regional metrics."""

from __future__ import annotations

from typing import Optional

from sqlalchemy import String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel


class Region(BaseModel):
    """Model representing a geographic region/territory."""

    __tablename__ = "regions"

    # Region identifier
    name: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)

    # ZIP codes in this region
    zip_codes: Mapped[Optional[list]] = mapped_column(JSONB, nullable=True, default=list)

    # Regional metrics (JSONB for flexibility)
    metrics: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True, default=dict)
    # Expected structure:
    # {
    #     "total_technicians": int,
    #     "active_technicians": int,
    #     "avg_tenure_days": float,
    #     "avg_jobs_per_week": float,
    #     "avg_rating": float,
    #     "retention_rate_6mo": float,
    #     "job_density": float,
    #     "competition_index": float,
    #     "avg_payout_per_job": float,
    # }

    def __repr__(self) -> str:
        return f"<Region(id={self.id}, name={self.name})>"
