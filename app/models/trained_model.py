"""TrainedModel model for storing ML model metadata."""

from __future__ import annotations

from typing import Optional

from sqlalchemy import Boolean, Float, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel


class TrainedModel(BaseModel):
    """Model representing a trained ML model."""

    __tablename__ = "trained_models"

    # Model identification
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    version: Mapped[str] = mapped_column(String(50), nullable=False)
    model_type: Mapped[str] = mapped_column(
        String(50), index=True, nullable=False
    )  # EBM, DECISION_TREE, PROPHET

    # Model purpose
    target: Mapped[str] = mapped_column(String(50), nullable=False)  # RETENTION, CHURN, FORECAST

    # File storage
    file_path: Mapped[str] = mapped_column(Text, nullable=False)

    # Model metrics (JSONB for flexibility)
    metrics: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True, default=dict)
    # Expected structure for classification:
    # {
    #     "accuracy": 0.85,
    #     "precision": 0.82,
    #     "recall": 0.88,
    #     "f1_score": 0.85,
    #     "auc_roc": 0.91,
    #     "training_samples": 10000,
    #     "feature_count": 15,
    # }
    # Expected structure for Prophet:
    # {
    #     "mape": 0.12,
    #     "rmse": 45.2,
    #     "mae": 32.1,
    #     "training_samples": 730,
    # }

    # Feature importance (for interpretable models)
    feature_importance: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    # Expected structure:
    # {
    #     "features": ["tenure_days", "avg_rating", ...],
    #     "importances": [0.25, 0.18, ...],
    #     "std": [0.02, 0.01, ...],  # optional
    # }

    # Training configuration
    training_config: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Score for model comparison
    primary_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    def __repr__(self) -> str:
        return f"<TrainedModel(id={self.id}, name={self.name}, version={self.version})>"
