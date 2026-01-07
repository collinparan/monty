"""Prediction model for storing model predictions."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Float, ForeignKey, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import BaseModel

if TYPE_CHECKING:
    from app.models.technician import Technician
    from app.models.trained_model import TrainedModel


class Prediction(BaseModel):
    """Model representing a prediction for a technician."""

    __tablename__ = "predictions"

    # Foreign keys
    technician_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("technicians.id"), index=True, nullable=False
    )
    model_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("trained_models.id"), index=True, nullable=False
    )

    # Prediction type
    prediction_type: Mapped[str] = mapped_column(
        String(50), index=True, nullable=False
    )  # RETENTION, RECRUITMENT, CHURN_RISK

    # Prediction result
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_class: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True
    )  # HIGH_RISK, MEDIUM_RISK, LOW_RISK

    # Explanation (local feature contributions)
    explanation: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    # Expected structure:
    # {
    #     "feature_contributions": [
    #         {"feature": "tenure_days", "value": 45, "contribution": -0.15},
    #         {"feature": "avg_rating", "value": 4.2, "contribution": 0.08},
    #         ...
    #     ],
    #     "base_value": 0.5,
    #     "output_value": 0.73,
    #     "natural_language": "This technician has a 73% risk of churning..."
    # }

    # Relationships
    technician: Mapped[Technician] = relationship("Technician", lazy="selectin")
    model: Mapped[TrainedModel] = relationship("TrainedModel", lazy="selectin")

    def __repr__(self) -> str:
        return (
            f"<Prediction(id={self.id}, type={self.prediction_type}, prob={self.probability:.2f})>"
        )
