"""Pydantic schemas for ML model endpoints."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelType(str, Enum):
    """Supported model types."""

    EBM = "EBM"
    DECISION_TREE = "DECISION_TREE"
    PROPHET = "PROPHET"


class TrainingStatus(str, Enum):
    """Training job status."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


# Request schemas
class TrainEBMRequest(BaseModel):
    """Request to train an EBM model."""

    target_column: str = Field(default="churned", description="Target column name")
    feature_columns: Optional[list[str]] = Field(
        default=None, description="Feature columns (if None, uses all except target)"
    )
    test_size: float = Field(default=0.2, ge=0.1, le=0.5, description="Fraction for test split")
    random_state: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    version: Optional[str] = Field(default=None, description="Model version")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "target_column": "churned",
                "test_size": 0.2,
                "random_state": 42,
            }
        }
    )


class TrainDecisionTreeRequest(BaseModel):
    """Request to train a Decision Tree model."""

    target_column: str = Field(default="churned", description="Target column name")
    feature_columns: Optional[list[str]] = Field(default=None)
    max_depth: int = Field(default=5, ge=2, le=20, description="Maximum tree depth")
    test_size: float = Field(default=0.2, ge=0.1, le=0.5)
    random_state: Optional[int] = Field(default=None)
    version: Optional[str] = Field(default=None)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "target_column": "churned",
                "max_depth": 5,
                "test_size": 0.2,
            }
        }
    )


class TrainProphetRequest(BaseModel):
    """Request to train a Prophet forecast model."""

    date_column: str = Field(default="ds", description="Date column name")
    value_column: str = Field(default="y", description="Value column name")
    forecast_type: str = Field(
        default="headcount", description="Type of forecast (headcount, demand)"
    )
    periods: int = Field(default=90, ge=1, le=365, description="Number of periods to forecast")
    freq: str = Field(default="D", description="Frequency (D=daily, W=weekly, M=monthly)")
    yearly_seasonality: bool = Field(default=True)
    weekly_seasonality: bool = Field(default=True)
    daily_seasonality: bool = Field(default=False)
    version: Optional[str] = Field(default=None)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "forecast_type": "headcount",
                "periods": 90,
                "freq": "D",
            }
        }
    )


class PredictRequest(BaseModel):
    """Request for prediction with local explanation."""

    features: dict[str, Any] = Field(description="Feature values for prediction")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "features": {
                    "tenure_days": 45,
                    "avg_jobs_per_week": 8.5,
                    "avg_rating": 3.2,
                    "completion_rate": 0.75,
                    "jobs_last_30d": 25,
                    "region_job_density": 0.8,
                    "competition_index": 0.7,
                }
            }
        }
    )


# Response schemas
class MetricsResponse(BaseModel):
    """Model metrics response."""

    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc_roc: Optional[float] = None
    mae: Optional[float] = None
    mape: Optional[float] = None
    rmse: Optional[float] = None
    training_samples: Optional[int] = None
    test_samples: Optional[int] = None
    feature_count: Optional[int] = None
    training_points: Optional[int] = None


class FeatureImportanceResponse(BaseModel):
    """Feature importance response."""

    features: list[str]
    importances: list[float]


class TrainedModelResponse(BaseModel):
    """Response for a trained model."""

    id: UUID
    name: str
    version: str
    model_type: ModelType
    target: str
    file_path: str
    metrics: Optional[MetricsResponse] = None
    feature_importance: Optional[FeatureImportanceResponse] = None
    training_config: Optional[dict[str, Any]] = None
    is_active: bool
    primary_score: Optional[float] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class TrainedModelListResponse(BaseModel):
    """Response for list of trained models."""

    models: list[TrainedModelResponse]
    total: int


class TrainingJobResponse(BaseModel):
    """Response for a training job."""

    job_id: str
    status: TrainingStatus
    model_type: ModelType
    message: str
    model_id: Optional[UUID] = None


class GlobalExplanationResponse(BaseModel):
    """Response for global model explanations."""

    type: str = "global"
    features: list[str]
    scores: list[float]


class LocalExplanationResponse(BaseModel):
    """Response for local prediction explanation."""

    type: str = "local"
    prediction: int
    probability: float
    predicted_class: str
    feature_contributions: list[dict[str, Any]]
    base_value: float
    natural_language: str


class ForecastResponse(BaseModel):
    """Response for Prophet forecast."""

    dates: list[str]
    yhat: list[float]
    yhat_lower: list[float]
    yhat_upper: list[float]
    periods: int
    freq: str
