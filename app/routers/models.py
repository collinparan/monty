"""API endpoints for model training and management."""

from __future__ import annotations

import uuid
from typing import Any, Optional

import redis.asyncio as redis
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import TrainedModel
from app.schemas.models import (
    ForecastResponse,
    GlobalExplanationResponse,
    LocalExplanationResponse,
    ModelType,
    PredictRequest,
    TrainDecisionTreeRequest,
    TrainEBMRequest,
    TrainedModelListResponse,
    TrainedModelResponse,
    TrainingJobResponse,
    TrainingStatus,
    TrainProphetRequest,
)
from app.services.ml import InterpretMLService, ProphetService

router = APIRouter(prefix="/api/v1/models", tags=["Models"])


def get_redis_client() -> redis.Redis:
    """Get Redis client from main module."""
    from app.dependencies import get_redis

    return get_redis()


def get_interpret_service() -> InterpretMLService:
    """Get InterpretML service instance."""
    return InterpretMLService()


def get_prophet_service() -> ProphetService:
    """Get Prophet service instance."""
    return ProphetService()


async def _store_training_status(
    redis_client: redis.Redis, job_id: str, status: str, message: str, model_id: str | None = None
) -> None:
    """Store training job status in Redis."""
    data = {"status": status, "message": message}
    if model_id:
        data["model_id"] = model_id
    await redis_client.hset(f"training:{job_id}", mapping=data)  # type: ignore[misc]
    await redis_client.expire(f"training:{job_id}", 3600)  # 1 hour TTL


async def _save_trained_model(
    session: AsyncSession, result: dict[str, Any], model_type: str
) -> TrainedModel:
    """Save trained model metadata to database."""
    model = TrainedModel(
        name=result["name"],
        version=result["version"],
        model_type=model_type,
        target=result.get("target", result.get("forecast_type", "unknown")),
        file_path=result["file_path"],
        metrics=result.get("metrics"),
        feature_importance=result.get("feature_importance"),
        training_config=result.get("training_config"),
        is_active=True,
        primary_score=result.get("primary_score"),
    )
    session.add(model)
    await session.commit()
    await session.refresh(model)
    return model


@router.post("/train/ebm", response_model=TrainingJobResponse)
async def train_ebm_model(
    request: TrainEBMRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_db),
    redis_client: redis.Redis = Depends(get_redis_client),
    interpret_service: InterpretMLService = Depends(get_interpret_service),
) -> TrainingJobResponse:
    """Train an Explainable Boosting Machine model.

    The training runs as a background task. Use the returned job_id
    to check training status via GET /api/v1/models/training/{job_id}/status
    """
    job_id = str(uuid.uuid4())

    async def train_task():
        try:
            await _store_training_status(
                redis_client, job_id, TrainingStatus.RUNNING.value, "Training EBM model..."
            )

            # For now, use sample data - in production, fetch from Snowflake
            import numpy as np
            import pandas as pd

            np.random.seed(42)
            n_samples = 1000
            df = pd.DataFrame(
                {
                    "tenure_days": np.random.randint(1, 365, n_samples),
                    "avg_jobs_per_week": np.random.uniform(5, 25, n_samples),
                    "avg_rating": np.random.uniform(3.0, 5.0, n_samples),
                    "completion_rate": np.random.uniform(0.7, 1.0, n_samples),
                    "jobs_last_30d": np.random.randint(10, 100, n_samples),
                    "region_job_density": np.random.uniform(0.5, 2.0, n_samples),
                    "competition_index": np.random.uniform(0.1, 0.9, n_samples),
                    "churned": (np.random.random(n_samples) < 0.3).astype(int),
                }
            )

            result = interpret_service.train_ebm(
                df=df,
                target_column=request.target_column,
                feature_columns=request.feature_columns,
                test_size=request.test_size,
                random_state=request.random_state,
                version=request.version,
            )

            # Save to database
            model = await _save_trained_model(session, result, ModelType.EBM.value)

            await _store_training_status(
                redis_client,
                job_id,
                TrainingStatus.COMPLETED.value,
                f"Training completed. AUC-ROC: {result['metrics']['auc_roc']:.4f}",
                str(model.id),
            )

        except Exception as e:
            await _store_training_status(
                redis_client, job_id, TrainingStatus.FAILED.value, f"Training failed: {e!s}"
            )

    background_tasks.add_task(train_task)

    return TrainingJobResponse(
        job_id=job_id,
        status=TrainingStatus.PENDING,
        model_type=ModelType.EBM,
        message="Training job submitted",
    )


@router.post("/train/decision-tree", response_model=TrainingJobResponse)
async def train_decision_tree_model(
    request: TrainDecisionTreeRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_db),
    redis_client: redis.Redis = Depends(get_redis_client),
    interpret_service: InterpretMLService = Depends(get_interpret_service),
) -> TrainingJobResponse:
    """Train a Decision Tree model for explainability dashboard."""
    job_id = str(uuid.uuid4())

    async def train_task():
        try:
            await _store_training_status(
                redis_client, job_id, TrainingStatus.RUNNING.value, "Training Decision Tree..."
            )

            # Sample data
            import numpy as np
            import pandas as pd

            np.random.seed(42)
            n_samples = 1000
            df = pd.DataFrame(
                {
                    "tenure_days": np.random.randint(1, 365, n_samples),
                    "avg_jobs_per_week": np.random.uniform(5, 25, n_samples),
                    "avg_rating": np.random.uniform(3.0, 5.0, n_samples),
                    "completion_rate": np.random.uniform(0.7, 1.0, n_samples),
                    "jobs_last_30d": np.random.randint(10, 100, n_samples),
                    "region_job_density": np.random.uniform(0.5, 2.0, n_samples),
                    "competition_index": np.random.uniform(0.1, 0.9, n_samples),
                    "churned": (np.random.random(n_samples) < 0.3).astype(int),
                }
            )

            result = interpret_service.train_decision_tree(
                df=df,
                target_column=request.target_column,
                feature_columns=request.feature_columns,
                max_depth=request.max_depth,
                test_size=request.test_size,
                random_state=request.random_state,
                version=request.version,
            )

            model = await _save_trained_model(session, result, ModelType.DECISION_TREE.value)

            await _store_training_status(
                redis_client,
                job_id,
                TrainingStatus.COMPLETED.value,
                f"Training completed. Accuracy: {result['metrics']['accuracy']:.4f}",
                str(model.id),
            )

        except Exception as e:
            await _store_training_status(
                redis_client, job_id, TrainingStatus.FAILED.value, f"Training failed: {e!s}"
            )

    background_tasks.add_task(train_task)

    return TrainingJobResponse(
        job_id=job_id,
        status=TrainingStatus.PENDING,
        model_type=ModelType.DECISION_TREE,
        message="Training job submitted",
    )


@router.post("/train/prophet", response_model=TrainingJobResponse)
async def train_prophet_model(
    request: TrainProphetRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_db),
    redis_client: redis.Redis = Depends(get_redis_client),
    prophet_service: ProphetService = Depends(get_prophet_service),
) -> TrainingJobResponse:
    """Train a Prophet forecasting model."""
    job_id = str(uuid.uuid4())

    async def train_task():
        try:
            await _store_training_status(
                redis_client, job_id, TrainingStatus.RUNNING.value, "Training Prophet model..."
            )

            # Sample time series data
            from datetime import datetime, timedelta

            import numpy as np
            import pandas as pd

            np.random.seed(42)
            start_date = datetime(2023, 1, 1)
            dates = [start_date + timedelta(days=i) for i in range(365)]
            trend = np.linspace(100, 150, 365)
            weekly = np.array([10 if d.weekday() < 5 else -5 for d in dates])
            yearly = 20 * np.sin(2 * np.pi * np.arange(365) / 365)
            noise = np.random.normal(0, 5, 365)
            values = trend + weekly + yearly + noise

            df = pd.DataFrame({"ds": dates, "y": values})

            result = prophet_service.train_forecast(
                df=df,
                date_column=request.date_column,
                value_column=request.value_column,
                forecast_type=request.forecast_type,
                periods=request.periods,
                freq=request.freq,
                yearly_seasonality=request.yearly_seasonality,
                weekly_seasonality=request.weekly_seasonality,
                daily_seasonality=request.daily_seasonality,
                version=request.version,
            )

            # Adapt result for database storage
            result["target"] = request.forecast_type
            result["primary_score"] = -result["metrics"]["mape"]  # Lower is better

            model = await _save_trained_model(session, result, ModelType.PROPHET.value)

            await _store_training_status(
                redis_client,
                job_id,
                TrainingStatus.COMPLETED.value,
                f"Training completed. MAPE: {result['metrics']['mape']:.2f}%",
                str(model.id),
            )

        except Exception as e:
            await _store_training_status(
                redis_client, job_id, TrainingStatus.FAILED.value, f"Training failed: {e!s}"
            )

    background_tasks.add_task(train_task)

    return TrainingJobResponse(
        job_id=job_id,
        status=TrainingStatus.PENDING,
        model_type=ModelType.PROPHET,
        message="Training job submitted",
    )


@router.get("/training/{job_id}/status", response_model=TrainingJobResponse)
async def get_training_status(
    job_id: str,
    redis_client: redis.Redis = Depends(get_redis_client),
) -> TrainingJobResponse:
    """Get the status of a training job."""
    data = await redis_client.hgetall(f"training:{job_id}")  # type: ignore[misc]

    if not data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training job {job_id} not found",
        )

    model_id = data.get("model_id")
    return TrainingJobResponse(
        job_id=job_id,
        status=TrainingStatus(data["status"]),
        model_type=ModelType.EBM,  # Not stored, default
        message=data["message"],
        model_id=uuid.UUID(model_id) if model_id else None,
    )


@router.get("", response_model=TrainedModelListResponse)
async def list_models(
    model_type: Optional[ModelType] = None,
    is_active: Optional[bool] = None,
    limit: int = 50,
    offset: int = 0,
    session: AsyncSession = Depends(get_db),
) -> TrainedModelListResponse:
    """List trained models with optional filtering."""
    query = select(TrainedModel)

    if model_type:
        query = query.where(TrainedModel.model_type == model_type.value)
    if is_active is not None:
        query = query.where(TrainedModel.is_active == is_active)

    query = query.order_by(TrainedModel.created_at.desc()).offset(offset).limit(limit)

    result = await session.execute(query)
    models = result.scalars().all()

    # Get total count
    count_query = select(TrainedModel)
    if model_type:
        count_query = count_query.where(TrainedModel.model_type == model_type.value)
    if is_active is not None:
        count_query = count_query.where(TrainedModel.is_active == is_active)

    count_result = await session.execute(count_query)
    total = len(count_result.scalars().all())

    return TrainedModelListResponse(
        models=[TrainedModelResponse.model_validate(m) for m in models],
        total=total,
    )


@router.get("/{model_id}", response_model=TrainedModelResponse)
async def get_model(
    model_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
) -> TrainedModelResponse:
    """Get a specific trained model by ID."""
    result = await session.execute(select(TrainedModel).where(TrainedModel.id == model_id))
    model = result.scalar_one_or_none()

    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found",
        )

    return TrainedModelResponse.model_validate(model)


@router.get("/{model_id}/explanations", response_model=GlobalExplanationResponse)
async def get_global_explanations(
    model_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
    interpret_service: InterpretMLService = Depends(get_interpret_service),
) -> GlobalExplanationResponse:
    """Get global feature importance from a trained model."""
    result = await session.execute(select(TrainedModel).where(TrainedModel.id == model_id))
    model = result.scalar_one_or_none()

    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found",
        )

    if model.model_type not in (ModelType.EBM.value, ModelType.DECISION_TREE.value):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Global explanations only available for EBM and Decision Tree models",
        )

    explanation = interpret_service.get_global_explanations(model.file_path)

    return GlobalExplanationResponse(
        type="global",
        features=explanation["features"],
        scores=explanation["scores"],
    )


@router.post("/{model_id}/predict", response_model=LocalExplanationResponse)
async def predict_with_explanation(
    model_id: uuid.UUID,
    request: PredictRequest,
    session: AsyncSession = Depends(get_db),
    interpret_service: InterpretMLService = Depends(get_interpret_service),
) -> LocalExplanationResponse:
    """Get prediction with local explanation for a single instance."""
    result = await session.execute(select(TrainedModel).where(TrainedModel.id == model_id))
    model = result.scalar_one_or_none()

    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found",
        )

    if model.model_type not in (ModelType.EBM.value, ModelType.DECISION_TREE.value):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Local explanations only available for EBM and Decision Tree models",
        )

    # Get feature columns from training config
    feature_columns = (
        model.training_config.get("feature_columns", []) if model.training_config else []
    )
    if not feature_columns:
        feature_columns = list(request.features.keys())

    explanation = interpret_service.get_local_explanation(
        model_path=model.file_path,
        features=request.features,
        feature_columns=feature_columns,
    )

    return LocalExplanationResponse(
        type="local",
        prediction=explanation["prediction"],
        probability=explanation["probability"],
        predicted_class=explanation["predicted_class"],
        feature_contributions=explanation["feature_contributions"],
        base_value=explanation["base_value"],
        natural_language=explanation["natural_language"],
    )


@router.get("/{model_id}/forecast", response_model=ForecastResponse)
async def get_forecast(
    model_id: uuid.UUID,
    periods: int = 30,
    session: AsyncSession = Depends(get_db),
    prophet_service: ProphetService = Depends(get_prophet_service),
) -> ForecastResponse:
    """Get forecast predictions from a Prophet model."""
    result = await session.execute(select(TrainedModel).where(TrainedModel.id == model_id))
    model = result.scalar_one_or_none()

    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found",
        )

    if model.model_type != ModelType.PROPHET.value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Forecast only available for Prophet models",
        )

    forecast = prophet_service.predict_future(
        model_path=model.file_path,
        periods=periods,
    )

    return ForecastResponse(
        dates=forecast["dates"],
        yhat=forecast["yhat"],
        yhat_lower=forecast["yhat_lower"],
        yhat_upper=forecast["yhat_upper"],
        periods=forecast["periods"],
        freq=forecast["freq"],
    )


@router.delete("/{model_id}")
async def deactivate_model(
    model_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    """Deactivate a trained model (soft delete)."""
    result = await session.execute(select(TrainedModel).where(TrainedModel.id == model_id))
    model = result.scalar_one_or_none()

    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found",
        )

    model.is_active = False
    await session.commit()

    return {"message": f"Model {model_id} deactivated"}
