"""Tests for Prophet forecasting service."""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.services.ml.prophet_service import ProphetService


@pytest.fixture
def sample_timeseries() -> pd.DataFrame:
    """Create sample time series data for testing."""
    np.random.seed(42)

    # Generate 365 days of data with trend and seasonality
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(365)]

    # Base value with trend
    trend = np.linspace(100, 150, 365)

    # Add weekly seasonality (higher on weekdays)
    weekly = np.array([10 if d.weekday() < 5 else -5 for d in dates])

    # Add yearly seasonality (peak in summer)
    yearly = 20 * np.sin(2 * np.pi * np.arange(365) / 365)

    # Add noise
    noise = np.random.normal(0, 5, 365)

    values = trend + weekly + yearly + noise

    return pd.DataFrame({"ds": dates, "y": values})


@pytest.fixture
def temp_models_dir() -> Path:
    """Create temporary directory for model storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestProphetService:
    """Tests for ProphetService."""

    def test_train_forecast(self, sample_timeseries: pd.DataFrame, temp_models_dir: Path):
        """Test basic forecast training."""
        service = ProphetService(models_dir=temp_models_dir)

        result = service.train_forecast(
            df=sample_timeseries,
            date_column="ds",
            value_column="y",
            forecast_type="test_headcount",
            periods=30,
            freq="D",
            version="test_v1",
        )

        # Check result structure
        assert "model_id" in result
        assert result["name"] == "Prophet Test_Headcount Forecast"
        assert result["version"] == "test_v1"
        assert result["model_type"] == "PROPHET"
        assert result["forecast_type"] == "test_headcount"

        # Check metrics
        assert "metrics" in result
        assert "mae" in result["metrics"]
        assert "mape" in result["metrics"]
        assert "rmse" in result["metrics"]

        # Check forecast data
        assert "forecast" in result
        assert "historical" in result["forecast"]
        assert "forecast" in result["forecast"]
        assert len(result["forecast"]["forecast"]["dates"]) == 30

        # Check model file exists
        assert Path(result["file_path"]).exists()

    def test_predict_future(self, sample_timeseries: pd.DataFrame, temp_models_dir: Path):
        """Test future predictions."""
        service = ProphetService(models_dir=temp_models_dir)

        # Train model first
        result = service.train_forecast(
            df=sample_timeseries,
            forecast_type="future_test",
            periods=30,
            version="future_v1",
        )

        # Predict future
        future_pred = service.predict_future(
            model_path=result["file_path"],
            periods=14,
            freq="D",
        )

        assert len(future_pred["dates"]) == 14
        assert len(future_pred["yhat"]) == 14
        assert len(future_pred["yhat_lower"]) == 14
        assert len(future_pred["yhat_upper"]) == 14

        # Check confidence intervals make sense
        for i in range(14):
            assert future_pred["yhat_lower"][i] <= future_pred["yhat"][i]
            assert future_pred["yhat"][i] <= future_pred["yhat_upper"][i]

    def test_get_trend_components(self, sample_timeseries: pd.DataFrame, temp_models_dir: Path):
        """Test trend component extraction."""
        service = ProphetService(models_dir=temp_models_dir)

        # Train model first
        result = service.train_forecast(
            df=sample_timeseries,
            forecast_type="components_test",
            periods=30,
            yearly_seasonality=True,
            weekly_seasonality=True,
            version="comp_v1",
        )

        # Get components
        components = service.get_trend_components(result["file_path"])

        assert "trend" in components
        assert "dates" in components["trend"]
        assert "values" in components["trend"]

        # Should have yearly seasonality
        assert "yearly_seasonality" in components

        # Should have weekly seasonality
        assert "weekly_seasonality" in components

    def test_forecast_technician_headcount(
        self, sample_timeseries: pd.DataFrame, temp_models_dir: Path
    ):
        """Test convenience method for headcount forecasting."""
        service = ProphetService(models_dir=temp_models_dir)

        result = service.forecast_technician_headcount(
            df=sample_timeseries,
            region="chicago",
            periods=60,
            version="headcount_v1",
        )

        assert result["forecast_type"] == "headcount_chicago"
        assert result["training_config"]["periods"] == 60
        assert len(result["forecast"]["forecast"]["dates"]) == 60

    def test_forecast_job_demand(self, sample_timeseries: pd.DataFrame, temp_models_dir: Path):
        """Test convenience method for demand forecasting."""
        service = ProphetService(models_dir=temp_models_dir)

        result = service.forecast_job_demand(
            df=sample_timeseries,
            region="dallas",
            periods=45,
            version="demand_v1",
        )

        assert result["forecast_type"] == "demand_dallas"
        assert result["training_config"]["periods"] == 45
        assert len(result["forecast"]["forecast"]["dates"]) == 45

    def test_forecast_without_region(self, sample_timeseries: pd.DataFrame, temp_models_dir: Path):
        """Test forecast without region filter."""
        service = ProphetService(models_dir=temp_models_dir)

        # Headcount without region
        result1 = service.forecast_technician_headcount(
            df=sample_timeseries,
            periods=30,
            version="no_region_v1",
        )
        assert result1["forecast_type"] == "headcount"

        # Demand without region
        result2 = service.forecast_job_demand(
            df=sample_timeseries,
            periods=30,
            version="no_region_v2",
        )
        assert result2["forecast_type"] == "demand"

    def test_metrics_calculation(self, sample_timeseries: pd.DataFrame, temp_models_dir: Path):
        """Test that metrics are calculated correctly."""
        service = ProphetService(models_dir=temp_models_dir)

        result = service.train_forecast(
            df=sample_timeseries,
            forecast_type="metrics_test",
            periods=30,
            version="metrics_v1",
        )

        metrics = result["metrics"]

        # MAE should be positive
        assert metrics["mae"] >= 0

        # MAPE should be positive and reasonable (< 50% for this synthetic data)
        assert 0 <= metrics["mape"] < 50

        # RMSE should be positive
        assert metrics["rmse"] >= 0

        # Training points should match input
        assert metrics["training_points"] == len(sample_timeseries)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
