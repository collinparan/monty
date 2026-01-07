"""Prophet forecasting service for time-series predictions."""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from prophet import Prophet

from app.config import get_settings

settings = get_settings()


class ProphetService:
    """Service for time-series forecasting with Facebook Prophet."""

    def __init__(self, models_dir: Optional[Path] = None):
        """Initialize the service.

        Args:
            models_dir: Directory to store trained models. Defaults to settings.models_dir.
        """
        self.models_dir = models_dir or settings.models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def _get_model_path(self, forecast_type: str, version: str) -> Path:
        """Get the path for a model file."""
        model_dir = self.models_dir / "prophet" / forecast_type.lower()
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir / f"{version}.sav"

    def train_forecast(
        self,
        df: pd.DataFrame,
        date_column: str = "ds",
        value_column: str = "y",
        forecast_type: str = "headcount",
        periods: int = 90,
        freq: str = "D",
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False,
        version: Optional[str] = None,
    ) -> dict[str, Any]:
        """Train a Prophet model for time-series forecasting.

        Args:
            df: DataFrame with date and value columns
            date_column: Name of the date column
            value_column: Name of the value column to forecast
            forecast_type: Type of forecast (headcount, demand, etc.)
            periods: Number of periods to forecast
            freq: Frequency of forecast (D=daily, W=weekly, M=monthly)
            yearly_seasonality: Enable yearly seasonality
            weekly_seasonality: Enable weekly seasonality
            daily_seasonality: Enable daily seasonality
            version: Model version string. If None, generates from timestamp.

        Returns:
            Dictionary with model metadata, forecast, and components
        """
        version = (
            version
            or f"{settings.model_version_prefix}{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        )

        # Prepare data for Prophet (requires 'ds' and 'y' columns)
        prophet_df = df[[date_column, value_column]].copy()
        prophet_df.columns = ["ds", "y"]
        prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])

        # Initialize and train Prophet model
        model = Prophet(
            yearly_seasonality=yearly_seasonality,  # type: ignore[arg-type]
            weekly_seasonality=weekly_seasonality,  # type: ignore[arg-type]
            daily_seasonality=daily_seasonality,  # type: ignore[arg-type]
            interval_width=0.95,  # 95% confidence interval
        )
        model.fit(prophet_df)

        # Create future dataframe and predict
        future = model.make_future_dataframe(periods=periods, freq=freq)
        forecast = model.predict(future)

        # Extract forecast data in chart-friendly format
        forecast_data = self._extract_forecast_data(forecast, len(prophet_df))

        # Extract trend components
        components = self._extract_components(model, forecast)

        # Calculate metrics on historical data
        metrics = self._calculate_metrics(prophet_df, forecast)  # type: ignore[arg-type]

        # Save model
        model_path = self._get_model_path(forecast_type, version)
        joblib.dump(model, model_path)

        return {
            "model_id": str(uuid.uuid4()),
            "name": f"Prophet {forecast_type.title()} Forecast",
            "version": version,
            "model_type": "PROPHET",
            "forecast_type": forecast_type,
            "file_path": str(model_path),
            "metrics": metrics,
            "forecast": forecast_data,
            "components": components,
            "training_config": {
                "periods": periods,
                "freq": freq,
                "yearly_seasonality": yearly_seasonality,
                "weekly_seasonality": weekly_seasonality,
                "daily_seasonality": daily_seasonality,
                "training_points": len(prophet_df),
            },
        }

    def predict_future(
        self,
        model_path: str,
        periods: int = 30,
        freq: str = "D",
    ) -> dict[str, Any]:
        """Generate predictions for future periods using a trained model.

        Args:
            model_path: Path to the saved Prophet model
            periods: Number of future periods to predict
            freq: Frequency of predictions

        Returns:
            Dictionary with forecast data in chart-friendly format
        """
        model = self.load_model(model_path)

        # Create future dataframe
        future = model.make_future_dataframe(periods=periods, freq=freq)
        forecast = model.predict(future)

        # Return only future predictions (after training data)
        history_end = len(future) - periods
        future_forecast = forecast.iloc[history_end:]

        return {
            "dates": future_forecast["ds"].dt.strftime("%Y-%m-%d").tolist(),
            "yhat": future_forecast["yhat"].round(2).tolist(),
            "yhat_lower": future_forecast["yhat_lower"].round(2).tolist(),
            "yhat_upper": future_forecast["yhat_upper"].round(2).tolist(),
            "periods": periods,
            "freq": freq,
        }

    def get_trend_components(self, model_path: str) -> dict[str, Any]:
        """Get decomposed trend components from a trained model.

        Args:
            model_path: Path to the saved Prophet model

        Returns:
            Dictionary with trend, seasonality components
        """
        model = self.load_model(model_path)

        # Generate predictions to get components
        future = model.make_future_dataframe(periods=0)
        forecast = model.predict(future)

        return self._extract_components(model, forecast)

    def _extract_forecast_data(self, forecast: pd.DataFrame, history_length: int) -> dict[str, Any]:
        """Extract forecast data in chart-friendly format.

        Args:
            forecast: Prophet forecast DataFrame
            history_length: Number of historical data points

        Returns:
            Dictionary with dates and forecast values
        """
        # Split into historical fit and future prediction
        historical = forecast.iloc[:history_length]
        future = forecast.iloc[history_length:]

        return {
            "historical": {
                "dates": historical["ds"].dt.strftime("%Y-%m-%d").tolist(),
                "yhat": historical["yhat"].round(2).tolist(),
                "yhat_lower": historical["yhat_lower"].round(2).tolist(),
                "yhat_upper": historical["yhat_upper"].round(2).tolist(),
            },
            "forecast": {
                "dates": future["ds"].dt.strftime("%Y-%m-%d").tolist(),
                "yhat": future["yhat"].round(2).tolist(),
                "yhat_lower": future["yhat_lower"].round(2).tolist(),
                "yhat_upper": future["yhat_upper"].round(2).tolist(),
            },
            "combined": {
                "dates": forecast["ds"].dt.strftime("%Y-%m-%d").tolist(),
                "yhat": forecast["yhat"].round(2).tolist(),
                "yhat_lower": forecast["yhat_lower"].round(2).tolist(),
                "yhat_upper": forecast["yhat_upper"].round(2).tolist(),
            },
        }

    def _extract_components(
        self,
        model: Prophet,  # noqa: ARG002
        forecast: pd.DataFrame,
    ) -> dict[str, Any]:
        """Extract trend and seasonality components.

        Args:
            model: Trained Prophet model
            forecast: Prophet forecast DataFrame

        Returns:
            Dictionary with trend and seasonality data
        """
        components: dict[str, Any] = {
            "trend": {
                "dates": forecast["ds"].dt.strftime("%Y-%m-%d").tolist(),
                "values": forecast["trend"].round(2).tolist(),
            }
        }

        # Add seasonality components if they exist
        if "yearly" in forecast.columns:
            components["yearly_seasonality"] = {
                "dates": forecast["ds"].dt.strftime("%Y-%m-%d").tolist(),
                "values": forecast["yearly"].round(4).tolist(),
            }

        if "weekly" in forecast.columns:
            components["weekly_seasonality"] = {
                "dates": forecast["ds"].dt.strftime("%Y-%m-%d").tolist(),
                "values": forecast["weekly"].round(4).tolist(),
            }

        if "daily" in forecast.columns:
            components["daily_seasonality"] = {
                "dates": forecast["ds"].dt.strftime("%Y-%m-%d").tolist(),
                "values": forecast["daily"].round(4).tolist(),
            }

        return components

    def _calculate_metrics(
        self, actual_df: pd.DataFrame, forecast: pd.DataFrame
    ) -> dict[str, float]:
        """Calculate forecast accuracy metrics.

        Args:
            actual_df: DataFrame with actual values (columns: ds, y)
            forecast: Prophet forecast DataFrame

        Returns:
            Dictionary with MAE, MAPE, RMSE metrics
        """
        # Merge actual and predicted
        merged = actual_df.merge(forecast[["ds", "yhat"]], on="ds", how="inner")

        if len(merged) == 0:
            return {
                "mae": 0.0,
                "mape": 0.0,
                "rmse": 0.0,
                "training_points": len(actual_df),
            }

        actual = np.array(merged["y"].values)
        predicted = np.array(merged["yhat"].values)

        # Mean Absolute Error
        mae = float(np.abs(actual - predicted).mean())

        # Mean Absolute Percentage Error (avoid division by zero)
        non_zero_mask = actual != 0
        if non_zero_mask.any():
            mape = float(
                np.abs(
                    (actual[non_zero_mask] - predicted[non_zero_mask]) / actual[non_zero_mask]
                ).mean()
                * 100
            )
        else:
            mape = 0.0

        # Root Mean Squared Error
        rmse = float(np.sqrt(((actual - predicted) ** 2).mean()))

        return {
            "mae": round(mae, 4),
            "mape": round(mape, 2),
            "rmse": round(rmse, 4),
            "training_points": len(actual_df),
        }

    def load_model(self, model_path: str) -> Prophet:
        """Load a trained Prophet model from disk.

        Args:
            model_path: Path to the saved model

        Returns:
            Loaded Prophet model
        """
        return joblib.load(model_path)

    def forecast_technician_headcount(
        self,
        df: pd.DataFrame,
        region: Optional[str] = None,
        periods: int = 90,
        version: Optional[str] = None,
    ) -> dict[str, Any]:
        """Convenience method to forecast technician headcount.

        Args:
            df: DataFrame with date and headcount columns
            region: Optional region filter for the forecast name
            periods: Number of days to forecast
            version: Model version

        Returns:
            Forecast result dictionary
        """
        forecast_type = f"headcount_{region}" if region else "headcount"

        return self.train_forecast(
            df=df,
            date_column="ds",
            value_column="y",
            forecast_type=forecast_type,
            periods=periods,
            freq="D",
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            version=version,
        )

    def forecast_job_demand(
        self,
        df: pd.DataFrame,
        region: Optional[str] = None,
        periods: int = 90,
        version: Optional[str] = None,
    ) -> dict[str, Any]:
        """Convenience method to forecast job demand.

        Args:
            df: DataFrame with date and job_count columns
            region: Optional region filter for the forecast name
            periods: Number of days to forecast
            version: Model version

        Returns:
            Forecast result dictionary
        """
        forecast_type = f"demand_{region}" if region else "demand"

        return self.train_forecast(
            df=df,
            date_column="ds",
            value_column="y",
            forecast_type=forecast_type,
            periods=periods,
            freq="D",
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            version=version,
        )
