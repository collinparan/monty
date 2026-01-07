"""End-to-end tests for the training pipeline.

Tests the flow: data fetch -> model training -> prediction
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from app.services.ml.interpret_service import InterpretMLService
from app.services.ml.prophet_service import ProphetService


class TestE2ETrainingPipeline:
    """End-to-end tests for the ML training pipeline."""

    @pytest.mark.asyncio
    async def test_ebm_training_end_to_end(self, sample_training_data):
        """Test complete EBM training flow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            service = InterpretMLService(models_dir=Path(tmpdir))

            # Train EBM model
            result = service.train_ebm(
                df=sample_training_data,
                target_column="churned",
                test_size=0.2,
                random_state=42,
            )

            # Verify training output
            assert "model_id" in result
            assert result["model_type"] == "EBM"
            assert "metrics" in result
            assert result["metrics"]["auc_roc"] > 0.5  # Better than random
            assert "feature_importance" in result

            # Verify model file was saved
            model_path = Path(result["file_path"])
            assert model_path.exists()

            # Test predictions
            predictions = service.predict_batch(
                model_path=str(model_path),
                df=sample_training_data.head(5),
            )

            assert len(predictions) == 5
            for pred in predictions:
                assert "probability" in pred
                assert "risk_level" in pred
                assert pred["risk_level"] in ["LOW", "MEDIUM", "HIGH"]

    @pytest.mark.asyncio
    async def test_decision_tree_training_end_to_end(self, sample_training_data):
        """Test complete Decision Tree training flow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            service = InterpretMLService(models_dir=Path(tmpdir))

            # Train Decision Tree model
            result = service.train_decision_tree(
                df=sample_training_data,
                target_column="churned",
                max_depth=3,
                test_size=0.2,
                random_state=42,
            )

            # Verify training output
            assert "model_id" in result
            assert result["model_type"] == "DECISION_TREE"
            assert "metrics" in result
            assert "feature_importance" in result

            # Verify model file was saved
            model_path = Path(result["file_path"])
            assert model_path.exists()

    @pytest.mark.asyncio
    async def test_prophet_training_end_to_end(self, sample_forecast_data):
        """Test complete Prophet forecasting flow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            service = ProphetService(models_dir=Path(tmpdir))

            # Train Prophet model
            result = service.train_forecast(
                df=sample_forecast_data,
                date_column="ds",
                value_column="y",
                forecast_type="headcount",
                periods=30,
                freq="D",
            )

            # Verify training output
            assert "model_id" in result
            assert result["model_type"] == "PROPHET"
            assert "forecast" in result
            assert "components" in result
            assert "metrics" in result

            # Verify forecast data
            forecast = result["forecast"]
            assert len(forecast["dates"]) == 30
            assert len(forecast["yhat"]) == 30
            assert len(forecast["yhat_lower"]) == 30
            assert len(forecast["yhat_upper"]) == 30

            # Verify model file was saved
            model_path = Path(result["file_path"])
            assert model_path.exists()

            # Test future predictions
            future_result = service.predict_future(
                model_path=str(model_path),
                periods=14,
                freq="D",
            )

            assert len(future_result["dates"]) == 14

    @pytest.mark.asyncio
    async def test_local_explanation_generation(self, sample_training_data):
        """Test generating local explanations for predictions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            service = InterpretMLService(models_dir=Path(tmpdir))

            # Train model first
            training_result = service.train_ebm(
                df=sample_training_data,
                target_column="churned",
                random_state=42,
            )

            # Get local explanation for a single instance
            test_instance = sample_training_data.iloc[0:1].drop(columns=["churned"])
            explanation = service.get_local_explanation(
                model_path=training_result["file_path"],
                instance=test_instance.to_dict("records")[0],
            )

            assert "prediction" in explanation
            assert "probability" in explanation
            assert "feature_contributions" in explanation
            assert "natural_language" in explanation

            # Check natural language explanation format
            nl_explanation = explanation["natural_language"]
            assert isinstance(nl_explanation, str)
            assert len(nl_explanation) > 0

    @pytest.mark.asyncio
    async def test_global_explanation_retrieval(self, sample_training_data):
        """Test retrieving global explanations from trained model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            service = InterpretMLService(models_dir=Path(tmpdir))

            # Train model first
            training_result = service.train_ebm(
                df=sample_training_data,
                target_column="churned",
                random_state=42,
            )

            # Get global explanations
            explanations = service.get_global_explanations(
                model_path=training_result["file_path"],
            )

            assert "features" in explanations
            assert "importances" in explanations
            assert len(explanations["features"]) == len(explanations["importances"])
            assert len(explanations["features"]) > 0

            # Importances should sum to approximately 1
            total_importance = sum(explanations["importances"])
            assert 0.9 <= total_importance <= 1.1  # Allow some tolerance


class TestTrainingAPIEndpoints:
    """Test training via API endpoints."""

    @pytest.mark.asyncio
    async def test_trigger_training_job(self, test_client, sample_training_data):
        """Test triggering a training job via API."""
        # Note: This test requires the API to accept training data
        # In production, data comes from Snowflake
        # Here we test the endpoint responds correctly
        response = await test_client.post(
            "/api/v1/models/train/ebm",
            json={
                "target_column": "churned",
                "test_size": 0.2,
            },
        )

        # Should return 202 Accepted for background task
        # or 400 if Snowflake not available
        assert response.status_code in [202, 400, 503]

    @pytest.mark.asyncio
    async def test_list_models(self, test_client):
        """Test listing trained models via API."""
        response = await test_client.get("/api/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_get_training_status(self, test_client):
        """Test getting training job status."""
        # Try to get status of a non-existent job
        response = await test_client.get("/api/v1/models/training/nonexistent-job-id/status")

        # Should return 404 for non-existent job
        assert response.status_code == 404


class TestTrainingMetrics:
    """Test that training produces expected metrics."""

    @pytest.mark.asyncio
    async def test_ebm_metrics_reasonable(self, sample_training_data):
        """Test that EBM produces reasonable metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            service = InterpretMLService(models_dir=Path(tmpdir))

            result = service.train_ebm(
                df=sample_training_data,
                target_column="churned",
                random_state=42,
            )

            metrics = result["metrics"]

            # Check all expected metrics present
            assert "accuracy" in metrics
            assert "precision" in metrics
            assert "recall" in metrics
            assert "f1_score" in metrics
            assert "auc_roc" in metrics

            # Metrics should be in valid ranges
            assert 0 <= metrics["accuracy"] <= 1
            assert 0 <= metrics["precision"] <= 1
            assert 0 <= metrics["recall"] <= 1
            assert 0 <= metrics["f1_score"] <= 1
            assert 0 <= metrics["auc_roc"] <= 1

    @pytest.mark.asyncio
    async def test_prophet_metrics_reasonable(self, sample_forecast_data):
        """Test that Prophet produces reasonable metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            service = ProphetService(models_dir=Path(tmpdir))

            result = service.train_forecast(
                df=sample_forecast_data,
                forecast_type="test",
                periods=30,
            )

            metrics = result["metrics"]

            # Check expected metrics present
            assert "mae" in metrics or "mape" in metrics

            # Metrics should be non-negative
            for _key, value in metrics.items():
                if value is not None:
                    assert value >= 0
