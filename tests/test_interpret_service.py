"""Tests for InterpretML service."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.services.ml.interpret_service import InterpretMLService


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create sample technician data for testing."""
    np.random.seed(42)
    n_samples = 500

    return pd.DataFrame(
        {
            "tenure_days": np.random.randint(1, 365, n_samples),
            "avg_jobs_per_week": np.random.uniform(5, 25, n_samples),
            "avg_rating": np.random.uniform(3.0, 5.0, n_samples),
            "completion_rate": np.random.uniform(0.7, 1.0, n_samples),
            "jobs_last_30d": np.random.randint(10, 100, n_samples),
            "region_job_density": np.random.uniform(0.5, 2.0, n_samples),
            "competition_index": np.random.uniform(0.1, 0.9, n_samples),
            # Target: higher churn for low tenure, low rating, high competition
            "churned": (
                (np.random.random(n_samples) < 0.3)
                | (np.random.randint(1, 365, n_samples) < 90)
                & (np.random.uniform(3.0, 5.0, n_samples) < 3.5)
            ).astype(int),
        }
    )


@pytest.fixture
def temp_models_dir() -> Path:
    """Create temporary directory for model storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestInterpretMLService:
    """Tests for InterpretMLService."""

    def test_train_ebm(self, sample_data: pd.DataFrame, temp_models_dir: Path):
        """Test EBM training."""
        service = InterpretMLService(models_dir=temp_models_dir)

        result = service.train_ebm(
            df=sample_data,
            target_column="churned",
            test_size=0.2,
            version="test_v1",
        )

        # Check result structure
        assert "model_id" in result
        assert result["name"] == "EBM Retention Predictor"
        assert result["version"] == "test_v1"
        assert result["model_type"] == "EBM"

        # Check metrics
        assert "metrics" in result
        assert "accuracy" in result["metrics"]
        assert "auc_roc" in result["metrics"]
        assert result["metrics"]["accuracy"] > 0.5  # Better than random

        # Check feature importance
        assert "feature_importance" in result
        assert "features" in result["feature_importance"]
        assert "importances" in result["feature_importance"]

        # Check model file exists
        assert Path(result["file_path"]).exists()

    def test_train_decision_tree(self, sample_data: pd.DataFrame, temp_models_dir: Path):
        """Test Decision Tree training."""
        service = InterpretMLService(models_dir=temp_models_dir)

        result = service.train_decision_tree(
            df=sample_data,
            target_column="churned",
            max_depth=4,
            test_size=0.2,
            version="tree_v1",
        )

        # Check result structure
        assert result["name"] == "Decision Tree Explainer"
        assert result["model_type"] == "DECISION_TREE"
        assert result["metrics"]["max_depth"] == 4

        # Check model file exists
        assert Path(result["file_path"]).exists()

    def test_get_local_explanation(self, sample_data: pd.DataFrame, temp_models_dir: Path):
        """Test local explanation generation."""
        service = InterpretMLService(models_dir=temp_models_dir)

        # Train model first
        feature_cols = [col for col in sample_data.columns if col != "churned"]
        result = service.train_ebm(
            df=sample_data,
            target_column="churned",
            feature_columns=feature_cols,
            version="explain_v1",
        )

        # Get explanation for a single instance
        test_features = {
            "tenure_days": 45,
            "avg_jobs_per_week": 8.5,
            "avg_rating": 3.2,
            "completion_rate": 0.75,
            "jobs_last_30d": 25,
            "region_job_density": 0.8,
            "competition_index": 0.7,
        }

        explanation = service.get_local_explanation(
            model_path=result["file_path"],
            features=test_features,
            feature_columns=feature_cols,
        )

        # Check explanation structure
        assert explanation["type"] == "local"
        assert "prediction" in explanation
        assert "probability" in explanation
        assert "predicted_class" in explanation
        assert "feature_contributions" in explanation
        assert "natural_language" in explanation

        # Check natural language explanation is generated
        assert len(explanation["natural_language"]) > 0
        assert (
            "probability" in explanation["natural_language"].lower()
            or "%" in explanation["natural_language"]
        )

    def test_get_global_explanations(self, sample_data: pd.DataFrame, temp_models_dir: Path):
        """Test global explanation extraction."""
        service = InterpretMLService(models_dir=temp_models_dir)

        # Train model first
        result = service.train_ebm(
            df=sample_data,
            target_column="churned",
            version="global_v1",
        )

        # Get global explanations
        global_exp = service.get_global_explanations(result["file_path"])

        assert global_exp["type"] == "global"
        assert "features" in global_exp
        assert "scores" in global_exp
        assert len(global_exp["features"]) == len(global_exp["scores"])

    def test_predict_batch(self, sample_data: pd.DataFrame, temp_models_dir: Path):
        """Test batch predictions."""
        service = InterpretMLService(models_dir=temp_models_dir)

        # Train model
        feature_cols = [col for col in sample_data.columns if col != "churned"]
        result = service.train_ebm(
            df=sample_data,
            target_column="churned",
            feature_columns=feature_cols,
            version="batch_v1",
        )

        # Make batch predictions
        test_df = sample_data.head(10).copy()
        predictions = service.predict_batch(
            model_path=result["file_path"],
            df=test_df,
            feature_columns=feature_cols,
        )

        assert "prediction" in predictions.columns
        assert "probability" in predictions.columns
        assert "risk_class" in predictions.columns
        assert len(predictions) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
