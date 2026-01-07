"""InterpretML service for training explainable models."""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from interpret.glassbox import ClassificationTree, ExplainableBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from app.config import get_settings

settings = get_settings()


class InterpretMLService:
    """Service for training and using InterpretML models."""

    def __init__(self, models_dir: Optional[Path] = None):
        """Initialize the service.

        Args:
            models_dir: Directory to store trained models. Defaults to settings.models_dir.
        """
        self.models_dir = models_dir or settings.models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def _get_model_path(self, model_type: str, version: str) -> Path:
        """Get the path for a model file."""
        model_dir = self.models_dir / model_type.lower()
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir / f"{version}.sav"

    def train_ebm(
        self,
        df: pd.DataFrame,
        target_column: str = "churned",
        feature_columns: Optional[list[str]] = None,
        test_size: float = 0.2,
        random_state: Optional[int] = None,
        version: Optional[str] = None,
    ) -> dict[str, Any]:
        """Train an Explainable Boosting Machine for retention/churn prediction.

        Args:
            df: DataFrame with features and target
            target_column: Name of the target column (0/1 for classification)
            feature_columns: List of feature columns to use. If None, uses all except target.
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
            version: Model version string. If None, generates from timestamp.

        Returns:
            Dictionary with model metadata, metrics, and feature importance
        """
        random_state = random_state or settings.interpret_random_state
        version = (
            version
            or f"{settings.model_version_prefix}{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        )

        # Prepare features
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col != target_column]

        X = df[feature_columns]
        y = df[target_column]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Train EBM
        ebm = ExplainableBoostingClassifier(
            n_jobs=settings.interpret_n_jobs,
            random_state=random_state,
            feature_names=feature_columns,
        )
        ebm.fit(X_train, y_train)

        # Evaluate
        y_pred = ebm.predict(X_test)
        y_proba = ebm.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),  # type: ignore[arg-type]
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),  # type: ignore[arg-type]
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),  # type: ignore[arg-type]
            "auc_roc": float(roc_auc_score(y_test, y_proba)),
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "feature_count": len(feature_columns),
        }

        # Extract feature importance
        feature_importance = self._extract_ebm_importance(ebm, feature_columns)

        # Save model
        model_path = self._get_model_path("ebm", version)
        joblib.dump(ebm, model_path)

        return {
            "model_id": str(uuid.uuid4()),
            "name": "EBM Retention Predictor",
            "version": version,
            "model_type": "EBM",
            "target": target_column,
            "file_path": str(model_path),
            "metrics": metrics,
            "feature_importance": feature_importance,
            "training_config": {
                "test_size": test_size,
                "random_state": random_state,
                "feature_columns": feature_columns,
            },
            "primary_score": metrics["auc_roc"],
        }

    def train_decision_tree(
        self,
        df: pd.DataFrame,
        target_column: str = "churned",
        feature_columns: Optional[list[str]] = None,
        max_depth: int = 5,
        test_size: float = 0.2,
        random_state: Optional[int] = None,
        version: Optional[str] = None,
    ) -> dict[str, Any]:
        """Train a Decision Tree for explainability dashboard.

        Args:
            df: DataFrame with features and target
            target_column: Name of the target column
            feature_columns: List of feature columns to use
            max_depth: Maximum depth of the tree (for interpretability)
            test_size: Fraction of data for testing
            random_state: Random seed
            version: Model version string

        Returns:
            Dictionary with model metadata and metrics
        """
        random_state = random_state or settings.interpret_random_state
        version = (
            version
            or f"{settings.model_version_prefix}{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        )

        # Prepare features
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col != target_column]

        X = df[feature_columns]
        y = df[target_column]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Train Decision Tree
        tree = ClassificationTree(
            max_depth=max_depth,
            random_state=random_state,
            feature_names=feature_columns,
        )
        tree.fit(X_train, y_train)

        # Evaluate
        y_pred = tree.predict(X_test)
        y_proba = tree.predict_proba(X_test)[:, 1]  # type: ignore[index]

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),  # type: ignore[arg-type]
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),  # type: ignore[arg-type]
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),  # type: ignore[arg-type]
            "auc_roc": float(roc_auc_score(y_test, y_proba)),
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "feature_count": len(feature_columns),
            "max_depth": max_depth,
        }

        # Extract feature importance
        feature_importance = self._extract_tree_importance(tree, feature_columns)

        # Save model
        model_path = self._get_model_path("decision_tree", version)
        joblib.dump(tree, model_path)

        return {
            "model_id": str(uuid.uuid4()),
            "name": "Decision Tree Explainer",
            "version": version,
            "model_type": "DECISION_TREE",
            "target": target_column,
            "file_path": str(model_path),
            "metrics": metrics,
            "feature_importance": feature_importance,
            "training_config": {
                "max_depth": max_depth,
                "test_size": test_size,
                "random_state": random_state,
                "feature_columns": feature_columns,
            },
            "primary_score": metrics["auc_roc"],
        }

    def _extract_ebm_importance(
        self, ebm: ExplainableBoostingClassifier, feature_columns: list[str]
    ) -> dict[str, Any]:
        """Extract feature importance from EBM."""
        # Get global explanation
        global_exp = ebm.explain_global()

        # Extract importances
        importances = []
        for i, name in enumerate(global_exp.data()["names"]):
            if name in feature_columns:
                scores = global_exp.data()["scores"][i]
                # Use mean absolute score as importance
                importance = (
                    float(np.mean(np.abs(scores)))
                    if hasattr(scores, "__len__")
                    else float(abs(scores))
                )
                importances.append({"feature": name, "importance": importance})

        # Sort by importance
        importances.sort(key=lambda x: x["importance"], reverse=True)

        return {
            "features": [item["feature"] for item in importances],
            "importances": [item["importance"] for item in importances],
        }

    def _extract_tree_importance(
        self, tree: ClassificationTree, feature_columns: list[str]
    ) -> dict[str, Any]:
        """Extract feature importance from Decision Tree."""
        # Get global explanation
        global_exp = tree.explain_global()
        exp_data = global_exp.data()

        # Decision tree may have different data structure
        importances = []

        # Try different keys that interpret might use
        if "names" in exp_data and "scores" in exp_data:
            for i, name in enumerate(exp_data["names"]):
                if name in feature_columns:
                    importance = float(exp_data["scores"][i])
                    importances.append({"feature": name, "importance": importance})
        elif hasattr(tree, "feature_importances_"):
            # Fallback to sklearn-style feature importances
            for name, importance in zip(feature_columns, tree.feature_importances_):  # type: ignore[union-attr]
                importances.append({"feature": name, "importance": float(importance)})
        else:
            # Last resort: use uniform importance
            for name in feature_columns:
                importances.append({"feature": name, "importance": 1.0 / len(feature_columns)})

        # Sort by importance
        importances.sort(key=lambda x: x["importance"], reverse=True)

        return {
            "features": [item["feature"] for item in importances],
            "importances": [item["importance"] for item in importances],
        }

    def load_model(self, model_path: str) -> Any:
        """Load a trained model from disk."""
        return joblib.load(model_path)

    def get_global_explanations(self, model_path: str) -> dict[str, Any]:
        """Get global feature importance from a trained model.

        Args:
            model_path: Path to the saved model

        Returns:
            Dictionary with feature names and their global importance scores
        """
        model = self.load_model(model_path)
        global_exp = model.explain_global()

        return {
            "type": "global",
            "features": list(global_exp.data()["names"]),
            "scores": [
                float(s) if isinstance(s, (int, float)) else float(np.mean(np.abs(s)))
                for s in global_exp.data()["scores"]
            ],
        }

    def get_local_explanation(
        self,
        model_path: str,
        features: dict[str, Any],
        feature_columns: list[str],
    ) -> dict[str, Any]:
        """Get local explanation for a single prediction.

        Args:
            model_path: Path to the saved model
            features: Dictionary of feature name -> value for the instance
            feature_columns: List of feature column names in correct order

        Returns:
            Dictionary with prediction, probability, and feature contributions
        """
        model = self.load_model(model_path)

        # Create DataFrame with single row
        X = pd.DataFrame([features])[feature_columns]

        # Get prediction
        prediction = int(model.predict(X)[0])
        probability = float(model.predict_proba(X)[0, 1])

        # Get local explanation
        local_exp = model.explain_local(X)

        # Extract feature contributions
        contributions = []
        exp_data = local_exp.data(0)

        for i, name in enumerate(exp_data["names"]):
            if name in feature_columns:
                contribution = exp_data["scores"][i]
                if hasattr(contribution, "__len__"):
                    contribution = float(np.mean(contribution))
                else:
                    contribution = float(contribution)

                contributions.append(
                    {
                        "feature": name,
                        "value": features.get(name),
                        "contribution": contribution,
                    }
                )

        # Sort by absolute contribution
        contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)

        # Generate natural language explanation
        nl_explanation = self._generate_natural_language_explanation(
            prediction, probability, contributions[:5]
        )

        return {
            "type": "local",
            "prediction": prediction,
            "probability": probability,
            "predicted_class": "HIGH_RISK"
            if probability > 0.7
            else "MEDIUM_RISK"
            if probability > 0.4
            else "LOW_RISK",
            "feature_contributions": contributions,
            "base_value": float(exp_data.get("perf", {}).get("base_value", 0.5))
            if "perf" in exp_data
            else 0.5,
            "natural_language": nl_explanation,
        }

    def _generate_natural_language_explanation(
        self,
        prediction: int,
        probability: float,
        top_contributions: list[dict[str, Any]],
    ) -> str:
        """Generate a natural language explanation of the prediction.

        Args:
            prediction: Binary prediction (0 or 1)
            probability: Probability of positive class
            top_contributions: Top feature contributions

        Returns:
            Natural language explanation string
        """
        risk_level = "high" if probability > 0.7 else "moderate" if probability > 0.4 else "low"
        outcome = "churning" if prediction == 1 else "staying"

        # Build explanation parts
        parts = []
        for contrib in top_contributions[:3]:
            feature = contrib["feature"].replace("_", " ")
            value = contrib["value"]
            contribution = contrib["contribution"]

            direction = "increases" if contribution > 0 else "decreases"
            impact = abs(contribution)

            if isinstance(value, (int, float)):
                parts.append(f"their {feature} of {value:.1f} {direction} risk by {impact:.2f}")
            else:
                parts.append(f"their {feature} {direction} risk by {impact:.2f}")

        explanation = f"This technician has a {probability:.0%} probability of {outcome} ({risk_level} risk). "

        if parts:
            explanation += "Key factors: " + "; ".join(parts) + "."

        return explanation

    def predict_batch(
        self,
        model_path: str,
        df: pd.DataFrame,
        feature_columns: list[str],
    ) -> pd.DataFrame:
        """Make predictions for multiple instances.

        Args:
            model_path: Path to the saved model
            df: DataFrame with features
            feature_columns: List of feature columns

        Returns:
            DataFrame with predictions and probabilities added
        """
        model = self.load_model(model_path)

        X = df[feature_columns]

        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]

        result = df.copy()
        result["prediction"] = predictions
        result["probability"] = probabilities
        result["risk_class"] = pd.cut(
            probabilities,
            bins=[0, 0.4, 0.7, 1.0],
            labels=["LOW_RISK", "MEDIUM_RISK", "HIGH_RISK"],
        )

        return result
