"""Tests for technician analytics agent tools."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from app.services.agent_tools import (
    _calculate_risk,
    _generate_basic_explanation,
    _generate_roi_recommendation,
    _get_region_health,
    _interpret_feature,
    create_technician_analytics_tools,
)


class MockSession:
    """Mock async database session."""

    def __init__(self, data=None):
        self.data = data or []

    async def execute(self, query):
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = self.data
        mock_result.scalar_one_or_none.return_value = self.data[0] if self.data else None
        mock_result.all.return_value = self.data
        return mock_result


@asynccontextmanager
async def mock_db_factory(technicians=None, models=None):
    """Mock database session factory."""
    session = MagicMock()

    async def mock_execute(query):
        result = MagicMock()
        # Check query type by inspecting the query
        query_str = str(query)

        if "technician" in query_str.lower() or technicians:
            result.scalars.return_value.all.return_value = technicians or []
            result.scalar_one_or_none.return_value = technicians[0] if technicians else None
        elif models:
            result.scalars.return_value.all.return_value = models
            result.scalar_one_or_none.return_value = models[0] if models else None
        else:
            result.scalars.return_value.all.return_value = []
            result.scalar_one_or_none.return_value = None
            result.all.return_value = []

        return result

    session.execute = mock_execute
    yield session


class TestCalculateRisk:
    """Tests for risk calculation helper."""

    def test_churned_technician_is_high_risk(self):
        """Churned technicians should always be high risk."""
        tech = MagicMock()
        tech.status = "CHURNED"
        tech.tenure_days = 500  # Doesn't matter if churned

        risk_score, risk_level = _calculate_risk(tech)

        assert risk_score == 1.0
        assert risk_level == "HIGH"

    def test_new_technician_high_risk(self):
        """New technicians (<90 days) should be high risk."""
        tech = MagicMock()
        tech.status = "ACTIVE"
        tech.tenure_days = 30

        risk_score, risk_level = _calculate_risk(tech)

        assert risk_score == 0.7
        assert risk_level == "HIGH"

    def test_medium_tenure_medium_risk(self):
        """Technicians with 90-180 days should be medium risk."""
        tech = MagicMock()
        tech.status = "ACTIVE"
        tech.tenure_days = 120

        risk_score, risk_level = _calculate_risk(tech)

        assert risk_score == 0.5
        assert risk_level == "MEDIUM"

    def test_longer_tenure_lower_risk(self):
        """Technicians with 180-365 days should be lower risk."""
        tech = MagicMock()
        tech.status = "ACTIVE"
        tech.tenure_days = 250

        risk_score, risk_level = _calculate_risk(tech)

        assert risk_score == 0.3
        assert risk_level == "LOW"

    def test_veteran_technician_low_risk(self):
        """Veteran technicians (>365 days) should be low risk."""
        tech = MagicMock()
        tech.status = "ACTIVE"
        tech.tenure_days = 500

        risk_score, risk_level = _calculate_risk(tech)

        assert risk_score == 0.15
        assert risk_level == "LOW"


class TestGenerateBasicExplanation:
    """Tests for basic explanation generation."""

    def test_high_risk_explanation(self):
        """High risk technicians get appropriate explanation."""
        tech = MagicMock()
        tech.tenure_days = 30

        explanation = _generate_basic_explanation(tech, 0.8)

        assert "high churn risk" in explanation.lower()

    def test_low_risk_explanation(self):
        """Low risk technicians get appropriate explanation."""
        tech = MagicMock()
        tech.tenure_days = 500

        explanation = _generate_basic_explanation(tech, 0.2)

        assert "low churn risk" in explanation.lower()

    def test_tenure_mentioned_in_explanation(self):
        """Tenure should be mentioned in explanations."""
        tech = MagicMock()
        tech.tenure_days = 30

        explanation = _generate_basic_explanation(tech, 0.7)

        assert "30 days" in explanation


class TestGenerateROIRecommendation:
    """Tests for ROI recommendation generation."""

    def test_high_roi_strongly_recommended(self):
        """High ROI should be strongly recommended."""
        rec = _generate_roi_recommendation(2.5, 100, {"HIGH": 10, "MEDIUM": 40, "LOW": 50})
        assert "Strongly recommended" in rec

    def test_good_roi_recommended(self):
        """Good ROI should be recommended."""
        rec = _generate_roi_recommendation(1.5, 100, {"HIGH": 10, "MEDIUM": 40, "LOW": 50})
        assert "Recommended" in rec
        assert "Strongly" not in rec

    def test_marginal_roi_consider_carefully(self):
        """Marginal ROI should suggest careful consideration."""
        rec = _generate_roi_recommendation(0.3, 100, {"HIGH": 10, "MEDIUM": 40, "LOW": 50})
        assert "Consider carefully" in rec

    def test_negative_roi_not_recommended(self):
        """Negative ROI should not be recommended."""
        rec = _generate_roi_recommendation(-0.5, 100, {"HIGH": 10, "MEDIUM": 40, "LOW": 50})
        assert "Not recommended" in rec

    def test_high_risk_concentration_noted(self):
        """High concentration of at-risk technicians should be noted."""
        rec = _generate_roi_recommendation(1.5, 100, {"HIGH": 60, "MEDIUM": 30, "LOW": 10})
        assert "at-risk technicians" in rec.lower()


class TestInterpretFeature:
    """Tests for feature interpretation."""

    def test_tenure_interpretation(self):
        """Tenure should have specific interpretation."""
        interp = _interpret_feature("tenure_days")
        assert "long" in interp.lower() or "company" in interp.lower()

    def test_region_interpretation(self):
        """Region should have specific interpretation."""
        interp = _interpret_feature("region")
        assert "geographic" in interp.lower() or "market" in interp.lower()

    def test_unknown_feature_default(self):
        """Unknown features should get default interpretation."""
        interp = _interpret_feature("unknown_feature_xyz")
        assert "retention" in interp.lower() or "factor" in interp.lower()


class TestGetRegionHealth:
    """Tests for region health status."""

    def test_critical_health(self):
        """High churn rate should be critical."""
        health = _get_region_health(0.35)
        assert health == "CRITICAL"

    def test_at_risk_health(self):
        """Moderate-high churn rate should be at risk."""
        health = _get_region_health(0.25)
        assert health == "AT_RISK"

    def test_moderate_health(self):
        """Moderate churn rate should be moderate."""
        health = _get_region_health(0.15)
        assert health == "MODERATE"

    def test_healthy(self):
        """Low churn rate should be healthy."""
        health = _get_region_health(0.05)
        assert health == "HEALTHY"


class TestToolRegistry:
    """Tests for tool registry creation."""

    def test_create_registry_has_all_tools(self):
        """Registry should have all expected tools."""

        @asynccontextmanager
        async def mock_factory():
            yield MagicMock()

        registry = create_technician_analytics_tools(mock_factory)

        expected_tools = [
            "query_technicians",
            "get_prediction",
            "get_forecast",
            "calculate_roi",
            "get_feature_importance",
            "get_regional_summary",
        ]

        for tool_name in expected_tools:
            assert tool_name in registry.list_tools(), f"Missing tool: {tool_name}"

    def test_tool_schemas_valid(self):
        """Tool schemas should be valid for LLM export."""

        @asynccontextmanager
        async def mock_factory():
            yield MagicMock()

        registry = create_technician_analytics_tools(mock_factory)
        schemas = registry.export_for_llm()

        assert len(schemas) == 6

        for schema in schemas:
            assert "name" in schema
            assert "description" in schema
            assert "input_schema" in schema
            assert len(schema["description"]) > 20  # Meaningful description


class TestQueryTechnicians:
    """Tests for query_technicians tool."""

    @pytest.mark.asyncio
    async def test_query_technicians_returns_results(self):
        """Query should return formatted technician data."""
        tech = MagicMock()
        tech.id = uuid4()
        tech.external_id = "TECH-001"
        tech.region = "US-WEST"
        tech.status = "ACTIVE"
        tech.tenure_days = 100
        tech.created_at = datetime.now()

        @asynccontextmanager
        async def mock_factory():
            session = MagicMock()

            async def mock_execute(query):
                result = MagicMock()
                result.scalars.return_value.all.return_value = [tech]
                return result

            session.execute = mock_execute
            yield session

        registry = create_technician_analytics_tools(mock_factory)
        result = await registry.execute("query_technicians", {"limit": 10})

        assert result.success is True
        assert result.result["count"] == 1
        assert result.result["technicians"][0]["external_id"] == "TECH-001"


class TestCalculateROI:
    """Tests for calculate_roi tool."""

    @pytest.mark.asyncio
    async def test_calculate_roi_basic(self):
        """ROI calculation should return expected fields."""
        tech = MagicMock()
        tech.id = uuid4()
        tech.status = "ACTIVE"
        tech.tenure_days = 100

        @asynccontextmanager
        async def mock_factory():
            session = MagicMock()

            async def mock_execute(query):
                result = MagicMock()
                result.scalars.return_value.all.return_value = [tech]
                return result

            session.execute = mock_execute
            yield session

        registry = create_technician_analytics_tools(mock_factory)
        result = await registry.execute(
            "calculate_roi",
            {
                "intervention_cost": 500,
                "retention_improvement": 0.2,
                "technician_ltv": 50000,
            },
        )

        assert result.success is True
        assert "total_technicians" in result.result
        assert "roi_percentage" in result.result
        assert "recommendation" in result.result


class TestGetFeatureImportance:
    """Tests for get_feature_importance tool."""

    @pytest.mark.asyncio
    async def test_get_feature_importance_no_model(self):
        """Should handle missing model gracefully."""

        @asynccontextmanager
        async def mock_factory():
            session = MagicMock()

            async def mock_execute(query):
                result = MagicMock()
                result.scalar_one_or_none.return_value = None
                return result

            session.execute = mock_execute
            yield session

        registry = create_technician_analytics_tools(mock_factory)
        result = await registry.execute("get_feature_importance", {"model_type": "EBM"})

        assert result.success is True
        assert "error" in result.result

    @pytest.mark.asyncio
    async def test_get_feature_importance_with_model(self):
        """Should return feature importance from model."""
        model = MagicMock()
        model.model_type = "EBM"
        model.version = "1.0"
        model.is_active = True
        model.feature_importance = {
            "features": ["tenure_days", "region", "jobs_completed"],
            "importances": [0.45, 0.30, 0.25],
        }

        @asynccontextmanager
        async def mock_factory():
            session = MagicMock()

            async def mock_execute(query):
                result = MagicMock()
                result.scalar_one_or_none.return_value = model
                return result

            session.execute = mock_execute
            yield session

        registry = create_technician_analytics_tools(mock_factory)
        result = await registry.execute("get_feature_importance", {"model_type": "EBM", "top_n": 5})

        assert result.success is True
        assert "features" in result.result
        assert len(result.result["features"]) == 3
        assert result.result["features"][0]["feature"] == "tenure_days"
