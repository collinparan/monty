"""Tests for appliance recommendation tools."""

from __future__ import annotations

import pytest

from app.services.appliance_tools import (
    APPLIANCE_AGENT_PROMPT,
    APPLIANCE_CONFIGS,
    CONDITION_MULTIPLIERS,
    _calculate_repair_vs_replace,
    _get_appliance_assessment,
    create_appliance_recommendation_tools,
)


class TestApplianceConfigs:
    """Tests for appliance configuration data."""

    def test_all_appliances_have_required_fields(self):
        """Test that all appliances have required configuration."""
        required_fields = [
            "avg_lifespan_years",
            "typical_repair_cost",
            "replacement_cost_range",
            "critical_components",
        ]

        for appliance, config in APPLIANCE_CONFIGS.items():
            for field in required_fields:
                assert field in config, f"{appliance} missing {field}"

    def test_condition_multipliers_complete(self):
        """Test all conditions have multipliers."""
        expected_conditions = ["excellent", "good", "fair", "poor", "critical"]
        for condition in expected_conditions:
            assert condition in CONDITION_MULTIPLIERS

    def test_condition_multipliers_ordered(self):
        """Test condition multipliers increase with severity."""
        assert CONDITION_MULTIPLIERS["excellent"] < CONDITION_MULTIPLIERS["good"]
        assert CONDITION_MULTIPLIERS["good"] < CONDITION_MULTIPLIERS["fair"]
        assert CONDITION_MULTIPLIERS["fair"] < CONDITION_MULTIPLIERS["poor"]
        assert CONDITION_MULTIPLIERS["poor"] < CONDITION_MULTIPLIERS["critical"]


class TestApplianceAssessment:
    """Tests for appliance assessment tool."""

    @pytest.mark.asyncio
    async def test_assessment_new_appliance(self):
        """Test assessment of a new appliance in good condition."""
        result = await _get_appliance_assessment(
            appliance_type="refrigerator",
            age_years=2,
            condition="good",
            repair_count=0,
        )

        assert result["risk_score"] < 3  # Low risk for new appliance
        assert result["risk_level"] == "LOW"
        assert "REPAIR RECOMMENDED" in result["recommendation"]

    @pytest.mark.asyncio
    async def test_assessment_old_appliance(self):
        """Test assessment of an old appliance past lifespan."""
        result = await _get_appliance_assessment(
            appliance_type="refrigerator",
            age_years=18,  # Past 15-year lifespan
            condition="poor",
            repair_count=3,
        )

        assert result["risk_score"] >= 7  # High risk
        assert result["risk_level"] == "HIGH"
        assert "REPLACE RECOMMENDED" in result["recommendation"]

    @pytest.mark.asyncio
    async def test_assessment_includes_risk_factors(self):
        """Test that assessment includes detailed risk factors."""
        result = await _get_appliance_assessment(
            appliance_type="washer",
            age_years=10,
            condition="fair",
            repair_count=2,
        )

        assert "risk_factors" in result
        assert len(result["risk_factors"]) >= 2  # At least age and condition

        # Check risk factors have required fields
        for factor in result["risk_factors"]:
            assert "factor" in factor
            assert "value" in factor
            assert "contribution" in factor
            assert "explanation" in factor

    @pytest.mark.asyncio
    async def test_assessment_critical_component(self):
        """Test that critical component repairs increase risk."""
        # Without critical component repair
        result_without = await _get_appliance_assessment(
            appliance_type="refrigerator",
            age_years=8,
            condition="fair",
            repair_count=1,
        )

        # With critical component repair
        result_with = await _get_appliance_assessment(
            appliance_type="refrigerator",
            age_years=8,
            condition="fair",
            repair_count=1,
            last_repair_component="compressor",  # Critical component
        )

        assert result_with["risk_score"] > result_without["risk_score"]

    @pytest.mark.asyncio
    async def test_assessment_unknown_appliance(self):
        """Test handling of unknown appliance type."""
        result = await _get_appliance_assessment(
            appliance_type="unknown_device",
            age_years=5,
            condition="good",
        )

        assert "error" in result
        assert "supported_types" in result

    @pytest.mark.asyncio
    async def test_assessment_includes_appliance_info(self):
        """Test that assessment includes appliance reference info."""
        result = await _get_appliance_assessment(
            appliance_type="dishwasher",
            age_years=5,
            condition="good",
        )

        assert "appliance_info" in result
        info = result["appliance_info"]
        assert "avg_lifespan" in info
        assert "typical_repair_cost" in info
        assert "replacement_cost_range" in info
        assert "critical_components" in info

    @pytest.mark.asyncio
    async def test_assessment_natural_language_explanation(self):
        """Test that risk factors include natural language explanations."""
        result = await _get_appliance_assessment(
            appliance_type="dryer",
            age_years=14,
            condition="fair",
        )

        # Check age factor explanation mentions specific numbers
        age_factor = next((f for f in result["risk_factors"] if f["factor"] == "age"), None)
        assert age_factor is not None
        assert "14" in age_factor["explanation"]
        assert "risk score" in age_factor["explanation"].lower()


class TestRepairVsReplace:
    """Tests for repair vs replace calculation tool."""

    @pytest.mark.asyncio
    async def test_repair_recommendation_new_appliance(self):
        """Test repair is recommended for newer appliances."""
        result = await _calculate_repair_vs_replace(
            appliance_type="washer",
            age_years=3,
        )

        assert result["recommendation"] == "REPAIR"
        assert result["remaining_years_if_repaired"] > 0
        assert result["repair_cost"] > 0
        assert result["replacement_cost"] > 0

    @pytest.mark.asyncio
    async def test_replace_recommendation_old_appliance(self):
        """Test replacement is recommended for old appliances."""
        result = await _calculate_repair_vs_replace(
            appliance_type="washer",
            age_years=11,  # Near 12-year lifespan
        )

        assert result["recommendation"] == "REPLACE"

    @pytest.mark.asyncio
    async def test_custom_repair_cost(self):
        """Test using custom repair cost."""
        result = await _calculate_repair_vs_replace(
            appliance_type="refrigerator",
            age_years=5,
            estimated_repair_cost=800,  # Higher than typical
        )

        assert result["repair_cost"] == 800

    @pytest.mark.asyncio
    async def test_profit_breakdown_included(self):
        """Test that profit breakdown is included."""
        result = await _calculate_repair_vs_replace(
            appliance_type="oven",
            age_years=7,
        )

        assert "profit_breakdown" in result
        assert "repair" in result["profit_breakdown"]
        assert "replacement" in result["profit_breakdown"]

        repair_profit = result["profit_breakdown"]["repair"]
        assert "total_revenue" in repair_profit
        assert "technician_income" in repair_profit
        assert "company_profit" in repair_profit

    @pytest.mark.asyncio
    async def test_roi_calculations(self):
        """Test ROI calculations are present."""
        result = await _calculate_repair_vs_replace(
            appliance_type="dishwasher",
            age_years=5,
        )

        assert "repair_roi" in result
        assert "replacement_roi" in result
        assert result["repair_roi"] > 0
        assert result["replacement_roi"] > 0

    @pytest.mark.asyncio
    async def test_explanation_generated(self):
        """Test natural language explanation is generated."""
        result = await _calculate_repair_vs_replace(
            appliance_type="microwave",
            age_years=5,
        )

        assert "explanation" in result
        explanation = result["explanation"]

        # Should contain key information
        assert "Repair Option" in explanation
        assert "Replacement Option" in explanation
        assert "Recommendation" in explanation

    @pytest.mark.asyncio
    async def test_budget_consideration(self):
        """Test budget constraint is considered."""
        result = await _calculate_repair_vs_replace(
            appliance_type="hvac",
            age_years=10,
            customer_budget=1000,  # Below both options
        )

        assert result["customer_budget"] == 1000
        assert "budget" in result["explanation"].lower()

    @pytest.mark.asyncio
    async def test_unknown_appliance_error(self):
        """Test handling unknown appliance type."""
        result = await _calculate_repair_vs_replace(
            appliance_type="unknown",
            age_years=5,
        )

        assert "error" in result


class TestToolRegistry:
    """Tests for tool registry creation."""

    def test_registry_has_both_tools(self):
        """Test registry includes both appliance tools."""
        registry = create_appliance_recommendation_tools()

        tools = registry.list_tools()
        assert "get_appliance_assessment" in tools
        assert "calculate_repair_vs_replace" in tools

    def test_tool_schemas_valid(self):
        """Test tool schemas are valid for LLM export."""
        registry = create_appliance_recommendation_tools()
        schemas = registry.export_for_llm()

        assert len(schemas) == 2
        for schema in schemas:
            assert "name" in schema
            assert "description" in schema
            assert "input_schema" in schema
            # Should have meaningful descriptions
            assert len(schema["description"]) > 50

    @pytest.mark.asyncio
    async def test_execute_assessment_tool(self):
        """Test executing assessment through registry."""
        registry = create_appliance_recommendation_tools()

        result = await registry.execute(
            "get_appliance_assessment",
            {
                "appliance_type": "refrigerator",
                "age_years": 10,
                "condition": "fair",
            },
        )

        assert result.success is True
        assert "risk_score" in result.result

    @pytest.mark.asyncio
    async def test_execute_repair_replace_tool(self):
        """Test executing repair vs replace through registry."""
        registry = create_appliance_recommendation_tools()

        result = await registry.execute(
            "calculate_repair_vs_replace",
            {
                "appliance_type": "washer",
                "age_years": 8,
            },
        )

        assert result.success is True
        assert "recommendation" in result.result


class TestAgentPrompt:
    """Tests for agent prompt template."""

    def test_prompt_exists(self):
        """Test agent prompt is defined."""
        assert APPLIANCE_AGENT_PROMPT is not None
        assert len(APPLIANCE_AGENT_PROMPT) > 100

    def test_prompt_mentions_tools(self):
        """Test prompt references available tools."""
        assert "get_appliance_assessment" in APPLIANCE_AGENT_PROMPT
        assert "calculate_repair_vs_replace" in APPLIANCE_AGENT_PROMPT

    def test_prompt_customer_focused(self):
        """Test prompt emphasizes customer focus."""
        prompt_lower = APPLIANCE_AGENT_PROMPT.lower()
        assert "customer" in prompt_lower
        assert "empathetic" in prompt_lower or "helpful" in prompt_lower


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_zero_age_appliance(self):
        """Test assessment of brand new appliance."""
        result = await _get_appliance_assessment(
            appliance_type="refrigerator",
            age_years=0,
            condition="excellent",
        )

        assert result["risk_score"] < 1
        assert result["remaining_lifespan_estimate_years"] == 15

    @pytest.mark.asyncio
    async def test_very_old_appliance(self):
        """Test assessment of very old appliance."""
        result = await _get_appliance_assessment(
            appliance_type="washer",
            age_years=25,  # Way past 12-year lifespan
            condition="poor",
            repair_count=5,
        )

        # Risk score should be capped at 10
        assert result["risk_score"] <= 10
        assert result["risk_level"] == "HIGH"

    @pytest.mark.asyncio
    async def test_high_repair_count(self):
        """Test assessment with many repairs."""
        result = await _get_appliance_assessment(
            appliance_type="dryer",
            age_years=5,
            condition="fair",
            repair_count=10,  # Many repairs
        )

        # Repair contribution should be capped
        repair_factor = next(
            (f for f in result["risk_factors"] if f["factor"] == "repair_history"),
            None,
        )
        assert repair_factor is not None
        assert repair_factor["contribution"] <= 2.5
