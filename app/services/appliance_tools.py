"""Agent tools for customer-facing appliance repair/replace recommendations.

These tools enable the AI agent to assess appliances and provide
natural language recommendations about repair vs replacement decisions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from app.services.tools import ToolRegistry, create_json_schema

logger = logging.getLogger(__name__)


# Appliance type configurations
APPLIANCE_CONFIGS = {
    "refrigerator": {
        "avg_lifespan_years": 15,
        "typical_repair_cost": 350,
        "replacement_cost_range": (800, 3000),
        "critical_components": ["compressor", "evaporator", "condenser"],
    },
    "washer": {
        "avg_lifespan_years": 12,
        "typical_repair_cost": 250,
        "replacement_cost_range": (500, 1500),
        "critical_components": ["motor", "pump", "transmission", "drum bearings"],
    },
    "dryer": {
        "avg_lifespan_years": 13,
        "typical_repair_cost": 200,
        "replacement_cost_range": (400, 1200),
        "critical_components": ["motor", "heating element", "drum belt"],
    },
    "dishwasher": {
        "avg_lifespan_years": 10,
        "typical_repair_cost": 200,
        "replacement_cost_range": (400, 1200),
        "critical_components": ["pump", "motor", "control board"],
    },
    "oven": {
        "avg_lifespan_years": 15,
        "typical_repair_cost": 300,
        "replacement_cost_range": (600, 2500),
        "critical_components": ["heating element", "igniter", "control board"],
    },
    "microwave": {
        "avg_lifespan_years": 8,
        "typical_repair_cost": 150,
        "replacement_cost_range": (100, 600),
        "critical_components": ["magnetron", "door switch", "control panel"],
    },
    "hvac": {
        "avg_lifespan_years": 20,
        "typical_repair_cost": 500,
        "replacement_cost_range": (3000, 10000),
        "critical_components": ["compressor", "condenser coil", "evaporator coil", "blower motor"],
    },
}

# Condition multipliers for risk score
CONDITION_MULTIPLIERS = {
    "excellent": 0.5,
    "good": 0.75,
    "fair": 1.0,
    "poor": 1.5,
    "critical": 2.0,
}


@dataclass
class ApplianceAssessment:
    """Result of an appliance assessment."""

    appliance_type: str
    age_years: int
    condition: str
    repair_count: int
    risk_score: float
    risk_factors: list[dict[str, Any]]
    remaining_lifespan_estimate: int
    recommendation: str


@dataclass
class RepairVsReplaceAnalysis:
    """ROI analysis for repair vs replacement decision."""

    repair_cost: float
    replacement_cost: float
    expected_remaining_years: int
    repair_roi: float
    replacement_roi: float
    recommendation: str
    explanation: str
    profit_breakdown: dict[str, float]


def create_appliance_recommendation_tools() -> ToolRegistry:
    """Create and register appliance recommendation tools.

    Returns:
        ToolRegistry with appliance tools registered
    """
    registry = ToolRegistry()

    registry.register(
        name="get_appliance_assessment",
        description=(
            "Assess an appliance's condition and risk of failure based on age, condition, "
            "and repair history. Returns a risk score (0-10) with detailed factors and "
            "natural language explanation."
        ),
        parameters=create_json_schema(
            properties={
                "appliance_type": {
                    "type": "string",
                    "enum": list(APPLIANCE_CONFIGS.keys()),
                    "description": "Type of appliance being assessed",
                },
                "age_years": {
                    "type": "number",
                    "description": "Age of the appliance in years",
                    "minimum": 0,
                    "maximum": 50,
                },
                "condition": {
                    "type": "string",
                    "enum": ["excellent", "good", "fair", "poor", "critical"],
                    "description": "Current overall condition of the appliance",
                },
                "repair_count": {
                    "type": "integer",
                    "description": "Number of repairs in the past 3 years",
                    "minimum": 0,
                    "default": 0,
                },
                "last_repair_component": {
                    "type": "string",
                    "description": "Component that was repaired most recently (optional)",
                },
            },
            required=["appliance_type", "age_years", "condition"],
        ),
        handler=_get_appliance_assessment,
    )

    registry.register(
        name="calculate_repair_vs_replace",
        description=(
            "Calculate the ROI and financial analysis for repairing vs replacing an appliance. "
            "Includes cost analysis, expected value, and profit margins for the company "
            "and 1099 technician partner."
        ),
        parameters=create_json_schema(
            properties={
                "appliance_type": {
                    "type": "string",
                    "enum": list(APPLIANCE_CONFIGS.keys()),
                    "description": "Type of appliance",
                },
                "age_years": {
                    "type": "number",
                    "description": "Age of the appliance in years",
                },
                "estimated_repair_cost": {
                    "type": "number",
                    "description": "Estimated cost of repair (optional - uses typical if not provided)",
                },
                "condition_after_repair": {
                    "type": "string",
                    "enum": ["excellent", "good", "fair"],
                    "description": "Expected condition after repair",
                    "default": "good",
                },
                "customer_budget": {
                    "type": "number",
                    "description": "Customer's budget constraint (optional)",
                },
            },
            required=["appliance_type", "age_years"],
        ),
        handler=_calculate_repair_vs_replace,
    )

    return registry


async def _get_appliance_assessment(
    appliance_type: str,
    age_years: float,
    condition: str,
    repair_count: int = 0,
    last_repair_component: Optional[str] = None,
) -> dict[str, Any]:
    """Assess an appliance's condition and failure risk.

    Args:
        appliance_type: Type of appliance
        age_years: Age in years
        condition: Current condition rating
        repair_count: Number of repairs in past 3 years
        last_repair_component: Most recent repaired component

    Returns:
        Assessment with risk score and factors
    """
    config = APPLIANCE_CONFIGS.get(appliance_type)
    if not config:
        return {
            "error": f"Unknown appliance type: {appliance_type}",
            "supported_types": list(APPLIANCE_CONFIGS.keys()),
        }

    # Calculate base risk from age
    avg_lifespan = config["avg_lifespan_years"]
    age_factor = age_years / avg_lifespan

    # Build risk factors list with natural language
    risk_factors = []
    risk_score = 0.0

    # Age contribution
    if age_years >= avg_lifespan:
        age_contribution = min(3.0, (age_years - avg_lifespan) / 5 + 2)
        risk_factors.append(
            {
                "factor": "age",
                "value": age_years,
                "contribution": round(age_contribution, 2),
                "explanation": (
                    f"Your {appliance_type}'s {age_years}-year age exceeds the "
                    f"{avg_lifespan}-year average lifespan, adding {age_contribution:.1f} "
                    "to the risk score."
                ),
            }
        )
    elif age_years > avg_lifespan * 0.7:
        age_contribution = (age_years / avg_lifespan - 0.5) * 2
        risk_factors.append(
            {
                "factor": "age",
                "value": age_years,
                "contribution": round(age_contribution, 2),
                "explanation": (
                    f"At {age_years} years, your {appliance_type} is approaching the end "
                    f"of its typical {avg_lifespan}-year lifespan, adding {age_contribution:.1f} "
                    "to the risk score."
                ),
            }
        )
    else:
        age_contribution = age_factor * 0.5
        risk_factors.append(
            {
                "factor": "age",
                "value": age_years,
                "contribution": round(age_contribution, 2),
                "explanation": (
                    f"Your {appliance_type} is {age_years} years old, which is within "
                    f"normal range for its {avg_lifespan}-year expected lifespan."
                ),
            }
        )

    risk_score += age_contribution

    # Condition contribution
    condition_multiplier = CONDITION_MULTIPLIERS.get(condition, 1.0)
    condition_contribution = (condition_multiplier - 0.5) * 2
    risk_factors.append(
        {
            "factor": "condition",
            "value": condition,
            "contribution": round(condition_contribution, 2),
            "explanation": (
                f"The '{condition}' condition rating "
                + ("significantly increases" if condition_contribution > 1 else "adds")
                + f" {condition_contribution:.1f} to the risk score."
            ),
        }
    )
    risk_score += condition_contribution

    # Repair history contribution
    if repair_count > 0:
        repair_contribution = min(2.5, repair_count * 0.7)
        risk_factors.append(
            {
                "factor": "repair_history",
                "value": repair_count,
                "contribution": round(repair_contribution, 2),
                "explanation": (
                    f"The {repair_count} repair(s) in the past 3 years adds {repair_contribution:.1f} "
                    "to the risk score, indicating potential ongoing issues."
                ),
            }
        )
        risk_score += repair_contribution

    # Critical component repair
    if last_repair_component and last_repair_component.lower() in [
        c.lower() for c in config["critical_components"]
    ]:
        critical_contribution = 1.5
        risk_factors.append(
            {
                "factor": "critical_component",
                "value": last_repair_component,
                "contribution": critical_contribution,
                "explanation": (
                    f"Recent repair to the {last_repair_component} (a critical component) "
                    f"adds {critical_contribution} to the risk score as it may indicate "
                    "systemic issues."
                ),
            }
        )
        risk_score += critical_contribution

    # Cap and normalize risk score
    risk_score = min(10.0, max(0.0, risk_score))

    # Estimate remaining lifespan
    expected_remaining = max(0, int(avg_lifespan - age_years))
    if condition in ["poor", "critical"]:
        expected_remaining = max(0, expected_remaining - 3)
    if repair_count >= 3:
        expected_remaining = max(0, expected_remaining - 2)

    # Generate recommendation
    if risk_score >= 7:
        recommendation = (
            f"REPLACE RECOMMENDED: With a risk score of {risk_score:.1f}/10, "
            f"this {appliance_type} has a high probability of failure. "
            "Replacement is strongly recommended to avoid unexpected breakdowns and "
            "potentially more costly emergency repairs."
        )
    elif risk_score >= 4:
        recommendation = (
            f"REPAIR WITH CAUTION: With a risk score of {risk_score:.1f}/10, "
            f"this {appliance_type} may benefit from repair, but consider the "
            f"repair cost vs remaining lifespan ({expected_remaining} years estimated). "
            "Budget for potential replacement within the next 2-3 years."
        )
    else:
        recommendation = (
            f"REPAIR RECOMMENDED: With a risk score of {risk_score:.1f}/10, "
            f"this {appliance_type} is a good candidate for repair. "
            f"Expected remaining lifespan: {expected_remaining} years."
        )

    return {
        "appliance_type": appliance_type,
        "age_years": age_years,
        "condition": condition,
        "repair_count": repair_count,
        "risk_score": round(risk_score, 1),
        "risk_level": "HIGH" if risk_score >= 7 else "MEDIUM" if risk_score >= 4 else "LOW",
        "risk_factors": risk_factors,
        "remaining_lifespan_estimate_years": expected_remaining,
        "recommendation": recommendation,
        "appliance_info": {
            "avg_lifespan": avg_lifespan,
            "typical_repair_cost": config["typical_repair_cost"],
            "replacement_cost_range": config["replacement_cost_range"],
            "critical_components": config["critical_components"],
        },
    }


async def _calculate_repair_vs_replace(
    appliance_type: str,
    age_years: float,
    estimated_repair_cost: Optional[float] = None,
    condition_after_repair: str = "good",
    customer_budget: Optional[float] = None,
) -> dict[str, Any]:
    """Calculate ROI for repair vs replacement.

    Args:
        appliance_type: Type of appliance
        age_years: Age in years
        estimated_repair_cost: Repair cost (or use typical)
        condition_after_repair: Expected condition after repair
        customer_budget: Customer's budget constraint

    Returns:
        Analysis with ROI and profit breakdown
    """
    config = APPLIANCE_CONFIGS.get(appliance_type)
    if not config:
        return {
            "error": f"Unknown appliance type: {appliance_type}",
            "supported_types": list(APPLIANCE_CONFIGS.keys()),
        }

    # Use provided or typical repair cost
    repair_cost = estimated_repair_cost or config["typical_repair_cost"]

    # Calculate replacement cost (use midpoint of range)
    replacement_min, replacement_max = config["replacement_cost_range"]
    replacement_cost = (replacement_min + replacement_max) / 2

    # Calculate expected remaining years
    avg_lifespan = config["avg_lifespan_years"]
    remaining_if_repaired = max(1, int((avg_lifespan - age_years) * 0.7))
    remaining_if_replaced = avg_lifespan

    # Calculate cost per year of service
    repair_cost_per_year = (
        repair_cost / remaining_if_repaired if remaining_if_repaired > 0 else float("inf")
    )
    replacement_cost_per_year = replacement_cost / remaining_if_replaced

    # Calculate ROI (value per dollar spent)
    repair_roi = remaining_if_repaired / repair_cost * 100 if repair_cost > 0 else 0
    replacement_roi = remaining_if_replaced / replacement_cost * 100 if replacement_cost > 0 else 0

    # Profit breakdown for company and technician
    # Assume: 35% parts cost, 40% labor to technician, 25% company margin
    repair_parts_cost = repair_cost * 0.35
    repair_technician_labor = repair_cost * 0.40
    repair_company_margin = repair_cost * 0.25

    # For replacement, assume installation labor
    # 15% of replacement cost goes to technician for install
    replacement_technician_income = replacement_cost * 0.15
    # Company margin on appliance sale (typically 10-15%)
    replacement_company_margin = replacement_cost * 0.12

    profit_breakdown = {
        "repair": {
            "total_revenue": repair_cost,
            "parts_cost": round(repair_parts_cost, 2),
            "technician_income": round(repair_technician_labor, 2),
            "company_profit": round(repair_company_margin, 2),
        },
        "replacement": {
            "total_revenue": replacement_cost,
            "product_cost": round(replacement_cost * 0.70, 2),  # ~70% wholesale cost
            "technician_income": round(replacement_technician_income, 2),
            "company_profit": round(replacement_company_margin, 2),
        },
    }

    # Determine recommendation
    if age_years >= avg_lifespan * 0.9:
        recommendation = "REPLACE"
        reason = "near or past average lifespan"
    elif repair_cost > replacement_cost * 0.5:
        recommendation = "REPLACE"
        reason = "repair cost exceeds 50% of replacement"
    elif repair_cost_per_year < replacement_cost_per_year * 0.6:
        recommendation = "REPAIR"
        reason = "repair offers better cost-per-year value"
    elif remaining_if_repaired <= 2:
        recommendation = "REPLACE"
        reason = "limited remaining lifespan after repair"
    else:
        recommendation = "REPAIR"
        reason = "good value for remaining lifespan"

    # Generate natural language explanation
    explanation_parts = [
        f"Analysis for {age_years}-year-old {appliance_type}:",
        "",
        f"**Repair Option:** ${repair_cost:.0f}",
        f"- Expected additional lifespan: {remaining_if_repaired} years",
        f"- Cost per year: ${repair_cost_per_year:.0f}",
        f"- Technician earns: ${repair_technician_labor:.0f}",
        "",
        f"**Replacement Option:** ${replacement_cost:.0f}",
        f"- Expected lifespan: {remaining_if_replaced} years",
        f"- Cost per year: ${replacement_cost_per_year:.0f}",
        f"- Technician earns: ${replacement_technician_income:.0f}",
        "",
        f"**Recommendation: {recommendation}**",
        f"Reason: {reason.capitalize()}.",
    ]

    # Budget considerations
    if customer_budget:
        if customer_budget < repair_cost:
            explanation_parts.append(
                f"\nNote: Customer budget (${customer_budget:.0f}) is below repair cost. "
                "Consider financing options or prioritized partial repair."
            )
        elif customer_budget < replacement_cost and recommendation == "REPLACE":
            explanation_parts.append(
                f"\nNote: Customer budget (${customer_budget:.0f}) doesn't cover replacement. "
                "Repair may be necessary despite recommendation."
            )

    return {
        "appliance_type": appliance_type,
        "age_years": age_years,
        "repair_cost": repair_cost,
        "replacement_cost": replacement_cost,
        "remaining_years_if_repaired": remaining_if_repaired,
        "remaining_years_if_replaced": remaining_if_replaced,
        "repair_cost_per_year": round(repair_cost_per_year, 2),
        "replacement_cost_per_year": round(replacement_cost_per_year, 2),
        "repair_roi": round(repair_roi, 2),
        "replacement_roi": round(replacement_roi, 2),
        "recommendation": recommendation,
        "recommendation_reason": reason,
        "profit_breakdown": profit_breakdown,
        "explanation": "\n".join(explanation_parts),
        "customer_budget": customer_budget,
    }


# Customer-facing agent prompt template
APPLIANCE_AGENT_PROMPT = """You are a helpful appliance advisor for Sears Home Services.
Your role is to help customers understand their appliance's condition and make informed
decisions about repair vs replacement.

When providing recommendations:
- Always explain risk scores in natural language (e.g., "Your 14-year age adds 0.6 to the risk score...")
- Be empathetic and acknowledge the customer's situation
- Provide clear cost comparisons
- Mention financing options if budget is a concern
- Highlight both immediate and long-term value
- Be honest about expected remaining lifespan

When using tools:
- Use get_appliance_assessment to evaluate the appliance's condition
- Use calculate_repair_vs_replace for cost analysis and recommendations
- Present findings in a clear, customer-friendly way

Remember: The customer's best interest comes first, but also consider the technician's
livelihood - they depend on both repairs and installations for income."""
