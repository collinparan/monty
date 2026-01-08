"""Specialized agent tools for technician analytics.

These tools allow the AI agent to query and analyze technician data,
make predictions, get forecasts, and calculate ROI.
"""

from __future__ import annotations

import logging
from typing import Any, Optional
from uuid import UUID

from sqlalchemy import func, select

from app.models import Technician, TrainedModel
from app.schemas.models import ModelType
from app.services.ml import InterpretMLService, ProphetService
from app.services.recommendations import recommendation_engine
from app.services.tools import ToolRegistry, create_json_schema

logger = logging.getLogger(__name__)


def create_technician_analytics_tools(
    db_session_factory,
) -> ToolRegistry:
    """Create and register all technician analytics tools.

    Args:
        db_session_factory: Async context manager for database sessions

    Returns:
        ToolRegistry with all tools registered
    """
    registry = ToolRegistry()

    # Register all tools
    registry.register(
        name="query_technicians",
        description=(
            "Search and filter technicians by various criteria including region, status, "
            "and risk level. Returns a list of technicians matching the criteria with their "
            "basic information and risk scores."
        ),
        parameters=create_json_schema(
            properties={
                "region": {
                    "type": "string",
                    "description": "Filter by region code (e.g., 'US-WEST', 'US-EAST')",
                },
                "status": {
                    "type": "string",
                    "enum": ["ACTIVE", "CHURNED", "PENDING"],
                    "description": "Filter by technician status",
                },
                "risk_level": {
                    "type": "string",
                    "enum": ["LOW", "MEDIUM", "HIGH"],
                    "description": "Filter by risk level classification",
                },
                "min_tenure_days": {
                    "type": "integer",
                    "description": "Minimum tenure in days",
                },
                "max_tenure_days": {
                    "type": "integer",
                    "description": "Maximum tenure in days",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 10)",
                    "default": 10,
                },
            },
            required=[],
        ),
        handler=_create_query_technicians_handler(db_session_factory),
    )

    registry.register(
        name="get_prediction",
        description=(
            "Get a retention or recruitment prediction for a specific technician with "
            "detailed explanation of contributing factors. Returns probability score, "
            "risk level, and natural language explanation."
        ),
        parameters=create_json_schema(
            properties={
                "technician_id": {
                    "type": "string",
                    "description": "UUID of the technician",
                },
                "prediction_type": {
                    "type": "string",
                    "enum": ["retention", "recruitment"],
                    "description": "Type of prediction to make",
                    "default": "retention",
                },
            },
            required=["technician_id"],
        ),
        handler=_create_get_prediction_handler(db_session_factory),
    )

    registry.register(
        name="get_forecast",
        description=(
            "Get Prophet time-series forecast for technician headcount or job demand "
            "in a specific region. Returns forecast values with confidence intervals "
            "for the specified time horizon."
        ),
        parameters=create_json_schema(
            properties={
                "region": {
                    "type": "string",
                    "description": "Region code to forecast for (optional, omit for all regions)",
                },
                "forecast_type": {
                    "type": "string",
                    "enum": ["headcount", "demand"],
                    "description": "Type of forecast",
                    "default": "headcount",
                },
                "periods": {
                    "type": "integer",
                    "description": "Number of days to forecast ahead (default: 30)",
                    "default": 30,
                    "minimum": 1,
                    "maximum": 365,
                },
            },
            required=[],
        ),
        handler=_create_get_forecast_handler(db_session_factory),
    )

    registry.register(
        name="calculate_roi",
        description=(
            "Calculate return on investment for retention interventions targeting "
            "specific technicians or risk groups. Considers intervention cost, "
            "retention improvement probability, and lifetime value."
        ),
        parameters=create_json_schema(
            properties={
                "technician_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of technician UUIDs to calculate ROI for",
                },
                "risk_level": {
                    "type": "string",
                    "enum": ["LOW", "MEDIUM", "HIGH"],
                    "description": "Target all technicians with this risk level",
                },
                "intervention_cost": {
                    "type": "number",
                    "description": "Cost per technician for the intervention (default: $500)",
                    "default": 500.0,
                },
                "retention_improvement": {
                    "type": "number",
                    "description": "Expected retention probability improvement (0-1, default: 0.2)",
                    "default": 0.2,
                },
                "technician_ltv": {
                    "type": "number",
                    "description": "Average technician lifetime value (default: $50,000)",
                    "default": 50000.0,
                },
            },
            required=[],
        ),
        handler=_create_calculate_roi_handler(db_session_factory),
    )

    registry.register(
        name="get_feature_importance",
        description=(
            "Get the top factors (features) that affect technician retention outcomes "
            "from the trained ML model. Returns ranked list of features with their "
            "importance scores and interpretations."
        ),
        parameters=create_json_schema(
            properties={
                "model_type": {
                    "type": "string",
                    "enum": ["EBM", "DECISION_TREE"],
                    "description": "Which model to get feature importance from (default: EBM)",
                    "default": "EBM",
                },
                "top_n": {
                    "type": "integer",
                    "description": "Number of top features to return (default: 10)",
                    "default": 10,
                },
            },
            required=[],
        ),
        handler=_create_get_feature_importance_handler(db_session_factory),
    )

    registry.register(
        name="get_regional_summary",
        description=(
            "Get a summary of technician metrics by region including headcount, "
            "churn rate, average tenure, and risk distribution."
        ),
        parameters=create_json_schema(
            properties={
                "regions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of region codes (omit for all regions)",
                },
            },
            required=[],
        ),
        handler=_create_get_regional_summary_handler(db_session_factory),
    )

    registry.register(
        name="get_strategic_recommendations",
        description=(
            "Get AI-powered strategic recommendations for improving 1099 technician "
            "efficiency, job completion rates, and repair vs replace decisions. "
            "Uses Cicero-inspired strategic reasoning that models technician intent "
            "and provides mutually beneficial recommendations grounded in actual data. "
            "Returns prioritized recommendations with predicted impact, rationale, "
            "and action items."
        ),
        parameters=create_json_schema(
            properties={
                "state": {
                    "type": "string",
                    "description": "US state abbreviation to filter recommendations (e.g., 'CA', 'TX'). Omit for all states.",
                },
                "focus_areas": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["efficiency", "quality", "scheduling", "skill", "routing", "csat", "financial"],
                    },
                    "description": "Specific areas to focus recommendations on",
                },
            },
            required=[],
        ),
        handler=_create_get_recommendations_handler(),
    )

    return registry


def _create_query_technicians_handler(db_session_factory):
    """Create handler for query_technicians tool."""

    async def query_technicians(
        region: Optional[str] = None,
        status: Optional[str] = None,
        risk_level: Optional[str] = None,
        min_tenure_days: Optional[int] = None,
        max_tenure_days: Optional[int] = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Query technicians with filters."""
        async with db_session_factory() as session:
            query = select(Technician)

            if region:
                query = query.where(Technician.region == region)
            if status:
                query = query.where(Technician.status == status)
            if min_tenure_days is not None:
                query = query.where(Technician.tenure_days >= min_tenure_days)
            if max_tenure_days is not None:
                query = query.where(Technician.tenure_days <= max_tenure_days)

            query = query.order_by(Technician.created_at.desc()).limit(limit)

            result = await session.execute(query)
            technicians = result.scalars().all()

            # Calculate risk scores and filter by risk level
            tech_data = []
            for tech in technicians:
                risk_score, tech_risk_level = _calculate_risk(tech)

                if risk_level and tech_risk_level != risk_level:
                    continue

                tech_data.append(
                    {
                        "id": str(tech.id),
                        "external_id": tech.external_id,
                        "region": tech.region,
                        "status": tech.status,
                        "tenure_days": tech.tenure_days,
                        "risk_score": round(risk_score, 3),
                        "risk_level": tech_risk_level,
                    }
                )

            return {
                "count": len(tech_data),
                "technicians": tech_data,
                "filters_applied": {
                    "region": region,
                    "status": status,
                    "risk_level": risk_level,
                    "min_tenure_days": min_tenure_days,
                    "max_tenure_days": max_tenure_days,
                },
            }

    return query_technicians


def _create_get_prediction_handler(db_session_factory):
    """Create handler for get_prediction tool."""

    async def get_prediction(
        technician_id: str,
        prediction_type: str = "retention",
    ) -> dict[str, Any]:
        """Get prediction for a technician."""
        async with db_session_factory() as session:
            # Get technician
            tech_uuid = UUID(technician_id)
            result = await session.execute(select(Technician).where(Technician.id == tech_uuid))
            technician = result.scalar_one_or_none()

            if not technician:
                return {"error": f"Technician {technician_id} not found"}

            # Find latest active model
            model_result = await session.execute(
                select(TrainedModel)
                .where(TrainedModel.model_type == ModelType.EBM.value)
                .where(TrainedModel.is_active == True)  # noqa: E712
                .order_by(TrainedModel.created_at.desc())
                .limit(1)
            )
            model = model_result.scalar_one_or_none()

            if not model:
                # Return basic risk assessment without model
                risk_score, risk_level = _calculate_risk(technician)
                return {
                    "technician_id": technician_id,
                    "prediction_type": prediction_type,
                    "probability": risk_score,
                    "risk_level": risk_level,
                    "explanation": _generate_basic_explanation(technician, risk_score),
                    "model_used": None,
                    "note": "No trained model available - using heuristic assessment",
                }

            # Use InterpretML service for prediction
            interpret_service = InterpretMLService()

            try:
                # Build feature dict from technician data
                features = _build_feature_dict(technician)

                local_explanation = interpret_service.get_local_explanation(
                    model_path=model.file_path,
                    features=features,
                )

                return {
                    "technician_id": technician_id,
                    "prediction_type": prediction_type,
                    "probability": local_explanation.get("probability", 0.5),
                    "risk_level": local_explanation.get("risk_level", "MEDIUM"),
                    "explanation": local_explanation.get("explanation", ""),
                    "feature_contributions": local_explanation.get("feature_contributions", []),
                    "model_used": {
                        "name": model.name,
                        "version": model.version,
                        "type": model.model_type,
                    },
                }
            except Exception as e:
                logger.exception(f"Prediction error: {e}")
                risk_score, risk_level = _calculate_risk(technician)
                return {
                    "technician_id": technician_id,
                    "prediction_type": prediction_type,
                    "probability": risk_score,
                    "risk_level": risk_level,
                    "explanation": _generate_basic_explanation(technician, risk_score),
                    "model_used": None,
                    "error": str(e),
                }

    return get_prediction


def _create_get_forecast_handler(db_session_factory):
    """Create handler for get_forecast tool."""

    async def get_forecast(
        region: Optional[str] = None,
        forecast_type: str = "headcount",
        periods: int = 30,
    ) -> dict[str, Any]:
        """Get Prophet forecast."""
        async with db_session_factory() as session:
            # Find latest Prophet model
            query = (
                select(TrainedModel)
                .where(TrainedModel.model_type == ModelType.PROPHET.value)
                .where(TrainedModel.is_active == True)  # noqa: E712
            )

            if forecast_type:
                query = query.where(TrainedModel.target == forecast_type)

            query = query.order_by(TrainedModel.created_at.desc()).limit(1)

            result = await session.execute(query)
            model = result.scalar_one_or_none()

            if not model:
                return {
                    "error": f"No Prophet model found for {forecast_type}",
                    "forecast_type": forecast_type,
                    "region": region,
                }

            prophet_service = ProphetService()

            try:
                forecast = prophet_service.predict_future(
                    model_path=model.file_path,
                    periods=periods,
                )

                # Summarize forecast
                return {
                    "forecast_type": forecast_type,
                    "region": region or "all",
                    "periods": periods,
                    "forecast_start": forecast["dates"][0] if forecast["dates"] else None,
                    "forecast_end": forecast["dates"][-1] if forecast["dates"] else None,
                    "summary": {
                        "mean_forecast": round(sum(forecast["yhat"]) / len(forecast["yhat"]), 1)
                        if forecast["yhat"]
                        else None,
                        "min_forecast": round(min(forecast["yhat_lower"]), 1)
                        if forecast["yhat_lower"]
                        else None,
                        "max_forecast": round(max(forecast["yhat_upper"]), 1)
                        if forecast["yhat_upper"]
                        else None,
                    },
                    "model_used": {
                        "name": model.name,
                        "version": model.version,
                    },
                    "data_points": len(forecast["dates"]),
                }
            except Exception as e:
                logger.exception(f"Forecast error: {e}")
                return {
                    "error": str(e),
                    "forecast_type": forecast_type,
                    "region": region,
                }

    return get_forecast


def _create_calculate_roi_handler(db_session_factory):
    """Create handler for calculate_roi tool."""

    async def calculate_roi(
        technician_ids: Optional[list[str]] = None,
        risk_level: Optional[str] = None,
        intervention_cost: float = 500.0,
        retention_improvement: float = 0.2,
        technician_ltv: float = 50000.0,
    ) -> dict[str, Any]:
        """Calculate ROI for retention intervention."""
        async with db_session_factory() as session:
            # Build query
            query = select(Technician).where(Technician.status == "ACTIVE")

            if technician_ids:
                tech_uuids = [UUID(tid) for tid in technician_ids]
                query = query.where(Technician.id.in_(tech_uuids))

            result = await session.execute(query)
            technicians = list(result.scalars().all())

            # Filter by risk level if specified
            if risk_level:
                technicians = [t for t in technicians if _calculate_risk(t)[1] == risk_level]

            if not technicians:
                return {
                    "error": "No technicians found matching criteria",
                    "technician_ids": technician_ids,
                    "risk_level": risk_level,
                }

            # Calculate ROI
            total_techs = len(technicians)
            total_cost = total_techs * intervention_cost

            # Expected retained technicians
            expected_retained = total_techs * retention_improvement
            expected_value = expected_retained * technician_ltv

            roi = (expected_value - total_cost) / total_cost if total_cost > 0 else 0

            # Risk distribution
            risk_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
            for tech in technicians:
                _, level = _calculate_risk(tech)
                risk_counts[level] += 1

            return {
                "total_technicians": total_techs,
                "intervention_cost_per_tech": intervention_cost,
                "total_investment": total_cost,
                "retention_improvement_rate": retention_improvement,
                "technician_ltv": technician_ltv,
                "expected_retained_technicians": round(expected_retained, 1),
                "expected_value_retained": round(expected_value, 2),
                "roi_percentage": round(roi * 100, 1),
                "roi_ratio": round(roi, 2),
                "risk_distribution": risk_counts,
                "recommendation": _generate_roi_recommendation(roi, total_techs, risk_counts),
            }

    return calculate_roi


def _create_get_feature_importance_handler(db_session_factory):
    """Create handler for get_feature_importance tool."""

    async def get_feature_importance(
        model_type: str = "EBM",
        top_n: int = 10,
    ) -> dict[str, Any]:
        """Get feature importance from trained model."""
        async with db_session_factory() as session:
            # Find latest model of specified type
            result = await session.execute(
                select(TrainedModel)
                .where(TrainedModel.model_type == model_type)
                .where(TrainedModel.is_active == True)  # noqa: E712
                .order_by(TrainedModel.created_at.desc())
                .limit(1)
            )
            model = result.scalar_one_or_none()

            if not model:
                return {
                    "error": f"No active {model_type} model found",
                    "model_type": model_type,
                }

            if not model.feature_importance:
                return {
                    "error": "Model does not have feature importance data",
                    "model_type": model_type,
                }

            features = model.feature_importance.get("features", [])[:top_n]
            importances = model.feature_importance.get("importances", [])[:top_n]

            # Format with interpretations
            feature_data = []
            for i, (feature, importance) in enumerate(zip(features, importances)):
                feature_data.append(
                    {
                        "rank": i + 1,
                        "feature": feature,
                        "importance": round(importance, 4),
                        "interpretation": _interpret_feature(feature),
                    }
                )

            return {
                "model_type": model_type,
                "model_version": model.version,
                "top_n": top_n,
                "features": feature_data,
                "insight": _generate_feature_insight(features[:3]),
            }

    return get_feature_importance


def _create_get_regional_summary_handler(db_session_factory):
    """Create handler for get_regional_summary tool."""

    async def get_regional_summary(
        regions: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Get regional summary metrics."""
        async with db_session_factory() as session:
            # Base query for aggregation
            query = select(
                Technician.region,
                Technician.status,
                func.count(Technician.id).label("count"),
                func.avg(Technician.tenure_days).label("avg_tenure"),
            ).group_by(Technician.region, Technician.status)

            if regions:
                query = query.where(Technician.region.in_(regions))

            result = await session.execute(query)
            rows = result.all()

            # Aggregate by region
            region_data: dict[str, dict] = {}
            for row in rows:
                region_name = row.region
                if region_name not in region_data:
                    region_data[region_name] = {
                        "total": 0,
                        "active": 0,
                        "churned": 0,
                        "tenure_sum": 0,
                        "tenure_count": 0,
                    }

                region_data[region_name]["total"] += row.count
                if row.status == "ACTIVE":
                    region_data[region_name]["active"] = row.count
                elif row.status == "CHURNED":
                    region_data[region_name]["churned"] = row.count

                if row.avg_tenure:
                    region_data[region_name]["tenure_sum"] += float(row.avg_tenure) * row.count
                    region_data[region_name]["tenure_count"] += row.count

            # Build summary
            summaries = []
            for region_name, data in region_data.items():
                total = data["total"]
                churned = data["churned"]
                churn_rate = churned / total if total > 0 else 0

                avg_tenure = None
                if data["tenure_count"] > 0:
                    avg_tenure = data["tenure_sum"] / data["tenure_count"]

                summaries.append(
                    {
                        "region": region_name,
                        "total_technicians": total,
                        "active_technicians": data["active"],
                        "churned_technicians": churned,
                        "churn_rate": round(churn_rate, 3),
                        "avg_tenure_days": round(avg_tenure, 1) if avg_tenure else None,
                        "health_status": _get_region_health(churn_rate),
                    }
                )

            # Sort by churn rate descending
            summaries.sort(key=lambda x: x["churn_rate"], reverse=True)

            return {
                "region_count": len(summaries),
                "regions": summaries,
                "overall_insight": _generate_regional_insight(summaries),
            }

    return get_regional_summary


# Helper functions


def _calculate_risk(technician: Technician) -> tuple[float, str]:
    """Calculate risk score and level for a technician."""
    risk_score = 0.5  # Base risk

    if technician.status == "CHURNED":
        return (1.0, "HIGH")

    # Tenure-based risk
    if technician.tenure_days is not None:
        if technician.tenure_days < 90:
            risk_score = 0.7
        elif technician.tenure_days < 180:
            risk_score = 0.5
        elif technician.tenure_days < 365:
            risk_score = 0.3
        else:
            risk_score = 0.15

    # Determine level
    if risk_score >= 0.6:
        return (risk_score, "HIGH")
    elif risk_score >= 0.35:
        return (risk_score, "MEDIUM")
    else:
        return (risk_score, "LOW")


def _build_feature_dict(technician: Technician) -> dict[str, Any]:
    """Build feature dictionary for prediction."""
    features = {
        "tenure_days": technician.tenure_days or 0,
        "region": technician.region or "UNKNOWN",
        "status": technician.status or "UNKNOWN",
    }

    # Add metrics if available
    if technician.metrics:
        features.update(technician.metrics)

    return features


def _generate_basic_explanation(technician: Technician, risk_score: float) -> str:
    """Generate basic natural language explanation without model."""
    parts = []

    if risk_score >= 0.7:
        parts.append("This technician shows high churn risk")
    elif risk_score >= 0.4:
        parts.append("This technician shows moderate churn risk")
    else:
        parts.append("This technician shows low churn risk")

    if technician.tenure_days is not None:
        if technician.tenure_days < 90:
            parts.append(
                f"Their short tenure of {technician.tenure_days} days "
                "increases risk as new technicians are more likely to leave"
            )
        elif technician.tenure_days > 365:
            parts.append(
                f"Their tenure of {technician.tenure_days} days suggests "
                "stability and commitment to the role"
            )

    return ". ".join(parts) + "."


def _generate_roi_recommendation(roi: float, total_techs: int, risk_counts: dict[str, int]) -> str:
    """Generate ROI-based recommendation."""
    if roi > 2:
        rec = "Strongly recommended - excellent ROI potential"
    elif roi > 1:
        rec = "Recommended - good ROI potential"
    elif roi > 0:
        rec = "Consider carefully - marginal positive ROI"
    else:
        rec = "Not recommended - negative ROI expected"

    if risk_counts["HIGH"] > total_techs * 0.5:
        rec += ". High concentration of at-risk technicians makes intervention timely."

    return rec


def _interpret_feature(feature: str) -> str:
    """Generate interpretation for a feature name."""
    interpretations = {
        "tenure_days": "How long the technician has been with the company",
        "region": "Geographic area affects local market conditions and competition",
        "jobs_completed": "Work volume indicates engagement and performance",
        "avg_rating": "Customer satisfaction reflects service quality",
        "response_time": "Speed of accepting jobs shows availability",
        "cancellation_rate": "Job cancellations may indicate reliability issues",
        "training_hours": "Investment in skills development",
        "certification_count": "Professional qualifications held",
    }
    return interpretations.get(feature, "Factor affecting retention outcome")


def _generate_feature_insight(top_features: list[str]) -> str:
    """Generate insight about top features."""
    if not top_features:
        return "No features available for insight generation."

    return (
        f"The top factors affecting technician retention are: "
        f"{', '.join(top_features)}. "
        f"Focus retention efforts on improving these areas for at-risk technicians."
    )


def _get_region_health(churn_rate: float) -> str:
    """Get health status based on churn rate."""
    if churn_rate > 0.3:
        return "CRITICAL"
    elif churn_rate > 0.2:
        return "AT_RISK"
    elif churn_rate > 0.1:
        return "MODERATE"
    else:
        return "HEALTHY"


def _generate_regional_insight(summaries: list[dict]) -> str:
    """Generate insight about regional performance."""
    if not summaries:
        return "No regional data available."

    critical = [s["region"] for s in summaries if s["health_status"] == "CRITICAL"]
    healthy = [s["region"] for s in summaries if s["health_status"] == "HEALTHY"]

    parts = []
    if critical:
        parts.append(f"Regions requiring immediate attention: {', '.join(critical)}")
    if healthy:
        parts.append(f"Best performing regions: {', '.join(healthy[:3])}")

    return ". ".join(parts) if parts else "All regions performing within normal parameters."


def _create_get_recommendations_handler():
    """Create handler for get_strategic_recommendations tool."""

    async def get_strategic_recommendations(
        state: Optional[str] = None,
        focus_areas: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Get strategic recommendations using Cicero-inspired engine."""
        import httpx

        # Call the recommendations API endpoint
        try:
            async with httpx.AsyncClient() as client:
                url = "http://localhost:8000/api/v1/dashboard/recommendations"
                params = {}
                if state:
                    params["state"] = state

                response = await client.get(url, params=params, timeout=30.0)
                response.raise_for_status()
                data = response.json()

                # Format for agent consumption
                recommendations = data.get("recommendations", [])

                # Filter by focus areas if specified
                if focus_areas:
                    recommendations = [
                        r for r in recommendations
                        if r.get("type") in focus_areas
                    ]

                # Build agent-friendly response
                result = {
                    "state": state or "all",
                    "context": {
                        "total_jobs": data.get("context", {}).get("total_jobs", 0),
                        "completion_rate": data.get("context", {}).get("completion_rate", 0),
                        "avg_profit_per_job": data.get("context", {}).get("avg_profit_per_job", 0),
                        "trend": data.get("context", {}).get("trend_direction", "stable"),
                    },
                    "intent": {
                        "primary_goal": data.get("intent", {}).get("primary_goal", "maximize_earnings"),
                        "description": _get_intent_description(data.get("intent", {})),
                    },
                    "recommendation_count": len(recommendations),
                    "recommendations": [
                        {
                            "priority": r.get("priority", "medium"),
                            "type": r.get("type", "general"),
                            "title": r.get("title", ""),
                            "description": r.get("description", ""),
                            "rationale": r.get("rationale", ""),
                            "confidence": r.get("confidence", 0),
                            "predicted_impact": r.get("predicted_impact", {}),
                            "technician_benefit": r.get("technician_benefit", ""),
                            "company_benefit": r.get("company_benefit", ""),
                            "actions": r.get("actions", []),
                        }
                        for r in recommendations
                    ],
                    "summary": _generate_recommendations_summary(recommendations),
                }

                return result

        except httpx.HTTPError as e:
            logger.error(f"Failed to fetch recommendations: {e}")
            return {
                "error": f"Failed to fetch recommendations: {str(e)}",
                "state": state,
            }
        except Exception as e:
            logger.exception(f"Unexpected error in recommendations: {e}")
            return {
                "error": f"Unexpected error: {str(e)}",
                "state": state,
            }

    return get_strategic_recommendations


def _get_intent_description(intent: dict) -> str:
    """Get human-readable description of predicted intent."""
    goal = intent.get("primary_goal", "maximize_earnings")
    descriptions = {
        "maximize_earnings": "Technicians are primarily focused on maximizing their earnings.",
        "efficiency": "Technicians prioritize completing jobs efficiently.",
        "skill_growth": "Technicians are interested in developing new skills.",
        "work_life_balance": "Technicians value work-life balance.",
    }
    return descriptions.get(goal, "Analyzing technician intent...")


def _generate_recommendations_summary(recommendations: list[dict]) -> str:
    """Generate a summary of recommendations for the agent."""
    if not recommendations:
        return "No recommendations at this time. Operations appear healthy."

    critical = [r for r in recommendations if r.get("priority") == "critical"]
    high = [r for r in recommendations if r.get("priority") == "high"]

    parts = []
    if critical:
        parts.append(f"{len(critical)} critical recommendation(s) requiring immediate action")
    if high:
        parts.append(f"{len(high)} high-priority recommendation(s)")

    total = len(recommendations)
    if not parts:
        parts.append(f"{total} recommendation(s) for optimization")

    return ". ".join(parts) + "."
