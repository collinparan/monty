"""API endpoints for the analytics dashboard."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Technician, TrainedModel
from app.schemas.dashboard import (
    BarChartData,
    DashboardOverview,
    FeatureImportanceChartResponse,
    ForecastChartResponse,
    LineChartData,
    RegionListResponse,
    RegionSummary,
    TechnicianListResponse,
    TechnicianSummary,
)
from app.schemas.models import ModelType
from app.services.ml import ProphetService

router = APIRouter(prefix="/api/v1/dashboard", tags=["Dashboard"])


def get_prophet_service() -> ProphetService:
    """Get Prophet service instance."""
    return ProphetService()


@router.get("/overview", response_model=DashboardOverview)
async def get_dashboard_overview(
    session: AsyncSession = Depends(get_db),
) -> DashboardOverview:
    """Get high-level dashboard metrics."""
    # Count technicians by status
    total_query = select(func.count(Technician.id))
    active_query = select(func.count(Technician.id)).where(Technician.status == "ACTIVE")
    churned_query = select(func.count(Technician.id)).where(Technician.status == "CHURNED")

    total_result = await session.execute(total_query)
    active_result = await session.execute(active_query)
    churned_result = await session.execute(churned_query)

    total = total_result.scalar() or 0
    active = active_result.scalar() or 0
    churned = churned_result.scalar() or 0

    # Calculate churn rate
    churn_rate = churned / total if total > 0 else 0.0

    # Average tenure
    tenure_query = select(func.avg(Technician.tenure_days))
    tenure_result = await session.execute(tenure_query)
    avg_tenure = tenure_result.scalar() or 0.0

    # Count regions
    regions_query = select(func.count(func.distinct(Technician.region)))
    regions_result = await session.execute(regions_query)
    regions_count = regions_result.scalar() or 0

    # Risk levels from predictions (simplified - in production would join with latest predictions)
    # For now, return placeholder counts based on status
    high_risk = int(churned * 0.6) if churned > 0 else 0
    medium_risk = int((total - active) * 0.3) if total > active else 0
    low_risk = active

    return DashboardOverview(
        total_technicians=total,
        active_technicians=active,
        churned_technicians=churned,
        churn_rate=round(churn_rate, 4),
        avg_tenure_days=round(float(avg_tenure), 1),
        high_risk_count=high_risk,
        medium_risk_count=medium_risk,
        low_risk_count=low_risk,
        regions_count=regions_count,
    )


@router.get("/forecast", response_model=ForecastChartResponse)
async def get_forecast_chart(
    forecast_type: str = Query(default="headcount", description="Type of forecast"),
    periods: int = Query(default=90, ge=1, le=365, description="Forecast periods"),
    session: AsyncSession = Depends(get_db),
    prophet_service: ProphetService = Depends(get_prophet_service),
) -> ForecastChartResponse:
    """Get Prophet forecast data for Chart.js line chart."""
    # Find latest Prophet model for this forecast type
    query = (
        select(TrainedModel)
        .where(TrainedModel.model_type == ModelType.PROPHET.value)
        .where(TrainedModel.is_active == True)  # noqa: E712
        .where(TrainedModel.target == forecast_type)
        .order_by(TrainedModel.created_at.desc())
        .limit(1)
    )

    result = await session.execute(query)
    model = result.scalar_one_or_none()

    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No active Prophet model found for forecast type: {forecast_type}",
        )

    # Get forecast predictions
    forecast = prophet_service.predict_future(
        model_path=model.file_path,
        periods=periods,
    )

    # Format for Chart.js
    chart_data = LineChartData(
        labels=forecast["dates"],
        datasets=[
            {
                "label": "Forecast",
                "data": forecast["yhat"],
                "borderColor": "rgb(75, 192, 192)",
                "backgroundColor": "rgba(75, 192, 192, 0.2)",
                "fill": False,
            },
            {
                "label": "Upper Bound",
                "data": forecast["yhat_upper"],
                "borderColor": "rgba(75, 192, 192, 0.5)",
                "borderDash": [5, 5],
                "fill": False,
            },
            {
                "label": "Lower Bound",
                "data": forecast["yhat_lower"],
                "borderColor": "rgba(75, 192, 192, 0.5)",
                "borderDash": [5, 5],
                "fill": "-1",
                "backgroundColor": "rgba(75, 192, 192, 0.1)",
            },
        ],
    )

    return ForecastChartResponse(
        chart_data=chart_data,
        metrics=model.metrics or {},
        forecast_type=forecast_type,
        periods=periods,
        last_updated=model.updated_at,
    )


@router.get("/feature-importance", response_model=FeatureImportanceChartResponse)
async def get_feature_importance_chart(
    model_type: str = Query(default="EBM", description="Model type (EBM or DECISION_TREE)"),
    session: AsyncSession = Depends(get_db),
) -> FeatureImportanceChartResponse:
    """Get feature importance data for Chart.js bar chart."""
    # Find latest model of specified type
    valid_types = [ModelType.EBM.value, ModelType.DECISION_TREE.value]
    if model_type not in valid_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model type must be one of: {valid_types}",
        )

    query = (
        select(TrainedModel)
        .where(TrainedModel.model_type == model_type)
        .where(TrainedModel.is_active == True)  # noqa: E712
        .order_by(TrainedModel.created_at.desc())
        .limit(1)
    )

    result = await session.execute(query)
    model = result.scalar_one_or_none()

    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No active {model_type} model found",
        )

    if not model.feature_importance:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model does not have feature importance data",
        )

    features = model.feature_importance.get("features", [])
    importances = model.feature_importance.get("importances", [])

    # Take top 10 features
    top_n = 10
    features = features[:top_n]
    importances = importances[:top_n]

    # Format for Chart.js horizontal bar chart
    chart_data = BarChartData(
        labels=features,
        datasets=[
            {
                "label": "Feature Importance",
                "data": importances,
                "backgroundColor": [
                    "rgba(255, 99, 132, 0.8)",
                    "rgba(54, 162, 235, 0.8)",
                    "rgba(255, 206, 86, 0.8)",
                    "rgba(75, 192, 192, 0.8)",
                    "rgba(153, 102, 255, 0.8)",
                    "rgba(255, 159, 64, 0.8)",
                    "rgba(199, 199, 199, 0.8)",
                    "rgba(83, 102, 255, 0.8)",
                    "rgba(255, 99, 255, 0.8)",
                    "rgba(99, 255, 132, 0.8)",
                ][: len(features)],
                "borderColor": [
                    "rgb(255, 99, 132)",
                    "rgb(54, 162, 235)",
                    "rgb(255, 206, 86)",
                    "rgb(75, 192, 192)",
                    "rgb(153, 102, 255)",
                    "rgb(255, 159, 64)",
                    "rgb(199, 199, 199)",
                    "rgb(83, 102, 255)",
                    "rgb(255, 99, 255)",
                    "rgb(99, 255, 132)",
                ][: len(features)],
                "borderWidth": 1,
            }
        ],
    )

    # Format top features for display
    top_features = [
        {"feature": f, "importance": round(i, 4), "rank": idx + 1}
        for idx, (f, i) in enumerate(zip(features, importances))
    ]

    return FeatureImportanceChartResponse(
        chart_data=chart_data,
        model_type=model_type,
        model_version=model.version,
        top_features=top_features,
    )


@router.get("/regions", response_model=RegionListResponse)
async def get_regions_summary(
    session: AsyncSession = Depends(get_db),
) -> RegionListResponse:
    """Get regional summary with risk scores."""
    # Get technician counts by region and status
    query = select(
        Technician.region,
        Technician.status,
        func.count(Technician.id).label("count"),
        func.avg(Technician.tenure_days).label("avg_tenure"),
    ).group_by(Technician.region, Technician.status)

    result = await session.execute(query)
    rows = result.all()

    # Aggregate by region
    region_data: dict[str, dict] = {}
    for row in rows:
        region = row.region
        if region not in region_data:
            region_data[region] = {
                "total": 0,
                "active": 0,
                "churned": 0,
                "tenure_sum": 0,
                "tenure_count": 0,
            }

        region_data[region]["total"] += row.count
        if row.status == "ACTIVE":
            region_data[region]["active"] = row.count
        elif row.status == "CHURNED":
            region_data[region]["churned"] = row.count

        if row.avg_tenure:
            region_data[region]["tenure_sum"] += row.avg_tenure * row.count
            region_data[region]["tenure_count"] += row.count

    # Build region summaries
    regions = []
    total_technicians = 0
    total_churned = 0

    for region_name, data in region_data.items():
        total = data["total"]
        churned = data["churned"]
        active = data["active"]

        # Calculate risk score (simplified: churn rate)
        risk_score = churned / total if total > 0 else 0.0

        # Determine risk level
        if risk_score > 0.3:
            risk_level = "HIGH"
        elif risk_score > 0.15:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # Average tenure
        avg_tenure = None
        if data["tenure_count"] > 0:
            avg_tenure = data["tenure_sum"] / data["tenure_count"]

        regions.append(
            RegionSummary(
                region=region_name,
                technician_count=total,
                active_count=active,
                churned_count=churned,
                avg_risk_score=round(risk_score, 4),
                risk_level=risk_level,
                avg_tenure_days=round(avg_tenure, 1) if avg_tenure else None,
            )
        )

        total_technicians += total
        total_churned += churned

    # Sort by risk score descending
    regions.sort(key=lambda x: x.avg_risk_score, reverse=True)

    overall_churn_rate = total_churned / total_technicians if total_technicians > 0 else 0.0

    return RegionListResponse(
        regions=regions,
        total_technicians=total_technicians,
        overall_churn_rate=round(overall_churn_rate, 4),
    )


@router.get("/technicians", response_model=TechnicianListResponse)
async def get_technicians_list(
    region: Optional[str] = Query(default=None, description="Filter by region"),
    status: Optional[str] = Query(default=None, description="Filter by status"),
    risk_level: Optional[str] = Query(default=None, description="Filter by risk level"),
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page"),
    session: AsyncSession = Depends(get_db),
) -> TechnicianListResponse:
    """Get paginated list of technicians with predictions."""
    # Build query with filters
    query = select(Technician)

    if region:
        query = query.where(Technician.region == region)
    if status:
        query = query.where(Technician.status == status)

    # Get total count
    count_query = select(func.count(Technician.id))
    if region:
        count_query = count_query.where(Technician.region == region)
    if status:
        count_query = count_query.where(Technician.status == status)

    count_result = await session.execute(count_query)
    total = count_result.scalar() or 0

    # Apply pagination
    offset = (page - 1) * page_size
    query = query.order_by(Technician.created_at.desc()).offset(offset).limit(page_size)

    result = await session.execute(query)
    technicians = result.scalars().all()

    # Build response
    technician_summaries = []
    for tech in technicians:
        # Calculate risk score from metrics (simplified)
        risk_score = None
        risk_level_value = None

        if tech.status == "CHURNED":
            risk_score = 1.0
            risk_level_value = "HIGH"
        elif tech.tenure_days and tech.tenure_days < 90:
            risk_score = 0.6
            risk_level_value = "HIGH"
        elif tech.tenure_days and tech.tenure_days < 180:
            risk_score = 0.4
            risk_level_value = "MEDIUM"
        else:
            risk_score = 0.2
            risk_level_value = "LOW"

        # Filter by risk level if specified
        if risk_level and risk_level_value != risk_level:
            continue

        technician_summaries.append(
            TechnicianSummary(
                id=tech.id,
                external_id=tech.external_id,
                region=tech.region,
                status=tech.status,
                tenure_days=tech.tenure_days,
                risk_score=risk_score,
                risk_level=risk_level_value,
                metrics=tech.metrics,
                created_at=tech.created_at,
            )
        )

    total_pages = (total + page_size - 1) // page_size if total > 0 else 1

    return TechnicianListResponse(
        technicians=technician_summaries,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )
