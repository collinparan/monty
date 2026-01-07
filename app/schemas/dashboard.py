"""Pydantic schemas for dashboard endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


# Chart.js friendly data structures
class LineChartData(BaseModel):
    """Data for Chart.js line chart."""

    labels: list[str] = Field(description="X-axis labels (dates)")
    datasets: list[dict[str, Any]] = Field(description="Chart.js datasets")


class BarChartData(BaseModel):
    """Data for Chart.js bar chart."""

    labels: list[str] = Field(description="Category labels")
    datasets: list[dict[str, Any]] = Field(description="Chart.js datasets")


# Forecast response
class ForecastChartResponse(BaseModel):
    """Prophet forecast data optimized for Chart.js."""

    chart_data: LineChartData
    metrics: dict[str, float]
    forecast_type: str
    periods: int
    last_updated: datetime


# Feature importance response
class FeatureImportanceChartResponse(BaseModel):
    """Feature importance data optimized for Chart.js bar chart."""

    chart_data: BarChartData
    model_type: str
    model_version: str
    top_features: list[dict[str, Any]]


# Region summary
class RegionSummary(BaseModel):
    """Summary statistics for a region."""

    region: str
    technician_count: int
    active_count: int
    churned_count: int
    avg_risk_score: float
    risk_level: str  # LOW, MEDIUM, HIGH
    avg_tenure_days: Optional[float] = None
    avg_rating: Optional[float] = None


class RegionListResponse(BaseModel):
    """Response for regional summary."""

    regions: list[RegionSummary]
    total_technicians: int
    overall_churn_rate: float


# Technician list
class TechnicianSummary(BaseModel):
    """Summary of a technician for dashboard list."""

    id: UUID
    external_id: str
    region: str
    status: str
    tenure_days: Optional[int] = None
    risk_score: Optional[float] = None
    risk_level: Optional[str] = None
    metrics: Optional[dict[str, Any]] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class TechnicianListResponse(BaseModel):
    """Paginated list of technicians."""

    technicians: list[TechnicianSummary]
    total: int
    page: int
    page_size: int
    total_pages: int


# Dashboard overview
class DashboardOverview(BaseModel):
    """High-level dashboard metrics."""

    total_technicians: int
    active_technicians: int
    churned_technicians: int
    churn_rate: float
    avg_tenure_days: float
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    regions_count: int


# Query parameters
class DashboardFilters(BaseModel):
    """Common filters for dashboard endpoints."""

    region: Optional[str] = None
    risk_level: Optional[str] = None
    status: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
