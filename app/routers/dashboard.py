"""API endpoints for the analytics dashboard."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert

from app.database import get_db
from app.dependencies import get_redis
from app.models import Technician, TrainedModel
from app.services.snowflake import SnowflakeService
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
from app.services.recommendations import (
    recommendation_engine,
    RecommendationsResponse,
    RecommendationType,
    StrategicRecommendation,
)

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


class SyncResponse(BaseModel):
    """Response for data sync operation."""

    status: str
    message: str
    records_synced: int = 0


class TimeSeriesPoint(BaseModel):
    """A single point in a time series."""
    date: str
    value: float


class TimeSeriesResponse(BaseModel):
    """Response for time series data."""
    status: str
    metric: str
    data: list[TimeSeriesPoint]
    total_points: int


class ForecastResponse(BaseModel):
    """Response for Prophet forecast with decomposition."""
    status: str
    metric: str
    historical: dict
    forecast: dict
    components: dict
    metrics: dict


class StateTimeSeriesPoint(BaseModel):
    """A single point in a state time series."""
    date: str
    state: str
    value: float


class StateTimeSeriesResponse(BaseModel):
    """Response for state-level time series data."""
    status: str
    metric: str
    states: list[str]
    data: dict[str, list[TimeSeriesPoint]]  # state -> list of points
    total_points: int


# ZIP code prefix to US state mapping
ZIP_TO_STATE = {
    "005": "NY", "006": "PR", "007": "PR", "008": "PR", "009": "PR",
    "010": "MA", "011": "MA", "012": "MA", "013": "MA", "014": "MA", "015": "MA", "016": "MA", "017": "MA", "018": "MA", "019": "MA",
    "020": "MA", "021": "MA", "022": "MA", "023": "MA", "024": "MA", "025": "MA", "026": "MA", "027": "MA",
    "028": "RI", "029": "RI",
    "030": "NH", "031": "NH", "032": "NH", "033": "NH", "034": "NH", "035": "NH", "036": "NH", "037": "NH", "038": "NH",
    "039": "ME", "040": "ME", "041": "ME", "042": "ME", "043": "ME", "044": "ME", "045": "ME", "046": "ME", "047": "ME", "048": "ME", "049": "ME",
    "050": "VT", "051": "VT", "052": "VT", "053": "VT", "054": "VT", "055": "VT", "056": "VT", "057": "VT", "058": "VT", "059": "VT",
    "060": "CT", "061": "CT", "062": "CT", "063": "CT", "064": "CT", "065": "CT", "066": "CT", "067": "CT", "068": "CT", "069": "CT",
    "070": "NJ", "071": "NJ", "072": "NJ", "073": "NJ", "074": "NJ", "075": "NJ", "076": "NJ", "077": "NJ", "078": "NJ", "079": "NJ",
    "080": "NJ", "081": "NJ", "082": "NJ", "083": "NJ", "084": "NJ", "085": "NJ", "086": "NJ", "087": "NJ", "088": "NJ", "089": "NJ",
    "100": "NY", "101": "NY", "102": "NY", "103": "NY", "104": "NY", "105": "NY", "106": "NY", "107": "NY", "108": "NY", "109": "NY",
    "110": "NY", "111": "NY", "112": "NY", "113": "NY", "114": "NY", "115": "NY", "116": "NY", "117": "NY", "118": "NY", "119": "NY",
    "120": "NY", "121": "NY", "122": "NY", "123": "NY", "124": "NY", "125": "NY", "126": "NY", "127": "NY", "128": "NY", "129": "NY",
    "130": "NY", "131": "NY", "132": "NY", "133": "NY", "134": "NY", "135": "NY", "136": "NY", "137": "NY", "138": "NY", "139": "NY",
    "140": "NY", "141": "NY", "142": "NY", "143": "NY", "144": "NY", "145": "NY", "146": "NY", "147": "NY", "148": "NY", "149": "NY",
    "150": "PA", "151": "PA", "152": "PA", "153": "PA", "154": "PA", "155": "PA", "156": "PA", "157": "PA", "158": "PA", "159": "PA",
    "160": "PA", "161": "PA", "162": "PA", "163": "PA", "164": "PA", "165": "PA", "166": "PA", "167": "PA", "168": "PA", "169": "PA",
    "170": "PA", "171": "PA", "172": "PA", "173": "PA", "174": "PA", "175": "PA", "176": "PA", "177": "PA", "178": "PA", "179": "PA",
    "180": "PA", "181": "PA", "182": "PA", "183": "PA", "184": "PA", "185": "PA", "186": "PA", "187": "PA", "188": "PA", "189": "PA",
    "190": "PA", "191": "PA", "192": "PA", "193": "PA", "194": "PA", "195": "PA", "196": "PA",
    "197": "DE", "198": "DE", "199": "DE",
    "200": "DC", "201": "VA", "202": "DC", "203": "DC", "204": "DC", "205": "DC",
    "206": "MD", "207": "MD", "208": "MD", "209": "MD", "210": "MD", "211": "MD", "212": "MD", "214": "MD", "215": "MD", "216": "MD", "217": "MD", "218": "MD", "219": "MD",
    "220": "VA", "221": "VA", "222": "VA", "223": "VA", "224": "VA", "225": "VA", "226": "VA", "227": "VA", "228": "VA", "229": "VA",
    "230": "VA", "231": "VA", "232": "VA", "233": "VA", "234": "VA", "235": "VA", "236": "VA", "237": "VA", "238": "VA", "239": "VA",
    "240": "VA", "241": "VA", "242": "VA", "243": "VA", "244": "VA", "245": "VA", "246": "VA",
    "247": "WV", "248": "WV", "249": "WV", "250": "WV", "251": "WV", "252": "WV", "253": "WV", "254": "WV", "255": "WV", "256": "WV",
    "257": "WV", "258": "WV", "259": "WV", "260": "WV", "261": "WV", "262": "WV", "263": "WV", "264": "WV", "265": "WV", "266": "WV", "267": "WV", "268": "WV",
    "270": "NC", "271": "NC", "272": "NC", "273": "NC", "274": "NC", "275": "NC", "276": "NC", "277": "NC", "278": "NC", "279": "NC",
    "280": "NC", "281": "NC", "282": "NC", "283": "NC", "284": "NC", "285": "NC", "286": "NC", "287": "NC", "288": "NC", "289": "NC",
    "290": "SC", "291": "SC", "292": "SC", "293": "SC", "294": "SC", "295": "SC", "296": "SC", "297": "SC", "298": "SC", "299": "SC",
    "300": "GA", "301": "GA", "302": "GA", "303": "GA", "304": "GA", "305": "GA", "306": "GA", "307": "GA", "308": "GA", "309": "GA",
    "310": "GA", "311": "GA", "312": "GA", "313": "GA", "314": "GA", "315": "GA", "316": "GA", "317": "GA", "318": "GA", "319": "GA",
    "320": "FL", "321": "FL", "322": "FL", "323": "FL", "324": "FL", "325": "FL", "326": "FL", "327": "FL", "328": "FL", "329": "FL",
    "330": "FL", "331": "FL", "332": "FL", "333": "FL", "334": "FL", "335": "FL", "336": "FL", "337": "FL", "338": "FL", "339": "FL",
    "340": "FL", "341": "FL", "342": "FL", "344": "FL", "346": "FL", "347": "FL", "349": "FL",
    "350": "AL", "351": "AL", "352": "AL", "354": "AL", "355": "AL", "356": "AL", "357": "AL", "358": "AL", "359": "AL",
    "360": "AL", "361": "AL", "362": "AL", "363": "AL", "364": "AL", "365": "AL", "366": "AL", "367": "AL", "368": "AL", "369": "AL",
    "370": "TN", "371": "TN", "372": "TN", "373": "TN", "374": "TN", "375": "TN", "376": "TN", "377": "TN", "378": "TN", "379": "TN",
    "380": "TN", "381": "TN", "382": "TN", "383": "TN", "384": "TN", "385": "TN",
    "386": "MS", "387": "MS", "388": "MS", "389": "MS", "390": "MS", "391": "MS", "392": "MS", "393": "MS", "394": "MS", "395": "MS", "396": "MS", "397": "MS",
    "400": "KY", "401": "KY", "402": "KY", "403": "KY", "404": "KY", "405": "KY", "406": "KY", "407": "KY", "408": "KY", "409": "KY",
    "410": "KY", "411": "KY", "412": "KY", "413": "KY", "414": "KY", "415": "KY", "416": "KY", "417": "KY", "418": "KY",
    "420": "KY", "421": "KY", "422": "KY", "423": "KY", "424": "KY", "425": "KY", "426": "KY", "427": "KY",
    "430": "OH", "431": "OH", "432": "OH", "433": "OH", "434": "OH", "435": "OH", "436": "OH", "437": "OH", "438": "OH", "439": "OH",
    "440": "OH", "441": "OH", "442": "OH", "443": "OH", "444": "OH", "445": "OH", "446": "OH", "447": "OH", "448": "OH", "449": "OH",
    "450": "OH", "451": "OH", "452": "OH", "453": "OH", "454": "OH", "455": "OH", "456": "OH", "457": "OH", "458": "OH", "459": "OH",
    "460": "IN", "461": "IN", "462": "IN", "463": "IN", "464": "IN", "465": "IN", "466": "IN", "467": "IN", "468": "IN", "469": "IN",
    "470": "IN", "471": "IN", "472": "IN", "473": "IN", "474": "IN", "475": "IN", "476": "IN", "477": "IN", "478": "IN", "479": "IN",
    "480": "MI", "481": "MI", "482": "MI", "483": "MI", "484": "MI", "485": "MI", "486": "MI", "487": "MI", "488": "MI", "489": "MI",
    "490": "MI", "491": "MI", "492": "MI", "493": "MI", "494": "MI", "495": "MI", "496": "MI", "497": "MI", "498": "MI", "499": "MI",
    "500": "IA", "501": "IA", "502": "IA", "503": "IA", "504": "IA", "505": "IA", "506": "IA", "507": "IA", "508": "IA", "509": "IA",
    "510": "IA", "511": "IA", "512": "IA", "513": "IA", "514": "IA", "515": "IA", "516": "IA",
    "520": "IA", "521": "IA", "522": "IA", "523": "IA", "524": "IA", "525": "IA", "526": "IA", "527": "IA", "528": "IA",
    "530": "WI", "531": "WI", "532": "WI", "534": "WI", "535": "WI", "537": "WI", "538": "WI", "539": "WI",
    "540": "WI", "541": "WI", "542": "WI", "543": "WI", "544": "WI", "545": "WI", "546": "WI", "547": "WI", "548": "WI", "549": "WI",
    "550": "MN", "551": "MN", "553": "MN", "554": "MN", "555": "MN", "556": "MN", "557": "MN", "558": "MN", "559": "MN",
    "560": "MN", "561": "MN", "562": "MN", "563": "MN", "564": "MN", "565": "MN", "566": "MN", "567": "MN",
    "570": "SD", "571": "SD", "572": "SD", "573": "SD", "574": "SD", "575": "SD", "576": "SD", "577": "SD",
    "580": "ND", "581": "ND", "582": "ND", "583": "ND", "584": "ND", "585": "ND", "586": "ND", "587": "ND", "588": "ND",
    "590": "MT", "591": "MT", "592": "MT", "593": "MT", "594": "MT", "595": "MT", "596": "MT", "597": "MT", "598": "MT", "599": "MT",
    "600": "IL", "601": "IL", "602": "IL", "603": "IL", "604": "IL", "605": "IL", "606": "IL", "607": "IL", "608": "IL", "609": "IL",
    "610": "IL", "611": "IL", "612": "IL", "613": "IL", "614": "IL", "615": "IL", "616": "IL", "617": "IL", "618": "IL", "619": "IL",
    "620": "IL", "621": "IL", "622": "IL", "623": "IL", "624": "IL", "625": "IL", "626": "IL", "627": "IL", "628": "IL", "629": "IL",
    "630": "MO", "631": "MO", "633": "MO", "634": "MO", "635": "MO", "636": "MO", "637": "MO", "638": "MO", "639": "MO",
    "640": "MO", "641": "MO", "644": "MO", "645": "MO", "646": "MO", "647": "MO", "648": "MO", "649": "MO",
    "650": "MO", "651": "MO", "652": "MO", "653": "MO", "654": "MO", "655": "MO", "656": "MO", "657": "MO", "658": "MO",
    "660": "KS", "661": "KS", "662": "KS", "664": "KS", "665": "KS", "666": "KS", "667": "KS", "668": "KS", "669": "KS",
    "670": "KS", "671": "KS", "672": "KS", "673": "KS", "674": "KS", "675": "KS", "676": "KS", "677": "KS", "678": "KS", "679": "KS",
    "680": "NE", "681": "NE", "683": "NE", "684": "NE", "685": "NE", "686": "NE", "687": "NE", "688": "NE", "689": "NE",
    "690": "NE", "691": "NE", "692": "NE", "693": "NE",
    "700": "LA", "701": "LA", "703": "LA", "704": "LA", "705": "LA", "706": "LA", "707": "LA", "708": "LA",
    "710": "LA", "711": "LA", "712": "LA", "713": "LA", "714": "LA",
    "716": "AR", "717": "AR", "718": "AR", "719": "AR", "720": "AR", "721": "AR", "722": "AR", "723": "AR", "724": "AR", "725": "AR",
    "726": "AR", "727": "AR", "728": "AR", "729": "AR",
    "730": "OK", "731": "OK", "733": "OK", "734": "OK", "735": "OK", "736": "OK", "737": "OK", "738": "OK", "739": "OK",
    "740": "OK", "741": "OK", "743": "OK", "744": "OK", "745": "OK", "746": "OK", "747": "OK", "748": "OK", "749": "OK",
    "750": "TX", "751": "TX", "752": "TX", "753": "TX", "754": "TX", "755": "TX", "756": "TX", "757": "TX", "758": "TX", "759": "TX",
    "760": "TX", "761": "TX", "762": "TX", "763": "TX", "764": "TX", "765": "TX", "766": "TX", "767": "TX", "768": "TX", "769": "TX",
    "770": "TX", "771": "TX", "772": "TX", "773": "TX", "774": "TX", "775": "TX", "776": "TX", "777": "TX", "778": "TX", "779": "TX",
    "780": "TX", "781": "TX", "782": "TX", "783": "TX", "784": "TX", "785": "TX", "786": "TX", "787": "TX", "788": "TX", "789": "TX",
    "790": "TX", "791": "TX", "792": "TX", "793": "TX", "794": "TX", "795": "TX", "796": "TX", "797": "TX", "798": "TX", "799": "TX",
    "800": "CO", "801": "CO", "802": "CO", "803": "CO", "804": "CO", "805": "CO", "806": "CO", "807": "CO", "808": "CO", "809": "CO",
    "810": "CO", "811": "CO", "812": "CO", "813": "CO", "814": "CO", "815": "CO", "816": "CO",
    "820": "WY", "821": "WY", "822": "WY", "823": "WY", "824": "WY", "825": "WY", "826": "WY", "827": "WY", "828": "WY", "829": "WY", "830": "WY", "831": "WY",
    "832": "ID", "833": "ID", "834": "ID", "835": "ID", "836": "ID", "837": "ID", "838": "ID",
    "840": "UT", "841": "UT", "842": "UT", "843": "UT", "844": "UT", "845": "UT", "846": "UT", "847": "UT",
    "850": "AZ", "851": "AZ", "852": "AZ", "853": "AZ", "855": "AZ", "856": "AZ", "857": "AZ", "859": "AZ",
    "860": "AZ", "863": "AZ", "864": "AZ", "865": "AZ",
    "870": "NM", "871": "NM", "872": "NM", "873": "NM", "874": "NM", "875": "NM", "877": "NM", "878": "NM", "879": "NM",
    "880": "NM", "881": "NM", "882": "NM", "883": "NM", "884": "NM",
    "889": "NV", "890": "NV", "891": "NV", "893": "NV", "894": "NV", "895": "NV", "896": "NV", "897": "NV", "898": "NV",
    "900": "CA", "901": "CA", "902": "CA", "903": "CA", "904": "CA", "905": "CA", "906": "CA", "907": "CA", "908": "CA", "909": "CA",
    "910": "CA", "911": "CA", "912": "CA", "913": "CA", "914": "CA", "915": "CA", "916": "CA", "917": "CA", "918": "CA", "919": "CA",
    "920": "CA", "921": "CA", "922": "CA", "923": "CA", "924": "CA", "925": "CA", "926": "CA", "927": "CA", "928": "CA",
    "930": "CA", "931": "CA", "932": "CA", "933": "CA", "934": "CA", "935": "CA", "936": "CA", "937": "CA", "938": "CA", "939": "CA",
    "940": "CA", "941": "CA", "942": "CA", "943": "CA", "944": "CA", "945": "CA", "946": "CA", "947": "CA", "948": "CA", "949": "CA",
    "950": "CA", "951": "CA", "952": "CA", "953": "CA", "954": "CA", "955": "CA", "956": "CA", "957": "CA", "958": "CA", "959": "CA",
    "960": "CA", "961": "CA",
    "967": "HI", "968": "HI",
    "970": "OR", "971": "OR", "972": "OR", "973": "OR", "974": "OR", "975": "OR", "976": "OR", "977": "OR", "978": "OR", "979": "OR",
    "980": "WA", "981": "WA", "982": "WA", "983": "WA", "984": "WA", "985": "WA", "986": "WA", "987": "WA", "988": "WA", "989": "WA",
    "990": "WA", "991": "WA", "992": "WA", "993": "WA", "994": "WA",
    "995": "AK", "996": "AK", "997": "AK", "998": "AK", "999": "AK",
}


def zip_to_state(zip_code: str) -> str | None:
    """Convert ZIP code to US state abbreviation."""
    if not zip_code or len(zip_code) < 3:
        return None
    prefix = zip_code[:3]
    return ZIP_TO_STATE.get(prefix)


@router.get("/timeseries/jobs/by-state")
async def get_jobs_timeseries_by_state(
    days: int = Query(default=90, description="Number of days of history"),
    top_n: int = Query(default=10, description="Number of top states to return"),
) -> StateTimeSeriesResponse:
    """Get job assignments over time by US state from Snowflake."""
    from app.snowflake import execute_query

    # Query jobs with vendor ZIP codes to derive state
    query = f"""
    SELECT
        DATE(J.ASSIGNED_AT) AS period,
        SUBSTRING(REPLACE(REPLACE(V.ZIP_CODES, '{{', ''), '}}', ''), 1, 3) AS zip_prefix,
        COUNT(*) AS job_count
    FROM IH_DATASCIENCE.HS_1099.JOBS_ASSIGNMENTS J
    JOIN IH_DATASCIENCE.PUBLIC.VENDORS V ON J.VENDOR_ID = V.ID
    WHERE J.ASSIGNED_AT >= DATEADD(day, -{days}, CURRENT_DATE())
      AND V.ZIP_CODES IS NOT NULL
    GROUP BY period, zip_prefix
    ORDER BY period ASC
    """

    try:
        result = await execute_query(query)

        # Aggregate by state
        state_data: dict[str, dict[str, float]] = {}  # state -> date -> count
        state_totals: dict[str, float] = {}  # state -> total jobs

        for row in result:
            period = row.get("PERIOD")
            zip_prefix = row.get("ZIP_PREFIX", "")
            job_count = float(row.get("JOB_COUNT", 0) or 0)

            if not period or not zip_prefix:
                continue

            state = zip_to_state(zip_prefix)
            if not state:
                continue

            date_str = period.strftime("%Y-%m-%d") if hasattr(period, "strftime") else str(period)[:10]

            if state not in state_data:
                state_data[state] = {}
                state_totals[state] = 0

            if date_str not in state_data[state]:
                state_data[state][date_str] = 0

            state_data[state][date_str] += job_count
            state_totals[state] += job_count

        # Get top N states by total jobs
        top_states = sorted(state_totals.keys(), key=lambda s: state_totals[s], reverse=True)[:top_n]

        # Build response with only top states
        data: dict[str, list[TimeSeriesPoint]] = {}
        total_points = 0

        for state in top_states:
            dates_values = state_data[state]
            sorted_dates = sorted(dates_values.keys())
            data[state] = [
                TimeSeriesPoint(date=d, value=dates_values[d])
                for d in sorted_dates
            ]
            total_points += len(data[state])

        return StateTimeSeriesResponse(
            status="success",
            metric="jobs_by_state",
            states=top_states,
            data=data,
            total_points=total_points,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch state time series data: {e}",
        )


@router.post("/forecast/jobs/by-state")
async def forecast_jobs_by_state(
    state: str = Query(..., description="US state abbreviation (e.g., CA, TX, FL)"),
    days_history: int = Query(default=90, description="Days of historical data to use"),
    days_forecast: int = Query(default=30, description="Days to forecast"),
) -> ForecastResponse:
    """Train Prophet model on job data for a specific state and return forecast with decomposition."""
    from app.snowflake import execute_query
    import pandas as pd

    # Get ZIP prefixes for the state
    state_zips = [prefix for prefix, st in ZIP_TO_STATE.items() if st == state.upper()]
    if not state_zips:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid state abbreviation: {state}",
        )

    # Build ZIP prefix filter
    zip_conditions = " OR ".join([f"V.ZIP_CODES LIKE '%{prefix}%'" for prefix in state_zips[:20]])  # Limit to avoid huge query

    query = f"""
    SELECT
        DATE(J.ASSIGNED_AT) AS ds,
        COUNT(*) AS y
    FROM IH_DATASCIENCE.HS_1099.JOBS_ASSIGNMENTS J
    JOIN IH_DATASCIENCE.PUBLIC.VENDORS V ON J.VENDOR_ID = V.ID
    WHERE J.ASSIGNED_AT >= DATEADD(day, -{days_history}, CURRENT_DATE())
      AND ({zip_conditions})
    GROUP BY ds
    ORDER BY ds ASC
    """

    try:
        result = await execute_query(query)

        if not result or len(result) < 7:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Not enough historical data for {state}. Need at least 7 days.",
            )

        # Convert to DataFrame
        df = pd.DataFrame(result)
        df.columns = [c.lower() for c in df.columns]
        df["ds"] = pd.to_datetime(df["ds"])
        df["y"] = df["y"].astype(float)

        # Train Prophet model
        prophet_service = get_prophet_service()
        forecast_result = prophet_service.train_forecast(
            df=df,
            date_column="ds",
            value_column="y",
            forecast_type=f"jobs_{state}",
            periods=days_forecast,
            freq="D",
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
        )

        return ForecastResponse(
            status="success",
            metric=f"jobs_forecast_{state}",
            historical=forecast_result["forecast"]["historical"],
            forecast=forecast_result["forecast"]["forecast"],
            components=forecast_result["components"],
            metrics=forecast_result["metrics"],
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate forecast for {state}: {e}",
        )


@router.get("/timeseries/jobs")
async def get_jobs_timeseries(
    days: int = Query(default=90, description="Number of days of history"),
    granularity: str = Query(default="day", description="Aggregation: day, week, month"),
) -> TimeSeriesResponse:
    """Get job assignments over time from Snowflake."""
    from app.snowflake import execute_query

    # Map granularity to Snowflake date_trunc
    granularity_map = {
        "day": "DAY",
        "week": "WEEK",
        "month": "MONTH",
    }
    sf_granularity = granularity_map.get(granularity, "DAY")

    query = f"""
    SELECT
        DATE_TRUNC('{sf_granularity}', ASSIGNED_AT) AS period,
        COUNT(*) AS job_count,
        COUNT(DISTINCT VENDOR_ID) AS vendor_count,
        SUM(CASE WHEN STATUS = 'completed' THEN 1 ELSE 0 END) AS completed_count
    FROM IH_DATASCIENCE.HS_1099.JOBS_ASSIGNMENTS
    WHERE ASSIGNED_AT >= DATEADD(day, -{days}, CURRENT_DATE())
    GROUP BY period
    ORDER BY period ASC
    """

    try:
        result = await execute_query(query)

        data = []
        for row in result:
            period = row.get("PERIOD")
            if period:
                date_str = period.strftime("%Y-%m-%d") if hasattr(period, "strftime") else str(period)[:10]
                data.append(TimeSeriesPoint(
                    date=date_str,
                    value=float(row.get("JOB_COUNT", 0) or 0)
                ))

        return TimeSeriesResponse(
            status="success",
            metric="jobs_assigned",
            data=data,
            total_points=len(data),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch time series data: {e}",
        )


@router.get("/timeseries/revenue")
async def get_revenue_timeseries(
    days: int = Query(default=90, description="Number of days of history"),
    granularity: str = Query(default="day", description="Aggregation: day, week, month"),
) -> TimeSeriesResponse:
    """Get revenue over time from Snowflake."""
    from app.snowflake import execute_query

    granularity_map = {"day": "DAY", "week": "WEEK", "month": "MONTH"}
    sf_granularity = granularity_map.get(granularity, "DAY")

    query = f"""
    SELECT
        DATE_TRUNC('{sf_granularity}', ASSIGNED_AT) AS period,
        SUM(COALESCE(TOT_REV, 0)) AS total_revenue,
        SUM(COALESCE(ACTUAL_PPT, 0)) AS total_profit
    FROM IH_DATASCIENCE.HS_1099.ASSIGNED_JOBS_FINANCIALS
    WHERE ASSIGNED_AT >= DATEADD(day, -{days}, CURRENT_DATE())
    GROUP BY period
    ORDER BY period ASC
    """

    try:
        result = await execute_query(query)

        data = []
        for row in result:
            period = row.get("PERIOD")
            if period:
                date_str = period.strftime("%Y-%m-%d") if hasattr(period, "strftime") else str(period)[:10]
                data.append(TimeSeriesPoint(
                    date=date_str,
                    value=float(row.get("TOTAL_REVENUE", 0) or 0)
                ))

        return TimeSeriesResponse(
            status="success",
            metric="revenue",
            data=data,
            total_points=len(data),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch revenue data: {e}",
        )


@router.post("/forecast/jobs")
async def forecast_jobs(
    days_history: int = Query(default=90, description="Days of historical data to use"),
    days_forecast: int = Query(default=30, description="Days to forecast"),
) -> ForecastResponse:
    """Train Prophet model on job data and return forecast with decomposition."""
    from app.snowflake import execute_query
    import pandas as pd

    # Fetch historical job data
    query = f"""
    SELECT
        DATE(ASSIGNED_AT) AS ds,
        COUNT(*) AS y
    FROM IH_DATASCIENCE.HS_1099.JOBS_ASSIGNMENTS
    WHERE ASSIGNED_AT >= DATEADD(day, -{days_history}, CURRENT_DATE())
    GROUP BY ds
    ORDER BY ds ASC
    """

    try:
        result = await execute_query(query)

        if not result or len(result) < 7:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Not enough historical data for forecasting. Need at least 7 days.",
            )

        # Convert to DataFrame
        df = pd.DataFrame(result)
        df.columns = [c.lower() for c in df.columns]
        df["ds"] = pd.to_datetime(df["ds"])
        df["y"] = df["y"].astype(float)

        # Train Prophet model
        prophet_service = get_prophet_service()
        forecast_result = prophet_service.train_forecast(
            df=df,
            date_column="ds",
            value_column="y",
            forecast_type="jobs",
            periods=days_forecast,
            freq="D",
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
        )

        return ForecastResponse(
            status="success",
            metric="jobs_forecast",
            historical=forecast_result["forecast"]["historical"],
            forecast=forecast_result["forecast"]["forecast"],
            components=forecast_result["components"],
            metrics=forecast_result["metrics"],
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate forecast: {e}",
        )


@router.post("/forecast/revenue")
async def forecast_revenue(
    days_history: int = Query(default=90, description="Days of historical data to use"),
    days_forecast: int = Query(default=30, description="Days to forecast"),
) -> ForecastResponse:
    """Train Prophet model on revenue data and return forecast with decomposition."""
    from app.snowflake import execute_query
    import pandas as pd

    query = f"""
    SELECT
        DATE(ASSIGNED_AT) AS ds,
        SUM(COALESCE(TOT_REV, 0)) AS y
    FROM IH_DATASCIENCE.HS_1099.ASSIGNED_JOBS_FINANCIALS
    WHERE ASSIGNED_AT >= DATEADD(day, -{days_history}, CURRENT_DATE())
    GROUP BY ds
    ORDER BY ds ASC
    """

    try:
        result = await execute_query(query)

        if not result or len(result) < 7:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Not enough historical data for forecasting. Need at least 7 days.",
            )

        df = pd.DataFrame(result)
        df.columns = [c.lower() for c in df.columns]
        df["ds"] = pd.to_datetime(df["ds"])
        df["y"] = df["y"].astype(float)

        prophet_service = get_prophet_service()
        forecast_result = prophet_service.train_forecast(
            df=df,
            date_column="ds",
            value_column="y",
            forecast_type="revenue",
            periods=days_forecast,
            freq="D",
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
        )

        return ForecastResponse(
            status="success",
            metric="revenue_forecast",
            historical=forecast_result["forecast"]["historical"],
            forecast=forecast_result["forecast"]["forecast"],
            components=forecast_result["components"],
            metrics=forecast_result["metrics"],
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate forecast: {e}",
        )


@router.get("/snowflake/tables")
async def list_snowflake_tables() -> dict:
    """List available tables in Snowflake schema."""
    from app.snowflake import execute_query

    try:
        tables = await execute_query("SHOW TABLES")
        return {"status": "success", "tables": tables}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list tables: {e}",
        )


@router.get("/snowflake/describe/{table_name}")
async def describe_snowflake_table(table_name: str) -> dict:
    """Describe a Snowflake table structure."""
    from app.snowflake import execute_query

    try:
        columns = await execute_query(f"DESCRIBE TABLE {table_name}")
        sample = await execute_query(f"SELECT * FROM {table_name} LIMIT 5")
        return {"status": "success", "columns": columns, "sample_data": sample}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to describe table: {e}",
        )


@router.get("/snowflake/query")
async def run_snowflake_query(q: str = Query(..., description="SQL query to execute")) -> dict:
    """Run a custom Snowflake query (for testing)."""
    from app.snowflake import execute_query

    try:
        result = await execute_query(q)
        return {"status": "success", "rows": len(result), "data": result[:100]}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {e}",
        )


@router.post("/sync", response_model=SyncResponse)
async def sync_snowflake_data(
    session: AsyncSession = Depends(get_db),
) -> SyncResponse:
    """Sync 1099 vendor data from Snowflake IH_DATASCIENCE.HS_1099 to local PostgreSQL."""
    from app.snowflake import execute_query

    try:
        # Query vendor data with job metrics from IH_DATASCIENCE
        vendor_query = """
        SELECT
            V.ID AS VENDOR_ID,
            V.NAME AS VENDOR_NAME,
            V.PHONE,
            V.ZIP_CODES,
            V.SKILL_SETS,
            V.IS_ACTIVE,
            V.CREATED_AT,
            COALESCE(JM.TOTAL_JOBS, 0) AS TOTAL_JOBS,
            COALESCE(JM.COMPLETED_JOBS, 0) AS COMPLETED_JOBS,
            COALESCE(JM.CANCELLED_JOBS, 0) AS CANCELLED_JOBS,
            COALESCE(JM.TOTAL_REVENUE, 0) AS TOTAL_REVENUE,
            COALESCE(JM.TOTAL_PROFIT, 0) AS TOTAL_PROFIT,
            COALESCE(JM.AVG_PROFIT_PER_JOB, 0) AS AVG_PROFIT_PER_JOB,
            CASE
                WHEN JM.TOTAL_JOBS > 0 THEN ROUND(JM.COMPLETED_JOBS * 100.0 / JM.TOTAL_JOBS, 2)
                ELSE 0
            END AS COMPLETION_RATE
        FROM IH_DATASCIENCE.PUBLIC.VENDORS V
        LEFT JOIN (
            SELECT
                VENDOR_ID,
                COUNT(*) AS TOTAL_JOBS,
                SUM(CASE WHEN COMPL_OWNER = 'Completed by 1099' THEN 1 ELSE 0 END) AS COMPLETED_JOBS,
                SUM(CASE WHEN SO_STS_DESC = 'CA - Cancelled' THEN 1 ELSE 0 END) AS CANCELLED_JOBS,
                SUM(COALESCE(TOT_REV, 0)) AS TOTAL_REVENUE,
                SUM(COALESCE(ACTUAL_PPT, 0)) AS TOTAL_PROFIT,
                AVG(COALESCE(ACTUAL_PPT, 0)) AS AVG_PROFIT_PER_JOB
            FROM IH_DATASCIENCE.HS_1099.ASSIGNED_JOBS_FINANCIALS
            GROUP BY VENDOR_ID
        ) JM ON V.ID = JM.VENDOR_ID
        WHERE V.IS_ACTIVE = TRUE
        ORDER BY JM.TOTAL_JOBS DESC NULLS LAST
        """

        vendors_data = await execute_query(vendor_query)

        if not vendors_data:
            return SyncResponse(
                status="warning",
                message="No vendor data returned from Snowflake",
                records_synced=0,
            )

        # Parse ZIP codes to determine region (use first ZIP prefix as region)
        def get_region_from_zip(zip_codes_str: str) -> str:
            if not zip_codes_str:
                return "Unknown"
            # Parse {12345,67890} format
            zips = zip_codes_str.strip("{}").split(",")
            if zips and zips[0]:
                first_zip = zips[0].strip()
                # Map ZIP prefix to region
                prefix = first_zip[:3] if len(first_zip) >= 3 else first_zip
                # Rough US region mapping by ZIP prefix
                prefix_num = int(prefix) if prefix.isdigit() else 0
                if prefix_num < 100:
                    return "Northeast"
                elif prefix_num < 300:
                    return "Northeast"
                elif prefix_num < 400:
                    return "Southeast"
                elif prefix_num < 500:
                    return "Midwest"
                elif prefix_num < 600:
                    return "Midwest"
                elif prefix_num < 700:
                    return "South"
                elif prefix_num < 800:
                    return "South"
                elif prefix_num < 900:
                    return "Mountain"
                else:
                    return "West"
            return "Unknown"

        # Parse skills from string format
        def parse_skills(skills_str: str) -> list:
            if not skills_str:
                return []
            return [s.strip() for s in skills_str.strip("{}").split(",") if s.strip()]

        # Upsert vendors into PostgreSQL as Technicians
        records_synced = 0
        for vendor in vendors_data:
            vendor_id = str(vendor.get("VENDOR_ID", ""))
            if not vendor_id:
                continue

            region = get_region_from_zip(vendor.get("ZIP_CODES", ""))
            skills = parse_skills(vendor.get("SKILL_SETS", ""))

            # Calculate days since vendor creation
            created_at = vendor.get("CREATED_AT")
            tenure_days = None
            if created_at:
                from datetime import datetime
                try:
                    if isinstance(created_at, str):
                        created_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    else:
                        created_dt = created_at
                    tenure_days = (datetime.now() - created_dt.replace(tzinfo=None)).days
                except Exception:
                    pass

            # Determine status based on activity
            total_jobs = vendor.get("TOTAL_JOBS", 0) or 0
            is_active = vendor.get("IS_ACTIVE", True)
            if not is_active:
                status_val = "INACTIVE"
            elif total_jobs == 0:
                status_val = "NEW"
            else:
                status_val = "ACTIVE"

            stmt = insert(Technician).values(
                external_id=vendor_id,
                region=region,
                status=status_val,
                tenure_days=tenure_days,
                metrics={
                    "vendor_name": vendor.get("VENDOR_NAME"),
                    "phone": vendor.get("PHONE"),
                    "zip_codes": vendor.get("ZIP_CODES"),
                    "total_jobs": total_jobs,
                    "completed_jobs": vendor.get("COMPLETED_JOBS", 0) or 0,
                    "cancelled_jobs": vendor.get("CANCELLED_JOBS", 0) or 0,
                    "total_revenue": float(vendor.get("TOTAL_REVENUE", 0) or 0),
                    "total_profit": float(vendor.get("TOTAL_PROFIT", 0) or 0),
                    "avg_profit_per_job": float(vendor.get("AVG_PROFIT_PER_JOB", 0) or 0),
                    "completion_rate": float(vendor.get("COMPLETION_RATE", 0) or 0),
                },
                skills=skills,
                certifications=[],
            ).on_conflict_do_update(
                index_elements=["external_id"],
                set_={
                    "region": region,
                    "status": status_val,
                    "tenure_days": tenure_days,
                    "metrics": {
                        "vendor_name": vendor.get("VENDOR_NAME"),
                        "phone": vendor.get("PHONE"),
                        "zip_codes": vendor.get("ZIP_CODES"),
                        "total_jobs": total_jobs,
                        "completed_jobs": vendor.get("COMPLETED_JOBS", 0) or 0,
                        "cancelled_jobs": vendor.get("CANCELLED_JOBS", 0) or 0,
                        "total_revenue": float(vendor.get("TOTAL_REVENUE", 0) or 0),
                        "total_profit": float(vendor.get("TOTAL_PROFIT", 0) or 0),
                        "avg_profit_per_job": float(vendor.get("AVG_PROFIT_PER_JOB", 0) or 0),
                        "completion_rate": float(vendor.get("COMPLETION_RATE", 0) or 0),
                    },
                    "skills": skills,
                },
            )
            await session.execute(stmt)
            records_synced += 1

        await session.commit()

        return SyncResponse(
            status="success",
            message=f"Successfully synced {records_synced} vendors from Snowflake IH_DATASCIENCE.HS_1099",
            records_synced=records_synced,
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Snowflake private key not found: {e}",
        )
    except Exception as e:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to sync data from Snowflake: {e}",
        )


# =============================================================================
# Strategic Recommendations Endpoints (Cicero-inspired)
# =============================================================================


@router.get("/recommendations")
async def get_strategic_recommendations(
    state: str | None = Query(default=None, description="Filter by US state"),
    include_forecast: bool = Query(default=True, description="Include trend data in analysis"),
) -> RecommendationsResponse:
    """
    Get strategy-grounded recommendations for improving technician efficiency.

    Inspired by Meta's Cicero research, this endpoint:
    1. Analyzes current business metrics (game state)
    2. Models technician intent and likely behaviors
    3. Generates mutually beneficial recommendations
    4. Grounds all recommendations in actual data

    Reference: https://ai.meta.com/research/cicero/
    """
    from app.snowflake import execute_query

    try:
        # Build aggregate metrics query
        if state:
            # Get ZIP prefixes for the state
            state_zips = [prefix for prefix, st in ZIP_TO_STATE.items() if st == state.upper()]
            if not state_zips:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid state: {state}",
                )
            zip_conditions = " OR ".join([f"V.ZIP_CODES LIKE '%{prefix}%'" for prefix in state_zips[:20]])
            state_filter = f"AND ({zip_conditions})"
        else:
            state_filter = ""

        # Query aggregate metrics
        metrics_query = f"""
        SELECT
            COUNT(DISTINCT J.VENDOR_ID) AS total_vendors,
            COUNT(*) AS total_jobs,
            SUM(CASE WHEN F.COMPL_OWNER = 'Completed by 1099' THEN 1 ELSE 0 END) AS completed_jobs,
            SUM(CASE WHEN F.SO_STS_DESC = 'CA - Cancelled' THEN 1 ELSE 0 END) AS cancelled_jobs,
            SUM(COALESCE(F.TOT_REV, 0)) AS total_revenue,
            SUM(COALESCE(F.ACTUAL_PPT, 0)) AS total_profit,
            AVG(COALESCE(F.ACTUAL_PPT, 0)) AS avg_profit_per_job,
            CASE
                WHEN COUNT(*) > 0 THEN
                    ROUND(SUM(CASE WHEN F.COMPL_OWNER = 'Completed by 1099' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2)
                ELSE 0
            END AS completion_rate
        FROM IH_DATASCIENCE.HS_1099.JOBS_ASSIGNMENTS J
        LEFT JOIN IH_DATASCIENCE.HS_1099.ASSIGNED_JOBS_FINANCIALS F ON J.JOB_ID = F.JOB_ID
        LEFT JOIN IH_DATASCIENCE.PUBLIC.VENDORS V ON J.VENDOR_ID = V.ID
        WHERE J.ASSIGNED_AT >= DATEADD(day, -90, CURRENT_DATE())
        {state_filter}
        """

        result = await execute_query(metrics_query)

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No data available for recommendations",
            )

        row = result[0]
        aggregate_metrics = {
            "total_vendors": row.get("TOTAL_VENDORS", 0),
            "total_jobs": row.get("TOTAL_JOBS", 0),
            "completed_jobs": row.get("COMPLETED_JOBS", 0),
            "cancelled_jobs": row.get("CANCELLED_JOBS", 0),
            "total_revenue": float(row.get("TOTAL_REVENUE", 0) or 0),
            "total_profit": float(row.get("TOTAL_PROFIT", 0) or 0),
            "avg_profit_per_job": float(row.get("AVG_PROFIT_PER_JOB", 0) or 0),
            "completion_rate": float(row.get("COMPLETION_RATE", 0) or 0) / 100,  # Convert to decimal
            "state": state,
        }

        # Get trend data if requested
        trend_data = None
        if include_forecast:
            try:
                # Fetch time series for trend analysis
                if state:
                    state_zips_sample = state_zips[:5]
                    zip_cond = " OR ".join([f"V.ZIP_CODES LIKE '%{z}%'" for z in state_zips_sample])
                    trend_query = f"""
                    SELECT DATE(J.ASSIGNED_AT) AS ds, COUNT(*) AS y
                    FROM IH_DATASCIENCE.HS_1099.JOBS_ASSIGNMENTS J
                    JOIN IH_DATASCIENCE.PUBLIC.VENDORS V ON J.VENDOR_ID = V.ID
                    WHERE J.ASSIGNED_AT >= DATEADD(day, -90, CURRENT_DATE())
                      AND ({zip_cond})
                    GROUP BY ds ORDER BY ds
                    """
                else:
                    trend_query = """
                    SELECT DATE(ASSIGNED_AT) AS ds, COUNT(*) AS y
                    FROM IH_DATASCIENCE.HS_1099.JOBS_ASSIGNMENTS
                    WHERE ASSIGNED_AT >= DATEADD(day, -90, CURRENT_DATE())
                    GROUP BY ds ORDER BY ds
                    """

                trend_result = await execute_query(trend_query)
                if trend_result and len(trend_result) >= 7:
                    values = [float(r.get("Y", 0)) for r in trend_result]
                    dates = [str(r.get("DS"))[:10] for r in trend_result]
                    trend_data = {
                        "trend": {
                            "dates": dates,
                            "values": values,
                        }
                    }
            except Exception:
                pass  # Continue without trend data

        # Generate recommendations
        response = await recommendation_engine.generate_for_aggregate(
            aggregate_metrics=aggregate_metrics,
            trend_data=trend_data,
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate recommendations: {e}",
        )


@router.get("/recommendations/by-state/{state}")
async def get_state_recommendations(
    state: str,
) -> RecommendationsResponse:
    """Get recommendations specific to a US state."""
    return await get_strategic_recommendations(state=state, include_forecast=True)
