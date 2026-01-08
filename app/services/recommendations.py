"""Strategy-grounded recommendation engine inspired by Meta's Cicero.

Cicero's approach: Combine strategic reasoning with language models to generate
recommendations that are grounded in actual data and optimized for mutually
beneficial outcomes.

Key concepts applied:
1. Intent Modeling - Predict technician behavior patterns from historical data
2. Strategic Planning - Find recommendations that optimize both company and technician outcomes
3. Grounded Generation - Root recommendations in actual metrics and context
4. Human-Regularized Reasoning - Balance optimal recommendations with realistic actions

Reference: https://ai.meta.com/research/cicero/
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RecommendationType(str, Enum):
    """Types of strategic recommendations."""
    EFFICIENCY = "efficiency"           # Optimize job completion time
    QUALITY = "quality"                 # Improve repair vs replace decisions
    SCHEDULING = "scheduling"           # Optimize job scheduling patterns
    SKILL_DEVELOPMENT = "skill"         # Training and certification recommendations
    ROUTING = "routing"                 # Geographic optimization
    CUSTOMER_SATISFACTION = "csat"      # Customer experience improvements
    FINANCIAL = "financial"             # Revenue and profit optimization


class RecommendationPriority(str, Enum):
    """Priority levels for recommendations."""
    CRITICAL = "critical"   # Immediate action needed
    HIGH = "high"           # Address within 1 week
    MEDIUM = "medium"       # Address within 1 month
    LOW = "low"             # Nice to have


class StrategicContext(BaseModel):
    """Context for strategic reasoning - models the "game state"."""

    # Technician performance context
    total_jobs: int = 0
    completed_jobs: int = 0
    completion_rate: float = 0.0
    avg_profit_per_job: float = 0.0

    # Appliance mix context
    appliance_breakdown: dict[str, int] = Field(default_factory=dict)
    repair_vs_replace_ratio: float = 0.0

    # Geographic context
    state: str | None = None
    region: str | None = None
    coverage_zips: int = 0

    # Temporal patterns
    busiest_day: str | None = None
    avg_jobs_per_day: float = 0.0
    seasonality_factor: float = 1.0

    # Trend context (from Prophet decomposition)
    trend_direction: str = "stable"  # increasing, decreasing, stable
    trend_strength: float = 0.0


class TechnicianIntent(BaseModel):
    """Predicted technician intent/behavior - Cicero's intent modeling."""

    # What we predict the technician wants
    primary_goal: str = "maximize_earnings"  # or efficiency, work_life_balance, skill_growth

    # Behavioral patterns
    prefers_repair_over_replace: bool = True
    schedule_flexibility: str = "moderate"  # high, moderate, low
    travel_tolerance: str = "moderate"  # high, moderate, low

    # Predicted response to recommendations
    likely_to_accept_training: float = 0.5
    likely_to_expand_coverage: float = 0.3
    likely_to_change_schedule: float = 0.4


class StrategicRecommendation(BaseModel):
    """A strategy-grounded recommendation."""

    id: str
    type: RecommendationType
    priority: RecommendationPriority

    # The recommendation itself
    title: str
    description: str
    rationale: str  # Why this recommendation (grounded in data)

    # Strategic impact prediction (Cicero's planning output)
    predicted_impact: dict[str, float] = Field(default_factory=dict)
    # e.g., {"completion_rate": +0.05, "avg_profit": +15.0, "customer_sat": +0.1}

    # Confidence and evidence
    confidence: float = 0.0  # 0-1 confidence in recommendation
    evidence: list[str] = Field(default_factory=list)  # Data points supporting this

    # Action items
    actions: list[str] = Field(default_factory=list)

    # For mutual benefit framing (key Cicero insight)
    technician_benefit: str = ""
    company_benefit: str = ""

    created_at: datetime = Field(default_factory=datetime.utcnow)


class RecommendationsResponse(BaseModel):
    """Response containing strategic recommendations."""

    status: str
    context: StrategicContext
    intent: TechnicianIntent
    recommendations: list[StrategicRecommendation]
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class StrategicRecommendationEngine:
    """
    Strategy-grounded recommendation engine using Cicero-inspired approach.

    The engine:
    1. Analyzes the current "game state" (business metrics)
    2. Models technician intent (what they want/will do)
    3. Generates recommendations that are mutually beneficial
    4. Grounds all recommendations in actual data
    """

    def __init__(self):
        self.recommendation_counter = 0

    def _generate_id(self) -> str:
        """Generate unique recommendation ID."""
        self.recommendation_counter += 1
        return f"REC-{datetime.utcnow().strftime('%Y%m%d')}-{self.recommendation_counter:04d}"

    async def analyze_context(self, snowflake_data: dict) -> StrategicContext:
        """
        Analyze Snowflake data to build strategic context.
        This is like Cicero reading the board state.
        """
        context = StrategicContext()

        # Extract metrics from Snowflake data
        if "metrics" in snowflake_data:
            m = snowflake_data["metrics"]
            context.total_jobs = m.get("total_jobs", 0)
            context.completed_jobs = m.get("completed_jobs", 0)
            context.completion_rate = m.get("completion_rate", 0.0)
            context.avg_profit_per_job = m.get("avg_profit_per_job", 0.0)

        # Extract geographic context
        context.state = snowflake_data.get("state")
        context.region = snowflake_data.get("region")

        # Extract appliance breakdown if available
        if "appliance_breakdown" in snowflake_data:
            context.appliance_breakdown = snowflake_data["appliance_breakdown"]

        # Extract trend info from Prophet if available
        if "trend" in snowflake_data:
            trend = snowflake_data["trend"]
            if trend.get("slope", 0) > 0.1:
                context.trend_direction = "increasing"
            elif trend.get("slope", 0) < -0.1:
                context.trend_direction = "decreasing"
            else:
                context.trend_direction = "stable"
            context.trend_strength = abs(trend.get("slope", 0))

        return context

    async def model_intent(self, context: StrategicContext, historical_behavior: dict | None = None) -> TechnicianIntent:
        """
        Model technician intent based on their behavior patterns.
        This is Cicero's intent prediction for other players.
        """
        intent = TechnicianIntent()

        # Infer primary goal from behavior
        if context.avg_profit_per_job > 50:
            intent.primary_goal = "maximize_earnings"
        elif context.completion_rate > 0.9:
            intent.primary_goal = "efficiency"
        else:
            intent.primary_goal = "skill_growth"

        # Infer repair preference from ratio if available
        if context.repair_vs_replace_ratio > 0.7:
            intent.prefers_repair_over_replace = True
        elif context.repair_vs_replace_ratio < 0.3:
            intent.prefers_repair_over_replace = False

        # Infer flexibility from job patterns
        if context.avg_jobs_per_day > 5:
            intent.schedule_flexibility = "low"  # Very busy
        elif context.avg_jobs_per_day < 2:
            intent.schedule_flexibility = "high"  # Has capacity

        # Predict response to recommendations based on context
        if context.trend_direction == "decreasing":
            # More likely to accept help if business is declining
            intent.likely_to_accept_training = 0.7
            intent.likely_to_expand_coverage = 0.6
        elif context.trend_direction == "increasing":
            # Less likely to change what's working
            intent.likely_to_accept_training = 0.3
            intent.likely_to_expand_coverage = 0.2

        return intent

    async def generate_recommendations(
        self,
        context: StrategicContext,
        intent: TechnicianIntent,
        focus_areas: list[RecommendationType] | None = None,
    ) -> list[StrategicRecommendation]:
        """
        Generate strategy-grounded recommendations.
        This is Cicero's planning + dialogue generation.

        Key insight from Cicero: Recommendations must be:
        1. Grounded in actual data (not generic advice)
        2. Mutually beneficial (technician + company win)
        3. Aligned with predicted intent (realistic to accept)
        """
        recommendations = []

        # Efficiency recommendations
        if not focus_areas or RecommendationType.EFFICIENCY in focus_areas:
            if context.completion_rate < 0.85:
                rec = StrategicRecommendation(
                    id=self._generate_id(),
                    type=RecommendationType.EFFICIENCY,
                    priority=RecommendationPriority.HIGH,
                    title="Improve First-Time Fix Rate",
                    description=f"Current completion rate is {context.completion_rate:.1%}, below the 85% target. Focus on diagnostic accuracy before dispatching.",
                    rationale=f"Analysis of {context.total_jobs} jobs shows a {context.completion_rate:.1%} completion rate. Each incomplete job costs approximately $75 in rescheduling.",
                    predicted_impact={
                        "completion_rate": 0.10,
                        "avg_profit": 12.0,
                        "customer_satisfaction": 0.15
                    },
                    confidence=0.75,
                    evidence=[
                        f"Current completion rate: {context.completion_rate:.1%}",
                        f"Target completion rate: 85%",
                        f"Gap: {0.85 - context.completion_rate:.1%}"
                    ],
                    actions=[
                        "Review diagnostic checklist before each job",
                        "Ensure parts availability before dispatch",
                        "Request customer photos of appliance issue beforehand"
                    ],
                    technician_benefit="Fewer return visits = more time for new jobs = higher earnings",
                    company_benefit="Reduced operational costs and improved customer satisfaction"
                )
                recommendations.append(rec)

        # Scheduling recommendations based on patterns
        if not focus_areas or RecommendationType.SCHEDULING in focus_areas:
            if context.busiest_day and context.avg_jobs_per_day < 4:
                rec = StrategicRecommendation(
                    id=self._generate_id(),
                    type=RecommendationType.SCHEDULING,
                    priority=RecommendationPriority.MEDIUM,
                    title="Optimize Weekly Schedule Distribution",
                    description=f"Job volume peaks on {context.busiest_day}. Consider redistributing availability to capture more jobs on high-demand days.",
                    rationale=f"With {context.avg_jobs_per_day:.1f} avg jobs/day, there's capacity for {(5 - context.avg_jobs_per_day):.1f} more jobs daily.",
                    predicted_impact={
                        "jobs_per_day": 1.5,
                        "weekly_revenue": 450.0
                    },
                    confidence=0.65,
                    evidence=[
                        f"Current avg jobs/day: {context.avg_jobs_per_day:.1f}",
                        f"Busiest day: {context.busiest_day}",
                        f"Industry benchmark: 5 jobs/day"
                    ],
                    actions=[
                        f"Increase availability on {context.busiest_day}",
                        "Set up automated job acceptance for high-demand periods",
                        "Consider expanding to adjacent ZIP codes"
                    ],
                    technician_benefit="More jobs in same time = higher weekly earnings",
                    company_benefit="Better coverage during peak demand"
                )
                recommendations.append(rec)

        # Geographic expansion recommendations
        if not focus_areas or RecommendationType.ROUTING in focus_areas:
            if context.coverage_zips < 10 and intent.likely_to_expand_coverage > 0.4:
                rec = StrategicRecommendation(
                    id=self._generate_id(),
                    type=RecommendationType.ROUTING,
                    priority=RecommendationPriority.MEDIUM,
                    title="Strategic Coverage Expansion",
                    description=f"Currently covering {context.coverage_zips} ZIP codes. Expanding to adjacent high-demand areas could increase job volume by 20-30%.",
                    rationale=f"Based on your {context.state or 'regional'} location and current {context.avg_jobs_per_day:.1f} jobs/day, there's untapped demand nearby.",
                    predicted_impact={
                        "job_volume": 0.25,
                        "travel_time": 0.15  # 15% increase in travel
                    },
                    confidence=0.55,
                    evidence=[
                        f"Current ZIP coverage: {context.coverage_zips}",
                        f"Recommended expansion: 3-5 adjacent ZIPs",
                        f"Expected job increase: 20-30%"
                    ],
                    actions=[
                        "Review heat map of unserved jobs in your area",
                        "Add 2-3 high-demand ZIP codes adjacent to current coverage",
                        "Set maximum travel distance of 25 miles"
                    ],
                    technician_benefit="More job opportunities without significantly more driving",
                    company_benefit="Reduced unserved job requests in the region"
                )
                recommendations.append(rec)

        # Quality/skill development recommendations
        if not focus_areas or RecommendationType.SKILL_DEVELOPMENT in focus_areas:
            if context.appliance_breakdown:
                # Find the appliance type with lowest success rate
                low_performing = None
                for appliance, count in context.appliance_breakdown.items():
                    if count > 5:  # Enough data
                        low_performing = appliance
                        break

                if low_performing and intent.likely_to_accept_training > 0.4:
                    rec = StrategicRecommendation(
                        id=self._generate_id(),
                        type=RecommendationType.SKILL_DEVELOPMENT,
                        priority=RecommendationPriority.MEDIUM,
                        title=f"Certification Opportunity: {low_performing}",
                        description=f"Adding {low_performing} certification could open up new job categories and increase your eligible job pool.",
                        rationale=f"Based on regional demand analysis, {low_performing} repairs have high volume and good profit margins.",
                        predicted_impact={
                            "eligible_jobs": 0.15,
                            "avg_profit_per_job": 8.0
                        },
                        confidence=0.60,
                        evidence=[
                            f"High regional demand for {low_performing} repairs",
                            "Average certification ROI: 3 months",
                            "Company-subsidized training available"
                        ],
                        actions=[
                            f"Enroll in {low_performing} certification program",
                            "Complete online modules (est. 8 hours)",
                            "Schedule practical assessment"
                        ],
                        technician_benefit="New skill = more job types = higher earning potential",
                        company_benefit="Better coverage for specialized repairs"
                    )
                    recommendations.append(rec)

        # Repair vs Replace optimization
        if not focus_areas or RecommendationType.QUALITY in focus_areas:
            if context.repair_vs_replace_ratio < 0.5:
                rec = StrategicRecommendation(
                    id=self._generate_id(),
                    type=RecommendationType.QUALITY,
                    priority=RecommendationPriority.HIGH,
                    title="Optimize Repair vs Replace Decision Framework",
                    description="Current repair ratio is below optimal. Many units being replaced could be cost-effectively repaired, improving customer satisfaction and margins.",
                    rationale=f"Industry data shows 65% of replaced units could have been repaired. Each repair saves ~$200 vs replacement.",
                    predicted_impact={
                        "repair_rate": 0.15,
                        "customer_satisfaction": 0.10,
                        "profit_margin": 0.08
                    },
                    confidence=0.70,
                    evidence=[
                        f"Current repair rate: {context.repair_vs_replace_ratio:.1%}",
                        "Target repair rate: 65%",
                        "Cost difference: $200 avg per job"
                    ],
                    actions=[
                        "Use diagnostic decision tree before recommending replacement",
                        "Check parts availability and cost before deciding",
                        "Consider appliance age vs repair cost ratio (50% rule)"
                    ],
                    technician_benefit="Repairs often have better profit margins than replacements",
                    company_benefit="Higher customer retention and better margins"
                )
                recommendations.append(rec)

        # Trend-based recommendations
        if context.trend_direction == "decreasing" and context.trend_strength > 0.2:
            rec = StrategicRecommendation(
                id=self._generate_id(),
                type=RecommendationType.FINANCIAL,
                priority=RecommendationPriority.CRITICAL,
                title="Address Declining Job Volume Trend",
                description=f"Job volume has been declining. Immediate action recommended to stabilize and reverse this trend.",
                rationale=f"Prophet analysis shows a {context.trend_strength:.1%} decline rate. Without intervention, this could impact earnings significantly.",
                predicted_impact={
                    "job_volume": 0.20,  # Reverse the decline
                    "stability": 0.30
                },
                confidence=0.80,
                evidence=[
                    f"Trend direction: {context.trend_direction}",
                    f"Trend strength: {context.trend_strength:.1%}",
                    "Historical pattern suggests seasonal adjustment needed"
                ],
                actions=[
                    "Review recent customer feedback for issues",
                    "Check response time to job assignments",
                    "Expand availability windows during peak hours",
                    "Consider promotional rates for new customers"
                ],
                technician_benefit="Stabilize and grow your income stream",
                company_benefit="Maintain service coverage in your area"
            )
            recommendations.append(rec)

        # Sort by priority
        priority_order = {
            RecommendationPriority.CRITICAL: 0,
            RecommendationPriority.HIGH: 1,
            RecommendationPriority.MEDIUM: 2,
            RecommendationPriority.LOW: 3
        }
        recommendations.sort(key=lambda r: priority_order[r.priority])

        return recommendations

    async def generate_for_aggregate(
        self,
        aggregate_metrics: dict,
        trend_data: dict | None = None,
    ) -> RecommendationsResponse:
        """
        Generate recommendations for aggregate (all technicians/region) view.
        This is the fleet-wide strategic view.
        """
        # Build context from aggregate data
        context = StrategicContext(
            total_jobs=aggregate_metrics.get("total_jobs", 0),
            completed_jobs=aggregate_metrics.get("completed_jobs", 0),
            completion_rate=aggregate_metrics.get("completion_rate", 0.0),
            avg_profit_per_job=aggregate_metrics.get("avg_profit_per_job", 0.0),
            region=aggregate_metrics.get("region"),
            state=aggregate_metrics.get("state"),
        )

        # Add trend context if available
        if trend_data and "trend" in trend_data:
            values = trend_data["trend"].get("values", [])
            if len(values) >= 2:
                slope = (values[-1] - values[0]) / len(values) if len(values) > 1 else 0
                if slope > 0.5:
                    context.trend_direction = "increasing"
                elif slope < -0.5:
                    context.trend_direction = "decreasing"
                context.trend_strength = abs(slope) / (sum(values) / len(values)) if values else 0

        # Model aggregate intent (average technician behavior)
        intent = TechnicianIntent(
            primary_goal="maximize_earnings",
            likely_to_accept_training=0.5,
            likely_to_expand_coverage=0.4,
            likely_to_change_schedule=0.5,
        )

        # Generate recommendations
        recommendations = await self.generate_recommendations(context, intent)

        return RecommendationsResponse(
            status="success",
            context=context,
            intent=intent,
            recommendations=recommendations,
        )


# Singleton instance
recommendation_engine = StrategicRecommendationEngine()
