"""End-to-end tests for the dashboard API.

Tests that dashboard endpoints return valid chart data.
"""

from __future__ import annotations

import pytest


class TestDashboardOverview:
    """Test dashboard overview endpoint."""

    @pytest.mark.asyncio
    async def test_overview_returns_metrics(self, test_client):
        """Test that overview endpoint returns expected metrics structure."""
        response = await test_client.get("/api/v1/dashboard/overview")

        assert response.status_code == 200
        data = response.json()

        # Check expected structure
        assert "total_technicians" in data
        assert "active_technicians" in data
        assert "at_risk_count" in data
        assert "average_tenure_days" in data

        # Values should be non-negative
        assert data["total_technicians"] >= 0
        assert data["active_technicians"] >= 0
        assert data["at_risk_count"] >= 0

    @pytest.mark.asyncio
    async def test_overview_with_region_filter(self, test_client):
        """Test overview endpoint with region filter."""
        response = await test_client.get("/api/v1/dashboard/overview", params={"region": "US-WEST"})

        # Should succeed even if no data
        assert response.status_code in [200, 404]


class TestDashboardForecast:
    """Test dashboard forecast endpoint."""

    @pytest.mark.asyncio
    async def test_forecast_returns_chart_data(self, test_client):
        """Test that forecast endpoint returns Chart.js compatible data."""
        response = await test_client.get("/api/v1/dashboard/forecast")

        # May return 404 if no models trained
        if response.status_code == 404:
            pytest.skip("No forecast models available")

        assert response.status_code == 200
        data = response.json()

        # Check Chart.js compatible structure
        assert "dates" in data or "labels" in data
        assert "yhat" in data or "data" in data

        # If data present, arrays should have same length
        if "dates" in data and "yhat" in data:
            assert len(data["dates"]) == len(data["yhat"])

    @pytest.mark.asyncio
    async def test_forecast_with_region(self, test_client):
        """Test forecast endpoint with region parameter."""
        response = await test_client.get("/api/v1/dashboard/forecast", params={"region": "US-WEST"})

        # Should not error
        assert response.status_code in [200, 404]


class TestDashboardFeatureImportance:
    """Test dashboard feature importance endpoint."""

    @pytest.mark.asyncio
    async def test_feature_importance_returns_chart_data(self, test_client):
        """Test that feature importance returns Chart.js bar chart data."""
        response = await test_client.get("/api/v1/dashboard/feature-importance")

        # May return 404 if no models trained
        if response.status_code == 404:
            pytest.skip("No trained models available")

        assert response.status_code == 200
        data = response.json()

        # Check for bar chart structure
        assert "features" in data or "labels" in data
        assert "importances" in data or "values" in data or "data" in data

        # Features and importances should have same length
        features = data.get("features") or data.get("labels") or []
        importances = data.get("importances") or data.get("values") or data.get("data") or []

        if features and importances:
            assert len(features) == len(importances)


class TestDashboardRegions:
    """Test dashboard regional summary endpoint."""

    @pytest.mark.asyncio
    async def test_regions_returns_summary(self, test_client):
        """Test that regions endpoint returns regional summary."""
        response = await test_client.get("/api/v1/dashboard/regions")

        assert response.status_code == 200
        data = response.json()

        # Should return list of regions
        assert isinstance(data, list)

        # Each region should have expected fields
        for region in data:
            assert "region" in region or "name" in region
            # May have additional metrics

    @pytest.mark.asyncio
    async def test_regions_include_risk_metrics(self, test_client):
        """Test that regional data includes risk information."""
        response = await test_client.get("/api/v1/dashboard/regions")

        assert response.status_code == 200
        data = response.json()

        # If regions have risk data, verify structure
        for region in data:
            if "risk_score" in region or "churn_rate" in region:
                risk_value = region.get("risk_score") or region.get("churn_rate")
                if risk_value is not None:
                    assert isinstance(risk_value, (int, float))


class TestDashboardTechnicians:
    """Test dashboard technicians list endpoint."""

    @pytest.mark.asyncio
    async def test_technicians_returns_paginated_list(self, test_client):
        """Test that technicians endpoint returns paginated results."""
        response = await test_client.get("/api/v1/dashboard/technicians")

        assert response.status_code == 200
        data = response.json()

        # Should have pagination structure
        assert "items" in data or "technicians" in data or isinstance(data, list)

        # Get list
        items = data.get("items") or data.get("technicians") or data

        # If items present, check structure
        if items:
            for tech in items[:5]:  # Check first 5
                # Should have identifier
                assert "id" in tech or "technician_id" in tech or "external_id" in tech

    @pytest.mark.asyncio
    async def test_technicians_pagination(self, test_client):
        """Test technicians endpoint pagination parameters."""
        response = await test_client.get(
            "/api/v1/dashboard/technicians", params={"page": 1, "page_size": 10}
        )

        assert response.status_code == 200

        # Test with different page size
        response2 = await test_client.get(
            "/api/v1/dashboard/technicians", params={"page": 1, "page_size": 5}
        )

        assert response2.status_code == 200

    @pytest.mark.asyncio
    async def test_technicians_filter_by_region(self, test_client):
        """Test filtering technicians by region."""
        response = await test_client.get(
            "/api/v1/dashboard/technicians", params={"region": "US-WEST"}
        )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_technicians_filter_by_risk_level(self, test_client):
        """Test filtering technicians by risk level."""
        response = await test_client.get(
            "/api/v1/dashboard/technicians", params={"risk_level": "HIGH"}
        )

        assert response.status_code == 200


class TestDashboardChartJsCompatibility:
    """Test that dashboard endpoints return Chart.js compatible data."""

    @pytest.mark.asyncio
    async def test_forecast_chartjs_format(self, test_client):
        """Test forecast data is Chart.js line chart compatible."""
        response = await test_client.get("/api/v1/dashboard/forecast")

        if response.status_code != 200:
            pytest.skip("No forecast data available")

        data = response.json()

        # Chart.js line chart needs x-axis labels and datasets
        has_labels = "dates" in data or "labels" in data
        has_data = "yhat" in data or "data" in data or "datasets" in data

        assert has_labels or has_data

    @pytest.mark.asyncio
    async def test_feature_importance_chartjs_format(self, test_client):
        """Test feature importance data is Chart.js bar chart compatible."""
        response = await test_client.get("/api/v1/dashboard/feature-importance")

        if response.status_code != 200:
            pytest.skip("No model data available")

        data = response.json()

        # Chart.js bar chart needs labels and values
        has_labels = "features" in data or "labels" in data
        has_values = "importances" in data or "values" in data or "data" in data

        assert has_labels or has_values


class TestDashboardErrorHandling:
    """Test dashboard error handling."""

    @pytest.mark.asyncio
    async def test_invalid_region_handled(self, test_client):
        """Test that invalid region parameter is handled gracefully."""
        response = await test_client.get(
            "/api/v1/dashboard/overview", params={"region": "INVALID-REGION"}
        )

        # Should either return empty data or 404, not 500
        assert response.status_code in [200, 404]

    @pytest.mark.asyncio
    async def test_invalid_pagination_handled(self, test_client):
        """Test that invalid pagination parameters are handled."""
        response = await test_client.get(
            "/api/v1/dashboard/technicians",
            params={"page": -1, "page_size": 10000},
        )

        # Should return 422 for validation error or handle gracefully
        assert response.status_code in [200, 422]
