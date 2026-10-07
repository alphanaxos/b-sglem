"""
Unit and integration tests for Phase 6 Dashboard API Client and UI Helpers.

Covers:
- URL configuration and default fallbacks.
- Connection error handling without raw stack traces.
- FastAPI 422 validation error formatting.
- Valid response parsing for forecasting, dispatch, and risk analysis.
- Infeasible dispatch handling.
- Component data helpers and Plotly figure generation.
"""
import os
from unittest.mock import patch, MagicMock
import pytest
import requests

from dashboard.api_client import BSGLEMClient, APIClientError
from dashboard.components import (
    get_default_weather_24,
    get_default_demand_24,
    get_default_baseline_features,
    plot_historical_load,
    plot_forecast_24h,
    plot_dispatch_schedule,
    plot_risk_quantiles,
    plot_scenario_comparison,
    plot_fuel_shares_donut,
)


# ══════════════════════════════════════════════════════════════
# 1. API CLIENT CONFIGURATION & CONNECTION ERROR TESTS
# ══════════════════════════════════════════════════════════════

def test_api_client_url_configuration():
    """Verifies that APIClient uses BSGLEM_API_URL or defaults to localhost."""
    # Default URL
    with patch.dict(os.environ, {}, clear=True):
        client = BSGLEMClient()
        assert client.base_url == "http://127.0.0.1:8000"

    # Environment variable override
    with patch.dict(os.environ, {"BSGLEM_API_URL": "http://api.grid.internal:9000/"}):
        client = BSGLEMClient()
        assert client.base_url == "http://api.grid.internal:9000"

    # Explicit argument takes precedence
    client_arg = BSGLEMClient(base_url="http://custom-host:8080/")
    assert client_arg.base_url == "http://custom-host:8080"


def test_api_client_connection_failure():
    """Verifies that connection failures raise clean APIClientError."""
    client = BSGLEMClient(base_url="http://non-existent-domain-12345.local")

    with patch("requests.request", side_effect=requests.exceptions.ConnectionError("Connection refused")):
        with pytest.raises(APIClientError) as exc_info:
            client.get_health()

        assert "Cannot connect to backend" in exc_info.value.message
        assert exc_info.value.status_code is None


def test_api_client_timeout():
    """Verifies that request timeouts raise clear APIClientError."""
    client = BSGLEMClient(timeout=1.0)

    with patch("requests.request", side_effect=requests.exceptions.Timeout("Request timed out")):
        with pytest.raises(APIClientError) as exc_info:
            client.get_health()

        assert "timed out" in exc_info.value.message


def test_api_client_validation_error_formatting():
    """Verifies that FastAPI 422 validation errors are cleanly parsed and formatted."""
    client = BSGLEMClient()

    mock_resp = MagicMock()
    mock_resp.ok = False
    mock_resp.status_code = 422
    mock_resp.json.return_value = {
        "detail": [
            {"loc": ["body", "demand_mw"], "msg": "Demand profile must contain exactly 24 hourly values"},
            {"loc": ["body", "n_simulations"], "msg": "Input should be greater than or equal to 10"},
        ]
    }

    with patch("requests.request", return_value=mock_resp):
        with pytest.raises(APIClientError) as exc_info:
            client.solve_dispatch(demand_mw=[100.0] * 5)

        assert exc_info.value.status_code == 422
        assert "Validation error:" in exc_info.value.message
        assert "body -> demand_mw" in exc_info.value.message
        assert "body -> n_simulations" in exc_info.value.message


# ══════════════════════════════════════════════════════════════
# 2. SUCCESSFUL API METHOD PARSING
# ══════════════════════════════════════════════════════════════

def test_api_client_health_and_latest():
    """Verifies parsing of health and latest telemetry endpoints."""
    client = BSGLEMClient()

    # Health
    mock_health = MagicMock()
    mock_health.ok = True
    mock_health.json.return_value = {"status": "healthy", "models_loaded": True, "data_source_mode": "historical"}

    with patch("requests.request", return_value=mock_health):
        h = client.get_health()
        assert h["status"] == "healthy"
        assert h["models_loaded"] is True

    # Latest
    mock_latest = MagicMock()
    mock_latest.ok = True
    mock_latest.json.return_value = {"load_mw": 2150.0, "timestamp": "2024-12-31T23:00:00", "is_simulated": False}

    with patch("requests.request", return_value=mock_latest):
        lat = client.get_latest_data()
        assert lat["load_mw"] == 2150.0
        assert lat["is_simulated"] is False


def test_api_client_forecast_24h_parsing():
    """Verifies parsing of 24h forecast response."""
    client = BSGLEMClient()

    mock_resp = MagicMock()
    mock_resp.ok = True
    mock_resp.json.return_value = {
        "forecast_origin": "2024-12-31T23:00:00",
        "predictions": [
            {"hour": i, "predicted_load_mw": 2000.0 + i * 10, "spinning_reserve_mw": 400.0}
            for i in range(24)
        ],
        "interval_confidence_level": 0.90,
    }

    with patch("requests.request", return_value=mock_resp):
        res = client.forecast_24h(weather_forecasts=[{"temp": 25.0}] * 24)
        assert len(res["predictions"]) == 24
        assert res["predictions"][0]["predicted_load_mw"] == 2000.0


def test_api_client_dispatch_infeasible_status():
    """Verifies that infeasible dispatch outcomes are correctly passed through."""
    client = BSGLEMClient()

    mock_resp = MagicMock()
    mock_resp.ok = True
    mock_resp.json.return_value = {
        "optimization_status": "infeasible (solver produced empty dispatch)",
        "is_feasible": False,
        "is_optimal": False,
        "total_cost": 0.0,
    }

    with patch("requests.request", return_value=mock_resp):
        res = client.solve_dispatch(demand_mw=[4000.0] * 24, include_load_shedding=False)
        assert res["is_feasible"] is False
        assert "infeasible" in res["optimization_status"]


def test_api_client_risk_metric_parsing():
    """Verifies parsing of Monte Carlo risk metrics."""
    client = BSGLEMClient()

    mock_resp = MagicMock()
    mock_resp.ok = True
    mock_resp.json.return_value = {
        "n_simulations": 100,
        "dispatch_engine": "merit_order",
        "risk_metrics": {
            "LOLP": 0.02,
            "EENS_mwh": 12.5,
            "reserve_sufficiency": 0.98,
            "mean_cost": 172000.0,
        },
        "quantile_summaries": [
            {"metric": "Demand peak (MW)", "p05": 2100.0, "p50": 2250.0, "p95": 2380.0}
        ],
    }

    with patch("requests.request", return_value=mock_resp):
        res = client.run_risk_analysis(demand_mw=[2000.0] * 24, n_simulations=100)
        assert res["risk_metrics"]["LOLP"] == 0.02
        assert res["risk_metrics"]["EENS_mwh"] == 12.5
        assert len(res["quantile_summaries"]) == 1


# ══════════════════════════════════════════════════════════════
# 3. COMPONENT DATA HELPERS & PLOTLY CHART TESTS
# ══════════════════════════════════════════════════════════════

def test_default_weather_and_demand_generators():
    """Verifies helper generators produce 24 hourly rows with valid numbers."""
    weather = get_default_weather_24()
    assert len(weather) == 24
    for w in weather:
        assert 0 <= w["hour"] <= 23
        assert 10.0 <= w["temperature"] <= 40.0
        assert 0.0 <= w["solar_irradiance"] <= 1200.0

    demand = get_default_demand_24()
    assert len(demand) == 24
    for d in demand:
        assert d > 0.0

    feats = get_default_baseline_features()
    assert len(feats) == 28


def test_plotly_chart_builders():
    """Verifies that all dashboard Plotly chart builders generate valid Figure objects."""
    # 1. Historical load chart
    records = [
        {"timestamp": f"2024-12-31T{h:02d}:00:00", "load_mw": 2000.0 + h * 20}
        for h in range(24)
    ]
    fig_hist = plot_historical_load(records)
    assert fig_hist is not None
    assert len(fig_hist.data) >= 1

    # 2. 24h forecast chart
    preds = [
        {
            "hour": h,
            "predicted_load_mw": 2100.0 + h * 15,
            "spinning_reserve_mw": 420.0,
            "lower_bound_mw": 1950.0,
            "upper_bound_mw": 2250.0,
        }
        for h in range(24)
    ]
    fig_fc = plot_forecast_24h(preds)
    assert fig_fc is not None
    assert len(fig_fc.data) >= 3

    # 3. Scenario comparison chart
    sc_results = [
        {"scenario_name": "Normal", "predicted_load_mw": 2000.0, "delta_mw": 0.0, "pct_change": 0.0},
        {"scenario_name": "Heatwave", "predicted_load_mw": 2150.0, "delta_mw": 150.0, "pct_change": 7.5},
    ]
    fig_sc = plot_scenario_comparison(sc_results, baseline_mw=2000.0)
    assert fig_sc is not None

    # 4. Dispatch schedule chart
    sched = [
        {"hour": h, "coal_mw": 1500.0, "gas_mw": 500.0, "solar_mw": 100.0, "unserved_mw": 0.0, "demand_mw": 2100.0}
        for h in range(24)
    ]
    fig_sched = plot_dispatch_schedule(sched)
    assert fig_sched is not None

    # 5. Fuel shares donut
    fig_donut = plot_fuel_shares_donut(coal_mwh=36000.0, gas_mwh=8000.0, solar_mwh=2400.0)
    assert fig_donut is not None

    # 6. Risk quantiles chart
    q_data = [
        {"metric": "Demand peak (MW)", "p05": 2100.0, "p50": 2250.0, "p95": 2380.0},
        {"metric": "Cost (₹)", "p05": 160000.0, "p50": 172000.0, "p95": 185000.0},
    ]
    fig_q = plot_risk_quantiles(q_data)
    assert fig_q is not None
