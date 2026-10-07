"""
Comprehensive test suite for Phase 5 Energy Management and Risk Analysis APIs.

Covers:
- POST /energy/dispatch: Valid solves, horizon handling, constraint enforcement,
  infeasibility handling, and Phase 4 forecast integration.
- POST /energy/risk: Monte Carlo uncertainty simulation, LOLP/EENS reliability metrics,
  seed reproducibility, simulation-count bounds, and residual immutability.
- Safe serialization and error boundaries.
"""
from typing import List
import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def valid_demand_24() -> List[float]:
    """
    Realistic 24-hour Bengaluru electrical demand profile in MW.
    Peak around 2,200 MW, base around 1,500 MW.
    """
    return [
        1650.0, 1580.0, 1520.0, 1490.0, 1510.0, 1600.0,
        1780.0, 1950.0, 2100.0, 2180.0, 2220.0, 2200.0,
        2150.0, 2100.0, 2080.0, 2050.0, 2100.0, 2240.0,
        2290.0, 2250.0, 2150.0, 2000.0, 1850.0, 1720.0,
    ]


# ══════════════════════════════════════════════════════════════
# 1. PYPSA ECONOMIC DISPATCH TESTS
# ══════════════════════════════════════════════════════════════

def test_dispatch_valid_demand_success(client: TestClient, valid_demand_24: List[float]):
    """
    Verifies that POST /energy/dispatch solves 24-hour economic dispatch,
    producing optimal schedules, positive costs in INR, and valid fuel shares.
    """
    payload = {
        "demand_mw": valid_demand_24,
        "include_load_shedding": False,
    }
    response = client.post("/energy/dispatch", json=payload)
    assert response.status_code == 200, response.text

    data = response.json()
    assert data["is_optimal"] is True
    assert data["is_feasible"] is True
    assert "optimal" in data["optimization_status"].lower()
    assert data["currency"] == "INR (₹)"
    assert data["total_cost"] > 0.0

    # Hourly schedule
    schedule = data["hourly_schedule"]
    assert len(schedule) == 24
    for idx, item in enumerate(schedule):
        assert item["hour"] == idx
        assert item["demand_mw"] == valid_demand_24[idx]
        assert item["coal_mw"] >= 0.0
        assert item["gas_mw"] >= 0.0
        assert item["solar_mw"] >= 0.0
        assert item["unserved_mw"] == 0.0
        assert item["hourly_cost"] >= 0.0

    # Total supply balances total demand
    total_supply = data["coal_total_mwh"] + data["gas_total_mwh"] + data["solar_total_mwh"]
    assert abs(total_supply - data["total_demand_mwh"]) < 1.0

    # Fuel shares
    total_share = data["coal_share_pct"] + data["gas_share_pct"] + data["solar_share_pct"]
    assert abs(total_share - 100.0) < 0.5


def test_dispatch_invalid_demand_length(client: TestClient, valid_demand_24: List[float]):
    """
    Verifies rejection with 422 when demand sequence length is not exactly 24.
    """
    # 23 hours
    res_23 = client.post("/energy/dispatch", json={"demand_mw": valid_demand_24[:23]})
    assert res_23.status_code == 422

    # 25 hours
    res_25 = client.post("/energy/dispatch", json={"demand_mw": valid_demand_24 + [1800.0]})
    assert res_25.status_code == 422


def test_dispatch_non_finite_and_negative_demand(client: TestClient, valid_demand_24: List[float]):
    """
    Verifies that negative and non-finite demand values are rejected with 422.
    """
    # Negative demand
    bad_demand = list(valid_demand_24)
    bad_demand[5] = -50.0
    res_neg = client.post("/energy/dispatch", json={"demand_mw": bad_demand})
    assert res_neg.status_code == 422

    # Non-numeric / NaN value
    bad_nan = list(valid_demand_24)
    bad_nan[10] = "NaN"
    res_nan = client.post("/energy/dispatch", json={"demand_mw": bad_nan})
    assert res_nan.status_code == 422


def test_dispatch_invalid_solar_profile(client: TestClient, valid_demand_24: List[float]):
    """
    Verifies rejection when solar profile has wrong length or values outside [0.0, 1.0].
    """
    # Length != 24
    res_len = client.post(
        "/energy/dispatch",
        json={"demand_mw": valid_demand_24, "solar_profile": [0.5] * 12},
    )
    assert res_len.status_code == 422

    # Value > 1.0
    res_val = client.post(
        "/energy/dispatch",
        json={"demand_mw": valid_demand_24, "solar_profile": [1.5] * 24},
    )
    assert res_val.status_code == 422


def test_dispatch_infeasible_without_load_shedding(client: TestClient):
    """
    Verifies that when demand exceeds total fleet capacity (2,700 MW) and
    load shedding is disabled, the system reports infeasibility cleanly
    without crashing or falsifying success.
    """
    extreme_demand = [4000.0] * 24
    response = client.post(
        "/energy/dispatch",
        json={"demand_mw": extreme_demand, "include_load_shedding": False},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["is_feasible"] is False
    assert data["is_optimal"] is False
    assert "infeasible" in data["optimization_status"].lower()


def test_dispatch_feasible_with_load_shedding(client: TestClient):
    """
    Verifies that when load shedding is enabled under supply deficit,
    the dispatch solves with unserved energy recorded at the penalty cost.
    """
    deficit_demand = [3000.0] * 24  # Exceeds max 2,700 MW
    response = client.post(
        "/energy/dispatch",
        json={
            "demand_mw": deficit_demand,
            "include_load_shedding": True,
            "load_shedding_cost": 150.0,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["is_feasible"] is True
    assert data["unserved_energy_mwh"] > 0.0
    assert data["total_cost"] > 0.0


def test_dispatch_integration_with_phase4_forecast(client: TestClient, valid_weather_24: list):
    """
    Verifies that POST /energy/dispatch can consume the Phase 4 24h recursive forecast
    via use_forecast_24h=True and valid weather forecasts.
    """
    response = client.post(
        "/energy/dispatch",
        json={
            "use_forecast_24h": True,
            "weather_forecasts": valid_weather_24,
            "include_load_shedding": True,
        },
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["is_optimal"] is True
    assert len(data["hourly_schedule"]) == 24
    assert data["total_demand_mwh"] > 0.0


def test_dispatch_ambiguous_and_missing_inputs(client: TestClient, valid_demand_24: List[float], valid_weather_24: list):
    """
    Verifies rejection when both demand_mw and use_forecast_24h are supplied,
    or when neither is supplied.
    """
    # Both supplied
    res_both = client.post(
        "/energy/dispatch",
        json={
            "demand_mw": valid_demand_24,
            "use_forecast_24h": True,
            "weather_forecasts": valid_weather_24,
        },
    )
    assert res_both.status_code == 400
    assert "ambiguous" in res_both.json()["detail"].lower()

    # Neither supplied
    res_neither = client.post("/energy/dispatch", json={})
    assert res_neither.status_code == 400
    assert "missing" in res_neither.json()["detail"].lower()


# ══════════════════════════════════════════════════════════════
# 2. MONTE CARLO RISK ANALYSIS TESTS
# ══════════════════════════════════════════════════════════════

def test_risk_valid_request(client: TestClient, valid_demand_24: List[float]):
    """
    Verifies POST /energy/risk computes valid reliability and economic risk metrics.
    """
    payload = {
        "demand_mw": valid_demand_24,
        "n_simulations": 50,
        "random_seed": 42,
        "dispatch_engine": "merit_order",
    }
    response = client.post("/energy/risk", json=payload)
    assert response.status_code == 200, response.text

    data = response.json()
    assert data["n_simulations"] == 50
    assert data["random_seed"] == 42
    assert data["dispatch_engine"] == "merit_order"

    # Base forecast summary
    base_sum = data["base_forecast_summary"]
    assert base_sum["peak_demand_mw"] == max(valid_demand_24)
    assert base_sum["total_demand_mwh"] == sum(valid_demand_24)

    # Risk metrics
    metrics = data["risk_metrics"]
    assert 0.0 <= metrics["LOLP"] <= 1.0
    assert metrics["EENS_mwh"] >= 0.0
    assert 0.0 <= metrics["reserve_sufficiency"] <= 1.0
    assert abs((metrics["LOLP"] + metrics["reserve_sufficiency"]) - 1.0) < 1e-3
    assert metrics["min_cost"] <= metrics["median_cost"] <= metrics["max_cost"]
    assert metrics["p95_cost"] >= metrics["median_cost"]

    # Quantile summaries
    q_summaries = data["quantile_summaries"]
    assert len(q_summaries) == 6
    for q in q_summaries:
        assert "metric" in q
        assert q["p05"] <= q["p50"] <= q["p95"]

    # Provenance note
    assert "residual bootstrapping" in data["uncertainty_provenance_note"].lower()


def test_risk_seed_reproducibility(client: TestClient, valid_demand_24: List[float]):
    """
    Verifies that identical random seeds yield identical simulation results.
    """
    payload = {
        "demand_mw": valid_demand_24,
        "n_simulations": 30,
        "random_seed": 98765,
        "dispatch_engine": "merit_order",
    }
    res1 = client.post("/energy/risk", json=payload).json()
    res2 = client.post("/energy/risk", json=payload).json()

    assert res1["risk_metrics"] == res2["risk_metrics"]
    assert res1["quantile_summaries"] == res2["quantile_summaries"]


def test_risk_simulation_count_bounds(client: TestClient, valid_demand_24: List[float]):
    """
    Verifies enforcement of simulation count bounds [10, 1000].
    """
    # Too small (< 10)
    res_low = client.post(
        "/energy/risk",
        json={"demand_mw": valid_demand_24, "n_simulations": 5},
    )
    assert res_low.status_code == 422

    # Too high (> 1000)
    res_high = client.post(
        "/energy/risk",
        json={"demand_mw": valid_demand_24, "n_simulations": 1500},
    )
    assert res_high.status_code == 422


def test_risk_residuals_not_mutated(client: TestClient, valid_demand_24: List[float]):
    """
    Verifies that running Monte Carlo simulations does not mutate
    the in-memory forecaster.residuals array.
    """
    forecaster = client.app.state.forecaster
    assert forecaster is not None
    orig_residuals = forecaster.residuals.copy()

    # Run with clipping
    client.post(
        "/energy/risk",
        json={
            "demand_mw": valid_demand_24,
            "n_simulations": 25,
            "enable_residual_clipping": True,
        },
    )

    # Assert exact array equality
    np.testing.assert_array_equal(forecaster.residuals, orig_residuals)


def test_risk_integration_with_phase4_forecast(client: TestClient, valid_weather_24: list):
    """
    Verifies that POST /energy/risk safely consumes the Phase 4 24h recursive forecast.
    """
    response = client.post(
        "/energy/risk",
        json={
            "use_forecast_24h": True,
            "weather_forecasts": valid_weather_24,
            "n_simulations": 20,
            "random_seed": 42,
        },
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["n_simulations"] == 20
    assert data["base_forecast_summary"]["total_demand_mwh"] > 0.0
    assert "LOLP" in data["risk_metrics"]
