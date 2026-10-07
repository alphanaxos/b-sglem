"""
Comprehensive test suite for Phase 4 Data API, Feature Pipeline, and 24-Hour Forecasting.
"""
import copy
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from config import DATA_DIR, FEATURE_COLS
from api.app import create_app
from services.data_service import DataService


# ══════════════════════════════════════════════════════════════
# 1. LATEST RECORD RETRIEVAL
# ══════════════════════════════════════════════════════════════

def test_get_latest_record(client: TestClient):
    """
    Verifies that GET /data/latest returns the latest historical record
    with load, weather, calendar, and explicit historical labeling.
    """
    response = client.get("/data/latest")
    assert response.status_code == 200

    data = response.json()
    assert "timestamp" in data
    assert data["timestamp"] == "2024-12-31T23:00:00"
    assert "load_mw" in data
    assert isinstance(data["load_mw"], float)
    assert data["load_mw"] > 0

    # Weather fields check
    weather = data["weather"]
    for key in ["temperature", "humidity", "wind_speed", "solar_irradiance"]:
        assert key in weather
        assert isinstance(weather[key], float)

    # Calendar fields check
    calendar = data["calendar"]
    for key in [
        "is_weekend", "is_holiday", "is_festival", "is_pre_holiday",
        "is_post_holiday", "is_pre_festival", "is_post_festival", "workday_after_holiday"
    ]:
        assert key in calendar
        assert calendar[key] in (0, 1)

    # Provenance and labeling: must NOT be labeled as simulated or real-time
    assert data["data_source_mode"] == "historical"
    assert data["is_simulated"] is False


# ══════════════════════════════════════════════════════════════
# 2. HISTORICAL QUERY & PAGINATION
# ══════════════════════════════════════════════════════════════

def test_get_history_defaults(client: TestClient):
    """
    Verifies that GET /data/history returns chronological records with
    sensible default limit (100) and explicit timezone metadata.
    """
    response = client.get("/data/history")
    assert response.status_code == 200

    data = response.json()
    assert data["total_records"] == 8616
    assert data["returned_records"] == 100
    assert data["limit"] == 100
    assert data["offset"] == 0
    assert data["timezone"] == "Asia/Kolkata (UTC+05:30)"
    assert data["is_simulated"] is False

    records = data["records"]
    assert len(records) == 100

    # Chronological ordering check
    for i in range(len(records) - 1):
        assert records[i]["timestamp"] < records[i + 1]["timestamp"]


def test_get_history_range_filtering(client: TestClient):
    """
    Verifies date range filtering on GET /data/history.
    """
    start = "2024-06-01T00:00:00"
    end = "2024-06-02T23:00:00"
    response = client.get(f"/data/history?start={start}&end={end}&limit=200")
    assert response.status_code == 200

    data = response.json()
    assert data["returned_records"] == 48  # Exactly 48 hours for 2 full days
    for r in data["records"]:
        assert r["timestamp"] >= start
        assert r["timestamp"] <= end


def test_get_history_invalid_date_range(client: TestClient):
    """
    Verifies that start > end or invalid formats return HTTP 400 Bad Request.
    """
    # start after end
    resp1 = client.get("/data/history?start=2024-06-05T00:00:00&end=2024-06-01T00:00:00")
    assert resp1.status_code == 400
    assert "cannot be after end" in resp1.json()["detail"].lower()

    # malformed date
    resp2 = client.get("/data/history?start=invalid-date")
    assert resp2.status_code == 400


def test_get_history_pagination_validation(client: TestClient):
    """
    Verifies that invalid limit (<1 or >1000) or negative offset return HTTP 422.
    """
    assert client.get("/data/history?limit=0").status_code == 422
    assert client.get("/data/history?limit=1001").status_code == 422
    assert client.get("/data/history?offset=-1").status_code == 422


# ══════════════════════════════════════════════════════════════
# 3. DEMO SIMULATION (OPT-IN & SAFETY)
# ══════════════════════════════════════════════════════════════

def test_simulation_disabled_by_default(client: TestClient):
    """
    Verifies that POST /data/generate returns HTTP 403 Forbidden
    when simulation mode is disabled (default).
    """
    response = client.post("/data/generate", json={"hours": 1})
    assert response.status_code == 403
    assert "simulation mode is disabled" in response.json()["detail"].lower()


def test_simulation_enabled_advance(monkeypatch):
    """
    Verifies controlled demo-mode hourly advancement when simulation is enabled.
    Records are clearly labeled is_simulated=true and preserve continuity.
    """
    monkeypatch.setenv("ENABLE_SIMULATION_MODE", "true")
    sim_app = create_app()

    with TestClient(sim_app) as sim_client:
        # Generate 3 hours
        gen_resp = sim_client.post("/data/generate", json={"hours": 3})
        assert gen_resp.status_code == 200

        data = gen_resp.json()
        assert data["generated_count"] == 3
        assert data["is_simulated"] is True
        assert data["data_source_mode"] == "simulated"

        records = data["records"]
        assert len(records) == 3

        # Check chronological continuity from 2024-12-31T23:00:00
        assert records[0]["timestamp"] == "2025-01-01T00:00:00"
        assert records[1]["timestamp"] == "2025-01-01T01:00:00"
        assert records[2]["timestamp"] == "2025-01-01T02:00:00"

        for r in records:
            assert r["is_simulated"] is True
            assert r["data_source_mode"] == "simulated"
            assert r["load_mw"] > 0

        # Latest record should now be the third simulated hour
        latest_resp = sim_client.get("/data/latest")
        assert latest_resp.status_code == 200
        latest_data = latest_resp.json()
        assert latest_data["timestamp"] == "2025-01-01T02:00:00"
        assert latest_data["is_simulated"] is True


def test_simulation_never_modifies_historical_csv(monkeypatch):
    """
    Verifies that generating simulated data does not touch or modify the underlying CSV files.
    """
    csv_path = Path(DATA_DIR) / "bengaluru_power_2024_advanced.csv"
    initial_mtime = csv_path.stat().st_mtime
    initial_size = csv_path.stat().st_size

    monkeypatch.setenv("ENABLE_SIMULATION_MODE", "true")
    sim_app = create_app()
    with TestClient(sim_app) as sim_client:
        sim_client.post("/data/generate", json={"hours": 5})

    assert csv_path.stat().st_mtime == initial_mtime
    assert csv_path.stat().st_size == initial_size


# ══════════════════════════════════════════════════════════════
# 4. 28-FEATURE PREPARATION & TARGET LEAKAGE
# ══════════════════════════════════════════════════════════════

def test_feature_preparation_canonical_28_and_no_leakage():
    """
    Verifies that prepare_28_features outputs exactly the 28 canonical features
    in exact order of config.FEATURE_COLS, without target leakage.
    """
    import pandas as pd
    ds = DataService()
    ds.initialize()

    # 168 synthetic historical loads
    hist_loads = [2000.0 + i for i in range(168)]
    target_dt = pd.to_datetime("2025-01-01T12:00:00")
    weather = {
        "temperature": 28.5,
        "humidity": 55.0,
        "wind_speed": 3.0,
        "solar_irradiance": 650.0,
    }

    feature_df = ds.prepare_28_features(
        target_dt=target_dt,
        load_history_168=hist_loads,
        weather_inputs=weather,
    )

    # Exactly 28 features in exact order
    assert list(feature_df.columns) == FEATURE_COLS
    assert len(feature_df) == 1

    # Verify lag math (zero target leakage)
    assert feature_df["lag_1"].iloc[0] == hist_loads[-1]
    assert feature_df["lag_24"].iloc[0] == hist_loads[-24]
    assert feature_df["lag_48"].iloc[0] == hist_loads[-48]
    assert feature_df["lag_72"].iloc[0] == hist_loads[-72]
    assert feature_df["lag_168"].iloc[0] == hist_loads[-168]

    # Verify interaction
    expected_interaction = 28.5 * feature_df["hour_sin"].iloc[0]
    assert abs(feature_df["temp_x_hour_sin"].iloc[0] - expected_interaction) < 1e-6


def test_feature_preparation_insufficient_history():
    """
    Verifies that providing fewer than 168 hours of load history raises ValueError.
    """
    import pandas as pd
    ds = DataService()
    target_dt = pd.to_datetime("2025-01-01T12:00:00")
    weather = {"temperature": 25.0, "humidity": 60.0, "wind_speed": 2.0, "solar_irradiance": 0.0}

    with pytest.raises(ValueError, match="Expected exactly 168"):
        ds.prepare_28_features(
            target_dt=target_dt,
            load_history_168=[2000.0] * 100,  # Only 100
            weather_inputs=weather,
        )


def test_feature_preparation_missing_weather_rejected():
    """
    Verifies that missing weather inputs are rejected with clear ValueError.
    """
    import pandas as pd
    ds = DataService()
    target_dt = pd.to_datetime("2025-01-01T12:00:00")
    incomplete_weather = {"temperature": 25.0, "humidity": 60.0}  # Missing wind_speed & solar

    with pytest.raises(ValueError, match="Missing required weather feature"):
        ds.prepare_28_features(
            target_dt=target_dt,
            load_history_168=[2000.0] * 168,
            weather_inputs=incomplete_weather,
        )


# ══════════════════════════════════════════════════════════════
# 5. 24-HOUR MULTI-STEP RECURSIVE FORECASTING
# ══════════════════════════════════════════════════════════════

def test_forecast_24h_success(client: TestClient, valid_weather_24):
    """
    Verifies POST /forecast/24h generates an authentic 24-hour day-ahead
    forecast with recursive lag updates and empirical confidence bands.
    """
    payload = {"weather_forecasts": valid_weather_24}
    response = client.post("/forecast/24h", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["forecast_horizon_hours"] == 24
    assert data["forecast_origin"] == "2024-12-31T23:00:00"
    assert data["data_source"] == "historical"
    assert data["is_simulated"] is False

    predictions = data["predictions"]
    assert len(predictions) == 24

    # Verify first hour (origin + 1h) uses ground-truth lag (is_recursive_lag=False)
    assert predictions[0]["timestamp"] == "2025-01-01T00:00:00"
    assert predictions[0]["hour"] == 0
    assert predictions[0]["is_recursive_lag"] is False
    assert predictions[0]["predicted_load_mw"] > 0
    assert predictions[0]["spinning_reserve_mw"] == round(predictions[0]["predicted_load_mw"] * 0.20, 2)

    # Verify subsequent hours (hour 2..24) are marked as recursive
    for i in range(1, 24):
        assert predictions[i]["is_recursive_lag"] is True
        assert predictions[i]["predicted_load_mw"] > 0

    assert predictions[23]["timestamp"] == "2025-01-01T23:00:00"

    # Provenance note check
    assert "recursive multi-step forecast" in data["interval_provenance_note"].lower()
    assert "not statistically calibrated" in data["interval_provenance_note"].lower()


def test_forecast_24h_missing_weather_rejected(client: TestClient, valid_weather_24):
    """
    Verifies that sending fewer than 24 weather items returns HTTP 422.
    """
    bad_payload = {"weather_forecasts": valid_weather_24[:20]}  # Only 20 hours
    response = client.post("/forecast/24h", json=bad_payload)
    assert response.status_code == 422


# ══════════════════════════════════════════════════════════════
# 6. DATASET TRANSITION & GAP DETECTION
# ══════════════════════════════════════════════════════════════

def test_gap_detection_combined_dataset():
    """
    Verifies that DataService.detect_gaps accurately detects the 168-hour transition
    gap between 2023-12-31 and 2024-01-08 when loading the combined dataset.
    """
    ds = DataService(active_dataset="combined")
    df = ds.load_historical_data("combined")
    gaps = ds.detect_gaps(df)

    assert len(gaps) == 1
    start_gap, end_gap, gap_hours = gaps[0]
    assert start_gap == "2023-12-31T23:00:00"
    assert end_gap == "2024-01-08T00:00:00"
    assert gap_hours == 168


# ══════════════════════════════════════════════════════════════
# 7. HEALTH CHECK DATA STATUS
# ══════════════════════════════════════════════════════════════

def test_health_reports_data_status(client: TestClient):
    """
    Verifies that GET /health reports data source mode and active dataset.
    """
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["data_source_mode"] == "historical"
    assert data["active_dataset"] == "2024"
    assert data["simulation_enabled"] is False
