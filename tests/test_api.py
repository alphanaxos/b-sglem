"""
Comprehensive API test suite for B-SGLEM Phase 3 ML API.
"""
import copy
import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.schemas import CANONICAL_FEATURE_COLUMNS


# ══════════════════════════════════════════════════════════════
# 1. HEALTH ENDPOINT TESTS
# ══════════════════════════════════════════════════════════════

def test_health_success(client: TestClient):
    """
    Verifies that /health returns 200 OK, reports models loaded,
    and does not expose sensitive host file paths.
    """
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert data["models_loaded"] is True
    assert set(data["loaded_models"]) == {"XGBoost", "BiLSTM", "CNN-LSTM"}
    assert data["feature_count"] == 28
    assert "prediction_interval_quantiles" in data
    assert "q05" in data["prediction_interval_quantiles"]
    assert "q95" in data["prediction_interval_quantiles"]

    # Security: Verify no absolute local filesystem paths leaked
    raw_text = response.text.lower()
    assert "c:\\" not in raw_text
    assert "/users/" not in raw_text
    assert "models_saved" not in raw_text


def test_health_and_prediction_when_artifacts_missing(monkeypatch, tmp_path):
    """
    Tests graceful degradation when model artifacts are absent.
    /health reports degraded, and prediction endpoints return HTTP 503.
    """
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "non_existent_models"))
    app_degraded = create_app()

    with TestClient(app_degraded) as deg_client:
        # Health check should report degraded but return 200
        health_resp = deg_client.get("/health")
        assert health_resp.status_code == 200
        health_data = health_resp.json()
        assert health_data["status"] == "degraded"
        assert health_data["models_loaded"] is False

        # Operational predict should return 503 Service Unavailable
        pred_resp = deg_client.post("/predict", json={
            "features": {col: 0.0 for col in CANONICAL_FEATURE_COLUMNS}
        })
        assert pred_resp.status_code == 503
        assert "unavailable" in pred_resp.json()["detail"].lower()

        # Scenario predict should return 503 Service Unavailable
        scen_resp = deg_client.post("/predict/scenario", json={
            "features": {col: 0.0 for col in CANONICAL_FEATURE_COLUMNS}
        })
        assert scen_resp.status_code == 503


# ══════════════════════════════════════════════════════════════
# 2. OPERATIONAL PREDICTION TESTS (/predict)
# ══════════════════════════════════════════════════════════════

def test_predict_valid_features(client: TestClient, valid_features_dict):
    """
    Verifies valid prediction response, spinning reserve (20%),
    empirical interval bounds, and limitation provenance documentation.
    """
    payload = {
        "features": valid_features_dict,
        "timestamp": "2026-10-06T23:00:00Z"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "predicted_load_mw" in data
    pred = data["predicted_load_mw"]
    assert isinstance(pred, float)
    assert pred > 0

    # Spinning reserve check (exactly 20%)
    expected_reserve = round(pred * 0.20, 2)
    assert data["spinning_reserve_mw"] == expected_reserve

    # Empirical interval checks
    health_data = client.get("/health").json()
    q05 = health_data["prediction_interval_quantiles"]["q05"]
    q95 = health_data["prediction_interval_quantiles"]["q95"]

    expected_lower = round(pred + q05, 2)
    expected_upper = round(pred + q95, 2)
    assert data["lower_bound_mw"] == expected_lower
    assert data["upper_bound_mw"] == expected_upper

    # Provenance note check
    assert "stacked out-of-fold" in data["interval_provenance_note"].lower()
    assert "not a statistically calibrated prediction interval" in data["interval_provenance_note"].lower()
    assert data["model_used"] == "XGBoost (Operational Next-Hour)"
    assert data["timestamp"] == "2026-10-06T23:00:00Z"


def test_predict_missing_feature(client: TestClient, valid_features_dict):
    """
    Verifies that omitting a required feature column produces HTTP 422.
    """
    bad_features = copy.deepcopy(valid_features_dict)
    del bad_features["temperature"]

    response = client.post("/predict", json={"features": bad_features})
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any("temperature" in str(err["loc"]) for err in errors)


def test_predict_extra_unexpected_field(client: TestClient, valid_features_dict):
    """
    Verifies that unknown/unexpected fields are strictly rejected (HTTP 422).
    """
    bad_features = copy.deepcopy(valid_features_dict)
    bad_features["unexpected_column_x"] = 999.0

    response = client.post("/predict", json={"features": bad_features})
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any("extra_forbidden" in err.get("type", "") for err in errors)


def test_predict_invalid_binary_flag(client: TestClient, valid_features_dict):
    """
    Verifies that binary calendar flags only accept 0 or 1.
    """
    bad_features = copy.deepcopy(valid_features_dict)
    bad_features["is_weekend"] = 2  # Invalid value

    response = client.post("/predict", json={"features": bad_features})
    assert response.status_code == 422


def test_predict_non_finite_number(client: TestClient, valid_features_dict):
    """
    Verifies that non-finite numbers (NaN, Inf) are rejected.
    """
    # JSON cannot serialize NaN natively, but strings or null will trigger validation errors
    bad_features = copy.deepcopy(valid_features_dict)
    bad_features["humidity"] = "NaN"

    response = client.post("/predict", json={"features": bad_features})
    assert response.status_code == 422


# ══════════════════════════════════════════════════════════════
# 3. SCENARIO ANALYSIS TESTS (/predict/scenario)
# ══════════════════════════════════════════════════════════════

def test_scenario_standard_five(client: TestClient, valid_features_dict):
    """
    Verifies that /predict/scenario evaluates all 5 standard scenarios
    and correctly calculates delta and percentage change vs Normal.
    """
    response = client.post("/predict/scenario", json={"features": valid_features_dict})
    assert response.status_code == 200

    data = response.json()
    baseline = data["baseline_load_mw"]
    assert baseline > 0

    results = data["results"]
    assert len(results) == 5

    names = [r["scenario_name"] for r in results]
    assert names == ["Normal", "Heatwave (+5°C)", "Festival Load", "Rainy Day", "Night Peak (22:00)"]

    for r in results:
        assert "predicted_load_mw" in r
        assert "delta_mw" in r
        assert "pct_change" in r
        assert "description" in r

        # Normal scenario must have zero delta
        if r["scenario_name"] == "Normal":
            assert r["delta_mw"] == 0.0
            assert r["pct_change"] == 0.0
            assert r["predicted_load_mw"] == baseline
        else:
            # Check mathematical relationship
            expected_delta = round(r["predicted_load_mw"] - baseline, 2)
            assert abs(r["delta_mw"] - expected_delta) < 0.05


def test_scenario_subset_selection(client: TestClient, valid_features_dict):
    """
    Verifies that requesting a subset of scenarios returns only those scenarios.
    """
    selected = ["Normal", "Festival Load"]
    response = client.post("/predict/scenario", json={
        "features": valid_features_dict,
        "scenarios": selected
    })
    assert response.status_code == 200
    data = response.json()
    results = data["results"]
    assert len(results) == 2
    assert [r["scenario_name"] for r in results] == selected


def test_scenario_unknown_name_rejected(client: TestClient, valid_features_dict):
    """
    Verifies that requesting an unrecognized scenario name returns HTTP 400.
    """
    response = client.post("/predict/scenario", json={
        "features": valid_features_dict,
        "scenarios": ["Bogus Scenario"]
    })
    assert response.status_code == 400
    assert "unknown scenario" in response.json()["detail"].lower()


# ══════════════════════════════════════════════════════════════
# 4. PRESERVATION & STABILITY TESTS
# ══════════════════════════════════════════════════════════════

def test_models_loaded_once_no_retraining(client: TestClient, valid_features_dict):
    """
    Verifies that models are loaded once during startup and never refitted or reloaded.
    """
    forecaster = client.app.state.forecaster
    initial_xgb = forecaster.xgb_model

    # Run predictions multiple times
    for _ in range(5):
        client.post("/predict", json={"features": valid_features_dict})

    # Assert model object identity remains strictly unchanged
    assert client.app.state.forecaster.xgb_model is initial_xgb


def test_canonical_feature_ordering_enforced(valid_features_dict):
    """
    Verifies that FeatureRowInput maintains canonical column ordering
    regardless of input dictionary key ordering.
    """
    from api.schemas import FeatureRowInput

    # Scramble keys into reversed order
    reversed_dict = {k: valid_features_dict[k] for k in sorted(valid_features_dict.keys(), reverse=True)}

    row_input = FeatureRowInput(**reversed_dict)
    df = row_input.to_dataframe()

    assert list(df.columns) == CANONICAL_FEATURE_COLUMNS
