"""
Pytest configuration and fixtures for API testing.
"""
import pytest
from typing import Dict, Any
from fastapi.testclient import TestClient

from api.app import create_app


@pytest.fixture(scope="session")
def valid_features_dict() -> Dict[str, Any]:
    """
    Returns a valid feature dictionary with all 28 canonical features,
    derived from genuine Bengaluru grid historical telemetry.
    """
    return {
        "is_weekend": 0,
        "is_holiday": 0,
        "is_festival": 0,
        "is_pre_holiday": 0,
        "is_post_holiday": 0,
        "is_pre_festival": 0,
        "is_post_festival": 0,
        "workday_after_holiday": 0,
        "temperature": 19.743,
        "humidity": 62.15,
        "wind_speed": 1.707,
        "solar_irradiance": 0.0,
        "lag_1": 2165.26,
        "lag_24": 2126.03,
        "lag_168": 2027.28,
        "rolling_mean_24": 1973.15,
        "rolling_std_24": 115.85,
        "hour_sin": 0.0,
        "hour_cos": 1.0,
        "day_sin": 0.9749,
        "day_cos": -0.2225,
        "month_sin": 0.5,
        "month_cos": 0.8660,
        "lag_48": 2155.39,
        "lag_72": 1832.71,
        "rolling_max_24": 2165.26,
        "rolling_min_24": 1785.32,
        "temp_x_hour_sin": 0.0,
    }


@pytest.fixture(scope="session")
def valid_weather_24() -> list:
    """
    Returns 24 hourly weather measurement forecasts.
    """
    return [
        {
            "temperature": round(21.0 + 6.0 * ((i % 12) / 12.0), 2),
            "humidity": 60.0,
            "wind_speed": 2.2,
            "solar_irradiance": 500.0 if 6 <= (i % 24) <= 18 else 0.0,
        }
        for i in range(24)
    ]


@pytest.fixture(scope="session")
def client():
    """
    TestClient fixture wrapping application with lifespan execution.
    Loads models once during startup.
    """
    test_app = create_app()
    with TestClient(test_app) as c:
        yield c
