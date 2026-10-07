"""
Day-ahead 24-hour forecasting route handler (Phase 4).
"""
from fastapi import APIRouter, Depends, HTTPException, status

from api.schemas import Forecast24hRequest, Forecast24hResponse, HourlyForecastItem
from api.dependencies import get_forecaster, get_data_service
from services.prediction import LoadForecaster
from services.data_service import DataService

router = APIRouter(tags=["Multi-Step Forecasting"])


@router.post("/forecast/24h", response_model=Forecast24hResponse)
async def forecast_24h_day_ahead(
    payload: Forecast24hRequest,
    forecaster: LoadForecaster = Depends(get_forecaster),
    data_service: DataService = Depends(get_data_service),
) -> Forecast24hResponse:
    """
    Generates an authentic 24-hour day-ahead recursive forecast.

    Requirements:
    - Uses the latest valid historical load sequence as the forecast origin.
    - Requires 168 preceding hours of load telemetry for lag/rolling features.
    - Requires 24 hourly weather forecasts (temperature, humidity, wind, solar).
    - Recursively advances the 24 hours: predicted load at hour k is fed back
      into the load history buffer as lag_1 and rolling window inputs for subsequent hours.
    - Uses the trained operational XGBoost model without retraining.
    - Returns 20% spinning reserve and empirical residual bounds.
    """
    weather_dicts = [w.model_dump() for w in payload.weather_forecasts]
    cal_dicts = [c.model_dump() for c in payload.calendar_overrides] if payload.calendar_overrides else None

    q05 = float(forecaster.q05) if forecaster.q05 is not None else 0.0
    q95 = float(forecaster.q95) if forecaster.q95 is not None else 0.0

    try:
        forecast_result = data_service.forecast_24h_recursive(
            xgb_model=forecaster.xgb_model,
            q05=q05,
            q95=q95,
            weather_forecasts=weather_dicts,
            calendar_overrides=cal_dicts,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"24-hour forecasting execution failed: {exc}",
        )

    prediction_items = [HourlyForecastItem(**item) for item in forecast_result["predictions"]]

    return Forecast24hResponse(
        forecast_origin=forecast_result["forecast_origin"],
        forecast_horizon_hours=forecast_result["forecast_horizon_hours"],
        data_source=forecast_result["data_source"],
        is_simulated=forecast_result["is_simulated"],
        predictions=prediction_items,
        interval_confidence_level=forecast_result["interval_confidence_level"],
        interval_provenance_note=forecast_result["interval_provenance_note"],
        model_used=forecast_result["model_used"],
    )
