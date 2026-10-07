"""
Health check route handler.
"""
import datetime
from fastapi import APIRouter, Request
from api.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def check_health(request: Request) -> HealthResponse:
    """
    Returns the health status of the API and in-memory model artifacts.
    Omits sensitive host paths or environment secrets.
    """
    forecaster = getattr(request.app.state, "forecaster", None)
    load_error = getattr(request.app.state, "load_error", None)

    is_loaded = forecaster is not None and getattr(forecaster, "xgb_model", None) is not None
    now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()

    data_service = getattr(request.app.state, "data_service", None)
    ds_mode = getattr(data_service, "data_source_mode", "historical") if data_service else "historical"
    ds_active = getattr(data_service, "active_dataset", "2024") if data_service else "2024"
    ds_sim_enabled = getattr(data_service, "simulation_enabled", False) if data_service else False

    if is_loaded:
        return HealthResponse(
            status="healthy",
            models_loaded=True,
            loaded_models=["XGBoost", "BiLSTM", "CNN-LSTM"],
            feature_count=forecaster.n_features or 28,
            ensemble_method="bias_corrected_convex_weighted_ensemble",
            prediction_interval_quantiles={
                "q05": round(float(forecaster.q05), 4) if forecaster.q05 is not None else 0.0,
                "q95": round(float(forecaster.q95), 4) if forecaster.q95 is not None else 0.0,
            },
            server_time=now_utc,
            details="All models and scalers loaded in memory and ready for inference.",
            data_source_mode=ds_mode,
            active_dataset=ds_active,
            simulation_enabled=ds_sim_enabled,
        )
    else:
        return HealthResponse(
            status="degraded",
            models_loaded=False,
            loaded_models=[],
            feature_count=28,
            ensemble_method="bias_corrected_convex_weighted_ensemble",
            prediction_interval_quantiles={"q05": 0.0, "q95": 0.0},
            server_time=now_utc,
            details=f"Models failed to load: {load_error}" if load_error else "Model artifacts not loaded.",
            data_source_mode=ds_mode,
            active_dataset=ds_active,
            simulation_enabled=ds_sim_enabled,
        )
