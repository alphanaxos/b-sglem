"""
FastAPI Application Entrypoint for B-SGLEM Load Forecasting System.

Artifacts and data services are loaded once during the lifespan startup context.
Zero retraining, zero parameter recalculation, zero disk re-reads per request.
"""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import (
    MODEL_DIR,
    DATA_DIR,
    ACTIVE_DATASET,
    DATA_SOURCE_MODE,
    ENABLE_SIMULATION_MODE,
)
from services.prediction import LoadForecaster
from services.data_service import DataService
from api.routes import health, predict, scenario, data, forecast, energy


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.
    Loads saved model artifacts and initializes the data service once during startup.
    Never reloads artifacts or refits models during HTTP request handling.
    """
    model_dir = os.getenv("MODEL_DIR", MODEL_DIR)
    data_dir = os.getenv("DATA_DIR", DATA_DIR)
    active_dataset = os.getenv("ACTIVE_DATASET", ACTIVE_DATASET)
    data_source_mode = os.getenv("DATA_SOURCE_MODE", DATA_SOURCE_MODE)
    simulation_enabled = os.getenv("ENABLE_SIMULATION_MODE", str(ENABLE_SIMULATION_MODE)).lower() in ("true", "1")

    # 1. Load ML models and scalers
    try:
        forecaster = LoadForecaster.load(model_dir)
        app.state.forecaster = forecaster
        app.state.load_error = None
    except Exception as exc:
        app.state.forecaster = None
        app.state.load_error = str(exc)

    # 2. Initialize Data Service
    try:
        data_service = DataService(
            data_dir=data_dir,
            active_dataset=active_dataset,
            data_source_mode=data_source_mode,
            simulation_enabled=simulation_enabled,
        )
        data_service.initialize()
        app.state.data_service = data_service
    except Exception as exc:
        app.state.data_service = None
        if app.state.load_error:
            app.state.load_error += f"; DataService error: {exc}"
        else:
            app.state.load_error = f"DataService error: {exc}"

    yield

    # Clean shutdown
    app.state.forecaster = None
    app.state.data_service = None


def create_app() -> FastAPI:
    """
    Application factory for the B-SGLEM ML API.
    """
    application = FastAPI(
        title="B-SGLEM Load Forecasting & Data API",
        description=(
            "Operational Load Forecasting, Historical Telemetry, Demo Simulation, "
            "and 24-Hour Day-Ahead Forecasting Service for the Bengaluru Smart Grid (Phase 4)."
        ),
        version="4.0.0",
        lifespan=lifespan,
    )

    # Environment-driven CORS with permissive fallback for local development
    cors_env = os.getenv("CORS_ORIGINS", "*").strip()
    if cors_env == "*" or not cors_env:
        allow_origins = ["*"]
    else:
        allow_origins = [origin.strip() for origin in cors_env.split(",") if origin.strip()]

    application.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register Route Handlers
    application.include_router(health.router, tags=["System Health"])
    application.include_router(predict.router, tags=["Operational Forecasting"])
    application.include_router(scenario.router, tags=["Scenario Analysis"])
    application.include_router(data.router)
    application.include_router(forecast.router)
    application.include_router(energy.router)

    return application


app = create_app()


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("api.app:app", host=host, port=port)
