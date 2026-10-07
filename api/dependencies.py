"""
FastAPI dependency injection for forecaster and data service access.
"""
from fastapi import Request, HTTPException, status
from services.prediction import LoadForecaster
from services.data_service import DataService


def get_forecaster(request: Request) -> LoadForecaster:
    """
    Retrieves the in-memory LoadForecaster singleton from application state.
    Raises HTTP 503 if artifacts failed to load during startup lifespan.
    """
    forecaster = getattr(request.app.state, "forecaster", None)
    if forecaster is None:
        load_error = getattr(request.app.state, "load_error", "Artifacts not loaded.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Forecasting service unavailable: {load_error}"
        )
    return forecaster


def get_data_service(request: Request) -> DataService:
    """
    Retrieves the in-memory DataService singleton from application state.
    Raises HTTP 503 if data service is not initialized.
    """
    data_service = getattr(request.app.state, "data_service", None)
    if data_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Data service unavailable: service not initialized."
        )
    return data_service
