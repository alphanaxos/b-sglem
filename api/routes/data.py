"""
Data API route handlers (Phase 4).
"""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.schemas import (
    DataRecordResponse,
    HistoryResponse,
    GenerateDataRequest,
    GenerateDataResponse,
)
from api.dependencies import get_data_service
from services.data_service import DataService

router = APIRouter(prefix="/data", tags=["Data Pipeline & Telemetry"])


@router.get("/latest", response_model=DataRecordResponse)
async def get_latest_record(
    data_service: DataService = Depends(get_data_service),
) -> DataRecordResponse:
    """
    Returns the latest available record from the active data source.
    Includes demand load (MW), weather measurements, and calendar flags.
    Clearly distinguishes whether the observation is historical or simulated.
    """
    try:
        record = data_service.get_latest_record()
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve latest record: {exc}",
        )

    return DataRecordResponse(**record)


@router.get("/history", response_model=HistoryResponse)
async def get_historical_records(
    start: Optional[str] = Query(None, description="Start timestamp (ISO-8601, e.g. 2024-01-08T00:00:00)"),
    end: Optional[str] = Query(None, description="End timestamp (ISO-8601, e.g. 2024-01-10T23:00:00)"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return (1-1000)"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    data_service: DataService = Depends(get_data_service),
) -> HistoryResponse:
    """
    Returns chronological historical telemetry records with query range filtering and pagination.
    """
    try:
        history = data_service.get_history(
            start=start,
            end=end,
            limit=limit,
            offset=offset,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to query historical data: {exc}",
        )

    return HistoryResponse(**history)


@router.post("/generate", response_model=GenerateDataResponse)
async def generate_simulated_hour(
    payload: GenerateDataRequest,
    data_service: DataService = Depends(get_data_service),
) -> GenerateDataResponse:
    """
    Advances simulated hourly telemetry data in controlled demo mode.

    Requirements:
    - Opt-in through ENABLE_SIMULATION_MODE=true configuration.
    - Preserves chronological continuity without duplicate timestamps.
    - Never modifies original historical CSV files.
    - Clearly marks all generated records as is_simulated=true.
    """
    weather_override_dict = payload.weather_override.model_dump() if payload.weather_override else None

    try:
        records = data_service.generate_simulated_hour(
            count=payload.hours,
            weather_override=weather_override_dict,
        )
    except PermissionError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate simulated data: {exc}",
        )

    record_objs = [DataRecordResponse(**r) for r in records]

    return GenerateDataResponse(
        message=f"Successfully generated {len(records)} simulated hourly observation(s).",
        generated_count=len(records),
        data_source_mode="simulated",
        is_simulated=True,
        records=record_objs,
    )
