"""
Energy Management and Risk Analysis route handler (Phase 5).

Provides:
- POST /energy/dispatch: PyPSA 24-hour linear optimal power flow economic dispatch.
- POST /energy/risk: Monte Carlo uncertainty and grid reliability risk analysis.
"""
from typing import List, Tuple, Optional
from fastapi import APIRouter, Depends, HTTPException, status

from api.schemas import (
    DispatchRequest,
    DispatchResponse,
    RiskAnalysisRequest,
    RiskAnalysisResponse,
)
from api.dependencies import get_forecaster, get_data_service
from services.prediction import LoadForecaster
from services.data_service import DataService
from services.energy_service import EnergyService

router = APIRouter(prefix="/energy", tags=["Energy Management & Risk"])


def _resolve_demand_and_timestamps(
    payload: DispatchRequest | RiskAnalysisRequest,
    forecaster: LoadForecaster,
    data_service: DataService,
) -> Tuple[List[float], Optional[List[str]]]:
    """
    Resolves the 24-hour demand vector and timestamps.
    Supports either explicit demand_mw or Phase 4 24h recursive forecasting.
    """
    if payload.demand_mw is not None and payload.use_forecast_24h:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Ambiguous request: specify either 'demand_mw' or 'use_forecast_24h=True', not both.",
        )

    if payload.demand_mw is not None:
        return payload.demand_mw, None

    if payload.use_forecast_24h:
        if not payload.weather_forecasts or len(payload.weather_forecasts) != 24:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="When 'use_forecast_24h' is True, exactly 24 'weather_forecasts' must be provided.",
            )
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
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to generate 24h recursive forecast: {exc}",
            )

        predicted_loads = [float(p["predicted_load_mw"]) for p in forecast_result["predictions"]]
        timestamps = [str(p["timestamp"]) for p in forecast_result["predictions"]]
        return predicted_loads, timestamps

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Demand profile missing: please provide 'demand_mw' (24 hourly values) or set 'use_forecast_24h=True' with 'weather_forecasts'.",
    )


@router.post("/dispatch", response_model=DispatchResponse)
async def dispatch_energy(
    payload: DispatchRequest,
    forecaster: LoadForecaster = Depends(get_forecaster),
    data_service: DataService = Depends(get_data_service),
) -> DispatchResponse:
    """
    Solves 24-hour economic dispatch using PyPSA Linear Optimal Power Flow.

    Features:
    - Minimizes operational generation costs across Coal (₹3.5/unit), Gas (₹6.0/unit), and Solar (₹0.0/unit).
    - Enforces thermal capacity, minimum stable output (Coal 40%), and diurnal solar profiles.
    - Supports optional load shedding (penalty ₹100/MWh) to assess supply adequacy.
    - Transparently reports optimization status and feasibility without altering physical constraints.
    - Accepts an explicit 24h demand sequence or integrates with the Phase 4 recursive forecast.
    """
    demand_mw, timestamps = _resolve_demand_and_timestamps(payload, forecaster, data_service)

    try:
        dispatch_dict = EnergyService.solve_dispatch(
            demand_mw=demand_mw,
            solar_profile=payload.solar_profile,
            include_load_shedding=payload.include_load_shedding,
            load_shedding_cost=payload.load_shedding_cost,
            timestamps=timestamps,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Economic dispatch optimization failed: {exc}",
        )

    return DispatchResponse(**dispatch_dict)


@router.post("/risk", response_model=RiskAnalysisResponse)
async def assess_risk(
    payload: RiskAnalysisRequest,
    forecaster: LoadForecaster = Depends(get_forecaster),
    data_service: DataService = Depends(get_data_service),
) -> RiskAnalysisResponse:
    """
    Executes Monte Carlo uncertainty and grid reliability risk assessment over a 24-hour horizon.

    Features:
    - Generates stochastic load scenarios using block-bootstrapped empirical residuals from the stacked ensemble.
    - Clips residual tails between 5th and 95th percentiles to avoid extreme unphysical spikes.
    - Evaluates Loss of Load Probability (LOLP), Expected Energy Not Served (EENS in MWh), and Reserve Sufficiency.
    - Computes empirical quantile bounds (p05, p50, p95) for peak demand, dispatch cost, and fuel utilization.
    - Provides reproducible draws via optional random seed.
    - Keeps residual distributions immutable in memory and operates without mutating model artifacts.
    """
    demand_mw, _ = _resolve_demand_and_timestamps(payload, forecaster, data_service)

    if forecaster.residuals is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Residual distribution artifact not available; risk analysis cannot proceed.",
        )

    try:
        risk_dict = EnergyService.run_risk_analysis(
            demand_mw=demand_mw,
            residuals=forecaster.residuals,
            n_simulations=payload.n_simulations,
            random_seed=payload.random_seed,
            enable_residual_clipping=payload.enable_residual_clipping,
            load_shedding_cost=payload.load_shedding_cost,
            dispatch_engine=payload.dispatch_engine,
            solar_profile=payload.solar_profile,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Risk analysis simulation failed: {exc}",
        )

    return RiskAnalysisResponse(**risk_dict)
