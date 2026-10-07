"""
Next-hour operational prediction route handler.
"""
from fastapi import APIRouter, Depends
from api.schemas import PredictRequest, PredictResponse
from api.dependencies import get_forecaster
from services.prediction import LoadForecaster

router = APIRouter()

INTERVAL_PROVENANCE_NOTE = (
    "Bounds calculated using empirical 5th and 95th percentile residuals "
    "(-146.87 / +145.05 MW) derived from the stacked out-of-fold ensemble. "
    "This offset provides an empirical reference band but is not a statistically "
    "calibrated prediction interval for the standalone XGBoost regressor."
)


@router.post("/predict", response_model=PredictResponse)
async def predict_next_hour(
    payload: PredictRequest,
    forecaster: LoadForecaster = Depends(get_forecaster)
) -> PredictResponse:
    """
    Generates a next-hour operational point forecast using the trained XGBoost model.

    Preserves the exact operational calculation path from Section 9 of train.py:
    - XGBoost regressor evaluated on canonical 28-feature input row.
    - Operating spinning reserve required = 20% of predicted load.
    - Empirical bounds computed from saved stacked ensemble residual quantiles (q05, q95).
    """
    df = payload.features.to_dataframe()

    # Predict using XGBoost operational path
    pred_val = float(forecaster.predict_xgb(df)[0])

    # Supplementary operational metrics
    spinning_reserve = round(pred_val * 0.20, 2)
    q05 = float(forecaster.q05) if forecaster.q05 is not None else 0.0
    q95 = float(forecaster.q95) if forecaster.q95 is not None else 0.0
    lower_bound = round(pred_val + q05, 2)
    upper_bound = round(pred_val + q95, 2)

    return PredictResponse(
        predicted_load_mw=round(pred_val, 2),
        spinning_reserve_mw=spinning_reserve,
        lower_bound_mw=lower_bound,
        upper_bound_mw=upper_bound,
        interval_confidence_level=0.90,
        interval_provenance_note=INTERVAL_PROVENANCE_NOTE,
        model_used="XGBoost (Operational Next-Hour)",
        timestamp=payload.timestamp
    )
