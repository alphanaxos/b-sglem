"""
Scenario analysis route handler.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from api.schemas import ScenarioRequest, ScenarioResponse, ScenarioItem
from api.dependencies import get_forecaster
from services.prediction import LoadForecaster
from services.scenario_engine import evaluate_scenarios

router = APIRouter()


@router.post("/predict/scenario", response_model=ScenarioResponse)
async def predict_scenarios(
    payload: ScenarioRequest,
    forecaster: LoadForecaster = Depends(get_forecaster)
) -> ScenarioResponse:
    """
    Evaluates what-if scenario forecasts using the operational XGBoost model.

    Preserves the exact scenario definitions and delta calculations from Section 10 of train.py:
    1. Normal: Unmodified baseline.
    2. Heatwave (+5°C): Temperature increased by 5°C and interaction recomputed.
    3. Festival Load: is_festival set to 1.
    4. Rainy Day: Humidity increased by 20%, solar irradiance attenuated by 70%.
    5. Night Peak (22:00): Fourier hour cyclical features set to 22:00.
    """
    df = payload.features.to_dataframe()

    try:
        evaluation = evaluate_scenarios(
            xgb_model=forecaster.xgb_model,
            baseline_df=df,
            selected_scenarios=payload.scenarios
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc)
        )

    results = [
        ScenarioItem(
            scenario_name=item["scenario_name"],
            predicted_load_mw=item["predicted_load_mw"],
            delta_mw=item["delta_mw"],
            pct_change=item["pct_change"],
            description=item["description"],
        )
        for item in evaluation["results"]
    ]

    return ScenarioResponse(
        baseline_load_mw=evaluation["baseline_load_mw"],
        results=results,
        model_used="XGBoost (Operational)"
    )
