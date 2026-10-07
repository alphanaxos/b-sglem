"""
Scenario analysis service extracted from train.py.

Preserves the exact scenario definitions, feature modifications,
and evaluation logic used in Phase 2 operational analysis.
"""
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd


STANDARD_SCENARIO_DESCRIPTIONS = {
    "Normal": "Baseline operational feature vector without modification.",
    "Heatwave (+5°C)": "Temperature increased by 5°C with updated temp_x_hour_sin interaction.",
    "Festival Load": "is_festival flag set to 1.",
    "Rainy Day": "Humidity increased by 20% and solar_irradiance attenuated by 70% (multiplied by 0.3).",
    "Night Peak (22:00)": "Fourier hour harmonics adjusted to 22:00 (hour_sin = sin(2π·22/24), hour_cos = cos(2π·22/24)).",
}


def build_scenario_dataframes(baseline_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Builds the 5 standard scenario DataFrames from a baseline feature row.
    Directly mirrors Section 10 of train.py.

    Args:
        baseline_df (pd.DataFrame): Single-row DataFrame containing the 28 engineered features.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping scenario names to modified feature DataFrames.
    """
    if len(baseline_df) != 1:
        raise ValueError(f"Baseline feature DataFrame must contain exactly 1 row, got {len(baseline_df)}")

    base_row = baseline_df.copy()

    scenarios = {
        "Normal": base_row.copy(),
        "Heatwave (+5°C)": base_row.copy(),
        "Festival Load": base_row.copy(),
        "Rainy Day": base_row.copy(),
        "Night Peak (22:00)": base_row.copy(),
    }

    # 1. Heatwave (+5°C)
    if 'temperature' in base_row.columns:
        scenarios["Heatwave (+5°C)"]['temperature'] += 5.0
        if 'hour_sin' in base_row.columns and 'temp_x_hour_sin' in base_row.columns:
            scenarios["Heatwave (+5°C)"]['temp_x_hour_sin'] = (
                scenarios["Heatwave (+5°C)"]['temperature'] *
                scenarios["Heatwave (+5°C)"]['hour_sin']
            )

    # 2. Festival Load
    if 'is_festival' in base_row.columns:
        scenarios["Festival Load"]['is_festival'] = 1

    # 3. Rainy Day
    if 'humidity' in base_row.columns:
        scenarios["Rainy Day"]['humidity'] += 20.0
    if 'solar_irradiance' in base_row.columns:
        scenarios["Rainy Day"]['solar_irradiance'] *= 0.3

    # 4. Night Peak (22:00)
    if 'hour_sin' in base_row.columns:
        scenarios["Night Peak (22:00)"]['hour_sin'] = float(np.sin(2 * np.pi * 22 / 24))
    if 'hour_cos' in base_row.columns:
        scenarios["Night Peak (22:00)"]['hour_cos'] = float(np.cos(2 * np.pi * 22 / 24))

    return scenarios


def evaluate_scenarios(
    xgb_model: Any,
    baseline_df: pd.DataFrame,
    selected_scenarios: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Evaluates scenario forecasts against the baseline using the operational XGBoost model.

    Args:
        xgb_model: Fitted XGBRegressor.
        baseline_df (pd.DataFrame): 1-row DataFrame containing engineered features.
        selected_scenarios (Optional[List[str]]): Specific scenarios to evaluate. Defaults to all 5.

    Returns:
        Dict[str, Any]:
            - baseline_load_mw: float
            - results: List[Dict[str, Any]] with keys:
                - scenario_name
                - predicted_load_mw
                - delta_mw
                - pct_change
                - description
    """
    scenario_dfs = build_scenario_dataframes(baseline_df)

    if selected_scenarios is not None:
        unknown = [s for s in selected_scenarios if s not in scenario_dfs]
        if unknown:
            raise ValueError(f"Unknown scenario(s): {unknown}. Available: {list(scenario_dfs.keys())}")
        target_scenarios = {name: scenario_dfs[name] for name in selected_scenarios}
    else:
        target_scenarios = scenario_dfs

    # Baseline prediction (Normal)
    normal_pred = float(xgb_model.predict(scenario_dfs["Normal"])[0])

    results = []
    for name, df_row in target_scenarios.items():
        pred = float(xgb_model.predict(df_row)[0])
        delta = pred - normal_pred
        pct = (delta / normal_pred * 100.0) if normal_pred != 0.0 else 0.0

        results.append({
            "scenario_name": name,
            "predicted_load_mw": round(pred, 2),
            "delta_mw": round(delta, 2),
            "pct_change": round(pct, 2),
            "description": STANDARD_SCENARIO_DESCRIPTIONS.get(name, "Custom scenario variant"),
        })

    return {
        "baseline_load_mw": round(normal_pred, 2),
        "results": results,
    }
