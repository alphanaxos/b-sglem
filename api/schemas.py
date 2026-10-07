"""
Pydantic schemas for the B-SGLEM ML API.

Validates all 28 required feature inputs with strict constraints:
- Binary calendar flags must strictly be 0 or 1.
- Numerical features must be finite (rejecting NaN, Inf, -Inf).
- Extra/unexpected fields are strictly forbidden.
- Preserves the canonical feature order matching models_saved/feature_config.json.
"""
import math
from typing import List, Optional, Literal, Dict, Any
import pandas as pd
from pydantic import BaseModel, Field, ConfigDict, field_validator


# Canonical feature ordering from models_saved/feature_config.json
CANONICAL_FEATURE_COLUMNS: List[str] = [
    "is_weekend",
    "is_holiday",
    "is_festival",
    "is_pre_holiday",
    "is_post_holiday",
    "is_pre_festival",
    "is_post_festival",
    "workday_after_holiday",
    "temperature",
    "humidity",
    "wind_speed",
    "solar_irradiance",
    "lag_1",
    "lag_24",
    "lag_168",
    "rolling_mean_24",
    "rolling_std_24",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "month_sin",
    "month_cos",
    "lag_48",
    "lag_72",
    "rolling_max_24",
    "rolling_min_24",
    "temp_x_hour_sin",
]

BINARY_FIELDS = {
    "is_weekend",
    "is_holiday",
    "is_festival",
    "is_pre_holiday",
    "is_post_holiday",
    "is_pre_festival",
    "is_post_festival",
    "workday_after_holiday",
}


class FeatureRowInput(BaseModel):
    """
    Validates a single feature observation row containing all 28 engineered features.
    Rejects unexpected fields and non-finite values.
    """
    model_config = ConfigDict(extra="forbid")

    # ── Binary calendar flags ───────────────────────────────────────────
    is_weekend: Literal[0, 1] = Field(..., description="1 if weekend (Sat/Sun), else 0")
    is_holiday: Literal[0, 1] = Field(..., description="1 if official holiday, else 0")
    is_festival: Literal[0, 1] = Field(..., description="1 if festival day, else 0")
    is_pre_holiday: Literal[0, 1] = Field(..., description="1 if day prior to holiday, else 0")
    is_post_holiday: Literal[0, 1] = Field(..., description="1 if day following holiday, else 0")
    is_pre_festival: Literal[0, 1] = Field(..., description="1 if day prior to festival, else 0")
    is_post_festival: Literal[0, 1] = Field(..., description="1 if day following festival, else 0")
    workday_after_holiday: Literal[0, 1] = Field(..., description="1 if first working day after holiday, else 0")

    # ── Weather variables ───────────────────────────────────────────────
    temperature: float = Field(..., description="Ambient temperature in °C")
    humidity: float = Field(..., description="Relative humidity in %")
    wind_speed: float = Field(..., description="Wind speed in m/s")
    solar_irradiance: float = Field(..., description="Global horizontal solar irradiance in W/m²")

    # ── Autoregressive lag features (MW) ────────────────────────────────
    lag_1: float = Field(..., description="Demand observation at t-1 hour (MW)")
    lag_24: float = Field(..., description="Demand observation at t-24 hours (MW)")
    lag_48: float = Field(..., description="Demand observation at t-48 hours (MW)")
    lag_72: float = Field(..., description="Demand observation at t-72 hours (MW)")
    lag_168: float = Field(..., description="Demand observation at t-168 hours (same hour last week) (MW)")

    # ── Rolling window statistics (MW) ──────────────────────────────────
    rolling_mean_24: float = Field(..., description="24-hour historical rolling mean load up to t-1 (MW)")
    rolling_std_24: float = Field(..., description="24-hour historical rolling standard deviation load up to t-1 (MW)")
    rolling_max_24: float = Field(..., description="24-hour historical rolling maximum load up to t-1 (MW)")
    rolling_min_24: float = Field(..., description="24-hour historical rolling minimum load up to t-1 (MW)")

    # ── Fourier cyclical encodings ─────────────────────────────────────
    hour_sin: float = Field(..., description="sin(2π · hour / 24)")
    hour_cos: float = Field(..., description="cos(2π · hour / 24)")
    day_sin: float = Field(..., description="sin(2π · day_of_week / 7)")
    day_cos: float = Field(..., description="cos(2π · day_of_week / 7)")
    month_sin: float = Field(..., description="sin(2π · month / 12)")
    month_cos: float = Field(..., description="cos(2π · month / 12)")

    # ── Interaction feature ─────────────────────────────────────────────
    temp_x_hour_sin: float = Field(..., description="Interaction product: temperature × hour_sin")

    @field_validator(
        "temperature", "humidity", "wind_speed", "solar_irradiance",
        "lag_1", "lag_24", "lag_48", "lag_72", "lag_168",
        "rolling_mean_24", "rolling_std_24", "rolling_max_24", "rolling_min_24",
        "hour_sin", "hour_cos", "day_sin", "day_cos", "month_sin", "month_cos",
        "temp_x_hour_sin"
    )
    @classmethod
    def validate_finite(cls, v: float, info) -> float:
        if not math.isfinite(v):
            raise ValueError(f"Field '{info.field_name}' must be a finite real number (got {v})")
        return v

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts validated inputs into a single-row pandas DataFrame
        strictly adhering to the canonical feature ordering.
        """
        data = {col: getattr(self, col) for col in CANONICAL_FEATURE_COLUMNS}
        return pd.DataFrame([data], columns=CANONICAL_FEATURE_COLUMNS)


class PredictRequest(BaseModel):
    """
    Request payload for next-hour operational load forecast.
    """
    model_config = ConfigDict(extra="forbid")
    features: FeatureRowInput = Field(..., description="Engineered 28-feature input row")
    timestamp: Optional[str] = Field(None, description="Optional ISO-8601 timestamp for the forecast target hour")


class PredictResponse(BaseModel):
    """
    Response payload for next-hour operational forecast.
    """
    predicted_load_mw: float = Field(..., description="XGBoost operational point forecast in MW")
    spinning_reserve_mw: float = Field(..., description="20% operating spinning reserve requirement in MW")
    lower_bound_mw: float = Field(..., description="Empirical reference lower bound (point forecast + q05) in MW")
    upper_bound_mw: float = Field(..., description="Empirical reference upper bound (point forecast + q95) in MW")
    interval_confidence_level: float = Field(0.90, description="Nominal confidence level of the residual reference band")
    interval_provenance_note: str = Field(
        ...,
        description="Clarification of interval derivation and limitations"
    )
    model_used: str = Field("XGBoost (Operational Next-Hour)", description="Model path utilized")
    timestamp: Optional[str] = Field(None, description="Echoed forecast target timestamp if provided")


class ScenarioRequest(BaseModel):
    """
    Request payload for scenario stress testing.
    """
    model_config = ConfigDict(extra="forbid")
    features: FeatureRowInput = Field(..., description="Baseline engineered 28-feature input row")
    scenarios: Optional[List[str]] = Field(
        None,
        description="Optional subset of scenarios to evaluate. Defaults to all 5 standard scenarios."
    )


class ScenarioItem(BaseModel):
    """
    Prediction result for an individual scenario.
    """
    scenario_name: str = Field(..., description="Name of the scenario")
    predicted_load_mw: float = Field(..., description="Forecasted load under scenario conditions (MW)")
    delta_mw: float = Field(..., description="Difference from baseline forecast (MW)")
    pct_change: float = Field(..., description="Percentage change from baseline forecast (%)")
    description: str = Field(..., description="Summary of perturbations applied in this scenario")


class ScenarioResponse(BaseModel):
    """
    Response payload for scenario analysis.
    """
    baseline_load_mw: float = Field(..., description="Forecast under baseline (Normal) conditions (MW)")
    results: List[ScenarioItem] = Field(..., description="List of evaluated scenario outcomes")
    model_used: str = Field("XGBoost (Operational)", description="Model utilized for scenario evaluation")


class HealthResponse(BaseModel):
    """
    Response payload for API and model health status.
    Excludes sensitive system paths.
    """
    status: Literal["healthy", "degraded"] = Field(..., description="Overall service status")
    models_loaded: bool = Field(..., description="Whether model artifacts are successfully loaded in memory")
    loaded_models: List[str] = Field(..., description="List of models available in the service")
    feature_count: int = Field(..., description="Number of expected input features")
    ensemble_method: str = Field(..., description="Configured stacking ensemble method")
    prediction_interval_quantiles: Dict[str, float] = Field(
        ...,
        description="Empirical residual quantiles (q05 and q95) in MW"
    )
    server_time: str = Field(..., description="Current ISO-8601 server UTC timestamp")
    details: Optional[str] = Field(None, description="Status or diagnostic message")
    data_source_mode: Optional[str] = Field("historical", description="Active telemetry source mode")
    active_dataset: Optional[str] = Field("2024", description="Active dataset identifier")
    simulation_enabled: Optional[bool] = Field(False, description="Whether demo simulation mode is enabled")


# ══════════════════════════════════════════════════════════════
# DATA API SCHEMAS (Phase 4)
# ══════════════════════════════════════════════════════════════

class WeatherMeasurements(BaseModel):
    """
    Weather measurements or forecasts.
    """
    model_config = ConfigDict(extra="forbid")
    temperature: float = Field(..., description="Ambient temperature in °C")
    humidity: float = Field(..., description="Relative humidity in %")
    wind_speed: float = Field(..., description="Wind speed in m/s")
    solar_irradiance: float = Field(..., description="Global horizontal solar irradiance in W/m²")

    @field_validator("temperature", "humidity", "wind_speed", "solar_irradiance")
    @classmethod
    def validate_finite(cls, v: float, info) -> float:
        if not math.isfinite(v):
            raise ValueError(f"Field '{info.field_name}' must be a finite real number (got {v})")
        return v


class CalendarAttributes(BaseModel):
    """
    Calendar and holiday binary attributes.
    """
    model_config = ConfigDict(extra="forbid")
    is_weekend: Literal[0, 1] = Field(0, description="1 if weekend (Sat/Sun), else 0")
    is_holiday: Literal[0, 1] = Field(0, description="1 if official holiday, else 0")
    is_festival: Literal[0, 1] = Field(0, description="1 if festival day, else 0")
    is_pre_holiday: Literal[0, 1] = Field(0, description="1 if day prior to holiday, else 0")
    is_post_holiday: Literal[0, 1] = Field(0, description="1 if day following holiday, else 0")
    is_pre_festival: Literal[0, 1] = Field(0, description="1 if day prior to festival, else 0")
    is_post_festival: Literal[0, 1] = Field(0, description="1 if day following festival, else 0")
    workday_after_holiday: Literal[0, 1] = Field(0, description="1 if first working day after holiday, else 0")


class DataRecordResponse(BaseModel):
    """
    Single hourly telemetry/historical record representation.
    """
    timestamp: str = Field(..., description="ISO-8601 observation timestamp (Asia/Kolkata)")
    load_mw: float = Field(..., description="Grid electrical demand in MW")
    weather: WeatherMeasurements = Field(..., description="Weather measurements for the hour")
    calendar: CalendarAttributes = Field(..., description="Calendar attributes for the hour")
    data_source_mode: str = Field("historical", description="Data source mode: 'historical' or 'simulated'")
    is_simulated: bool = Field(False, description="Whether this record is simulated (true) or historical observation (false)")


class HistoryResponse(BaseModel):
    """
    Response payload for historical telemetry query.
    """
    total_records: int = Field(..., description="Total matching records in query range")
    returned_records: int = Field(..., description="Number of records returned in this page")
    limit: int = Field(..., description="Applied pagination limit")
    offset: int = Field(..., description="Applied pagination offset")
    data_source_mode: str = Field("historical", description="Active data source mode")
    is_simulated: bool = Field(False, description="Whether returned records are simulated")
    timezone: str = Field("Asia/Kolkata (UTC+05:30)", description="Reference timezone for timestamps")
    records: List[DataRecordResponse] = Field(..., description="Chronological hourly records")


class GenerateDataRequest(BaseModel):
    """
    Request payload for controlled simulated hourly advance (demo mode).
    """
    model_config = ConfigDict(extra="forbid")
    hours: int = Field(1, ge=1, le=168, description="Number of hourly records to simulate (1 to 168)")
    weather_override: Optional[WeatherMeasurements] = Field(
        None,
        description="Optional custom weather override for generated hours"
    )


class GenerateDataResponse(BaseModel):
    """
    Response payload for demo data generation.
    """
    message: str = Field(..., description="Status summary message")
    generated_count: int = Field(..., description="Number of records generated")
    data_source_mode: str = Field("simulated", description="Data source mode for generated records")
    is_simulated: bool = Field(True, description="Always true for generated demo records")
    records: List[DataRecordResponse] = Field(..., description="List of generated records")


# ══════════════════════════════════════════════════════════════
# 24-HOUR FORECAST SCHEMAS (Phase 4)
# ══════════════════════════════════════════════════════════════

class Forecast24hRequest(BaseModel):
    """
    Request payload for day-ahead 24-hour multi-step recursive forecasting.
    Requires exactly 24 hourly weather forecast items.
    """
    model_config = ConfigDict(extra="forbid")
    weather_forecasts: List[WeatherMeasurements] = Field(
        ...,
        min_length=24,
        max_length=24,
        description="Hourly forecasted weather for the next 24 hours (temperature, humidity, wind_speed, solar_irradiance)"
    )
    calendar_overrides: Optional[List[CalendarAttributes]] = Field(
        None,
        min_length=24,
        max_length=24,
        description="Optional calendar flag overrides for each of the 24 hours"
    )


class HourlyForecastItem(BaseModel):
    """
    Forecast item for a single hour in the 24-hour horizon.
    """
    timestamp: str = Field(..., description="ISO-8601 target forecast timestamp")
    hour: int = Field(..., description="Hour of the day (0-23)")
    predicted_load_mw: float = Field(..., description="Predicted demand in MW")
    spinning_reserve_mw: float = Field(..., description="Operating spinning reserve requirement (20%) in MW")
    lower_bound_mw: float = Field(..., description="Empirical reference lower bound (predicted + q05) in MW")
    upper_bound_mw: float = Field(..., description="Empirical reference upper bound (predicted + q95) in MW")
    is_recursive_lag: bool = Field(..., description="True if lag_1 was fed recursively from prior predicted hours")


class Forecast24hResponse(BaseModel):
    """
    Response payload for genuine day-ahead 24-hour forecast.
    """
    forecast_origin: str = Field(..., description="ISO-8601 forecast origin timestamp (T)")
    forecast_horizon_hours: int = Field(24, description="Forecast horizon length in hours")
    data_source: str = Field(..., description="Telemetry source origin: 'historical' or 'simulated'")
    is_simulated: bool = Field(..., description="Whether forecast origin is from simulated data")
    predictions: List[HourlyForecastItem] = Field(..., description="24 consecutive hourly forecasts")
    interval_confidence_level: float = Field(0.90, description="Nominal confidence level of the residual reference band")
    interval_provenance_note: str = Field(..., description="Explanation of residual quantile origin and multi-step limitations")
    model_used: str = Field("XGBoost (Operational Multi-Step Recursive)", description="Forecasting model and method utilized")


# ══════════════════════════════════════════════════════════════
# ENERGY MANAGEMENT & RISK ANALYSIS SCHEMAS (Phase 5)
# ══════════════════════════════════════════════════════════════

class DispatchHourlyItem(BaseModel):
    """
    Hourly economic dispatch result item.
    """
    hour: int = Field(..., description="Hour of the dispatch horizon (0-23)")
    timestamp: str = Field(..., description="Snapshot timestamp for the hour")
    demand_mw: float = Field(..., description="Grid demand in MW")
    coal_mw: float = Field(..., description="Coal generation output in MW")
    gas_mw: float = Field(..., description="Gas generation output in MW")
    solar_mw: float = Field(..., description="Solar generation output in MW")
    unserved_mw: float = Field(..., description="Unserved energy / shortage in MW")
    hourly_cost: float = Field(..., description="Total generation and penalty cost for the hour in INR (₹)")


class DispatchRequest(BaseModel):
    """
    Request payload for 24-hour PyPSA economic dispatch.
    Accepts either an explicit 24-hour demand profile or activates Phase 4 24h forecasting.
    """
    model_config = ConfigDict(extra="forbid")
    demand_mw: Optional[List[float]] = Field(
        None,
        description="Optional 24 hourly demand values in MW. If provided, must contain exactly 24 non-negative finite numbers."
    )
    use_forecast_24h: bool = Field(
        False,
        description="Whether to generate and consume day-ahead 24h recursive forecast from Phase 4 data service."
    )
    weather_forecasts: Optional[List[WeatherMeasurements]] = Field(
        None,
        description="24 hourly weather forecasts (required if use_forecast_24h is True)."
    )
    calendar_overrides: Optional[List[CalendarAttributes]] = Field(
        None,
        description="Optional calendar flag overrides if use_forecast_24h is True."
    )
    solar_profile: Optional[List[float]] = Field(
        None,
        description="Optional 24 hourly solar availability factors [0.0 - 1.0]. Defaults to standard diurnal solar profile."
    )
    include_load_shedding: bool = Field(
        False,
        description="Whether to include unserved energy slack generator to prevent mathematical infeasibility."
    )
    load_shedding_cost: float = Field(
        100.0,
        gt=0.0,
        description="Penalty cost per MWh of unmet load (₹/MWh). Default ₹100/MWh."
    )

    @field_validator("demand_mw")
    @classmethod
    def validate_demand(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        if v is not None:
            if len(v) != 24:
                raise ValueError(f"Demand profile must contain exactly 24 hourly values, got {len(v)}")
            for idx, val in enumerate(v):
                if not math.isfinite(val):
                    raise ValueError(f"Demand value at index {idx} must be a finite number (got {val})")
                if val < 0.0:
                    raise ValueError(f"Demand value at index {idx} must be non-negative (got {val})")
        return v

    @field_validator("solar_profile")
    @classmethod
    def validate_solar(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        if v is not None:
            if len(v) != 24:
                raise ValueError(f"Solar profile must contain exactly 24 hourly values, got {len(v)}")
            for idx, val in enumerate(v):
                if not math.isfinite(val):
                    raise ValueError(f"Solar profile value at index {idx} must be finite (got {val})")
                if val < 0.0 or val > 1.0:
                    raise ValueError(f"Solar profile value at index {idx} must be between 0.0 and 1.0 (got {val})")
        return v


class DispatchResponse(BaseModel):
    """
    Response payload for 24-hour PyPSA economic dispatch.
    """
    optimization_status: str = Field(..., description="PyPSA / HiGHS optimization outcome status")
    is_optimal: bool = Field(..., description="Whether optimization achieved optimal status")
    is_feasible: bool = Field(..., description="Whether a valid dispatch schedule was successfully resolved")
    total_cost: float = Field(..., description="Total system dispatch cost over 24 hours in INR (₹)")
    currency: str = Field("INR (₹)", description="Currency unit for costs")
    peak_demand_mw: float = Field(..., description="Maximum hourly grid demand across the horizon (MW)")
    total_demand_mwh: float = Field(..., description="Total energy demanded over 24 hours (MWh)")
    coal_total_mwh: float = Field(..., description="Total coal energy dispatched (MWh)")
    gas_total_mwh: float = Field(..., description="Total gas energy dispatched (MWh)")
    solar_total_mwh: float = Field(..., description="Total solar energy dispatched (MWh)")
    unserved_energy_mwh: float = Field(..., description="Total unserved energy / load shed (MWh)")
    coal_share_pct: float = Field(..., description="Coal share of total generation supply (%)")
    gas_share_pct: float = Field(..., description="Gas share of total generation supply (%)")
    solar_share_pct: float = Field(..., description="Solar share of total generation supply (%)")
    hourly_schedule: List[DispatchHourlyItem] = Field(..., description="24-hour unit dispatch schedule")
    solver_used: str = Field(..., description="Optimization formulation and solver utilized")
    provenance_note: str = Field(..., description="Technical specification and fleet constraints description")


class RiskMetricsSummary(BaseModel):
    """
    Reliability and economic risk indicators from Monte Carlo simulation.
    """
    LOLP: float = Field(..., description="Loss of Load Probability [0.0 - 1.0] across simulated scenarios")
    EENS_mwh: float = Field(..., description="Expected Energy Not Served across all scenarios in MWh")
    reserve_sufficiency: float = Field(..., description="Reserve sufficiency probability (1.0 - LOLP)")
    mean_cost: float = Field(..., description="Expected dispatch cost across scenarios in INR (₹)")
    median_cost: float = Field(..., description="Median dispatch cost across scenarios in INR (₹)")
    min_cost: float = Field(..., description="Best-case dispatch cost across scenarios in INR (₹)")
    max_cost: float = Field(..., description="Worst-case dispatch cost across scenarios in INR (₹)")
    p95_cost: float = Field(..., description="95th percentile dispatch cost in INR (₹)")
    peak_demand_p95_mw: float = Field(..., description="95th percentile peak demand in MW")


class MetricQuantiles(BaseModel):
    """
    Empirical quantile bounds across Monte Carlo scenarios for a key metric.
    """
    metric: str = Field(..., description="Metric description and unit")
    p05: float = Field(..., description="5th percentile empirical scenario bound")
    p50: float = Field(..., description="50th percentile (median) scenario bound")
    p95: float = Field(..., description="95th percentile empirical scenario bound")


class BaseForecastSummary(BaseModel):
    """
    Summary statistics of the base 24-hour demand forecast.
    """
    peak_demand_mw: float = Field(..., description="Peak load in base demand forecast (MW)")
    mean_demand_mw: float = Field(..., description="Mean load in base demand forecast (MW)")
    total_demand_mwh: float = Field(..., description="Total energy demanded over 24 hours (MWh)")


class RiskAnalysisRequest(BaseModel):
    """
    Request payload for Monte Carlo uncertainty and reliability risk assessment.
    """
    model_config = ConfigDict(extra="forbid")
    demand_mw: Optional[List[float]] = Field(
        None,
        description="Optional 24 hourly base demand forecast in MW."
    )
    use_forecast_24h: bool = Field(
        False,
        description="Whether to generate and consume day-ahead 24h recursive forecast from Phase 4 data service."
    )
    weather_forecasts: Optional[List[WeatherMeasurements]] = Field(
        None,
        description="24 hourly weather forecasts (required if use_forecast_24h is True)."
    )
    calendar_overrides: Optional[List[CalendarAttributes]] = Field(
        None,
        description="Optional calendar flag overrides if use_forecast_24h is True."
    )
    n_simulations: int = Field(
        100,
        ge=10,
        le=1000,
        description="Number of Monte Carlo scenarios (10 to 1,000). Default is 100."
    )
    random_seed: Optional[int] = Field(
        42,
        description="Optional random seed for reproducible residual bootstrapping."
    )
    dispatch_engine: Literal["merit_order", "pypsa"] = Field(
        "merit_order",
        description="Dispatch algorithm: 'merit_order' (fast, recommended for API latency) or 'pypsa' (rigorous LP)."
    )
    enable_residual_clipping: bool = Field(
        True,
        description="Whether to clip residual bootstrap tails between 5th and 95th percentiles."
    )
    load_shedding_cost: float = Field(
        100.0,
        gt=0.0,
        description="Penalty cost per MWh of unmet load (₹/MWh). Default ₹100/MWh."
    )
    solar_profile: Optional[List[float]] = Field(
        None,
        description="Optional 24 hourly solar availability factors [0.0 - 1.0]."
    )

    @field_validator("demand_mw")
    @classmethod
    def validate_demand(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        if v is not None:
            if len(v) != 24:
                raise ValueError(f"Demand profile must contain exactly 24 hourly values, got {len(v)}")
            for idx, val in enumerate(v):
                if not math.isfinite(val):
                    raise ValueError(f"Demand value at index {idx} must be a finite number (got {val})")
                if val < 0.0:
                    raise ValueError(f"Demand value at index {idx} must be non-negative (got {val})")
        return v

    @field_validator("solar_profile")
    @classmethod
    def validate_solar(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        if v is not None:
            if len(v) != 24:
                raise ValueError(f"Solar profile must contain exactly 24 hourly values, got {len(v)}")
            for idx, val in enumerate(v):
                if not math.isfinite(val):
                    raise ValueError(f"Solar profile value at index {idx} must be finite (got {val})")
                if val < 0.0 or val > 1.0:
                    raise ValueError(f"Solar profile value at index {idx} must be between 0.0 and 1.0 (got {val})")
        return v


class RiskAnalysisResponse(BaseModel):
    """
    Response payload for Monte Carlo uncertainty and risk assessment.
    """
    n_simulations: int = Field(..., description="Number of Monte Carlo simulations evaluated")
    random_seed: Optional[int] = Field(..., description="Seed used for simulation draws")
    dispatch_engine: str = Field(..., description="Dispatch engine used across scenarios")
    base_forecast_summary: BaseForecastSummary = Field(..., description="Summary of the 24-hour base demand profile")
    risk_metrics: RiskMetricsSummary = Field(..., description="Grid reliability and economic risk metrics")
    quantile_summaries: List[MetricQuantiles] = Field(..., description="Quantile tables (p05, p50, p95) across key variables")
    uncertainty_provenance_note: str = Field(..., description="Technical note describing bootstrap source and quantile interpretation")


