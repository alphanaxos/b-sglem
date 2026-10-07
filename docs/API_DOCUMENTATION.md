# B-SGLEM Phase 4 Data API & Load Forecasting Documentation

**Bengaluru Smart Grid Load Forecasting and Energy Management System**  
*Phase 4: Data API, Historical Pipeline, Demo Simulation & 24-Hour Day-Ahead Forecasting*

---

## 1. Overview & Architectural Principles

Phase 4 introduces a dedicated, resilient historical data pipeline and telemetry API along with an authentic 24-hour day-ahead recursive forecasting engine:

* **Separation of Telemetry Modes:** Explicitly supports `historical` CSV playback, controlled `simulated` demo mode, and explicitly marks `live` mode as not implemented (no fake claims of BESCOM/SLDC SCADA access).
* **Controlled Demo Simulation:** Advancing simulated hourly observations is strictly **opt-in** (`ENABLE_SIMULATION_MODE=true`). Demo generation **never modifies** historical CSV files and clearly tags all generated records as `is_simulated=true`.
* **Zero Target Leakage Feature Pipeline:** Feature preparation computes lags and 24-hour rolling statistics exclusively from preceding historical load observations ($T-168 \dots T-1$). The load at target forecast hour $T$ is never used.
* **Exact Canonical Contract:** Prepares precisely 28 features in the canonical sequence defined by `config.FEATURE_COLS`.
* **Authentic 24-Hour Forecasting:** `POST /forecast/24h` generates future predictions across hours $T+1 \dots T+24$. Lag features are dynamically updated using prior predicted loads in a recursive multi-step loop.
* **Preservation of Phase 2 & Phase 3:** All Phase 3 endpoints (`/health`, `/predict`, `/predict/scenario`) and Phase 2 validation scripts remain 100% operational with bit-level reproducibility.

---

## 2. Installation & Server Startup

### Environment Setup

The service runs on **Python 3.12.10** with dependencies in `requirements.txt`:

```bash
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### Configuration Options (Environment Variables)

| Variable | Default | Description |
|---|---|---|
| `DATA_SOURCE_MODE` | `historical` | Telemetry mode: `historical`, `simulated`, or `live`. |
| `ACTIVE_DATASET` | `2024` | Historical dataset selection: `2024`, `2023`, or `combined`. |
| `ENABLE_SIMULATION_MODE` | `false` | Set to `true` to permit `POST /data/generate` demo data generation. |
| `DATA_DIR` | `<project>/data` | Directory containing historical CSVs. |
| `MODEL_DIR` | `<project>/models_saved` | Directory containing serialized model artifacts. |

### Starting the Server

```bash
# Operational startup (binds port 8000)
uvicorn api.app:app --host 0.0.0.0 --port 8000

# Enable simulation mode for demo sessions
$env:ENABLE_SIMULATION_MODE="true"; uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload
```

Interactive documentation:
* **Swagger UI:** `http://127.0.0.1:8000/docs`
* **Redoc UI:** `http://127.0.0.1:8000/redoc`

---

## 3. Canonical 28-Feature Contract

The XGBoost regressor and stacked ensemble require exactly **28 engineered features** in canonical order:

| # | Feature Column | Type | Allowed Format | Provenance / Formula |
|---|---|---|---|---|
| 1 | `is_weekend` | Integer | `0` or `1` | 1 if Saturday or Sunday |
| 2 | `is_holiday` | Integer | `0` or `1` | 1 if public holiday |
| 3 | `is_festival` | Integer | `0` or `1` | 1 if festival day |
| 4 | `is_pre_holiday` | Integer | `0` or `1` | 1 if day before holiday |
| 5 | `is_post_holiday` | Integer | `0` or `1` | 1 if day after holiday |
| 6 | `is_pre_festival` | Integer | `0` or `1` | 1 if day before festival |
| 7 | `is_post_festival` | Integer | `0` or `1` | 1 if day after festival |
| 8 | `workday_after_holiday` | Integer | `0` or `1` | 1 if first working day after holiday |
| 9 | `temperature` | Float | Finite (°C) | Hourly ambient temperature |
| 10 | `humidity` | Float | Finite (%) | Relative humidity |
| 11 | `wind_speed` | Float | Finite (m/s) | Wind speed |
| 12 | `solar_irradiance` | Float | Finite (W/m²) | Global horizontal solar irradiance |
| 13 | `lag_1` | Float | Finite (MW) | Historical load at $T-1$ hour |
| 14 | `lag_24` | Float | Finite (MW) | Historical load at $T-24$ hours |
| 15 | `lag_168` | Float | Finite (MW) | Historical load at $T-168$ hours (1 week prior) |
| 16 | `rolling_mean_24` | Float | Finite (MW) | Mean of loads from $T-24$ to $T-1$ |
| 17 | `rolling_std_24` | Float | Finite (MW) | Sample std (`ddof=1`) of loads from $T-24$ to $T-1$ |
| 18 | `hour_sin` | Float | $[-1, 1]$ | $\sin(2\pi \cdot \text{hour} / 24)$ |
| 19 | `hour_cos` | Float | $[-1, 1]$ | $\cos(2\pi \cdot \text{hour} / 24)$ |
| 20 | `day_sin` | Float | $[-1, 1]$ | $\sin(2\pi \cdot \text{day\_of\_week} / 7)$ |
| 21 | `day_cos` | Float | $[-1, 1]$ | $\cos(2\pi \cdot \text{day\_of\_week} / 7)$ |
| 22 | `month_sin` | Float | $[-1, 1]$ | $\sin(2\pi \cdot \text{month} / 12)$ |
| 23 | `month_cos` | Float | $[-1, 1]$ | $\cos(2\pi \cdot \text{month} / 12)$ |
| 24 | `lag_48` | Float | Finite (MW) | Historical load at $T-48$ hours |
| 25 | `lag_72` | Float | Finite (MW) | Historical load at $T-72$ hours |
| 26 | `rolling_max_24` | Float | Finite (MW) | Maximum of loads from $T-24$ to $T-1$ |
| 27 | `rolling_min_24` | Float | Finite (MW) | Minimum of loads from $T-24$ to $T-1$ |
| 28 | `temp_x_hour_sin` | Float | Finite | $\text{temperature} \times \text{hour\_sin}$ |

---

## 4. Complete API Endpoints

### 4.1 System & Model Health: `GET /health`

Reports service status, loaded model artifacts, feature count, data source mode, and prediction interval quantiles. Redacts internal file paths.

#### Response (`200 OK`)
```json
{
  "status": "healthy",
  "models_loaded": true,
  "loaded_models": ["XGBoost", "BiLSTM", "CNN-LSTM"],
  "feature_count": 28,
  "ensemble_method": "bias_corrected_convex_weighted_ensemble",
  "prediction_interval_quantiles": {
    "q05": -146.8706,
    "q95": 145.0495
  },
  "server_time": "2026-10-06T18:00:00.000000Z",
  "details": "All models and scalers loaded in memory and ready for inference.",
  "data_source_mode": "historical",
  "active_dataset": "2024",
  "simulation_enabled": false
}
```

---

### 4.2 Latest Telemetry Record: `GET /data/latest`

Returns the most recent hourly telemetry record. Clearly indicates whether the observation is historical or simulated.

#### Response (`200 OK`)
```json
{
  "timestamp": "2024-12-31T23:00:00",
  "load_mw": 2089.44,
  "weather": {
    "temperature": 22.39,
    "humidity": 68.81,
    "wind_speed": 1.57,
    "solar_irradiance": 117.47
  },
  "calendar": {
    "is_weekend": 0,
    "is_holiday": 0,
    "is_festival": 0,
    "is_pre_holiday": 0,
    "is_post_holiday": 0,
    "is_pre_festival": 0,
    "is_post_festival": 0,
    "workday_after_holiday": 0
  },
  "data_source_mode": "historical",
  "is_simulated": false
}
```

#### Example cURL
```bash
curl -X GET http://127.0.0.1:8000/data/latest
```

---

### 4.3 Historical Query & Pagination: `GET /data/history`

Retrieves chronological historical telemetry records with query range filtering and pagination.

#### Query Parameters
* `start` (optional string): Start timestamp (ISO-8601, e.g. `2024-06-01T00:00:00`).
* `end` (optional string): End timestamp (ISO-8601, e.g. `2024-06-02T23:00:00`).
* `limit` (integer, default `100`, range `1-1000`): Maximum records to return.
* `offset` (integer, default `0`, min `0`): Pagination offset.

#### Response (`200 OK`)
```json
{
  "total_records": 48,
  "returned_records": 48,
  "limit": 100,
  "offset": 0,
  "data_source_mode": "historical",
  "is_simulated": false,
  "timezone": "Asia/Kolkata (UTC+05:30)",
  "records": [
    {
      "timestamp": "2024-06-01T00:00:00",
      "load_mw": 2012.35,
      "weather": { "temperature": 23.4, "humidity": 65.0, "wind_speed": 1.8, "solar_irradiance": 0.0 },
      "calendar": { "is_weekend": 1, "is_holiday": 0, ... },
      "data_source_mode": "historical",
      "is_simulated": false
    }
  ]
}
```

#### Example cURL
```bash
curl -X GET "http://127.0.0.1:8000/data/history?start=2024-06-01T00:00:00&end=2024-06-02T23:00:00&limit=50"
```

---

### 4.4 Demo Data Generation: `POST /data/generate`

Advances simulated hourly observations in controlled demo mode.
* **Protection:** Rejects requests with `403 Forbidden` unless `ENABLE_SIMULATION_MODE=true` is set.
* **Safety:** Does **not** modify original CSV files. Simulated records reside in memory and preserve timestamp continuity.

#### Request Body
```json
{
  "hours": 3,
  "weather_override": {
    "temperature": 26.0,
    "humidity": 55.0,
    "wind_speed": 2.5,
    "solar_irradiance": 300.0
  }
}
```

#### Response (`200 OK`)
```json
{
  "message": "Successfully generated 3 simulated hourly observation(s).",
  "generated_count": 3,
  "data_source_mode": "simulated",
  "is_simulated": true,
  "records": [
    {
      "timestamp": "2025-01-01T00:00:00",
      "load_mw": 1678.86,
      "weather": { "temperature": 26.0, "humidity": 55.0, "wind_speed": 2.5, "solar_irradiance": 300.0 },
      "calendar": { "is_weekend": 0, "is_holiday": 0, ... },
      "data_source_mode": "simulated",
      "is_simulated": true
    }
  ]
}
```

---

### 4.5 Day-Ahead 24-Hour Forecast: `POST /forecast/24h`

Performs an authentic 24-hour day-ahead recursive forecast.
* **Requirements:** Exactly 24 hourly weather forecast items (`temperature`, `humidity`, `wind_speed`, `solar_irradiance`) must be provided.
* **Recursive Lag Updating:** At step $k=1$, lag features use actual history. At steps $k=2 \dots 24$, `lag_1` is updated with predicted load $\hat{y}_{T+k-1}$, and rolling statistics roll forward to incorporate predictions.
* **Provenance:** Clearly states that residual quantiles provide an empirical reference band from stacked OOF residuals, not calibrated intervals for recursive forecasting.

#### Request Body
```json
{
  "weather_forecasts": [
    { "temperature": 21.0, "humidity": 70.0, "wind_speed": 1.8, "solar_irradiance": 0.0 },
    { "temperature": 20.5, "humidity": 73.0, "wind_speed": 1.6, "solar_irradiance": 0.0 },
    ... (24 hourly items)
  ]
}
```

#### Response (`200 OK`)
```json
{
  "forecast_origin": "2024-12-31T23:00:00",
  "forecast_horizon_hours": 24,
  "data_source": "historical",
  "is_simulated": false,
  "predictions": [
    {
      "timestamp": "2025-01-01T00:00:00",
      "hour": 0,
      "predicted_load_mw": 1985.42,
      "spinning_reserve_mw": 397.08,
      "lower_bound_mw": 1838.55,
      "upper_bound_mw": 2130.47,
      "is_recursive_lag": false
    },
    {
      "timestamp": "2025-01-01T01:00:00",
      "hour": 1,
      "predicted_load_mw": 1942.15,
      "spinning_reserve_mw": 388.43,
      "lower_bound_mw": 1795.28,
      "upper_bound_mw": 2087.20,
      "is_recursive_lag": true
    }
  ],
  "interval_confidence_level": 0.90,
  "interval_provenance_note": "Recursive multi-step forecast using the operational XGBoost model. Bounds calculated using empirical 5th and 95th percentile residuals (-146.87 / +145.05 MW) from the stacked OOF ensemble. These provide an empirical reference band but are not statistically calibrated prediction intervals for recursive multi-step forecasting.",
  "model_used": "XGBoost (Operational Multi-Step Recursive)"
}
```

---

### 4.6 Operational Next-Hour Prediction: `POST /predict` (Phase 3 Preserved)

Single-row operational point forecast using the pre-loaded XGBoost model.

#### Request Body
```json
{
  "features": { ... 28 canonical features ... },
  "timestamp": "2026-10-06T23:00:00Z"
}
```

#### Response (`200 OK`)
```json
{
  "predicted_load_mw": 2070.23,
  "spinning_reserve_mw": 414.05,
  "lower_bound_mw": 1923.36,
  "upper_bound_mw": 2215.28,
  "interval_confidence_level": 0.90,
  "interval_provenance_note": "Bounds calculated using empirical 5th and 95th percentile residuals (-146.87 / +145.05 MW) derived from the stacked out-of-fold ensemble. This offset provides an empirical reference band but is not a statistically calibrated prediction interval for the standalone XGBoost regressor.",
  "model_used": "XGBoost (Operational Next-Hour)",
  "timestamp": "2026-10-06T23:00:00Z"
}
```

---

### 4.7 Operational Scenario Simulation: `POST /predict/scenario` (Phase 3 Preserved)

Runs what-if perturbations (`Normal`, `Heatwave (+5°C)`, `Festival Load`, `Rainy Day`, `Night Peak (22:00)`) against a baseline feature row.

---

### 4.8 Economic Power Dispatch: `POST /energy/dispatch` (Phase 5)

Solves a 24-hour linear optimal power flow (LOPF) unit commitment / dispatch optimization using **PyPSA** and the **HiGHS** linear programming solver. Minimizes total generation costs while adhering to physical generator capacities, ramping, minimum stable output, and solar profiles.

#### Operational Fleet Characteristics:
* **Coal Generator:** $1,500\text{ MW}$ capacity, $40\%$ minimum stable output ($600\text{ MW}$), marginal cost $₹3.5/\text{unit}$ ($₹3,500/\text{MWh}$).
* **Gas Peaker:** $800\text{ MW}$ capacity, fully flexible ($0\text{ MW}$ minimum), marginal cost $₹6.0/\text{unit}$ ($₹6,000/\text{MWh}$).
* **Solar Farm:** $400\text{ MW}$ nameplate capacity, diurnal solar availability curve, marginal cost $₹0.0/\text{unit}$.
* **Load Shedding / Shortage (Optional):** Slack generator at configurable penalty (default $₹100.0/\text{MWh}$).

#### Input Options:
Clients can supply an explicit 24-hour demand profile (`demand_mw`) or invoke automatic integration with the Phase 4 day-ahead recursive forecast (`use_forecast_24h=true` with `weather_forecasts`).

#### Request Body Example (Explicit Demand)
```json
{
  "demand_mw": [
    1650.0, 1580.0, 1520.0, 1490.0, 1510.0, 1600.0,
    1780.0, 1950.0, 2100.0, 2180.0, 2220.0, 2200.0,
    2150.0, 2100.0, 2080.0, 2050.0, 2100.0, 2240.0,
    2290.0, 2250.0, 2150.0, 2000.0, 1850.0, 1720.0
  ],
  "include_load_shedding": false
}
```

#### Response (`200 OK`)
```json
{
  "optimization_status": "optimal",
  "is_optimal": true,
  "is_feasible": true,
  "total_cost": 172450.0,
  "currency": "INR (₹)",
  "peak_demand_mw": 2290.0,
  "total_demand_mwh": 47290.0,
  "coal_total_mwh": 36000.0,
  "gas_total_mwh": 8790.0,
  "solar_total_mwh": 2500.0,
  "unserved_energy_mwh": 0.0,
  "coal_share_pct": 76.13,
  "gas_share_pct": 18.59,
  "solar_share_pct": 5.28,
  "hourly_schedule": [
    {
      "hour": 0,
      "timestamp": "2025-01-01T00:00:00",
      "demand_mw": 1650.0,
      "coal_mw": 1500.0,
      "gas_mw": 150.0,
      "solar_mw": 0.0,
      "unserved_mw": 0.0,
      "hourly_cost": 6150.0
    }
  ],
  "solver_used": "HiGHS / PyPSA Linear Optimal Power Flow",
  "provenance_note": "24-hour economic dispatch solved using PyPSA (Linear Optimal Power Flow with HiGHS solver). Generation fleet includes Coal (1,500 MW, 40% min stable, ₹3.5/unit), Gas (800 MW flexible, ₹6.0/unit), Solar (400 MW nameplate, ₹0.0/unit), and optional Load Shedding (₹100/MWh unserved penalty)."
}
```

---

### 4.9 Monte Carlo Risk & Reliability Analysis: `POST /energy/risk` (Phase 5)

Performs probabilistic uncertainty simulation over a 24-hour operational planning horizon using **residual bootstrapping** from the stacked out-of-fold ensemble.

#### Uncertainty Methodology & Limitations:
* **Residual Bootstrapping:** Draws 24-hour residual sequences using sliding-window block bootstrapping from the out-of-fold stacked ensemble residuals (`residuals.pkl`).
* **Tail Clipping:** Residual tails are clipped between the 5th and 95th percentiles (`-146.87 MW` to `+145.05 MW`) by default to prevent non-physical extreme spikes.
* **Dispatch Engine Selection:** Supports `merit_order` (fast analytical LP solution, ~15ms for 100 simulations; recommended for interactive API queries) or `pypsa` (rigorous numerical LP per simulation).
* **Provenance & Interpretation:** Quantile bounds represent **empirical scenario bounds** across simulated trajectories and must **not** be interpreted as statistically calibrated confidence or prediction intervals.

#### Request Body Example
```json
{
  "demand_mw": [
    1650.0, 1580.0, 1520.0, 1490.0, 1510.0, 1600.0,
    1780.0, 1950.0, 2100.0, 2180.0, 2220.0, 2200.0,
    2150.0, 2100.0, 2080.0, 2050.0, 2100.0, 2240.0,
    2290.0, 2250.0, 2150.0, 2000.0, 1850.0, 1720.0
  ],
  "n_simulations": 100,
  "random_seed": 42,
  "dispatch_engine": "merit_order",
  "enable_residual_clipping": true
}
```

#### Response (`200 OK`)
```json
{
  "n_simulations": 100,
  "random_seed": 42,
  "dispatch_engine": "merit_order",
  "base_forecast_summary": {
    "peak_demand_mw": 2290.0,
    "mean_demand_mw": 1970.42,
    "total_demand_mwh": 47290.0
  },
  "risk_metrics": {
    "LOLP": 0.0,
    "EENS_mwh": 0.0,
    "reserve_sufficiency": 1.0,
    "mean_cost": 172948.50,
    "median_cost": 172780.00,
    "min_cost": 168240.00,
    "max_cost": 178120.00,
    "p95_cost": 176450.00,
    "peak_demand_p95_mw": 2355.20
  },
  "quantile_summaries": [
    {
      "metric": "Demand peak (MW)",
      "p05": 2210.45,
      "p50": 2290.00,
      "p95": 2355.20
    },
    {
      "metric": "Dispatch cost (₹)",
      "p05": 169350.00,
      "p50": 172780.00,
      "p95": 176450.00
    },
    {
      "metric": "Coal usage (MWh)",
      "p05": 36000.00,
      "p50": 36000.00,
      "p95": 36000.00
    },
    {
      "metric": "Gas usage (MWh)",
      "p05": 8200.00,
      "p50": 8790.00,
      "p95": 9420.00
    },
    {
      "metric": "Solar usage (MWh)",
      "p05": 2500.00,
      "p50": 2500.00,
      "p95": 2500.00
    },
    {
      "metric": "Unserved energy (MWh)",
      "p05": 0.0,
      "p50": 0.0,
      "p95": 0.0
    }
  ],
  "uncertainty_provenance_note": "Probabilistic risk metrics evaluated via residual bootstrapping from the stacked out-of-fold ensemble. Tail clipping applied between 5th and 95th empirical percentiles. Reported quantiles represent empirical scenario bounds across simulated trajectories and must not be interpreted as statistically calibrated prediction intervals."
}
```

---

## 5. Verification & Test Suite

Run the full automated regression test suite:

```powershell
# Run all 50 unit, data, forecast, dispatch, risk, and dashboard tests
.\.venv\Scripts\python -m pytest tests/ -v

# Run Phase 2 model serialization validation
.\.venv\Scripts\python validate_inference.py data
```

**Results:**
* `pytest tests/`: **50 passed out of 50** ($100\%$ pass rate across Phase 3, Phase 4, Phase 5, and Phase 6).
* `validate_inference.py`: **All 7 checks passed** with bit-level match against training references and zero model retraining.

---

## 6. Interactive Academic Dashboard (Phase 6)

The Streamlit dashboard (`dashboard/app.py`) provides an interactive interface to demonstrate and test the system.

### Starting the Dashboard
```bash
# Terminal 1: Start FastAPI backend
uvicorn api.app:app --host 127.0.0.1 --port 8000

# Terminal 2: Start Streamlit dashboard
streamlit run dashboard/app.py
```

### Dashboard Architecture
* **Decoupled Architecture:** The dashboard contains zero machine learning, feature engineering, PyPSA optimization, or Monte Carlo code. All computation is executed via HTTP requests against the FastAPI backend.
* **Configuration:** Backend base URL is configured via the `BSGLEM_API_URL` environment variable (default: `http://127.0.0.1:8000`).
* **Interactive Pages:**
  1. **System Overview:** Displays backend health, active dataset mode, latest telemetry observation (historical vs. simulated), and past 7 days of load history via Plotly.
  2. **24-Hour Forecasting:** Editable 24-hour weather inputs table, calling `POST /forecast/24h` to display peak demand, 20% spinning reserve, and empirical scenario bounds.
  3. **Scenario Analysis:** What-if stress testing across standard perturbations (`Heatwave`, `Festival`, `Rainy Day`, `Night Peak`).
  4. **PyPSA Economic Dispatch:** 24-hour generation scheduling across Coal, Gas, Solar, and optional Load Shedding, displaying feasibility and stacked dispatch bar charts.
  5. **Monte Carlo Risk Analysis:** Residual-bootstrapped reliability assessment evaluating LOLP, EENS, reserve sufficiency, and empirical quantile ranges.
  6. **Demo Data Controls:** Safe controls to advance simulated demo telemetry (when `ENABLE_SIMULATION_MODE=true`).


