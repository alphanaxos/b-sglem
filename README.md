# Bengaluru Smart Grid Load Forecasting and Energy Management System (B-SGLEM)

Hybrid Machine Learning Forecasting System using XGBoost, BiLSTM, and CNN-LSTM with Bias-Corrected Convex Stacking and Operational Data API.

---

## System Overview

B-SGLEM is an intelligent power system load forecasting platform tailored for the Bengaluru electricity grid. The project integrates:
* **Hybrid Ensemble ML:** Tuned XGBoost regressor, Bidirectional LSTM, and Conv1D-LSTM stacked via convex non-negative weights $[0.35, 0.00, 0.65]$ and out-of-fold bias correction.
* **Operational ML API (Phase 3):** High-performance FastAPI service providing next-hour operational forecasts, 20% spinning reserve calculations, and scenario stress testing.
* **Data API & Pipeline (Phase 4):** Resilient historical telemetry ingestion, chronological querying, opt-in controlled demo simulation, feature engineering pipeline without target leakage, and genuine 24-hour day-ahead recursive forecasting.
* **Energy Management & Risk (Phase 5):** PyPSA linear optimal power flow 24-hour economic dispatch and Monte Carlo uncertainty risk analysis (LOLP, EENS, reserve sufficiency, quantile tables) with seamless Phase 4 day-ahead forecast integration.
* **Interactive Dashboard (Phase 6):** Streamlit and Plotly preliminary demonstration dashboard communicating with the FastAPI backend across telemetry, forecasting, stress testing, economic dispatch, and risk assessment workflows.

---

## Project Structure

```
├── config.py              # Central configuration (paths, hyperparameters, features)
├── train.py               # Model training and artifact serialization orchestrator
├── inference.py           # Standalone multi-model batch inference
├── validate_inference.py  # Cross-process serialization validation
├── api/                   # FastAPI application package
│   ├── app.py             # App factory and lifespan manager
│   ├── dependencies.py    # Forecaster and DataService dependency injectors
│   ├── schemas.py         # Pydantic v2 validation models
│   └── routes/            # Route handlers
│       ├── health.py      # GET /health
│       ├── predict.py     # POST /predict
│       ├── scenario.py    # POST /predict/scenario
│       ├── data.py        # GET /data/latest, GET /data/history, POST /data/generate
│       ├── forecast.py    # POST /forecast/24h
│       └── energy.py      # POST /energy/dispatch, POST /energy/risk
├── dashboard/             # Streamlit interactive UI package (Phase 6)
│   ├── app.py             # Dashboard main application & page orchestrator
│   ├── api_client.py      # Robust HTTP client communicating with backend
│   └── components.py      # Plotly interactive chart builders and sample inputs
├── services/              # Domain logic services
│   ├── artifact_manager.py# Artifact persistence and deserialization
│   ├── prediction.py      # LoadForecaster service
│   ├── scenario_engine.py # Operational scenario evaluation service
│   ├── data_service.py    # Historical data pipeline, demo simulation & 24h recursive forecast
│   ├── energy_service.py  # Energy dispatch & Monte Carlo risk orchestration service
│   ├── pypsa_dispatch.py  # PyPSA unit commitment / dispatch optimization
│   └── monte_carlo.py     # Monte Carlo risk and uncertainty simulation
├── data/                  # Historical telemetry datasets (2023, 2024)
├── features/              # Feature engineering formulas
├── models/                # Base learner model architectures & stacking
├── models_saved/          # Serialized production model artifacts
├── docs/                  # API and system documentation
└── tests/                 # Automated pytest test suites
```

---

## Quickstart

### 1. Installation

```bash
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install tested dependencies
pip install -r requirements.txt
```

### 2. Running the Backend API (Terminal 1)

```bash
# Operational backend startup (binds port 8000)
uvicorn api.app:app --host 127.0.0.1 --port 8000

# Optional: enable demo simulation advance mode
$env:ENABLE_SIMULATION_MODE="true"; uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload
```

Interactive API documentation:
* **Swagger UI:** `http://127.0.0.1:8000/docs`
* **Redoc UI:** `http://127.0.0.1:8000/redoc`

### 3. Running the Streamlit Dashboard (Terminal 2)

```bash
# Run the interactive dashboard (binds port 8501)
streamlit run dashboard/app.py

# Optional: configure custom backend URL if running on a different port or host
$env:BSGLEM_API_URL="http://127.0.0.1:8000"; streamlit run dashboard/app.py
```

---

## API Endpoints Summary

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service and model status, data mode, and configuration metadata. |
| `POST` | `/predict` | Operational next-hour load forecast using the single-row XGBoost path. |
| `POST` | `/predict/scenario` | What-if scenario analysis (`Normal`, `Heatwave`, `Festival`, `Rainy Day`, `Night Peak`). |
| `GET` | `/data/latest` | Most recent telemetry record (distinguishes historical vs simulated). |
| `GET` | `/data/history` | Chronological telemetry queries with date range filtering and pagination. |
| `POST` | `/data/generate` | Advances simulated hourly telemetry in controlled demo mode (requires `ENABLE_SIMULATION_MODE=true`). |
| `POST` | `/forecast/24h` | Authentic 24-hour day-ahead recursive forecast using future weather forecasts. |
| `POST` | `/energy/dispatch` | 24-hour PyPSA economic dispatch optimizing Coal, Gas, Solar, and optional Load Shedding. |
| `POST` | `/energy/risk` | Monte Carlo uncertainty risk analysis evaluating LOLP, EENS, and empirical quantile bounds. |

> **Notice on Telemetry Modes:** The APIs and dashboard operate on configured project inputs and historical models. They do **not** represent live grid SCADA telemetry or a direct connection to BESCOM or SLDC.

---

## Testing & Validation

```powershell
# Run the complete test suite (50 tests)
.\.venv\Scripts\python -m pytest tests/ -v

# Run Phase 2 model serialization validation
.\.venv\Scripts\python validate_inference.py data
```

Both suites run with 100% pass rates without retraining models or modifying original datasets.


