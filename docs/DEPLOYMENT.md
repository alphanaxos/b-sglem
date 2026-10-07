# B-SGLEM Cloud Deployment Runbook (Render)

**Project:** Bengaluru Smart Grid Load Forecasting and Energy Management System Using Hybrid Machine Learning (B-SGLEM)  
**Target Platform:** Render (Docker Web Services via `render.yaml`)

---

## 1. Architecture Overview

B-SGLEM is deployed as a decoupled two-tier architecture:

```
                          User Web Browser
                                 │
                                 ▼
         ┌─────────────────────────────────────────────────┐
         │          Streamlit Dashboard Service            │
         │      (Render Web Service: bsglem-dashboard)     │
         └───────────────────────┬─────────────────────────┘
                                 │ HTTPS REST
                                 │ BSGLEM_API_URL
                                 ▼
         ┌─────────────────────────────────────────────────┐
         │             FastAPI Backend Service             │
         │       (Render Web Service: bsglem-backend)      │
         └───────────────────────┬─────────────────────────┘
                                 │ In-Memory Singletons
                                 ▼
      ┌──────────────────────────┼──────────────────────────┐
      │                          │                          │
┌─────────────┐          ┌───────────────┐          ┌───────────────┐
│ ML Models   │          │ Data Service  │          │ Energy/PyPSA  │
│ (XGB, BiLSTM│          │ (Historical + │          │ (HiGHS LP &   │
│  CNN-LSTM)  │          │  24h Forecast)│          │  Monte Carlo) │
└─────────────┘          └───────────────┘          └───────────────┘
```

---

## 2. Environment Networking Contexts

The frontend dashboard communicates with the backend across different environments via `BSGLEM_API_URL`:

| Environment | Backend Base URL (`BSGLEM_API_URL`) | Notes |
|---|---|---|
| **Local Development** | `http://127.0.0.1:8000` | Both services running locally on host machine |
| **Docker Compose** | `http://backend:8000` | Internal bridge network resolving container service name |
| **Render Cloud** | `https://<actual-backend-service>.onrender.com` | Public HTTPS endpoint provided by Render |

> [!WARNING]
> Do NOT use `http://backend:8000` or `http://localhost:8000` in Render. Two separate Render Web Services communicate over their public HTTPS hostnames, not Docker Compose internal DNS aliases.

---

## 3. Step-by-Step Render Deployment Procedure

### Step 1: Push Repository to Git Provider
Ensure all deployment configuration files (`render.yaml`, `Dockerfile.api`, `Dockerfile.dashboard`, `requirements.txt`, `models_saved/`, `data/`) are committed and pushed to your GitHub or GitLab repository:
```bash
git add .
git commit -m "chore(deploy): prepare Render deployment blueprint and runbook"
git push origin main
```

### Step 2: Connect Repository to Render
1. Log in to your [Render Dashboard](https://dashboard.render.com).
2. Click **New +** in the top navigation bar and select **Blueprint**.
3. Select your connected repository (`Final Year Project` / `Load_forcast`).
4. Render will automatically parse `render.yaml` and identify two web services:
   - `bsglem-backend` (using `Dockerfile.api`)
   - `bsglem-dashboard` (using `Dockerfile.dashboard`)

### Step 3: Deploy Backend Service First
1. Click **Apply** to begin deploying the blueprint.
2. In the Render service list, open `bsglem-backend`.
3. Monitor the build logs:
   - Python 3.12-slim base image download.
   - Dependency installation from `requirements.txt`.
   - Copying source code, pre-trained models (`models_saved/`), and datasets (`data/`).
   - Uvicorn starting on `0.0.0.0:$PORT`.

### Step 4: Verify Backend Health Check
1. The backend lifespan context loads models and initializes the data service once on startup.
2. Render probes `/health`. The deployment succeeds when `/health` returns HTTP 200:
   ```json
   {
     "status": "healthy",
     "models_loaded": true,
     "loaded_models": ["XGBoost", "BiLSTM", "CNN-LSTM"]
   }
   ```
3. Test the interactive Swagger docs in your browser:
   `https://<actual-backend-service>.onrender.com/docs`

### Step 5: Obtain Backend Render HTTPS URL
Copy the assigned backend URL from the top of the `bsglem-backend` service page:
`https://<actual-backend-service>.onrender.com`

### Step 6: Configure Dashboard Environment
1. Navigate to the `bsglem-dashboard` service in Render.
2. Go to the **Environment** tab.
3. Update the `BSGLEM_API_URL` variable to your actual backend URL:
   ```
   BSGLEM_API_URL = https://<actual-backend-service>.onrender.com
   ```
4. Save the environment variable.

### Step 7: (Recommended) Configure Backend CORS
1. Navigate back to the `bsglem-backend` service.
2. In the **Environment** tab, update `CORS_ORIGINS` with the dashboard URL:
   ```
   CORS_ORIGINS = https://<actual-dashboard-service>.onrender.com
   ```
3. Save changes.

### Step 8: Deploy / Redeploy Dashboard
1. Trigger a deploy on `bsglem-dashboard` (or wait for the environment variable save to trigger an automatic redeploy).
2. Render builds the dashboard container and starts Streamlit headlessly on `$PORT`.

### Step 9: Open and Test the Dashboard
1. Open the dashboard URL:
   `https://<actual-dashboard-service>.onrender.com`
2. Verify that the sidebar reports:
   `Backend Status: Online (Operational)`
   `Telemetry Mode: Historical (2024)`

### Step 10: End-to-End Operational Verification
Perform the standard verification workflow directly in the deployed dashboard UI:
1. **Telemetry Tab:** Verify that latest grid load and 2024 historical observations load correctly.
2. **24h Forecast Tab:** Generate a day-ahead recursive forecast using sample weather inputs; verify predicted load curve and 20% spinning reserve.
3. **Scenarios Tab:** Test the 5 what-if conditions (Heatwave +5°C, Festival, etc.) and check load deltas.
4. **Energy Dispatch Tab:** Run PyPSA economic dispatch; verify thermal and solar dispatch schedules.
5. **Risk Analysis Tab:** Execute Monte Carlo simulation; verify LOLP, EENS, and reserve sufficiency metrics.

---

## 4. Cloud Platform Runtime Limitations & Operational Considerations

1. **Startup Latency & Model Loading:**
   All machine learning models (XGBoost regressor, Bidirectional LSTM, and Conv1D-LSTM) and scalers are loaded into memory during the backend lifespan startup context. Startup takes approximately 10–25 seconds. Do not terminate the instance prematurely during this window.
2. **Resource Footprint Differences:**
   The backend service packages heavy computational runtimes (TensorFlow, PyPSA, HiGHS solver, and XGBoost) and requires ~500 MB–1 GB RAM during Monte Carlo batch execution. The Streamlit dashboard is lightweight and consumes REST endpoints.
3. **Render Free Tier Spin-Down (Cold Starts):**
   On Render's free tier, services spin down after 15 minutes of inactivity. The first incoming request will trigger a cold start (~30–50 seconds). The dashboard will display a connection warning until the backend finishes booting.
4. **Data Scope & Scope of Operations:**
   - The system operates on authentic historical hourly telemetry (2023–2024 Bengaluru grid data) and controlled simulated sequences. It does **NOT** connect to live BESCOM / KPTCL / SLDC SCADA feeds.
   - The 24-hour day-ahead recursive forecasting endpoint requires 24 hourly weather input measurements (temperature, humidity, wind speed, solar irradiance) provided by the client or generated via synthetic diurnal profiles.
5. **Persistent Storage:**
   All model artifacts and historical CSV files are bundled directly into the container image. No external database or cloud object bucket (S3/GCS) is required.

---

## 5. Local Rollback and Fallback Run

If cloud services are unavailable, the entire system can always be operated locally:

**Option A: Multi-Terminal Native Run**
```powershell
# Terminal 1: Backend
.\.venv\Scripts\Activate.ps1
uvicorn api.app:app --host 127.0.0.1 --port 8000

# Terminal 2: Dashboard
.\.venv\Scripts\Activate.ps1
$env:BSGLEM_API_URL="http://127.0.0.1:8000"
streamlit run dashboard/app.py
```

**Option B: Docker Compose (on machines with Docker)**
```bash
docker compose up --build
```
