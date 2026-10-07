"""
Reusable HTTP API Client for B-SGLEM FastAPI Backend (Phase 6).

Handles communication with the backend service:
- Configurable base URL via BSGLEM_API_URL (defaults to http://127.0.0.1:8000).
- Explicit request timeouts.
- Safe error parsing without exposing internal stack traces or secrets.
- Zero local ML or optimization logic.
"""
import os
from typing import Dict, Any, List, Optional, Tuple
import requests


DEFAULT_API_URL = "http://127.0.0.1:8000"


class APIClientError(Exception):
    """Custom exception raised when an API request fails."""

    def __init__(self, message: str, status_code: Optional[int] = None, detail: Optional[Any] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.detail = detail

    def __str__(self) -> str:
        if self.status_code:
            return f"[{self.status_code}] {self.message}"
        return self.message


class BSGLEMClient:
    """
    Lightweight client for communicating with the B-SGLEM FastAPI backend.
    """

    def __init__(self, base_url: Optional[str] = None, timeout: float = 10.0):
        url = base_url or os.getenv("BSGLEM_API_URL", DEFAULT_API_URL)
        self.base_url = url.rstrip("/")
        self.default_timeout = timeout
        self.computation_timeout = 60.0  # longer timeout for PyPSA/Monte Carlo

    def _format_error(self, response: requests.Response) -> str:
        """Parses error responses into readable human messages."""
        try:
            data = response.json()
            if isinstance(data, dict) and "detail" in data:
                detail = data["detail"]
                if isinstance(detail, list):
                    # FastAPI Pydantic validation error list
                    err_lines = []
                    for item in detail:
                        loc = " -> ".join(str(x) for x in item.get("loc", []))
                        msg = item.get("msg", "Validation error")
                        err_lines.append(f"• {loc}: {msg}" if loc else f"• {msg}")
                    return "Validation error:\n" + "\n".join(err_lines)
                return str(detail)
            return response.text[:300]
        except Exception:
            return f"HTTP error {response.status_code}: {response.reason}"

    def _request(
        self,
        method: str,
        endpoint: str,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Executes HTTP request with error handling."""
        url = f"{self.base_url}{endpoint}"
        t = timeout or self.default_timeout

        try:
            resp = requests.request(
                method=method,
                url=url,
                json=json,
                params=params,
                timeout=t,
                headers={"Accept": "application/json"},
            )
        except requests.exceptions.ConnectionError:
            raise APIClientError(
                f"Cannot connect to backend at {self.base_url}. Please ensure FastAPI is running.",
                status_code=None,
            )
        except requests.exceptions.Timeout:
            raise APIClientError(
                f"Request to {endpoint} timed out after {t:.0f} seconds.",
                status_code=None,
            )
        except requests.exceptions.RequestException as exc:
            raise APIClientError(f"HTTP communication error: {exc}", status_code=None)

        if not resp.ok:
            error_msg = self._format_error(resp)
            raise APIClientError(error_msg, status_code=resp.status_code, detail=resp.text)

        try:
            return resp.json()
        except Exception as exc:
            raise APIClientError(f"Failed to parse JSON response: {exc}", status_code=resp.status_code)

    # ── 1. System Health & Metadata ─────────────────────────────────────

    def get_health(self) -> Dict[str, Any]:
        """GET /health - reports service status, model load state, data source mode."""
        return self._request("GET", "/health")

    # ── 2. Historical & Telemetry Data ──────────────────────────────────

    def get_latest_data(self) -> Dict[str, Any]:
        """GET /data/latest - retrieves most recent historical or simulated record."""
        return self._request("GET", "/data/latest")

    def get_data_history(
        self,
        limit: int = 168,
        offset: int = 0,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """GET /data/history - retrieves historical load and weather telemetry."""
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._request("GET", "/data/history", params=params)

    def generate_demo_data(
        self,
        hours: int = 1,
        weather_override: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """POST /data/generate - advances demo simulation by N hours."""
        payload: Dict[str, Any] = {"hours": hours}
        if weather_override:
            payload["weather_override"] = weather_override
        return self._request("POST", "/data/generate", json=payload)

    # ── 3. Forecasting Endpoints ────────────────────────────────────────

    def forecast_24h(
        self,
        weather_forecasts: List[Dict[str, float]],
        calendar_overrides: Optional[List[Dict[str, int]]] = None,
    ) -> Dict[str, Any]:
        """POST /forecast/24h - authentic day-ahead recursive 24-hour forecast."""
        payload: Dict[str, Any] = {"weather_forecasts": weather_forecasts}
        if calendar_overrides:
            payload["calendar_overrides"] = calendar_overrides
        return self._request("POST", "/forecast/24h", json=payload, timeout=20.0)

    def predict_scenario(
        self,
        features: Dict[str, Any],
        scenarios: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """POST /predict/scenario - what-if stress scenario evaluation."""
        payload: Dict[str, Any] = {"features": features}
        if scenarios:
            payload["scenarios"] = scenarios
        return self._request("POST", "/predict/scenario", json=payload)

    # ── 4. Energy Management & Risk Endpoints ───────────────────────────

    def solve_dispatch(
        self,
        demand_mw: Optional[List[float]] = None,
        use_forecast_24h: bool = False,
        weather_forecasts: Optional[List[Dict[str, float]]] = None,
        solar_profile: Optional[List[float]] = None,
        include_load_shedding: bool = False,
        load_shedding_cost: float = 100.0,
    ) -> Dict[str, Any]:
        """POST /energy/dispatch - PyPSA 24-hour linear optimal power flow economic dispatch."""
        payload: Dict[str, Any] = {
            "use_forecast_24h": use_forecast_24h,
            "include_load_shedding": include_load_shedding,
            "load_shedding_cost": load_shedding_cost,
        }
        if demand_mw is not None:
            payload["demand_mw"] = demand_mw
        if weather_forecasts is not None:
            payload["weather_forecasts"] = weather_forecasts
        if solar_profile is not None:
            payload["solar_profile"] = solar_profile

        return self._request("POST", "/energy/dispatch", json=payload, timeout=self.computation_timeout)

    def run_risk_analysis(
        self,
        demand_mw: Optional[List[float]] = None,
        use_forecast_24h: bool = False,
        weather_forecasts: Optional[List[Dict[str, float]]] = None,
        n_simulations: int = 100,
        random_seed: Optional[int] = 42,
        dispatch_engine: str = "merit_order",
        enable_residual_clipping: bool = True,
        load_shedding_cost: float = 100.0,
        solar_profile: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """POST /energy/risk - Monte Carlo uncertainty and grid reliability risk assessment."""
        payload: Dict[str, Any] = {
            "use_forecast_24h": use_forecast_24h,
            "n_simulations": n_simulations,
            "random_seed": random_seed,
            "dispatch_engine": dispatch_engine,
            "enable_residual_clipping": enable_residual_clipping,
            "load_shedding_cost": load_shedding_cost,
        }
        if demand_mw is not None:
            payload["demand_mw"] = demand_mw
        if weather_forecasts is not None:
            payload["weather_forecasts"] = weather_forecasts
        if solar_profile is not None:
            payload["solar_profile"] = solar_profile

        # Monte Carlo merit_order takes ~15ms; PyPSA engine can take ~30-60s
        timeout = 120.0 if dispatch_engine == "pypsa" else 30.0
        return self._request("POST", "/energy/risk", json=payload, timeout=timeout)
