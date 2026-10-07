"""
Energy Management and Risk Analysis Service for B-SGLEM (Phase 5).

Provides:
- PyPSA Economic Dispatch orchestration reusing services/pypsa_dispatch.py.
- Monte Carlo uncertainty & reliability risk analysis reusing residual bootstrapping
  and loss-of-load calculation methodology from services/monte_carlo.py.
- Safe integration with Phase 4 24-hour forecasting.
"""
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd

from config import (
    LOAD_SHEDDING_COST,
    ENABLE_RESIDUAL_TAIL_CLIPPING,
    RESIDUAL_CLIP_LOW_PERCENTILE,
    RESIDUAL_CLIP_HIGH_PERCENTILE,
)
from services.pypsa_dispatch import run_pypsa_dispatch, merit_order_dispatch


DEFAULT_SOLAR_PROFILE = np.clip(
    np.sin(np.pi * np.arange(24) / 24) * 1.6 - 0.3, 0.0, 1.0
)

DISPATCH_PROVENANCE_NOTE = (
    "24-hour economic dispatch solved using PyPSA (Linear Optimal Power Flow with HiGHS solver). "
    "Generation fleet includes Coal (1,500 MW, 40% min stable, ₹3.5/unit), "
    "Gas (800 MW flexible, ₹6.0/unit), Solar (400 MW nameplate, ₹0.0/unit), "
    "and optional Load Shedding (₹100/MWh unserved penalty)."
)

RISK_PROVENANCE_NOTE = (
    "Probabilistic risk metrics evaluated via residual bootstrapping from the stacked "
    "out-of-fold ensemble. Tail clipping applied between 5th and 95th empirical percentiles. "
    "Reported quantiles represent empirical scenario bounds across simulated trajectories "
    "and must not be interpreted as statistically calibrated prediction intervals."
)


class EnergyService:
    """
    Domain service orchestrating power dispatch optimization and Monte Carlo risk assessment.
    """

    @staticmethod
    def default_solar_profile() -> List[float]:
        """Returns the standard 24-hour Bengaluru solar generation profile."""
        return [round(float(v), 4) for v in DEFAULT_SOLAR_PROFILE]

    @classmethod
    def solve_dispatch(
        cls,
        demand_mw: List[float],
        solar_profile: Optional[List[float]] = None,
        include_load_shedding: bool = False,
        load_shedding_cost: float = LOAD_SHEDDING_COST,
        timestamps: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Solves 24-hour economic dispatch using PyPSA.

        Args:
            demand_mw: 24 hourly demand observations or forecasts in MW.
            solar_profile: Optional 24 hourly solar availability factors [0.0 - 1.0].
            include_load_shedding: Whether to include unserved energy generator.
            load_shedding_cost: Penalty cost per MWh of unmet load (₹/MWh).
            timestamps: Optional list of 24 ISO timestamp strings.

        Returns:
            Dict[str, Any]: Structured dispatch schedule, cost, and reliability metrics.
        """
        if len(demand_mw) != 24:
            raise ValueError(f"Demand profile must have exactly 24 hourly values, got {len(demand_mw)}")

        demand_arr = np.array(demand_mw, dtype=float)
        if not np.all(np.isfinite(demand_arr)):
            raise ValueError("Demand profile contains non-finite (NaN or Inf) values.")
        if np.any(demand_arr < 0.0):
            raise ValueError("Demand values must be non-negative.")

        if solar_profile is not None:
            if len(solar_profile) != 24:
                raise ValueError(f"Solar profile must have exactly 24 hourly values, got {len(solar_profile)}")
            solar_arr = np.clip(np.array(solar_profile, dtype=float), 0.0, 1.0)
            if not np.all(np.isfinite(solar_arr)):
                raise ValueError("Solar profile contains non-finite values.")
        else:
            solar_arr = DEFAULT_SOLAR_PROFILE.copy()

        # Build snapshots
        if timestamps and len(timestamps) == 24:
            snapshots = pd.to_datetime(timestamps)
        else:
            snapshots = pd.date_range("2025-01-01", periods=24, freq="h")

        # Solve via PyPSA
        network, dispatch_status, result = run_pypsa_dispatch(
            load_24h=demand_arr,
            snapshots=snapshots,
            solar_profile=solar_arr,
            include_load_shedding=include_load_shedding,
            load_shedding_cost=load_shedding_cost,
        )

        is_optimal = "optimal" in dispatch_status.lower()
        is_feasible = is_optimal or "fallback" in dispatch_status.lower()
        if dispatch_status == "empty_dispatch":
            display_status = "infeasible (solver produced empty dispatch)"
            is_feasible = False
            is_optimal = False
        else:
            display_status = dispatch_status

        coal_gen = np.asarray(result["coal_generation"], dtype=float)
        gas_gen = np.asarray(result["gas_generation"], dtype=float)
        solar_gen = np.asarray(result["solar_generation"], dtype=float)
        shortage = np.asarray(result["unserved_energy"], dtype=float)
        total_cost = float(result["total_cost"])

        total_demand_mwh = float(demand_arr.sum())
        coal_total_mwh = float(coal_gen.sum())
        gas_total_mwh = float(gas_gen.sum())
        solar_total_mwh = float(solar_gen.sum())
        unserved_mwh = float(shortage.sum())

        total_supply = coal_total_mwh + gas_total_mwh + solar_total_mwh
        coal_share = (coal_total_mwh / total_supply * 100.0) if total_supply > 0 else 0.0
        gas_share = (gas_total_mwh / total_supply * 100.0) if total_supply > 0 else 0.0
        solar_share = (solar_total_mwh / total_supply * 100.0) if total_supply > 0 else 0.0

        hourly_schedule = []
        for h in range(24):
            c_cost = coal_gen[h] * 3.5 + gas_gen[h] * 6.0 + shortage[h] * load_shedding_cost
            hourly_schedule.append({
                "hour": h,
                "timestamp": snapshots[h].strftime("%Y-%m-%dT%H:%M:%S") if hasattr(snapshots[h], "strftime") else str(snapshots[h]),
                "demand_mw": round(float(demand_arr[h]), 2),
                "coal_mw": round(float(coal_gen[h]), 2),
                "gas_mw": round(float(gas_gen[h]), 2),
                "solar_mw": round(float(solar_gen[h]), 2),
                "unserved_mw": round(float(shortage[h]), 2),
                "hourly_cost": round(float(c_cost), 2),
            })

        return {
            "optimization_status": display_status,
            "is_optimal": is_optimal,
            "is_feasible": is_feasible,
            "total_cost": round(total_cost, 2),
            "currency": "INR (₹)",
            "peak_demand_mw": round(float(demand_arr.max()), 2),
            "total_demand_mwh": round(total_demand_mwh, 2),
            "coal_total_mwh": round(coal_total_mwh, 2),
            "gas_total_mwh": round(gas_total_mwh, 2),
            "solar_total_mwh": round(solar_total_mwh, 2),
            "unserved_energy_mwh": round(unserved_mwh, 2),
            "coal_share_pct": round(coal_share, 2),
            "gas_share_pct": round(gas_share, 2),
            "solar_share_pct": round(solar_share, 2),
            "hourly_schedule": hourly_schedule,
            "solver_used": "HiGHS / PyPSA Linear Optimal Power Flow",
            "provenance_note": DISPATCH_PROVENANCE_NOTE,
        }

    @classmethod
    def run_risk_analysis(
        cls,
        demand_mw: List[float],
        residuals: np.ndarray,
        n_simulations: int = 100,
        random_seed: Optional[int] = 42,
        enable_residual_clipping: bool = ENABLE_RESIDUAL_TAIL_CLIPPING,
        clip_low_percentile: float = RESIDUAL_CLIP_LOW_PERCENTILE,
        clip_high_percentile: float = RESIDUAL_CLIP_HIGH_PERCENTILE,
        load_shedding_cost: float = LOAD_SHEDDING_COST,
        dispatch_engine: str = "merit_order",
        solar_profile: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Executes Monte Carlo uncertainty and risk assessment over a 24-hour demand horizon.

        Args:
            demand_mw: 24-hour base forecast load in MW.
            residuals: Raw residuals from stacked OOF ensemble.
            n_simulations: Number of Monte Carlo draws (10 to 1,000).
            random_seed: Random number seed for reproducibility.
            enable_residual_clipping: Whether to clip empirical residual tails.
            clip_low_percentile: Lower clipping quantile (default 5.0).
            clip_high_percentile: Upper clipping quantile (default 95.0).
            load_shedding_cost: Unserved energy penalty (₹/MWh).
            dispatch_engine: "merit_order" (fast, recommended for API) or "pypsa".
            solar_profile: Optional 24-hour solar availability profile.

        Returns:
            Dict[str, Any]: Structured risk metrics, quantiles, and reliability indicators.
        """
        if len(demand_mw) != 24:
            raise ValueError(f"Base demand forecast must have exactly 24 hourly values, got {len(demand_mw)}")

        demand_base = np.array(demand_mw, dtype=float)
        if not np.all(np.isfinite(demand_base)):
            raise ValueError("Base demand forecast contains non-finite values.")
        if np.any(demand_base < 0.0):
            raise ValueError("Base demand values must be non-negative.")

        if n_simulations < 10 or n_simulations > 1000:
            raise ValueError(f"Simulation count must be between 10 and 1,000, got {n_simulations}")

        if solar_profile is not None:
            if len(solar_profile) != 24:
                raise ValueError(f"Solar profile must have exactly 24 hourly values, got {len(solar_profile)}")
            solar_arr = np.clip(np.array(solar_profile, dtype=float), 0.0, 1.0)
        else:
            solar_arr = DEFAULT_SOLAR_PROFILE.copy()

        # Prepare residuals (DO NOT mutate original input array)
        residual_distribution = np.asarray(residuals, dtype=float).copy()
        residual_distribution = residual_distribution[np.isfinite(residual_distribution)]
        if residual_distribution.size == 0:
            raise ValueError("Residual distribution is empty; risk simulation cannot proceed.")

        if enable_residual_clipping:
            r_low = np.percentile(residual_distribution, clip_low_percentile)
            r_high = np.percentile(residual_distribution, clip_high_percentile)
            residual_distribution = np.clip(residual_distribution, r_low, r_high)

        rng = np.random.default_rng(random_seed)

        # Bootstrap residuals: block bootstrap if sufficient length
        if residual_distribution.size >= 24:
            residual_blocks = np.lib.stride_tricks.sliding_window_view(residual_distribution, 24)
            sampled_block_ids = rng.integers(0, len(residual_blocks), size=n_simulations)
            sampled_residuals = residual_blocks[sampled_block_ids].copy()
        else:
            sampled_residuals = rng.choice(residual_distribution, size=(n_simulations, 24), replace=True)

        # Generate simulated load trajectories
        simulated_loads = np.maximum(demand_base + sampled_residuals, 0.0)

        coal_energy = np.zeros(n_simulations)
        gas_energy = np.zeros(n_simulations)
        solar_energy = np.zeros(n_simulations)
        shortage_energy = np.zeros(n_simulations)
        dispatch_costs = np.zeros(n_simulations)
        peak_loads = simulated_loads.max(axis=1)

        snapshots = pd.date_range("2025-01-01", periods=24, freq="h")

        # Run dispatch per simulated scenario
        for sim_id in range(n_simulations):
            s_load = simulated_loads[sim_id]
            if dispatch_engine == "pypsa":
                _, _, d_res = run_pypsa_dispatch(
                    load_24h=s_load,
                    snapshots=snapshots,
                    solar_profile=solar_arr,
                    include_load_shedding=True,
                    load_shedding_cost=load_shedding_cost,
                )
            else:
                # Fast merit-order dispatch
                d_res = merit_order_dispatch(
                    load_24h=s_load,
                    solar_profile=solar_arr,
                    include_load_shedding=True,
                    load_shedding_cost=load_shedding_cost,
                )

            coal_energy[sim_id] = float(np.sum(d_res["coal_generation"]))
            gas_energy[sim_id] = float(np.sum(d_res["gas_generation"]))
            solar_energy[sim_id] = float(np.sum(d_res["solar_generation"]))
            shortage_energy[sim_id] = float(np.sum(d_res["unserved_energy"]))
            dispatch_costs[sim_id] = float(d_res["total_cost"])

        # Reliability metrics
        shortage_count = int((shortage_energy > 1e-4).sum())
        lolp = float(shortage_count / n_simulations)
        eens = float(shortage_energy.mean())
        reserve_sufficiency = float(1.0 - lolp)

        risk_metrics = {
            "LOLP": round(lolp, 4),
            "EENS_mwh": round(eens, 2),
            "reserve_sufficiency": round(reserve_sufficiency, 4),
            "mean_cost": round(float(np.mean(dispatch_costs)), 2),
            "median_cost": round(float(np.median(dispatch_costs)), 2),
            "min_cost": round(float(np.min(dispatch_costs)), 2),
            "max_cost": round(float(np.max(dispatch_costs)), 2),
            "p95_cost": round(float(np.percentile(dispatch_costs, 95)), 2),
            "peak_demand_p95_mw": round(float(np.percentile(peak_loads, 95)), 2),
        }

        # Quantile summaries across key dimensions
        quantile_summaries = [
            {
                "metric": "Demand peak (MW)",
                "p05": round(float(np.percentile(peak_loads, 5)), 2),
                "p50": round(float(np.percentile(peak_loads, 50)), 2),
                "p95": round(float(np.percentile(peak_loads, 95)), 2),
            },
            {
                "metric": "Dispatch cost (₹)",
                "p05": round(float(np.percentile(dispatch_costs, 5)), 2),
                "p50": round(float(np.percentile(dispatch_costs, 50)), 2),
                "p95": round(float(np.percentile(dispatch_costs, 95)), 2),
            },
            {
                "metric": "Coal usage (MWh)",
                "p05": round(float(np.percentile(coal_energy, 5)), 2),
                "p50": round(float(np.percentile(coal_energy, 50)), 2),
                "p95": round(float(np.percentile(coal_energy, 95)), 2),
            },
            {
                "metric": "Gas usage (MWh)",
                "p05": round(float(np.percentile(gas_energy, 5)), 2),
                "p50": round(float(np.percentile(gas_energy, 50)), 2),
                "p95": round(float(np.percentile(gas_energy, 95)), 2),
            },
            {
                "metric": "Solar usage (MWh)",
                "p05": round(float(np.percentile(solar_energy, 5)), 2),
                "p50": round(float(np.percentile(solar_energy, 50)), 2),
                "p95": round(float(np.percentile(solar_energy, 95)), 2),
            },
            {
                "metric": "Unserved energy (MWh)",
                "p05": round(float(np.percentile(shortage_energy, 5)), 2),
                "p50": round(float(np.percentile(shortage_energy, 50)), 2),
                "p95": round(float(np.percentile(shortage_energy, 95)), 2),
            },
        ]

        return {
            "n_simulations": n_simulations,
            "random_seed": random_seed,
            "dispatch_engine": dispatch_engine,
            "base_forecast_summary": {
                "peak_demand_mw": round(float(demand_base.max()), 2),
                "mean_demand_mw": round(float(demand_base.mean()), 2),
                "total_demand_mwh": round(float(demand_base.sum()), 2),
            },
            "risk_metrics": risk_metrics,
            "quantile_summaries": quantile_summaries,
            "uncertainty_provenance_note": RISK_PROVENANCE_NOTE,
        }
