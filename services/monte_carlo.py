import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from services.pypsa_dispatch import run_pypsa_dispatch

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable

from config import (
    N_SIMULATIONS,
    RANDOM_SEED,
    ENABLE_WEATHER_UNCERTAINTY,
    LOAD_SHEDDING_COST,
    ENABLE_RESIDUAL_TAIL_CLIPPING,
    RESIDUAL_CLIP_LOW_PERCENTILE,
    RESIDUAL_CLIP_HIGH_PERCENTILE,
    MONTE_CARLO_DIR
)

def run_monte_carlo_simulation(residuals_train, last_24_X, df_model, xgb, forecast_24h, hours_range, solar_profile, n_simulations_override=None):
    """
    Runs the Monte Carlo simulation for economic dispatch under weather and residual uncertainty.

    Args:
        residuals_train (np.ndarray): Historical residuals.
        last_24_X (pd.DataFrame): Exogenous features for the next 24 hours.
        df_model (pd.DataFrame): Training DataFrame containing full columns for scaling/std calculations.
        xgb (XGBRegressor): Trained XGBoost model.
        forecast_24h (np.ndarray): Base 24-hour forecast from XGBoost.
        hours_range (pd.DatetimeIndex): Datetime indexes forsnapshots.
        solar_profile (np.ndarray): Solar profile values.

    Returns:
        tuple: (mc_results_df, risk_metrics_df, ci_summary_df, simulated_loads)
    """
    os.makedirs(MONTE_CARLO_DIR, exist_ok=True)

    n_sims = n_simulations_override if n_simulations_override is not None else N_SIMULATIONS

    print(f"  Monte Carlo simulations        : {n_sims}")
    print(f"  Weather uncertainty enabled    : {ENABLE_WEATHER_UNCERTAINTY}")
    print(f"  Load-shedding penalty          : ₹{LOAD_SHEDDING_COST:,.0f}/MWh")
    print(f"  Residual tail clipping         : {ENABLE_RESIDUAL_TAIL_CLIPPING}")

    rng = np.random.default_rng(RANDOM_SEED)
    residual_distribution = np.asarray(residuals_train, dtype=float)
    residual_distribution = residual_distribution[np.isfinite(residual_distribution)]

    if residual_distribution.size == 0:
        raise ValueError("Residual distribution is empty; Monte Carlo simulation cannot proceed.")

    if ENABLE_RESIDUAL_TAIL_CLIPPING:
        residual_low = np.percentile(residual_distribution, RESIDUAL_CLIP_LOW_PERCENTILE)
        residual_high = np.percentile(residual_distribution, RESIDUAL_CLIP_HIGH_PERCENTILE)
        residual_distribution = np.clip(residual_distribution, residual_low, residual_high)
        print(
            f"  Residual clip range            : "
            f"[{residual_low:,.1f}, {residual_high:,.1f}] MW"
        )

    # Optional weather layer
    if ENABLE_WEATHER_UNCERTAINTY:
        weather_forecast_cube = np.tile(last_24_X.to_numpy(dtype=float), (n_sims, 1, 1))
        feature_index = {col: i for i, col in enumerate(last_24_X.columns)}

        for weather_col in ["temperature", "humidity", "solar_irradiance"]:
            if weather_col in feature_index and weather_col in df_model.columns:
                sigma = 0.1 * float(df_model[weather_col].std())
                shocks = rng.normal(0.0, sigma, size=(n_sims, 24))
                weather_forecast_cube[:, :, feature_index[weather_col]] += shocks

        if "solar_irradiance" in feature_index:
            solar_idx = feature_index["solar_irradiance"]
            weather_forecast_cube[:, :, solar_idx] = np.maximum(weather_forecast_cube[:, :, solar_idx], 0.0)

        if "humidity" in feature_index:
            humidity_idx = feature_index["humidity"]
            weather_forecast_cube[:, :, humidity_idx] = np.clip(weather_forecast_cube[:, :, humidity_idx], 0.0, 100.0)

        if "temp_x_hour_sin" in feature_index and "temperature" in feature_index and "hour_sin" in feature_index:
            weather_forecast_cube[:, :, feature_index["temp_x_hour_sin"]] = (
                weather_forecast_cube[:, :, feature_index["temperature"]]
                * weather_forecast_cube[:, :, feature_index["hour_sin"]]
            )

        weather_forecast_frame = pd.DataFrame(
            weather_forecast_cube.reshape(-1, len(last_24_X.columns)),
            columns=last_24_X.columns
        )
        base_forecasts_mc = xgb.predict(weather_forecast_frame).reshape(n_sims, 24)
    else:
        base_forecasts_mc = np.tile(forecast_24h, (n_sims, 1))

    # Bootstrap residuals
    if residual_distribution.size >= 24:
        residual_blocks = np.lib.stride_tricks.sliding_window_view(residual_distribution, 24)
        sampled_block_ids = rng.integers(0, len(residual_blocks), size=n_sims)
        sampled_residuals = residual_blocks[sampled_block_ids].copy()
    else:
        sampled_residuals = rng.choice(
            residual_distribution,
            size=(n_sims, 24),
            replace=True
        )
    simulated_loads = np.maximum(base_forecasts_mc + sampled_residuals, 0.0)

    scenario_loads = {
        simulation_id: simulated_loads[simulation_id].copy()
        for simulation_id in range(n_sims)
    }

    mc_records = []
    coal_energy = np.zeros(n_sims)
    gas_energy = np.zeros(n_sims)
    solar_energy = np.zeros(n_sims)
    shortage_energy = np.zeros(n_sims)
    dispatch_costs = np.zeros(n_sims)
    peak_loads = simulated_loads.max(axis=1)
    dispatch_statuses = []

    for simulation_id in tqdm(range(n_sims), desc="  Monte Carlo PyPSA dispatch"):
        _, dispatch_status, dispatch_result = run_pypsa_dispatch(
            scenario_loads[simulation_id],
            hours_range,
            solar_profile,
            include_load_shedding=True,
            load_shedding_cost=LOAD_SHEDDING_COST
        )

        coal_mwh = float(np.sum(dispatch_result["coal_generation"]))
        gas_mwh = float(np.sum(dispatch_result["gas_generation"]))
        solar_mwh = float(np.sum(dispatch_result["solar_generation"]))
        shortage_mwh = float(np.sum(dispatch_result["unserved_energy"]))
        generation_cost = coal_mwh * 3.5 + gas_mwh * 6.0
        shortage_penalty = shortage_mwh * LOAD_SHEDDING_COST
        total_dispatch_cost = float(dispatch_result["total_cost"])

        coal_energy[simulation_id] = coal_mwh
        gas_energy[simulation_id] = gas_mwh
        solar_energy[simulation_id] = solar_mwh
        shortage_energy[simulation_id] = shortage_mwh
        dispatch_costs[simulation_id] = total_dispatch_cost
        dispatch_statuses.append(dispatch_status)

        mc_records.append({
            "simulation_id": simulation_id,
            "peak_load": peak_loads[simulation_id],
            "total_cost": total_dispatch_cost,
            "coal_energy": coal_mwh,
            "gas_energy": gas_mwh,
            "solar_energy": solar_mwh,
            "shortage_mwh": shortage_mwh,
            "generation_cost": generation_cost,
            "shortage_penalty": shortage_penalty,
            "dispatch_status": dispatch_status,
        })

    mc_results_df = pd.DataFrame(mc_records)

    shortage_scenarios = int((shortage_energy > 1e-6).sum())
    lolp = shortage_scenarios / n_sims
    eens = float(shortage_energy.mean())
    reserve_sufficiency = 1.0 - lolp

    risk_metrics = {
        "LOLP": lolp,
        "EENS": eens,
        "mean_cost": float(np.mean(dispatch_costs)),
        "median_cost": float(np.median(dispatch_costs)),
        "min_cost": float(np.min(dispatch_costs)),
        "max_cost": float(np.max(dispatch_costs)),
        "p95_cost": float(np.percentile(dispatch_costs, 95)),
        "reserve_sufficiency": reserve_sufficiency,
        "peak_load_p95": float(np.percentile(peak_loads, 95)),
        "load_shedding_cost": LOAD_SHEDDING_COST,
    }
    risk_metrics_df = pd.DataFrame([risk_metrics])

    ci_summary_df = pd.DataFrame([
        {
            "metric": "Demand peak (MW)",
            "p05": np.percentile(peak_loads, 5),
            "p50": np.percentile(peak_loads, 50),
            "p95": np.percentile(peak_loads, 95),
        },
        {
            "metric": "Dispatch cost (₹)",
            "p05": np.percentile(dispatch_costs, 5),
            "p50": np.percentile(dispatch_costs, 50),
            "p95": np.percentile(dispatch_costs, 95),
        },
        {
            "metric": "Coal usage (MWh)",
            "p05": np.percentile(coal_energy, 5),
            "p50": np.percentile(coal_energy, 50),
            "p95": np.percentile(coal_energy, 95),
        },
        {
            "metric": "Gas usage (MWh)",
            "p05": np.percentile(gas_energy, 5),
            "p50": np.percentile(gas_energy, 50),
            "p95": np.percentile(gas_energy, 95),
        },
        {
            "metric": "Solar usage (MWh)",
            "p05": np.percentile(solar_energy, 5),
            "p50": np.percentile(solar_energy, 50),
            "p95": np.percentile(solar_energy, 95),
        },
    ])

    mc_results_path = os.path.join(MONTE_CARLO_DIR, "monte_carlo_summary.csv")
    risk_metrics_path = os.path.join(MONTE_CARLO_DIR, "risk_metrics.csv")
    ci_summary_path = os.path.join(MONTE_CARLO_DIR, "confidence_intervals.csv")
    mc_results_df.to_csv(mc_results_path, index=False)
    risk_metrics_df.to_csv(risk_metrics_path, index=False)
    ci_summary_df.to_csv(ci_summary_path, index=False)

    print("\n  Monte Carlo confidence intervals (5th / 50th / 95th percentile):")
    print(ci_summary_df.to_string(index=False, formatters={
        "p05": "{:,.2f}".format,
        "p50": "{:,.2f}".format,
        "p95": "{:,.2f}".format,
    }))

    print("\n  Risk metrics:")
    print(f"    LOLP                : {lolp * 100:.2f}%")
    print(f"    EENS                : {eens:.2f} MWh")
    print(f"    Reserve sufficiency : {reserve_sufficiency * 100:.2f}%")
    print(f"    Cost P95            : ₹{risk_metrics['p95_cost']:,.0f}")

    # Generate plots
    demand_p05 = np.percentile(simulated_loads, 5, axis=0)
    demand_p50 = np.percentile(simulated_loads, 50, axis=0)
    demand_p95 = np.percentile(simulated_loads, 95, axis=0)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(peak_loads, bins=35, color="#1f77b4", alpha=0.82, edgecolor="white")
    ax.axvline(np.percentile(peak_loads, 95), color="#d62728", lw=2, label="P95")
    ax.set_title("Monte Carlo Peak Demand Distribution", fontsize=12, weight="bold")
    ax.set_xlabel("Peak load (MW)")
    ax.set_ylabel("Scenario count")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(MONTE_CARLO_DIR, "demand_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(dispatch_costs, bins=35, color="#2ca02c", alpha=0.82, edgecolor="white")
    ax.axvline(np.percentile(dispatch_costs, 95), color="#d62728", lw=2, label="P95")
    ax.set_title("Monte Carlo Dispatch Cost Distribution", fontsize=12, weight="bold")
    ax.set_xlabel("Total dispatch cost (₹)")
    ax.set_ylabel("Scenario count")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(MONTE_CARLO_DIR, "cost_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(coal_energy, bins=30, alpha=0.62, label="Coal", color="#4d4d4d", edgecolor="white")
    ax.hist(gas_energy, bins=30, alpha=0.62, label="Gas", color="#e07b00", edgecolor="white")
    ax.hist(solar_energy, bins=30, alpha=0.62, label="Solar", color="#f4c430", edgecolor="white")
    ax.set_title("Generator Utilization Distribution", fontsize=12, weight="bold")
    ax.set_xlabel("Daily generation (MWh)")
    ax.set_ylabel("Scenario count")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(MONTE_CARLO_DIR, "generator_utilization_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 5))
    reliability_vals = [reserve_sufficiency * 100, lolp * 100]
    bars = ax.bar(["Fully met", "Shortage"], reliability_vals,
                  color=["#2ca02c", "#d62728"], edgecolor="white")
    for bar, value in zip(bars, reliability_vals):
        ax.text(bar.get_x() + bar.get_width()/2, value + 1,
                f"{value:.2f}%", ha="center", va="bottom", fontsize=9, weight="bold")
    ax.set_ylim(0, max(100, max(reliability_vals) * 1.15))
    ax.set_title("Reliability Distribution", fontsize=12, weight="bold")
    ax.set_ylabel("Probability (%)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(MONTE_CARLO_DIR, "reliability_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Monte Carlo Economic Dispatch Risk Dashboard", fontsize=14, weight="bold")

    ax = axes[0, 0]
    hours = np.arange(24)
    ax.fill_between(hours, demand_p05, demand_p95, color="#1f77b4", alpha=0.22, label="P05–P95")
    ax.plot(hours, demand_p50, color="#1f77b4", lw=2, label="Median")
    ax.plot(hours, forecast_24h, color="#111111", lw=1.6, linestyle="--", label="Base forecast")
    ax.set_title("Demand Uncertainty Envelope")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Load (MW)")
    ax.set_xticks(range(0, 24, 3))
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.hist(dispatch_costs, bins=35, color="#2ca02c", alpha=0.82, edgecolor="white")
    ax.axvline(np.mean(dispatch_costs), color="#111111", lw=1.8, label="Mean")
    ax.axvline(np.percentile(dispatch_costs, 95), color="#d62728", lw=1.8, label="P95")
    ax.set_title("Cost Distribution")
    ax.set_xlabel("Total dispatch cost (₹)")
    ax.set_ylabel("Scenario count")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 0]
    box = ax.boxplot([coal_energy, gas_energy, solar_energy],
                     tick_labels=["Coal", "Gas", "Solar"],
                     patch_artist=True,
                     showfliers=False)
    for patch, color in zip(box["boxes"], ["#4d4d4d", "#e07b00", "#f4c430"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)
    ax.set_title("Generator Utilization Distribution")
    ax.set_ylabel("Daily generation (MWh)")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 1]
    metric_names = ["LOLP", "EENS", "Reserve", "Peak P95"]
    metric_values = [lolp * 100, eens, reserve_sufficiency * 100, risk_metrics["peak_load_p95"]]
    metric_colors = ["#d62728", "#9467bd", "#2ca02c", "#1f77b4"]
    bars = ax.bar(metric_names, metric_values, color=metric_colors, edgecolor="white", alpha=0.86)
    for bar, value in zip(bars, metric_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                f"{value:,.2f}", ha="center", va="bottom", fontsize=8, weight="bold")
    ax.set_title("Reliability Metrics")
    ax.set_ylabel("Value")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    dashboard_path = os.path.join(MONTE_CARLO_DIR, "monte_carlo_dashboard.png")
    fig.savefig(dashboard_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n   {mc_results_path}")
    print(f"   {risk_metrics_path}")
    print(f"   {ci_summary_path}")
    print(f"   {dashboard_path}")

    print(f"""
  Monte Carlo Simulations : {n_sims}
  Expected Cost           : ₹{risk_metrics['mean_cost']:,.0f}
  Worst Case Cost         : ₹{risk_metrics['max_cost']:,.0f}
  Best Case Cost          : ₹{risk_metrics['min_cost']:,.0f}
  LOLP                    : {lolp * 100:.2f}%
  EENS                    : {eens:.2f} MWh
  Reserve Sufficiency     : {reserve_sufficiency * 100:.2f}%
  Peak Load P95           : {risk_metrics['peak_load_p95']:,.1f} MW
""")

    return mc_results_df, risk_metrics_df, ci_summary_df, simulated_loads
