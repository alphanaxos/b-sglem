"""
Bengaluru Smart Grid Load Forecasting & Energy Management System (B-SGLEM)
Phase 6 — Preliminary Interactive Academic Dashboard.

Built with Streamlit & Plotly. Connects to FastAPI backend without duplicating ML,
dispatch, or Monte Carlo domain logic.
"""
import os
from typing import Dict, Any, List
import pandas as pd
import streamlit as st

from dashboard.api_client import BSGLEMClient, APIClientError
from dashboard.components import (
    get_default_weather_24,
    get_default_demand_24,
    get_default_baseline_features,
    plot_historical_load,
    plot_forecast_24h,
    plot_scenario_comparison,
    plot_dispatch_schedule,
    plot_fuel_shares_donut,
    plot_risk_quantiles,
)

# ── Streamlit Page Configuration ────────────────────────────────────────
st.set_page_config(
    page_title="B-SGLEM Smart Grid Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Client Initialization ───────────────────────────────────────────────
api_url_env = os.getenv("BSGLEM_API_URL", "http://127.0.0.1:8000")
client = BSGLEMClient(base_url=api_url_env)


# ── Sidebar Navigation & System Monitor ─────────────────────────────────
st.sidebar.title("⚡ B-SGLEM Platform")
st.sidebar.caption("Bengaluru Smart Grid Load Forecasting & Energy Management")

# Backend Health Status Card
try:
    health_data = client.get_health()
    api_online = True
    status_label = "🟢 Online"
    mode_label = health_data.get("data_source_mode", "historical")
    sim_enabled = health_data.get("simulation_enabled", False)
except Exception:
    health_data = {}
    api_online = False
    status_label = "🔴 Offline"
    mode_label = "Unavailable"
    sim_enabled = False

st.sidebar.markdown(f"**Backend Status:** {status_label}")
st.sidebar.markdown(f"**Data Mode:** `{mode_label}`")
st.sidebar.markdown(f"**API URL:** `{client.base_url}`")
st.sidebar.markdown("---")

nav_choice = st.sidebar.radio(
    "Navigation",
    [
        "📊 System Overview",
        "🔮 24-Hour Forecasting",
        "⚡ Scenario Analysis",
        "⚙️ PyPSA Economic Dispatch",
        "🎲 Monte Carlo Risk Analysis",
        "🧪 Demo Data Controls",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "⚠️ **Operational Notice:** System operates on configured ML artifacts "
    "and optimization fleet models. It does not represent a live SCADA connection "
    "to BESCOM or SLDC."
)


# ══════════════════════════════════════════════════════════════
# PAGE 1: SYSTEM OVERVIEW
# ══════════════════════════════════════════════════════════════
if nav_choice == "📊 System Overview":
    st.header("📊 System Overview & Telemetry")
    st.markdown(
        "Monitor operational status, latest telemetry observations, "
        "and historical load behavior from the Bengaluru power grid."
    )

    col_btn, _ = st.columns([1, 5])
    with col_btn:
        refresh = st.button("🔄 Refresh Data")

    if not api_online:
        st.error(
            f"❌ Unable to communicate with the FastAPI backend at `{client.base_url}`.\n\n"
            "Please ensure the backend service is started via:\n"
            "`uvicorn api.app:app --host 127.0.0.1 --port 8000`"
        )
    else:
        try:
            latest = client.get_latest_data()
            history = client.get_data_history(limit=168)

            # KPI Metric Row
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            with kpi1:
                st.metric(
                    label="Latest Grid Demand",
                    value=f"{latest['load_mw']:,.1f} MW",
                    help="Most recent electrical load recorded in telemetry database",
                )
            with kpi2:
                st.metric(
                    label="Observation Timestamp",
                    value=latest["timestamp"].replace("T", " "),
                    help="Asia/Kolkata observation timestamp",
                )
            with kpi3:
                is_sim = latest.get("is_simulated", False)
                provenance_badge = "Simulated (Demo)" if is_sim else "Historical Record"
                st.metric(
                    label="Data Provenance",
                    value=provenance_badge,
                    help="Distinguishes genuine historical observations from simulated demo advances",
                )
            with kpi4:
                weather = latest.get("weather", {})
                st.metric(
                    label="Ambient Temperature",
                    value=f"{weather.get('temperature', 0.0):.1f} °C",
                    delta=f"{weather.get('humidity', 0.0):.0f}% Humidity",
                )

            # Historical Load Chart
            st.subheader("Historical Demand Profile (Past 7 Days)")
            records = history.get("records", [])
            if records:
                fig_hist = plot_historical_load(records)
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.info("No historical records available in the active dataset range.")

            # Model & Service Information
            with st.expander("ℹ️ Backend Configuration Details"):
                st.json({
                    "service_status": health_data.get("status"),
                    "loaded_models": health_data.get("loaded_models"),
                    "feature_count": health_data.get("feature_count"),
                    "ensemble_method": health_data.get("ensemble_method"),
                    "active_dataset": health_data.get("active_dataset"),
                    "data_source_mode": health_data.get("data_source_mode"),
                    "simulation_enabled": health_data.get("simulation_enabled"),
                })

        except APIClientError as err:
            st.error(f"Error loading system data: {err}")


# ══════════════════════════════════════════════════════════════
# PAGE 2: 24-HOUR FORECASTING
# ══════════════════════════════════════════════════════════════
elif nav_choice == "🔮 24-Hour Forecasting":
    st.header("🔮 Day-Ahead 24-Hour Recursive Load Forecast")
    st.markdown(
        "Generates genuine 24-hour day-ahead predictions using the trained operational "
        "**XGBoost** model. Autoregressive lags ($T-1$) and rolling statistics are "
        "dynamically updated in a recursive loop using prior predicted outputs."
    )

    st.info(
        "📝 **Weather Input Notice:** The weather values below are **demonstration example inputs** "
        "reflecting typical Bengaluru diurnal weather. They can be freely edited before running the forecast."
    )

    # Initialize or load weather dataframe in session state
    if "weather_24_df" not in st.session_state:
        st.session_state.weather_24_df = pd.DataFrame(get_default_weather_24())

    with st.expander("🛠️ View / Edit 24-Hour Weather Forecast Inputs", expanded=True):
        edited_weather_df = st.data_editor(
            st.session_state.weather_24_df,
            column_config={
                "hour": st.column_config.NumberColumn("Hour (0-23)", disabled=True),
                "temperature": st.column_config.NumberColumn("Temp (°C)", min_value=5.0, max_value=45.0, step=0.1),
                "humidity": st.column_config.NumberColumn("Humidity (%)", min_value=10.0, max_value=100.0, step=1.0),
                "wind_speed": st.column_config.NumberColumn("Wind (m/s)", min_value=0.0, max_value=25.0, step=0.1),
                "solar_irradiance": st.column_config.NumberColumn("Solar (W/m²)", min_value=0.0, max_value=1200.0, step=10.0),
            },
            hide_index=True,
            use_container_width=True,
        )

    col_fc_btn, _ = st.columns([1, 4])
    with col_fc_btn:
        run_forecast = st.button("🚀 Generate 24-Hour Forecast", type="primary")

    if run_forecast:
        if len(edited_weather_df) != 24:
            st.error(f"Exactly 24 hourly weather records are required. Found {len(edited_weather_df)}.")
        else:
            weather_payload = [
                {
                    "temperature": float(row["temperature"]),
                    "humidity": float(row["humidity"]),
                    "wind_speed": float(row["wind_speed"]),
                    "solar_irradiance": float(row["solar_irradiance"]),
                }
                for _, row in edited_weather_df.iterrows()
            ]

            with st.spinner("Generating recursive 24-hour load forecast..."):
                try:
                    fc_response = client.forecast_24h(weather_payload)
                    st.session_state.last_forecast = fc_response
                    st.success("✅ 24-hour day-ahead forecast generated successfully!")
                except APIClientError as err:
                    st.error(f"Forecasting failed: {err}")

    if "last_forecast" in st.session_state:
        fc = st.session_state.last_forecast
        preds = fc.get("predictions", [])
        if preds:
            df_preds = pd.DataFrame(preds)

            # KPI Summary
            peak_mw = df_preds["predicted_load_mw"].max()
            peak_hour = df_preds.loc[df_preds["predicted_load_mw"].idxmax(), "hour"]
            mean_mw = df_preds["predicted_load_mw"].mean()
            total_mwh = df_preds["predicted_load_mw"].sum()

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Peak Predicted Load", f"{peak_mw:,.1f} MW", delta=f"Hour {peak_hour:02d}:00")
            with c2:
                st.metric("Average Load", f"{mean_mw:,.1f} MW")
            with c3:
                st.metric("Total Demanded Energy", f"{total_mwh:,.1f} MWh")
            with c4:
                st.metric(
                    "Operating Reserve (20%)",
                    f"{peak_mw * 0.20:,.1f} MW",
                    help="Spinning reserve requirement calculated at peak demand",
                )

            # Interactive Plotly Chart
            fig_fc = plot_forecast_24h(preds)
            st.plotly_chart(fig_fc, use_container_width=True)

            # Data Table & Caveats
            with st.expander("📋 Tabular 24-Hour Forecast Schedule"):
                st.dataframe(
                    df_preds[[
                        "hour", "timestamp", "predicted_load_mw", "spinning_reserve_mw",
                        "lower_bound_mw", "upper_bound_mw", "is_recursive_lag"
                    ]],
                    use_container_width=True,
                )

            st.caption(f"**Forecast Origin:** {fc.get('forecast_origin')} | **Source:** `{fc.get('data_source')}`")
            st.caption(f"ℹ️ **Uncertainty Caveat:** {fc.get('interval_provenance_note')}")


# ══════════════════════════════════════════════════════════════
# PAGE 3: SCENARIO ANALYSIS
# ══════════════════════════════════════════════════════════════
elif nav_choice == "⚡ Scenario Analysis":
    st.header("⚡ Operational Scenario Stress Testing")
    st.markdown(
        "Evaluate grid vulnerability and demand sensitivity under exogenous perturbations: "
        "`Heatwave (+5°C)`, `Festival Load`, `Rainy Day`, `Night Peak (22:00)`."
    )

    default_features = get_default_baseline_features()

    col_sc1, col_sc2 = st.columns([1, 2])
    with col_sc1:
        st.subheader("Baseline Parameters")
        base_temp = st.number_input("Ambient Temperature (°C)", value=float(default_features["temperature"]), step=0.5)
        base_hum = st.number_input("Relative Humidity (%)", value=float(default_features["humidity"]), step=1.0)
        base_lag1 = st.number_input("Prior Hour Demand Lag_1 (MW)", value=float(default_features["lag_1"]), step=10.0)

        selected_scenarios = st.multiselect(
            "Scenarios to Evaluate",
            ["Normal", "Heatwave (+5°C)", "Festival Load", "Rainy Day", "Night Peak (22:00)"],
            default=["Normal", "Heatwave (+5°C)", "Festival Load", "Rainy Day", "Night Peak (22:00)"],
        )

        run_scenarios = st.button("⚡ Run Scenario Stress Test", type="primary")

    if run_scenarios:
        # Construct feature dict
        feats = default_features.copy()
        feats["temperature"] = float(base_temp)
        feats["humidity"] = float(base_hum)
        feats["lag_1"] = float(base_lag1)

        with st.spinner("Evaluating scenarios with operational XGBoost..."):
            try:
                sc_res = client.predict_scenario(features=feats, scenarios=selected_scenarios)
                st.session_state.last_scenarios = sc_res
                st.success("Scenarios evaluated successfully!")
            except APIClientError as err:
                st.error(f"Scenario analysis failed: {err}")

    if "last_scenarios" in st.session_state:
        sc = st.session_state.last_scenarios
        base_mw = sc.get("baseline_load_mw", 0.0)
        results = sc.get("results", [])

        with col_sc2:
            st.subheader("Stress Test Outcomes")
            fig_sc = plot_scenario_comparison(results, baseline_mw=base_mw)
            st.plotly_chart(fig_sc, use_container_width=True)

        # Tabular details
        st.markdown("### Scenario Impact Summary")
        df_sc = pd.DataFrame(results)
        st.dataframe(
            df_sc[["scenario_name", "predicted_load_mw", "delta_mw", "pct_change", "description"]],
            use_container_width=True,
            column_config={
                "predicted_load_mw": st.column_config.NumberColumn("Predicted Load (MW)", format="%.1f MW"),
                "delta_mw": st.column_config.NumberColumn("Delta vs Normal (MW)", format="%+.1f MW"),
                "pct_change": st.column_config.NumberColumn("Change (%)", format="%+.2f%%"),
            },
        )


# ══════════════════════════════════════════════════════════════
# PAGE 4: PYPSA ECONOMIC DISPATCH
# ══════════════════════════════════════════════════════════════
elif nav_choice == "⚙️ PyPSA Economic Dispatch":
    st.header("⚙️ PyPSA 24-Hour Economic Dispatch")
    st.markdown(
        "Solves Linear Optimal Power Flow (LOPF) unit commitment minimizing total operational "
        "cost while enforcing thermal fleet parameters, minimum stable output, and solar availability."
    )

    st.markdown(
        "**Fleet Configuration:** Coal (1,500 MW, 40% min stable, ₹3.5/unit) | "
        "Gas Peaker (800 MW, ₹6.0/unit) | Solar (400 MW nameplate, ₹0.0/unit)"
    )

    input_mode = st.radio(
        "Demand Input Source",
        ["Use Explicit 24-Hour Profile", "Consume Phase 4 24h Recursive Forecast"],
        horizontal=True,
    )

    with st.expander("Optimization Parameters", expanded=True):
        col_op1, col_op2 = st.columns(2)
        with col_op1:
            include_shedding = st.checkbox(
                "Include Unserved Energy Generator (Load Shedding)",
                value=False,
                help="Allows solver to shed load at penalty price when demand exceeds fleet capacity (2,700 MW).",
            )
        with col_op2:
            shed_cost = st.number_input(
                "Load Shedding Penalty (₹/MWh)",
                value=100.0,
                min_value=10.0,
                max_value=1000.0,
                step=10.0,
            )

    run_dispatch = st.button("⚡ Solve Dispatch Optimization", type="primary")

    if run_dispatch:
        with st.spinner("Solving PyPSA Linear Optimal Power Flow via HiGHS solver..."):
            try:
                if input_mode == "Use Explicit 24-Hour Profile":
                    demand_vals = get_default_demand_24()
                    res = client.solve_dispatch(
                        demand_mw=demand_vals,
                        include_load_shedding=include_shedding,
                        load_shedding_cost=float(shed_cost),
                    )
                else:
                    weather_vals = [
                        {
                            "temperature": float(row["temperature"]),
                            "humidity": float(row["humidity"]),
                            "wind_speed": float(row["wind_speed"]),
                            "solar_irradiance": float(row["solar_irradiance"]),
                        }
                        for _, row in pd.DataFrame(get_default_weather_24()).iterrows()
                    ]
                    res = client.solve_dispatch(
                        use_forecast_24h=True,
                        weather_forecasts=weather_vals,
                        include_load_shedding=include_shedding,
                        load_shedding_cost=float(shed_cost),
                    )

                st.session_state.last_dispatch = res
            except APIClientError as err:
                st.error(f"Dispatch optimization failed: {err}")

    if "last_dispatch" in st.session_state:
        d = st.session_state.last_dispatch
        is_feas = d.get("is_feasible", False)
        status_text = d.get("optimization_status", "unknown")

        if not is_feas:
            st.error(
                f"🚨 **Optimization Infeasible:** Solver status: `{status_text}`.\n\n"
                "Demand exceeded generator capacity (2,700 MW) with load-shedding disabled. "
                "Enable 'Include Unserved Energy Generator' to solve with explicit shortage penalty."
            )
        else:
            st.success(f"✅ Optimization Feasible & Solved: `{status_text}`")

            # Metrics
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Total 24h Cost", f"₹{d.get('total_cost', 0):,.2f}")
            with m2:
                st.metric("Total Energy Dispatched", f"{d.get('total_demand_mwh', 0):,.1f} MWh")
            with m3:
                st.metric("Peak Demand", f"{d.get('peak_demand_mw', 0):,.1f} MW")
            with m4:
                unserved = d.get("unserved_energy_mwh", 0.0)
                st.metric("Unserved Energy", f"{unserved:,.1f} MWh", delta=f"{d.get('solar_share_pct', 0):.1f}% Solar")

            col_chart1, col_chart2 = st.columns([2, 1])
            with col_chart1:
                fig_sched = plot_dispatch_schedule(d.get("hourly_schedule", []))
                st.plotly_chart(fig_sched, use_container_width=True)
            with col_chart2:
                fig_pie = plot_fuel_shares_donut(
                    coal_mwh=d.get("coal_total_mwh", 0),
                    gas_mwh=d.get("gas_total_mwh", 0),
                    solar_mwh=d.get("solar_total_mwh", 0),
                    unserved_mwh=d.get("unserved_energy_mwh", 0),
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with st.expander("📋 Detailed 24-Hour Dispatch Schedule Table"):
                st.dataframe(pd.DataFrame(d.get("hourly_schedule", [])), use_container_width=True)

            st.caption(f"ℹ️ **Fleet Specification & Solver:** {d.get('provenance_note')}")


# ══════════════════════════════════════════════════════════════
# PAGE 5: MONTE CARLO RISK ANALYSIS
# ══════════════════════════════════════════════════════════════
elif nav_choice == "🎲 Monte Carlo Risk Analysis":
    st.header("🎲 Monte Carlo Risk & Uncertainty Assessment")
    st.markdown(
        "Quantifies operational risk across simulated 24-hour scenarios using "
        "**block-bootstrapped empirical residuals** from the out-of-fold stacked ensemble."
    )

    st.markdown(
        "Evaluates **Loss of Load Probability (LOLP)**, **Expected Energy Not Served (EENS)**, "
        "and empirical scenario quantiles across peak load, total cost, and generator utilization."
    )

    col_rc1, col_rc2 = st.columns(2)
    with col_rc1:
        n_sims = st.slider("Number of Simulations", min_value=10, max_value=500, value=100, step=10)
        seed = st.number_input("Reproducibility Seed", value=42, min_value=1, step=1)
    with col_rc2:
        engine = st.selectbox(
            "Dispatch Engine",
            ["merit_order", "pypsa"],
            index=0,
            help="'merit_order' is fast (~15ms for 100 draws). 'pypsa' executes numerical LOPF per scenario (~1s per draw).",
        )
        clip_residuals = st.checkbox("Enable Residual Tail Clipping (P05–P95)", value=True)

    run_risk = st.button("🎲 Run Monte Carlo Risk Simulation", type="primary")

    if run_risk:
        with st.spinner(f"Simulating {n_sims} scenarios via {engine} engine..."):
            try:
                demand_vals = get_default_demand_24()
                risk_res = client.run_risk_analysis(
                    demand_mw=demand_vals,
                    n_simulations=int(n_sims),
                    random_seed=int(seed),
                    dispatch_engine=engine,
                    enable_residual_clipping=clip_residuals,
                )
                st.session_state.last_risk = risk_res
                st.success(f"Evaluated {n_sims} Monte Carlo scenarios successfully!")
            except APIClientError as err:
                st.error(f"Risk analysis failed: {err}")

    if "last_risk" in st.session_state:
        r = st.session_state.last_risk
        metrics = r.get("risk_metrics", {})
        quantiles = r.get("quantile_summaries", [])

        # Reliability KPIs
        rk1, rk2, rk3, rk4 = st.columns(4)
        with rk1:
            lolp = metrics.get("LOLP", 0.0)
            st.metric("Loss of Load Probability (LOLP)", f"{lolp * 100:.2f}%")
        with rk2:
            eens = metrics.get("EENS_mwh", 0.0)
            st.metric("Expected Unserved Energy (EENS)", f"{eens:.2f} MWh")
        with rk3:
            res_suf = metrics.get("reserve_sufficiency", 1.0)
            st.metric("Reserve Sufficiency", f"{res_suf * 100:.2f}%")
        with rk4:
            st.metric("Expected Cost (Mean)", f"₹{metrics.get('mean_cost', 0):,.2f}")

        # Cost Bounds
        ck1, ck2, ck3, ck4 = st.columns(4)
        with ck1:
            st.metric("Best-Case Cost (Min)", f"₹{metrics.get('min_cost', 0):,.2f}")
        with ck2:
            st.metric("Median Cost (P50)", f"₹{metrics.get('median_cost', 0):,.2f}")
        with ck3:
            st.metric("P95 Worst-Case Cost", f"₹{metrics.get('p95_cost', 0):,.2f}")
        with ck4:
            st.metric("Peak Demand P95", f"{metrics.get('peak_demand_p95_mw', 0):,.1f} MW")

        # Quantile Range Chart
        if quantiles:
            st.subheader("Empirical Scenario Quantile Ranges")
            fig_q = plot_risk_quantiles(quantiles)
            st.plotly_chart(fig_q, use_container_width=True)

            with st.expander("📋 Detailed Empirical Quantiles Table"):
                df_q = pd.DataFrame(quantiles)
                st.dataframe(
                    df_q,
                    use_container_width=True,
                    column_config={
                        "p05": st.column_config.NumberColumn("5th Percentile (P05)", format="%.2f"),
                        "p50": st.column_config.NumberColumn("50th Percentile (Median)", format="%.2f"),
                        "p95": st.column_config.NumberColumn("95th Percentile (P95)", format="%.2f"),
                    },
                )

        st.caption(f"ℹ️ **Interpretation Note:** {r.get('uncertainty_provenance_note')}")


# ══════════════════════════════════════════════════════════════
# PAGE 6: DEMO DATA CONTROLS
# ══════════════════════════════════════════════════════════════
elif nav_choice == "🧪 Demo Data Controls":
    st.header("🧪 Demo Simulation Controls")
    st.markdown(
        "Provides controlled hourly simulation advances for demonstration sessions. "
        "Generated records are explicitly tagged as `is_simulated=true` and **never** "
        "overwrite the genuine historical CSV dataset."
    )

    if not sim_enabled:
        st.warning(
            "🔒 **Demo Simulation Mode is Currently Disabled.**\n\n"
            "The backend is currently running in read-only historical playback mode (`ENABLE_SIMULATION_MODE=false`).\n\n"
            "To enable demo generation, restart the backend with:\n"
            "`$env:ENABLE_SIMULATION_MODE='true'; uvicorn api.app:app --host 127.0.0.1 --port 8000`"
        )
    else:
        st.success("🟢 Demo Simulation Mode is Active on the backend.")

        hours_advance = st.slider("Hours to Advance", min_value=1, max_value=24, value=1)
        confirm_demo = st.checkbox(
            "I understand this creates synthetic demo records in server memory for testing.",
            value=False,
        )

        btn_gen = st.button("🧪 Advance Demo Telemetry", type="primary", disabled=not confirm_demo)

        if btn_gen:
            with st.spinner(f"Advancing simulation by {hours_advance} hour(s)..."):
                try:
                    gen_res = client.generate_demo_data(hours=int(hours_advance))
                    st.success(f"Generated {gen_res.get('generated_count', 0)} simulated record(s)!")
                    st.dataframe(pd.DataFrame(gen_res.get("records", [])), use_container_width=True)
                except APIClientError as err:
                    st.error(f"Generation failed: {err}")
