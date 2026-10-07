"""
Visualization and data helper components for the B-SGLEM Streamlit dashboard.
"""
from typing import List, Dict, Any, Optional
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


# ── Color Palette & Styles ──────────────────────────────────────────────
COLOR_DEMAND = "#2563EB"       # Royal Blue
COLOR_COAL = "#4B5563"         # Charcoal / Slate
COLOR_GAS = "#F59E0B"          # Amber / Orange
COLOR_SOLAR = "#FBBF24"        # Yellow / Gold
COLOR_UNSERVED = "#EF4444"     # Red
COLOR_RESERVE = "#10B981"      # Emerald Green
COLOR_BOUNDS = "#93C5FD"       # Soft Light Blue


def get_default_weather_24() -> List[Dict[str, float]]:
    """
    Returns realistic 24-hour diurnal weather observations for Bengaluru.
    Labeled as demonstration inputs.
    """
    profile = []
    for h in range(24):
        # Diurnal temp cycle: min ~19C at 05:00, peak ~29C at 14:00
        t_cycle = 24.0 + 5.0 * ((h - 6) % 24 - 12) / 12.0
        temp = round(24.0 + 4.5 * ((h >= 6 and h <= 18) and ((12 - abs(h - 13)) / 7.0) or -0.8), 2)
        # Humidity inverse to temperature
        humidity = round(85.0 - (temp - 19.0) * 3.5, 1)
        humidity = min(max(humidity, 35.0), 95.0)
        # Solar irradiance during daylight (06:00 to 18:00)
        if 6 <= h <= 18:
            solar = round(max(0.0, 850.0 * (1.0 - ((h - 12) / 6.0) ** 2)), 1)
        else:
            solar = 0.0
        wind = round(1.8 + 0.8 * ((h % 5) / 5.0), 2)

        profile.append({
            "hour": h,
            "temperature": temp,
            "humidity": humidity,
            "wind_speed": wind,
            "solar_irradiance": solar,
        })
    return profile


def get_default_demand_24() -> List[float]:
    """Realistic 24-hour Bengaluru electrical demand profile in MW."""
    return [
        1650.0, 1580.0, 1520.0, 1490.0, 1510.0, 1600.0,
        1780.0, 1950.0, 2100.0, 2180.0, 2220.0, 2200.0,
        2150.0, 2100.0, 2080.0, 2050.0, 2100.0, 2240.0,
        2290.0, 2250.0, 2150.0, 2000.0, 1850.0, 1720.0,
    ]


def get_default_baseline_features() -> Dict[str, Any]:
    """Canonical 28-feature dictionary representing an operational hour."""
    return {
        "is_weekend": 0,
        "is_holiday": 0,
        "is_festival": 0,
        "is_pre_holiday": 0,
        "is_post_holiday": 0,
        "is_pre_festival": 0,
        "is_post_festival": 0,
        "workday_after_holiday": 0,
        "temperature": 24.5,
        "humidity": 62.0,
        "wind_speed": 2.1,
        "solar_irradiance": 420.0,
        "lag_1": 2165.26,
        "lag_24": 2126.03,
        "lag_168": 2027.28,
        "rolling_mean_24": 1973.15,
        "rolling_std_24": 115.85,
        "hour_sin": 0.0,
        "hour_cos": 1.0,
        "day_sin": 0.9749,
        "day_cos": -0.2225,
        "month_sin": 0.5,
        "month_cos": 0.8660,
        "lag_48": 2155.39,
        "lag_72": 1832.71,
        "rolling_max_24": 2165.26,
        "rolling_min_24": 1785.32,
        "temp_x_hour_sin": 0.0,
    }


# ── Plotly Visualizations ───────────────────────────────────────────────

def plot_historical_load(records: List[Dict[str, Any]]) -> go.Figure:
    """Plots historical/simulated load series over time."""
    df = pd.DataFrame(records)
    if "load_mw" not in df.columns or "timestamp" not in df.columns:
        fig = go.Figure()
        fig.update_layout(title="No load records found in query.")
        return fig

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["load_mw"],
        mode="lines",
        name="Historical Demand (MW)",
        line=dict(color=COLOR_DEMAND, width=2.5),
        hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>Demand: %{y:.2f} MW<extra></extra>",
    ))

    fig.update_layout(
        title="Historical Grid Electrical Demand (MW)",
        xaxis_title="Observation Timestamp (Asia/Kolkata)",
        yaxis_title="Electrical Demand (MW)",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_forecast_24h(predictions: List[Dict[str, Any]]) -> go.Figure:
    """Plots recursive 24-hour day-ahead forecast with empirical uncertainty bounds."""
    df = pd.DataFrame(predictions)
    fig = go.Figure()

    # Empirical residual bounds shading
    if "lower_bound_mw" in df.columns and "upper_bound_mw" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["hour"],
            y=df["upper_bound_mw"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=df["hour"],
            y=df["lower_bound_mw"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(147, 197, 253, 0.35)",
            name="Empirical Scenario Bounds (P05–P95)",
            hoverinfo="skip",
        ))

    # Point forecast line
    fig.add_trace(go.Scatter(
        x=df["hour"],
        y=df["predicted_load_mw"],
        mode="lines+markers",
        name="Predicted Load (MW)",
        line=dict(color=COLOR_DEMAND, width=3.0),
        marker=dict(size=6, color=COLOR_DEMAND),
        hovertemplate="Hour %{x}: <b>%{y:.2f} MW</b><extra></extra>",
    ))

    # Operating reserve line (20%)
    if "spinning_reserve_mw" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["hour"],
            y=df["spinning_reserve_mw"],
            mode="lines",
            name="20% Operating Reserve (MW)",
            line=dict(color=COLOR_RESERVE, width=1.8, dash="dot"),
            hovertemplate="Reserve: %{y:.2f} MW<extra></extra>",
        ))

    fig.update_layout(
        title="24-Hour Day-Ahead Recursive Load Forecast",
        xaxis_title="Forecast Horizon (Hour 0 to 23)",
        yaxis_title="Grid Demand (MW)",
        xaxis=dict(tickmode="linear", tick0=0, dtick=2),
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_scenario_comparison(results: List[Dict[str, Any]], baseline_mw: float) -> go.Figure:
    """Plots what-if scenario predictions compared against baseline."""
    df = pd.DataFrame(results)

    colors = [
        "#10B981" if val <= 0 else "#EF4444"
        for val in df["delta_mw"]
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["scenario_name"],
        y=df["predicted_load_mw"],
        marker_color="#3B82F6",
        text=df["predicted_load_mw"].apply(lambda v: f"{v:,.1f} MW"),
        textposition="outside",
        name="Scenario Forecast (MW)",
        hovertemplate="<b>%{x}</b><br>Forecast: %{y:.1f} MW<extra></extra>",
    ))

    # Baseline reference horizontal line
    fig.add_hline(
        y=baseline_mw,
        line_dash="dash",
        line_color="#1E293B",
        annotation_text=f"Baseline: {baseline_mw:,.1f} MW",
        annotation_position="top left",
    )

    fig.update_layout(
        title="Scenario Stress Testing Forecast Comparison",
        xaxis_title="Scenario",
        yaxis_title="Predicted Demand (MW)",
        margin=dict(l=40, r=40, t=50, b=60),
        template="plotly_white",
    )
    return fig


def plot_dispatch_schedule(schedule: List[Dict[str, Any]]) -> go.Figure:
    """Plots stacked generation dispatch schedule by generator fuel type."""
    df = pd.DataFrame(schedule)
    fig = go.Figure()

    # Stacked area/bar components
    fig.add_trace(go.Bar(
        x=df["hour"],
        y=df["solar_mw"],
        name="Solar (₹0.0)",
        marker_color=COLOR_SOLAR,
        hovertemplate="Solar: %{y:.1f} MW<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=df["hour"],
        y=df["coal_mw"],
        name="Coal (₹3.5)",
        marker_color=COLOR_COAL,
        hovertemplate="Coal: %{y:.1f} MW<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=df["hour"],
        y=df["gas_mw"],
        name="Gas (₹6.0)",
        marker_color=COLOR_GAS,
        hovertemplate="Gas: %{y:.1f} MW<extra></extra>",
    ))
    if "unserved_mw" in df.columns and df["unserved_mw"].sum() > 0:
        fig.add_trace(go.Bar(
            x=df["hour"],
            y=df["unserved_mw"],
            name="Unserved / Load Shed",
            marker_color=COLOR_UNSERVED,
            hovertemplate="Shortage: %{y:.1f} MW<extra></extra>",
        ))

    # Demand line overlay
    fig.add_trace(go.Scatter(
        x=df["hour"],
        y=df["demand_mw"],
        name="Grid Demand (MW)",
        line=dict(color="#1E1B4B", width=2.5, dash="solid"),
        hovertemplate="Demand: %{y:.1f} MW<extra></extra>",
    ))

    fig.update_layout(
        barmode="stack",
        title="24-Hour Economic Generation Dispatch Schedule",
        xaxis_title="Hour (0 to 23)",
        yaxis_title="Power (MW)",
        xaxis=dict(tickmode="linear", tick0=0, dtick=2),
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_fuel_shares_donut(coal_mwh: float, gas_mwh: float, solar_mwh: float, unserved_mwh: float = 0.0) -> go.Figure:
    """Renders energy mix donut chart."""
    labels = ["Coal Fleet", "Gas Peaker", "Solar Farm"]
    values = [coal_mwh, gas_mwh, solar_mwh]
    colors = [COLOR_COAL, COLOR_GAS, COLOR_SOLAR]

    if unserved_mwh > 0.0:
        labels.append("Unserved Shortage")
        values.append(unserved_mwh)
        colors.append(COLOR_UNSERVED)

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>%{value:,.1f} MWh (%{percent})<extra></extra>",
    )])

    fig.update_layout(
        title="24-Hour Energy Generation Mix (MWh)",
        margin=dict(l=20, r=20, t=40, b=20),
        template="plotly_white",
    )
    return fig


def plot_risk_quantiles(quantiles: List[Dict[str, Any]]) -> go.Figure:
    """Plots empirical quantile ranges (P05, P50, P95) for key risk metrics."""
    df = pd.DataFrame(quantiles)

    fig = go.Figure()
    for _, row in df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row["p05"], row["p50"], row["p95"]],
            y=[row["metric"], row["metric"], row["metric"]],
            mode="lines+markers",
            name=row["metric"],
            line=dict(color="#3B82F6", width=4),
            marker=dict(size=[8, 12, 8], color=["#93C5FD", "#1D4ED8", "#93C5FD"]),
            hovertemplate=(
                f"<b>{row['metric']}</b><br>"
                f"P05: {row['p05']:,.1f}<br>"
                f"P50 (Median): {row['p50']:,.1f}<br>"
                f"P95: {row['p95']:,.1f}<extra></extra>"
            ),
            showlegend=False,
        ))

    fig.update_layout(
        title="Empirical Scenario Quantile Ranges (P05 — Median P50 — P95)",
        xaxis_title="Metric Value",
        yaxis_title="",
        margin=dict(l=150, r=40, t=50, b=40),
        template="plotly_white",
    )
    return fig
