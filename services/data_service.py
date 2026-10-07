"""
Data Service for B-SGLEM Load Forecasting System.

Provides:
- Reliable historical dataset loading, caching, and chronological ordering.
- Validation of timestamps, duplicate detection, and gap detection.
- Retrieval of latest grid records and chronological history.
- Controlled, opt-in simulated hourly generation (without modifying historical CSVs).
- Feature preparation pipeline strictly producing the canonical 28 features
  without target leakage.
- Recursive 24-hour multi-step forecasting engine integrating the saved XGBoost model.
"""
import os
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd

from config import (
    BASE_DIR,
    DATA_DIR,
    ACTIVE_DATASET,
    DATA_SOURCE_MODE,
    ENABLE_SIMULATION_MODE,
    FEATURE_COLS,
)


class DataService:
    """
    Dedicated service managing grid telemetry data, feature preparation,
    controlled demo simulation, and multi-step recursive forecasting.
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        active_dataset: Optional[str] = None,
        data_source_mode: Optional[str] = None,
        simulation_enabled: Optional[bool] = None,
    ):
        self.data_dir = Path(data_dir or DATA_DIR)
        self.active_dataset = active_dataset or ACTIVE_DATASET
        self.data_source_mode = data_source_mode or DATA_SOURCE_MODE
        self.simulation_enabled = (
            simulation_enabled
            if simulation_enabled is not None
            else ENABLE_SIMULATION_MODE
        )

        # In-memory cache for loaded historical DataFrame
        self._cached_historical_df: Optional[pd.DataFrame] = None
        self._cached_dataset_name: Optional[str] = None

        # In-memory buffer for generated demo/simulated records (never written to historical CSVs)
        self._simulated_records: List[Dict[str, Any]] = []

    def initialize(self) -> None:
        """
        Pre-loads and validates the active historical dataset into memory.
        Called once during application startup.
        """
        self.load_historical_data(self.active_dataset)

    def load_historical_data(self, dataset_name: Optional[str] = None) -> pd.DataFrame:
        """
        Loads, validates, and sorts the specified historical dataset.
        Caches the result in memory to avoid repeated disk reads.
        """
        dataset_key = dataset_name or self.active_dataset

        if self._cached_historical_df is not None and self._cached_dataset_name == dataset_key:
            return self._cached_historical_df

        csv_files: List[Path] = []
        if dataset_key == "2024":
            csv_files = [self.data_dir / "bengaluru_power_2024_advanced.csv"]
        elif dataset_key == "2023":
            csv_files = [self.data_dir / "bengaluru_power_full_advanced.csv"]
        elif dataset_key == "combined":
            csv_files = [
                self.data_dir / "bengaluru_power_full_advanced.csv",
                self.data_dir / "bengaluru_power_2024_advanced.csv",
            ]
        else:
            # Custom file path support
            custom_path = self.data_dir / f"{dataset_key}.csv"
            if custom_path.exists():
                csv_files = [custom_path]
            else:
                raise FileNotFoundError(
                    f"Dataset '{dataset_key}' not found in data directory: {self.data_dir}"
                )

        dfs = []
        for file_path in csv_files:
            if not file_path.exists():
                raise FileNotFoundError(f"Required historical dataset file missing: {file_path}")
            df_part = pd.read_csv(file_path)
            dfs.append(df_part)

        if not dfs:
            raise ValueError(f"No data loaded for dataset key: {dataset_key}")

        combined_df = pd.concat(dfs, ignore_index=True)

        # Parse and validate datetime
        combined_df["parsed_dt"] = pd.to_datetime(combined_df["datetime"], dayfirst=True, errors="coerce")
        null_dates = combined_df["parsed_dt"].isna().sum()
        if null_dates > 0:
            combined_df = combined_df.dropna(subset=["parsed_dt"])

        # Sort chronologically and deduplicate timestamps
        combined_df = combined_df.sort_values("parsed_dt").drop_duplicates(subset=["parsed_dt"], keep="last").reset_index(drop=True)

        # Add formatted standard ISO-8601 timestamp string
        combined_df["timestamp_iso"] = combined_df["parsed_dt"].dt.strftime("%Y-%m-%dT%H:%M:%S")

        self._cached_historical_df = combined_df
        self._cached_dataset_name = dataset_key
        return self._cached_historical_df

    def detect_gaps(self, df: Optional[pd.DataFrame] = None) -> List[Tuple[str, str, int]]:
        """
        Detects missing hourly intervals within the dataset.
        Returns a list of tuples: (start_timestamp, end_timestamp, gap_hours).
        """
        target_df = df if df is not None else self.load_historical_data()
        if len(target_df) <= 1:
            return []

        dt_series = target_df["parsed_dt"]
        diffs = dt_series.diff()

        gaps = []
        gap_indices = diffs[diffs > pd.Timedelta(hours=1)].index
        for idx in gap_indices:
            prev_dt = dt_series.iloc[idx - 1]
            curr_dt = dt_series.iloc[idx]
            gap_hours = int((curr_dt - prev_dt).total_seconds() // 3600) - 1
            gaps.append((
                prev_dt.strftime("%Y-%m-%dT%H:%M:%S"),
                curr_dt.strftime("%Y-%m-%dT%H:%M:%S"),
                gap_hours
            ))

        return gaps

    def get_latest_record(self) -> Dict[str, Any]:
        """
        Retrieves the latest available grid record from the active data source.
        Prioritizes simulated records if simulation mode is active and has generated data.
        """
        if self.data_source_mode == "live":
            raise NotImplementedError(
                "Live grid SCADA/telemetry provider is not implemented or configured. "
                "Current system supports 'historical' and 'simulated' modes."
            )

        # Check if simulated records exist
        if self._simulated_records and self.data_source_mode in ("simulated", "historical"):
            latest_sim = self._simulated_records[-1]
            return {
                "timestamp": latest_sim["timestamp"],
                "load_mw": latest_sim["load_mw"],
                "weather": latest_sim["weather"],
                "calendar": latest_sim["calendar"],
                "data_source_mode": "simulated",
                "is_simulated": True,
            }

        df = self.load_historical_data()
        if df.empty:
            raise ValueError("No historical records available in active dataset.")

        last_row = df.iloc[-1]
        return {
            "timestamp": last_row["timestamp_iso"],
            "load_mw": round(float(last_row["load"]), 2),
            "weather": {
                "temperature": round(float(last_row["temperature"]), 2),
                "humidity": round(float(last_row["humidity"]), 2),
                "wind_speed": round(float(last_row["wind_speed"]), 2),
                "solar_irradiance": round(float(last_row["solar_irradiance"]), 2),
            },
            "calendar": {
                "is_weekend": int(last_row["is_weekend"]),
                "is_holiday": int(last_row["is_holiday"]),
                "is_festival": int(last_row["is_festival"]),
                "is_pre_holiday": int(last_row["is_pre_holiday"]),
                "is_post_holiday": int(last_row["is_post_holiday"]),
                "is_pre_festival": int(last_row["is_pre_festival"]),
                "is_post_festival": int(last_row["is_post_festival"]),
                "workday_after_holiday": int(last_row["workday_after_holiday"]),
            },
            "data_source_mode": "historical",
            "is_simulated": False,
        }

    def get_history(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """
        Retrieves historical records with chronological filtering and pagination.

        Args:
            start (Optional[str]): Start timestamp (ISO format, inclusive).
            end (Optional[str]): End timestamp (ISO format, inclusive).
            limit (int): Maximum records to return (1 to 1000).
            offset (int): Pagination offset (>= 0).

        Returns:
            Dict[str, Any]: History records and pagination metadata.
        """
        if limit < 1 or limit > 1000:
            raise ValueError(f"Limit must be between 1 and 1000, got {limit}")
        if offset < 0:
            raise ValueError(f"Offset must be non-negative, got {offset}")

        df = self.load_historical_data()

        # Parse start and end filters
        if start is not None:
            try:
                start_dt = pd.to_datetime(start)
            except Exception as e:
                raise ValueError(f"Invalid start timestamp format '{start}': {e}")
            df = df[df["parsed_dt"] >= start_dt]

        if end is not None:
            try:
                end_dt = pd.to_datetime(end)
            except Exception as e:
                raise ValueError(f"Invalid end timestamp format '{end}': {e}")
            df = df[df["parsed_dt"] <= end_dt]

        if start is not None and end is not None and start_dt > end_dt:
            raise ValueError(f"Start timestamp ({start}) cannot be after end timestamp ({end})")

        total_matching = len(df)
        paged_df = df.iloc[offset : offset + limit]

        records = []
        for _, row in paged_df.iterrows():
            records.append({
                "timestamp": row["timestamp_iso"],
                "load_mw": round(float(row["load"]), 2),
                "weather": {
                    "temperature": round(float(row["temperature"]), 2),
                    "humidity": round(float(row["humidity"]), 2),
                    "wind_speed": round(float(row["wind_speed"]), 2),
                    "solar_irradiance": round(float(row["solar_irradiance"]), 2),
                },
                "calendar": {
                    "is_weekend": int(row["is_weekend"]),
                    "is_holiday": int(row["is_holiday"]),
                    "is_festival": int(row["is_festival"]),
                    "is_pre_holiday": int(row["is_pre_holiday"]),
                    "is_post_holiday": int(row["is_post_holiday"]),
                    "is_pre_festival": int(row["is_pre_festival"]),
                    "is_post_festival": int(row["is_post_festival"]),
                    "workday_after_holiday": int(row["workday_after_holiday"]),
                },
                "data_source_mode": "historical",
                "is_simulated": False,
            })

        return {
            "total_records": total_matching,
            "returned_records": len(records),
            "limit": limit,
            "offset": offset,
            "data_source_mode": self.data_source_mode,
            "is_simulated": False,
            "timezone": "Asia/Kolkata (UTC+05:30)",
            "records": records,
        }

    def generate_simulated_hour(
        self,
        count: int = 1,
        weather_override: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Advances simulated hourly data in controlled demo mode.
        Simulated records are stored in memory and NEVER overwrite historical CSVs.
        """
        if not self.simulation_enabled:
            raise PermissionError(
                "Simulation mode is disabled in configuration. "
                "Set ENABLE_SIMULATION_MODE=true to enable demo data generation."
            )

        if self.data_source_mode == "live":
            raise ValueError("Live telemetry mode does not support simulation generation.")

        if count < 1 or count > 168:
            raise ValueError(f"Generation count must be between 1 and 168 hours, got {count}")

        # Determine last timestamp
        if self._simulated_records:
            last_dt = pd.to_datetime(self._simulated_records[-1]["timestamp"])
            last_load = self._simulated_records[-1]["load_mw"]
        else:
            df = self.load_historical_data()
            last_dt = df["parsed_dt"].iloc[-1]
            last_load = float(df["load"].iloc[-1])

        new_records = []
        for step in range(1, count + 1):
            next_dt = last_dt + pd.Timedelta(hours=step)
            h = next_dt.hour
            dow = next_dt.weekday()
            month = next_dt.month

            # Realistic deterministic diurnal cycle with mean reversion
            diurnal_offset = 260.0 * np.sin(2 * np.pi * (h - 7) / 24)
            weekday_offset = 40.0 if dow < 5 else -90.0
            seasonal_offset = 20.0 * np.sin(2 * np.pi * month / 12)
            sim_load = round(1880.0 + diurnal_offset + weekday_offset + seasonal_offset, 2)

            # Weather simulation or override
            if weather_override:
                temp = float(weather_override.get("temperature", 24.0))
                humidity = float(weather_override.get("humidity", 65.0))
                wind = float(weather_override.get("wind_speed", 2.0))
                solar = float(weather_override.get("solar_irradiance", 0.0))
            else:
                temp = round(21.5 + 6.5 * np.sin(2 * np.pi * (h - 9) / 24), 2)
                solar = round(max(0.0, 780.0 * np.sin(np.pi * (h - 6) / 12)), 2) if 6 <= h <= 18 else 0.0
                humidity = round(78.0 - 22.0 * np.sin(2 * np.pi * (h - 9) / 24), 2)
                wind = round(2.0 + 0.9 * np.cos(2 * np.pi * h / 24), 2)

            is_weekend = 1 if dow in (5, 6) else 0

            record = {
                "timestamp": next_dt.strftime("%Y-%m-%dT%H:%M:%S"),
                "load_mw": sim_load,
                "weather": {
                    "temperature": temp,
                    "humidity": humidity,
                    "wind_speed": wind,
                    "solar_irradiance": solar,
                },
                "calendar": {
                    "is_weekend": is_weekend,
                    "is_holiday": 0,
                    "is_festival": 0,
                    "is_pre_holiday": 0,
                    "is_post_holiday": 0,
                    "is_pre_festival": 0,
                    "is_post_festival": 0,
                    "workday_after_holiday": 0,
                },
                "data_source_mode": "simulated",
                "is_simulated": True,
            }
            new_records.append(record)
            self._simulated_records.append(record)

        return new_records

    def clear_simulated_records(self) -> None:
        """
        Clears in-memory simulated records buffer (used for testing).
        """
        self._simulated_records.clear()

    def prepare_28_features(
        self,
        target_dt: pd.Timestamp,
        load_history_168: List[float],
        weather_inputs: Dict[str, float],
        calendar_inputs: Optional[Dict[str, int]] = None,
    ) -> pd.DataFrame:
        """
        Prepares a single row containing the exact 28 canonical features in
        order of config.FEATURE_COLS for forecasting target_dt.

        Enforces zero target leakage: target_dt load is never used.
        Requires exactly 168 preceding historical load observations.
        """
        if len(load_history_168) != 168:
            raise ValueError(
                f"Expected exactly 168 preceding load observations, got {len(load_history_168)}"
            )

        loads = np.array(load_history_168, dtype=float)
        if not np.all(np.isfinite(loads)):
            raise ValueError("Load history contains non-finite (NaN or Inf) values.")

        # Lags: loads[-1] is t-1, loads[-24] is t-24, etc.
        lag_1 = loads[-1]
        lag_24 = loads[-24]
        lag_48 = loads[-48]
        lag_72 = loads[-72]
        lag_168 = loads[-168]

        # 24-hour rolling statistics preceding target hour: loads[-24:]
        window_24 = loads[-24:]
        rolling_mean_24 = float(np.mean(window_24))
        rolling_std_24 = float(np.std(window_24, ddof=1))
        rolling_max_24 = float(np.max(window_24))
        rolling_min_24 = float(np.min(window_24))

        # Fourier cyclical harmonics for target_dt
        h = target_dt.hour
        dow = target_dt.weekday()
        m = target_dt.month

        hour_sin = float(np.sin(2 * np.pi * h / 24))
        hour_cos = float(np.cos(2 * np.pi * h / 24))
        day_sin = float(np.sin(2 * np.pi * dow / 7))
        day_cos = float(np.cos(2 * np.pi * dow / 7))
        month_sin = float(np.sin(2 * np.pi * m / 12))
        month_cos = float(np.cos(2 * np.pi * m / 12))

        # Calendar features
        cal = calendar_inputs or {}
        is_weekend = cal.get("is_weekend", 1 if dow in (5, 6) else 0)
        is_holiday = cal.get("is_holiday", 0)
        is_festival = cal.get("is_festival", 0)
        is_pre_holiday = cal.get("is_pre_holiday", 0)
        is_post_holiday = cal.get("is_post_holiday", 0)
        is_pre_festival = cal.get("is_pre_festival", 0)
        is_post_festival = cal.get("is_post_festival", 0)
        workday_after_holiday = cal.get("workday_after_holiday", 0)

        # Weather features
        for key in ["temperature", "humidity", "wind_speed", "solar_irradiance"]:
            if key not in weather_inputs:
                raise ValueError(f"Missing required weather feature '{key}'")
            val = float(weather_inputs[key])
            if not np.isfinite(val):
                raise ValueError(f"Weather feature '{key}' is non-finite: {val}")

        temperature = float(weather_inputs["temperature"])
        humidity = float(weather_inputs["humidity"])
        wind_speed = float(weather_inputs["wind_speed"])
        solar_irradiance = float(weather_inputs["solar_irradiance"])

        # Interaction feature
        temp_x_hour_sin = temperature * hour_sin

        row_dict = {
            "is_weekend": is_weekend,
            "is_holiday": is_holiday,
            "is_festival": is_festival,
            "is_pre_holiday": is_pre_holiday,
            "is_post_holiday": is_post_holiday,
            "is_pre_festival": is_pre_festival,
            "is_post_festival": is_post_festival,
            "workday_after_holiday": workday_after_holiday,
            "temperature": temperature,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "solar_irradiance": solar_irradiance,
            "lag_1": lag_1,
            "lag_24": lag_24,
            "lag_168": lag_168,
            "rolling_mean_24": rolling_mean_24,
            "rolling_std_24": rolling_std_24,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "day_sin": day_sin,
            "day_cos": day_cos,
            "month_sin": month_sin,
            "month_cos": month_cos,
            "lag_48": lag_48,
            "lag_72": lag_72,
            "rolling_max_24": rolling_max_24,
            "rolling_min_24": rolling_min_24,
            "temp_x_hour_sin": temp_x_hour_sin,
        }

        # Validate exact 28 feature ordering
        return pd.DataFrame([row_dict], columns=FEATURE_COLS)

    def forecast_24h_recursive(
        self,
        xgb_model: Any,
        q05: float,
        q95: float,
        weather_forecasts: List[Dict[str, float]],
        calendar_overrides: Optional[List[Dict[str, int]]] = None,
    ) -> Dict[str, Any]:
        """
        Executes a genuine 24-hour day-ahead recursive forecast using the trained XGBoost model.

        Requirements:
        - 24 hourly weather forecasts must be explicitly supplied (no silent reuse of historical weather).
        - Origin is the latest available timestamp in active telemetry.
        - Previous 168 hours of actual loads are extracted for lag features.
        - Recursive loop: predicted load at hour k is fed back into load history for hours k+1..24.
        """
        if len(weather_forecasts) != 24:
            raise ValueError(
                f"24-hour forecasting requires exactly 24 hourly weather forecast items, got {len(weather_forecasts)}"
            )

        df = self.load_historical_data()
        if len(df) < 168:
            raise ValueError(f"Insufficient history: active dataset has {len(df)} rows, minimum 168 required.")

        # Determine origin timestamp and trailing 168 hours of load history
        if self._simulated_records and self.data_source_mode in ("simulated", "historical"):
            origin_dt = pd.to_datetime(self._simulated_records[-1]["timestamp"])
            origin_source = "simulated"
            # Blend historical and simulated loads if simulated has < 168 rows
            sim_loads = [r["load_mw"] for r in self._simulated_records]
            if len(sim_loads) >= 168:
                load_buffer = sim_loads[-168:]
            else:
                needed_hist = 168 - len(sim_loads)
                load_buffer = list(df["load"].iloc[-needed_hist:].astype(float)) + sim_loads
        else:
            origin_dt = df["parsed_dt"].iloc[-1]
            origin_source = "historical"
            load_buffer = list(df["load"].iloc[-168:].astype(float))

        # Check for gaps in the trailing 168 hours
        trailing_dt = df["parsed_dt"].iloc[-168:]
        diffs = trailing_dt.diff()
        if (diffs[1:] > pd.Timedelta(hours=1)).any():
            raise ValueError(
                "Gap detected in trailing 168-hour historical load sequence. "
                "Forecasting requires continuous hourly observations."
            )

        predictions = []
        for step in range(1, 25):
            target_dt = origin_dt + pd.Timedelta(hours=step)
            weather_step = weather_forecasts[step - 1]
            cal_step = calendar_overrides[step - 1] if calendar_overrides else None

            # Prepare 28 features using rolling load buffer (zero future actual load used)
            feature_df = self.prepare_28_features(
                target_dt=target_dt,
                load_history_168=load_buffer,
                weather_inputs=weather_step,
                calendar_inputs=cal_step,
            )

            # Generate prediction via operational XGBoost path
            pred_mw = float(xgb_model.predict(feature_df)[0])
            pred_mw = round(pred_mw, 2)

            # Update rolling load buffer with this forecasted value for subsequent hours
            load_buffer.pop(0)
            load_buffer.append(pred_mw)

            spinning_reserve = round(pred_mw * 0.20, 2)
            lower_bound = round(pred_mw + q05, 2)
            upper_bound = round(pred_mw + q95, 2)

            predictions.append({
                "timestamp": target_dt.strftime("%Y-%m-%dT%H:%M:%S"),
                "hour": target_dt.hour,
                "predicted_load_mw": pred_mw,
                "spinning_reserve_mw": spinning_reserve,
                "lower_bound_mw": lower_bound,
                "upper_bound_mw": upper_bound,
                "is_recursive_lag": step > 1,
            })

        provenance_note = (
            "Recursive multi-step forecast using the operational XGBoost model. "
            "Bounds calculated using empirical 5th and 95th percentile residuals "
            f"({q05:+.2f} / {q95:+.2f} MW) from the stacked OOF ensemble. "
            "These provide an empirical reference band but are not statistically "
            "calibrated prediction intervals for recursive multi-step forecasting."
        )

        return {
            "forecast_origin": origin_dt.strftime("%Y-%m-%dT%H:%M:%S"),
            "forecast_horizon_hours": 24,
            "data_source": origin_source,
            "is_simulated": origin_source == "simulated",
            "predictions": predictions,
            "interval_confidence_level": 0.90,
            "interval_provenance_note": provenance_note,
            "model_used": "XGBoost (Operational Multi-Step Recursive)",
        }
