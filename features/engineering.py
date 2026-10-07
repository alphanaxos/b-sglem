import numpy as np
import pandas as pd

def engineer_features(df):
    """
    Applies feature engineering to the preprocessed load DataFrame.
    Includes Fourier time encodings, lag variables, rolling window statistics,
    and temperature-time interaction features.

    Args:
        df (pd.DataFrame): Preprocessed DataFrame.

    Returns:
        pd.DataFrame: DataFrame with engineered features.
    """
    df = df.copy()

    # ── Fourier time encodings ───────────────────────────────
    df['hour_sin']  = np.sin(2 * np.pi * df['hour']        / 24)
    df['hour_cos']  = np.cos(2 * np.pi * df['hour']        / 24)
    df['day_sin']   = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos']   = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month']       / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']       / 12)
    print("  Fourier time encodings added (hour, day, month)")

    # ── Additional lag features ──────────────────────────────
    df['lag_48'] = df['load'].shift(48)
    df['lag_72'] = df['load'].shift(72)
    print("  lag_48, lag_72 added")

    # ── Rolling extremes ─────────────────────────────────────
    df['rolling_max_24'] = df['load'].shift(1).rolling(24).max()
    df['rolling_min_24'] = df['load'].shift(1).rolling(24).min()
    print("  rolling_max_24, rolling_min_24 added")

    # ── Temperature × hour_sin interaction ───────────────────
    df['temp_x_hour_sin'] = df['temperature'] * df['hour_sin']
    print("  temp_x_hour_sin interaction feature added")

    return df
