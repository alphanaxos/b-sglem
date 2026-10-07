import pandas as pd

def prepare_model_data(df, feature_cols, target_col):
    """
    Selects features and target, filters missing features, drops NaN rows,
    and returns model inputs while preserving the raw 'hour' information.

    Args:
        df (pd.DataFrame): Preprocessed DataFrame.
        feature_cols (list): List of feature column names.
        target_col (str): Target column name.

    Returns:
        tuple: (df_model, X, y, hours_all, available_features)
    """
    hour_series_full = df['hour'].copy()

    # Keep only usable columns
    available_features = [c for c in feature_cols if c in df.columns]
    missing_features   = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        print(f"  ⚠  Skipping missing columns: {missing_features}")

    df_model_subset = df[['datetime'] + available_features + [target_col]]
    surviving_idx = df_model_subset.dropna().index
    
    df_model = df_model_subset.dropna().reset_index(drop=True)
    hours_all = hour_series_full.iloc[surviving_idx].reset_index(drop=True)

    print(f"\n  Final dataset : {len(df_model):,} rows")
    print(f"  Date range    : {df_model['datetime'].min()} → {df_model['datetime'].max()}")
    print(f"  Features used : {len(available_features)}")

    X = df_model[available_features]
    y = df_model[target_col]

    return df_model, X, y, hours_all, available_features

def chronological_split(X, y, hours_all, train_ratio):
    """
    Splits the datasets and hours chronologically.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.
        hours_all (pd.Series): Survived hour series.
        train_ratio (float): Split ratio.

    Returns:
        tuple: (X_train, X_test, y_train, y_test, hours_test, split_idx)
    """
    split = int(len(X) * train_ratio)
    X_train = X.iloc[:split]
    X_test = X.iloc[split:]
    y_train = y.iloc[:split]
    y_test = y.iloc[split:]
    hours_test = hours_all.iloc[split:].values

    print(f"\n  Train : {len(X_train):,} rows | Test : {len(X_test):,} rows")
    return X_train, X_test, y_train, y_test, hours_test, split

def prepare_inference_data(df, feature_columns):
    """
    Prepares input features for inference WITHOUT requiring the target column.

    This is the feature-only preparation path: it selects the saved feature
    columns, validates they exist, drops NaN rows (from lag/rolling warmup),
    and returns the feature matrix.

    Args:
        df (pd.DataFrame): Preprocessed + feature-engineered DataFrame.
            Must contain all columns listed in feature_columns.
            Must contain a 'load' column with historical observations
            (needed by feature engineering for lags/rolling stats, but
            NOT used as a prediction target here).
        feature_columns (list): Exact feature column names and order
            as saved during training.

    Returns:
        tuple: (X, available_features)
            X (pd.DataFrame): Feature matrix with NaN rows dropped.
            available_features (list): Feature columns actually used.

    Raises:
        ValueError: If any required feature column is missing from the DataFrame.
    """
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required feature columns for inference: {missing}\n"
            f"The incoming DataFrame must contain all features used during training."
        )

    available_features = list(feature_columns)
    X = df[available_features].dropna().reset_index(drop=True)

    print(f"\n  Inference dataset : {len(X):,} rows")
    print(f"  Features used     : {len(available_features)}")

    return X, available_features

