from xgboost import XGBRegressor

def build_xgboost():
    """
    Builds the baseline XGBoost model with the exact pre-defined hyperparameters.

    Returns:
        XGBRegressor: Unfitted XGBoost model.
    """
    return XGBRegressor(
        n_estimators=400,
        learning_rate=0.04,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        random_state=42,
        verbosity=1,
    )

def train_xgboost(X_train, y_train, X_test=None, y_test=None):
    """
    Instantiates and trains the XGBoost model.

    Args:
        X_train (pd.DataFrame or np.ndarray): Training features.
        y_train (pd.Series or np.ndarray): Training target.
        X_test (pd.DataFrame or np.ndarray, optional): Validation/Test features. Defaults to None.
        y_test (pd.Series or np.ndarray, optional): Validation/Test target. Defaults to None.

    Returns:
        XGBRegressor: Fitted XGBoost model.
    """
    model = build_xgboost()
    if X_test is not None and y_test is not None:
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    else:
        model.fit(X_train, y_train, verbose=False)
    return model
