import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def compute_metrics(name, y_true, y_pred):
    """
    Computes standard regression metrics (MAE, RMSE, R2, MAPE) and prints them.

    Args:
        name (str): Model identifier.
        y_true (np.ndarray): Actual targets.
        y_pred (np.ndarray): Predicted targets.

    Returns:
        dict: Calculated metrics dictionary.
    """
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
    print(f"  {name:<18}  MAE={mae:7.2f}  RMSE={rmse:7.2f}  R²={r2:.4f}  MAPE={mape:.2f}%")
    return {"model": name, "MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}

def ramp_rate_error(y_true, y_pred):
    """
    Mean absolute error on hour-to-hour load change (MW/hour).

    Args:
        y_true (np.ndarray): Actual targets.
        y_pred (np.ndarray): Predicted targets.

    Returns:
        float: Calculated ramp-rate error.
    """
    ramp_true = np.diff(y_true)
    ramp_pred = np.diff(y_pred)
    return np.mean(np.abs(ramp_true - ramp_pred))
