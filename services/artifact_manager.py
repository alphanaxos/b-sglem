"""
Centralized model artifact serialization and deserialization.

Saves and loads all artifacts required for standalone inference:
XGBoost, BiLSTM, CNN-LSTM, scalers, ensemble config, residuals,
prediction intervals, feature config, metadata, and validation reference.
"""
import os
import json
import datetime
import numpy as np
import joblib

from config import (
    TIME_STEPS,
    TRAIN_RATIO,
    LSTM_EPOCHS,
    LSTM_UNITS,
    RANDOM_SEED,
    FEATURE_COLS,
    TARGET_COL,
)


def save_model_artifacts(
    model_dir,
    xgb_model,
    bilstm_model,
    cnn_lstm_model,
    scaler_X,
    scaler_y,
    ensemble_weights,
    model_bias,
    residuals_stacked_train,
    q05,
    q95,
    available_features,
    validation_preds,
):
    """
    Persists every artifact required for standalone inference.

    Args:
        model_dir (str): Directory to write artifacts into.
        xgb_model: Final trained XGBRegressor (NOT an OOF fold model).
        bilstm_model: Final trained BiLSTM Keras model.
        cnn_lstm_model: Final trained CNN-LSTM Keras model.
        scaler_X: Fitted MinMaxScaler for input features.
        scaler_y: Fitted MinMaxScaler for target variable.
        ensemble_weights (np.ndarray): Convex stacking weights [XGB, BiLSTM, CNN-LSTM].
        model_bias (np.ndarray): OOF bias corrections [XGB, BiLSTM, CNN-LSTM].
        residuals_stacked_train (np.ndarray): Raw stacked OOF residuals.
        q05 (float): 5th percentile of stacked residuals.
        q95 (float): 95th percentile of stacked residuals.
        available_features (list): Feature column names in exact training order.
        validation_preds (dict): Reference predictions for cross-process validation.
            Keys: y_pred_xgb, y_pred_bilstm, y_pred_cnnlstm, y_pred_hybrid,
                  lower_bound, upper_bound
    """
    os.makedirs(model_dir, exist_ok=True)

    # ── 1. XGBoost ──────────────────────────────────────────────
    xgb_path = os.path.join(model_dir, "xgboost_model.json")
    xgb_model.save_model(xgb_path)
    print(f"    Saved XGBoost model         → {xgb_path}")

    # ── 2. BiLSTM ───────────────────────────────────────────────
    bilstm_path = os.path.join(model_dir, "bilstm_model.keras")
    bilstm_model.save(bilstm_path)
    print(f"    Saved BiLSTM model          → {bilstm_path}")

    # ── 3. CNN-LSTM ─────────────────────────────────────────────
    cnn_path = os.path.join(model_dir, "cnn_lstm_model.keras")
    cnn_lstm_model.save(cnn_path)
    print(f"    Saved CNN-LSTM model         → {cnn_path}")

    # ── 4. Scalers ──────────────────────────────────────────────
    scaler_x_path = os.path.join(model_dir, "scaler_X.pkl")
    joblib.dump(scaler_X, scaler_x_path)
    print(f"    Saved scaler_X              → {scaler_x_path}")

    scaler_y_path = os.path.join(model_dir, "scaler_y.pkl")
    joblib.dump(scaler_y, scaler_y_path)
    print(f"    Saved scaler_y              → {scaler_y_path}")

    # ── 5. Ensemble config ──────────────────────────────────────
    ensemble_config = {
        "method": "bias_corrected_convex_weighted_ensemble",
        "models": ["XGBoost", "BiLSTM", "CNN-LSTM"],
        "model_order": ["XGBoost", "BiLSTM", "CNN-LSTM"],
        "bias": model_bias.tolist(),
        "weights": ensemble_weights.tolist(),
        "weight_step": 0.01,
    }
    ensemble_path = os.path.join(model_dir, "ensemble_config.json")
    with open(ensemble_path, "w") as f:
        json.dump(ensemble_config, f, indent=2)
    print(f"    Saved ensemble config       → {ensemble_path}")

    # ── 6. Residuals ────────────────────────────────────────────
    residuals_path = os.path.join(model_dir, "residuals.pkl")
    joblib.dump(residuals_stacked_train, residuals_path)
    print(f"    Saved residuals             → {residuals_path}")

    # ── 7. Prediction interval ──────────────────────────────────
    pi_config = {
        "method": "empirical_residual",
        "lower_quantile": 5,
        "upper_quantile": 95,
        "q05": float(q05),
        "q95": float(q95),
        "residual_source": "stacked OOF ensemble",
    }
    pi_path = os.path.join(model_dir, "prediction_interval.json")
    with open(pi_path, "w") as f:
        json.dump(pi_config, f, indent=2)
    print(f"    Saved prediction interval   → {pi_path}")

    # ── 8. Feature config ───────────────────────────────────────
    feature_config = {
        "feature_columns": list(available_features),
        "target_column": TARGET_COL,
        "time_steps": TIME_STEPS,
        "n_features": len(available_features),
    }
    feature_path = os.path.join(model_dir, "feature_config.json")
    with open(feature_path, "w") as f:
        json.dump(feature_config, f, indent=2)
    print(f"    Saved feature config        → {feature_path}")

    # ── 9. Model metadata ──────────────────────────────────────
    metadata = {
        "project": "Bengaluru Electricity Load Forecasting",
        "models": ["XGBoost", "BiLSTM", "CNN-LSTM"],
        "n_features": len(available_features),
        "time_steps": TIME_STEPS,
        "train_ratio": TRAIN_RATIO,
        "target_column": TARGET_COL,
        "random_seed": RANDOM_SEED,
        "lstm_epochs": LSTM_EPOCHS,
        "lstm_units": LSTM_UNITS,
        "saved_at": datetime.datetime.now().isoformat(),
    }
    meta_path = os.path.join(model_dir, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"    Saved model metadata        → {meta_path}")

    # ── 10. Validation reference ────────────────────────────────
    ref_path = os.path.join(model_dir, "validation_reference.npz")
    np.savez(
        ref_path,
        y_pred_xgb=validation_preds["y_pred_xgb"],
        y_pred_bilstm=validation_preds["y_pred_bilstm"],
        y_pred_cnnlstm=validation_preds["y_pred_cnnlstm"],
        y_pred_hybrid=validation_preds["y_pred_hybrid"],
        lower_bound=validation_preds["lower_bound"],
        upper_bound=validation_preds["upper_bound"],
        q05=np.array([q05]),
        q95=np.array([q95]),
    )
    print(f"    Saved validation reference  → {ref_path}")

    print(f"\n  All artifacts saved to {model_dir}/")


def load_model_artifacts(model_dir):
    """
    Loads all saved model artifacts from disk.

    Args:
        model_dir (str): Directory containing saved artifacts.

    Returns:
        dict: Dictionary with keys:
            xgb_model, bilstm_model, cnn_lstm_model,
            scaler_X, scaler_y,
            ensemble_weights, model_bias,
            residuals, q05, q95,
            feature_columns, target_column, time_steps, n_features

    Raises:
        FileNotFoundError: If any required artifact file is missing.
    """
    required_files = {
        "xgboost_model.json": "XGBoost model",
        "bilstm_model.keras": "BiLSTM model",
        "cnn_lstm_model.keras": "CNN-LSTM model",
        "scaler_X.pkl": "Feature scaler",
        "scaler_y.pkl": "Target scaler",
        "ensemble_config.json": "Ensemble configuration",
        "residuals.pkl": "Stacked residuals",
        "prediction_interval.json": "Prediction interval parameters",
        "feature_config.json": "Feature configuration",
    }

    for filename, description in required_files.items():
        path = os.path.join(model_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required artifact missing: {path}\n"
                f"  ({description})\n"
                f"  Run `python train.py` to generate the model artifacts."
            )

    # ── Load XGBoost ────────────────────────────────────────────
    from xgboost import XGBRegressor
    xgb_model = XGBRegressor()
    xgb_model.load_model(os.path.join(model_dir, "xgboost_model.json"))

    # ── Load Keras models ───────────────────────────────────────
    from tensorflow.keras.models import load_model as keras_load_model
    bilstm_model = keras_load_model(os.path.join(model_dir, "bilstm_model.keras"))
    cnn_lstm_model = keras_load_model(os.path.join(model_dir, "cnn_lstm_model.keras"))

    # ── Load scalers ────────────────────────────────────────────
    scaler_X = joblib.load(os.path.join(model_dir, "scaler_X.pkl"))
    scaler_y = joblib.load(os.path.join(model_dir, "scaler_y.pkl"))

    # ── Load ensemble config ────────────────────────────────────
    with open(os.path.join(model_dir, "ensemble_config.json"), "r") as f:
        ensemble_config = json.load(f)
    ensemble_weights = np.array(ensemble_config["weights"])
    model_bias = np.array(ensemble_config["bias"])

    # ── Load residuals ──────────────────────────────────────────
    residuals = joblib.load(os.path.join(model_dir, "residuals.pkl"))

    # ── Load prediction interval ────────────────────────────────
    with open(os.path.join(model_dir, "prediction_interval.json"), "r") as f:
        pi_config = json.load(f)
    q05 = pi_config["q05"]
    q95 = pi_config["q95"]

    # ── Load feature config ─────────────────────────────────────
    with open(os.path.join(model_dir, "feature_config.json"), "r") as f:
        feature_config = json.load(f)

    return {
        "xgb_model": xgb_model,
        "bilstm_model": bilstm_model,
        "cnn_lstm_model": cnn_lstm_model,
        "scaler_X": scaler_X,
        "scaler_y": scaler_y,
        "ensemble_weights": ensemble_weights,
        "model_bias": model_bias,
        "residuals": residuals,
        "q05": q05,
        "q95": q95,
        "feature_columns": feature_config["feature_columns"],
        "target_column": feature_config["target_column"],
        "time_steps": feature_config["time_steps"],
        "n_features": feature_config["n_features"],
    }
