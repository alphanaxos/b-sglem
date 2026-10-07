"""
Standalone inference script for Bengaluru electricity load forecasting.

Loads saved model artifacts from disk and generates predictions WITHOUT retraining.

Usage:
    python inference.py

This script:
    1. Loads all model artifacts from models_saved/
    2. Loads input CSV data
    3. Preprocesses and engineers features (shared modules)
    4. Generates predictions using saved models
    5. Applies saved bias correction and convex ensemble weights
    6. Applies saved q05/q95 for 90% prediction intervals
    7. Saves results to inference_results/

NO model.fit(), scaler.fit(), or scaler.fit_transform() calls are made.
NO OOF generation, no ensemble weight optimization, no residual recalculation.
"""
import sys
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Suppress TensorFlow verbosity before import
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from config import MODEL_DIR
from data.loader import load_data
from data.preprocessing import preprocess_data
from features.engineering import engineer_features
from services.prediction import LoadForecaster


INFERENCE_OUTPUT_DIR = "inference_results"


def section(title):
    print(f"\n{'═'*65}")
    print(f"  {title}")
    print('═'*65)


def main():
    # ══════════════════════════════════════════════════════════════
    # 1. LOAD SAVED MODEL ARTIFACTS
    # ══════════════════════════════════════════════════════════════
    section("1. LOADING SAVED MODEL ARTIFACTS")
    print(f"  Loading from: {MODEL_DIR}/")

    forecaster = LoadForecaster.load(MODEL_DIR)

    print(f"  Models loaded successfully")
    print(f"  Features expected : {forecaster.n_features}")
    print(f"  Time steps        : {forecaster.time_steps}")
    print(f"  Ensemble weights  : {forecaster.ensemble_weights}")
    print(f"  Model bias        : {forecaster.model_bias}")
    print(f"  q05               : {forecaster.q05:.4f}")
    print(f"  q95               : {forecaster.q95:.4f}")

    # ══════════════════════════════════════════════════════════════
    # 2. LOAD INPUT DATA
    # ══════════════════════════════════════════════════════════════
    section("2. LOADING INPUT DATA")
    default_folder = "data" if os.path.exists("data") and os.path.isdir("data") else "."
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        folder = sys.argv[1]
    else:
        try:
            folder = input(f"  Folder path containing CSV(s) [Enter = '{default_folder}']: ").strip() or default_folder
        except (EOFError, KeyboardInterrupt):
            folder = default_folder
    df = load_data(folder)

    # ══════════════════════════════════════════════════════════════
    # 3. PREPROCESS + FEATURE ENGINEERING
    # ══════════════════════════════════════════════════════════════
    section("3. PREPROCESSING + FEATURE ENGINEERING")
    df = preprocess_data(df)
    df = engineer_features(df)

    print(f"  DataFrame shape after engineering: {df.shape}")

    # ══════════════════════════════════════════════════════════════
    # 4. GENERATE PREDICTIONS
    # ══════════════════════════════════════════════════════════════
    section("4. GENERATING PREDICTIONS")
    result = forecaster.predict(df)

    min_len = result["min_len"]
    y_pred_xgb = result["y_pred_xgb"]
    y_pred_bilstm = result["y_pred_bilstm"]
    y_pred_cnnlstm = result["y_pred_cnnlstm"]
    y_pred_hybrid = result["y_pred_hybrid"]
    lower_bound = result["lower_bound"]
    upper_bound = result["upper_bound"]

    print(f"\n  Prediction length : {min_len}")
    print(f"\n  Sample predictions (first 5 hours):")
    print(f"  {'Hour':<6} {'XGBoost':>10} {'BiLSTM':>10} {'CNN-LSTM':>10} {'Stacked':>10} {'Lower 90%':>10} {'Upper 90%':>10}")
    for i in range(min(5, min_len)):
        print(f"  {i:<6} {y_pred_xgb[i]:>10.1f} {y_pred_bilstm[i]:>10.1f} {y_pred_cnnlstm[i]:>10.1f} "
              f"{y_pred_hybrid[i]:>10.1f} {lower_bound[i]:>10.1f} {upper_bound[i]:>10.1f}")

    # ══════════════════════════════════════════════════════════════
    # 5. SUMMARY STATISTICS
    # ══════════════════════════════════════════════════════════════
    section("5. PREDICTION SUMMARY")
    print(f"  XGBoost  — Mean: {np.mean(y_pred_xgb):,.1f} MW  |  Std: {np.std(y_pred_xgb):,.1f} MW")
    print(f"  BiLSTM   — Mean: {np.mean(y_pred_bilstm):,.1f} MW  |  Std: {np.std(y_pred_bilstm):,.1f} MW")
    print(f"  CNN-LSTM — Mean: {np.mean(y_pred_cnnlstm):,.1f} MW  |  Std: {np.std(y_pred_cnnlstm):,.1f} MW")
    print(f"  Stacked  — Mean: {np.mean(y_pred_hybrid):,.1f} MW  |  Std: {np.std(y_pred_hybrid):,.1f} MW")
    print(f"  90% PI   — Mean width: {np.mean(upper_bound - lower_bound):,.1f} MW")

    # ══════════════════════════════════════════════════════════════
    # 6. SAVE RESULTS
    # ══════════════════════════════════════════════════════════════
    section("6. SAVING RESULTS")
    os.makedirs(INFERENCE_OUTPUT_DIR, exist_ok=True)

    out_df = pd.DataFrame({
        "xgb_prediction": y_pred_xgb,
        "bilstm_prediction": y_pred_bilstm,
        "cnnlstm_prediction": y_pred_cnnlstm,
        "stacked_prediction": y_pred_hybrid,
        "lower_90": lower_bound,
        "upper_90": upper_bound,
    })

    out_path = os.path.join(INFERENCE_OUTPUT_DIR, "predictions.csv")
    out_df.to_csv(out_path, index=False)
    print(f"  Saved predictions to {out_path}")

    # ══════════════════════════════════════════════════════════════
    # COMPLETE
    # ══════════════════════════════════════════════════════════════
    section("INFERENCE COMPLETE")
    print(f"""
  ┌─────────────────────────────────────────────┐
  │  Predictions     : {min_len:>6} hours               │
  │  Stacked Mean    : {np.mean(y_pred_hybrid):>8,.1f} MW             │
  │  90% PI Width    : {np.mean(upper_bound - lower_bound):>8,.1f} MW             │
  │  Output          : {INFERENCE_OUTPUT_DIR}/predictions.csv    │
  │                                             │
  │  No models were retrained.                  │
  │  All predictions from saved artifacts.      │
  └─────────────────────────────────────────────┘
""")


if __name__ == "__main__":
    main()
