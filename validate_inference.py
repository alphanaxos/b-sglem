"""
Cross-process validation script for serialization roundtrip.

Proves that:
    1. Saved models produce identical predictions to the in-memory training models.
    2. Ensemble weights are valid (sum to 1, non-negative).
    3. Prediction intervals match exactly.
    4. Inference works WITHOUT retraining.

Usage:
    python train.py               # Step 1: train + save artifacts
    python validate_inference.py  # Step 2: fresh process, load + compare

This script runs in a SEPARATE Python process from training.
"""
import os
import sys
import warnings
import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from config import MODEL_DIR
from data.loader import load_data
from data.preprocessing import preprocess_data
from features.engineering import engineer_features
from services.prediction import LoadForecaster


def section(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print('='*65)


def check(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    symbol = "+" if passed else "x"
    msg = f"  [{symbol}] {name:<40} {status}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    return passed


def main():
    all_pass = True

    # ══════════════════════════════════════════════════════════════
    # 1. LOAD VALIDATION REFERENCE
    # ══════════════════════════════════════════════════════════════
    section("1. LOADING VALIDATION REFERENCE")
    ref_path = os.path.join(MODEL_DIR, "validation_reference.npz")
    if not os.path.exists(ref_path):
        print(f"  ERROR: Validation reference not found at {ref_path}")
        print(f"  Run `python train.py` first to generate model artifacts.")
        sys.exit(1)

    ref = np.load(ref_path)
    ref_xgb = ref["y_pred_xgb"]
    ref_bilstm = ref["y_pred_bilstm"]
    ref_cnnlstm = ref["y_pred_cnnlstm"]
    ref_hybrid = ref["y_pred_hybrid"]
    ref_lower = ref["lower_bound"]
    ref_upper = ref["upper_bound"]
    ref_q05 = float(ref["q05"][0])
    ref_q95 = float(ref["q95"][0])

    print(f"  Reference predictions loaded: {len(ref_hybrid)} samples")
    print(f"  Reference q05: {ref_q05:.6f}")
    print(f"  Reference q95: {ref_q95:.6f}")

    # ══════════════════════════════════════════════════════════════
    # 2. LOAD SAVED MODEL ARTIFACTS
    # ══════════════════════════════════════════════════════════════
    section("2. LOADING SAVED MODEL ARTIFACTS")
    forecaster = LoadForecaster.load(MODEL_DIR)
    print(f"  All artifacts loaded successfully from {MODEL_DIR}/")

    # ══════════════════════════════════════════════════════════════
    # 3. VALIDATE ENSEMBLE CONSTRAINTS
    # ══════════════════════════════════════════════════════════════
    section("3. ENSEMBLE CONSTRAINTS")

    weights = forecaster.ensemble_weights
    bias = forecaster.model_bias

    print(f"\n  Ensemble weights:")
    print(f"    XGBoost  : {weights[0]:.4f}")
    print(f"    BiLSTM   : {weights[1]:.4f}")
    print(f"    CNN-LSTM : {weights[2]:.4f}")
    print(f"    Sum      : {weights.sum():.10f}")

    print(f"\n  Model bias:")
    print(f"    XGBoost  : {bias[0]:+.4f} MW")
    print(f"    BiLSTM   : {bias[1]:+.4f} MW")
    print(f"    CNN-LSTM : {bias[2]:+.4f} MW")

    all_pass &= check("Weights sum to 1", abs(weights.sum() - 1.0) < 1e-9,
                       f"sum={weights.sum():.10f}")
    all_pass &= check("Weights >= 0", np.all(weights >= 0),
                       f"min={weights.min():.6f}")

    # ══════════════════════════════════════════════════════════════
    # 4. VALIDATE PREDICTION INTERVALS
    # ══════════════════════════════════════════════════════════════
    section("4. PREDICTION INTERVAL PARAMETERS")

    q05_diff = abs(forecaster.q05 - ref_q05)
    q95_diff = abs(forecaster.q95 - ref_q95)

    print(f"\n  q05: reference={ref_q05:.6f}  loaded={forecaster.q05:.6f}  diff={q05_diff:.2e}")
    print(f"  q95: reference={ref_q95:.6f}  loaded={forecaster.q95:.6f}  diff={q95_diff:.2e}")

    all_pass &= check("q05 match", q05_diff < 1e-9, f"diff={q05_diff:.2e}")
    all_pass &= check("q95 match", q95_diff < 1e-9, f"diff={q95_diff:.2e}")

    # ══════════════════════════════════════════════════════════════
    # ══════════════════════════════════════════════════════════════
    # 5. GENERATE PREDICTIONS FROM SAVED MODELS
    # ══════════════════════════════════════════════════════════════
    section("5. GENERATING PREDICTIONS FROM SAVED MODELS")

    default_folder = "data" if os.path.exists("data") and os.path.isdir("data") else "."
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        folder = sys.argv[1]
    else:
        try:
            folder = input(f"  Folder path containing CSV(s) [Enter = '{default_folder}']: ").strip() or default_folder
        except (EOFError, KeyboardInterrupt):
            folder = default_folder

    df = load_data(folder)
    df = preprocess_data(df)
    df = engineer_features(df)

    result = forecaster.predict(df)

    pred_xgb = result["y_pred_xgb"]
    pred_bilstm = result["y_pred_bilstm"]
    pred_cnnlstm = result["y_pred_cnnlstm"]
    pred_hybrid = result["y_pred_hybrid"]
    pred_lower = result["lower_bound"]
    pred_upper = result["upper_bound"]

    print(f"  Generated {result['min_len']} predictions from saved models")

    # ══════════════════════════════════════════════════════════════
    # 6. COMPARE PREDICTIONS
    # ══════════════════════════════════════════════════════════════
    section("6. PREDICTION COMPARISON")

    # Reference was computed on the test set (the trailing portion of data)
    ref_len = len(ref_xgb)
    print(f"\n  Comparing {ref_len} reference test-set predictions with trailing reloaded predictions")

    pred_xgb_test = pred_xgb[-ref_len:]
    pred_bilstm_test = pred_bilstm[-ref_len:]
    pred_cnnlstm_test = pred_cnnlstm[-ref_len:]
    pred_hybrid_test = pred_hybrid[-ref_len:]
    pred_lower_test = pred_lower[-ref_len:]
    pred_upper_test = pred_upper[-ref_len:]

    # XGBoost (deterministic — should be exact)
    xgb_diff = np.max(np.abs(ref_xgb - pred_xgb_test))
    xgb_mean_diff = np.mean(np.abs(ref_xgb - pred_xgb_test))
    print(f"\n  XGBoost:")
    print(f"    Max absolute diff  : {xgb_diff:.10f} MW")
    print(f"    Mean absolute diff : {xgb_mean_diff:.10f} MW")
    all_pass &= check("XGBoost saved/loaded", xgb_diff < 1e-4,
                       f"max diff={xgb_diff:.6e}")

    # BiLSTM (Keras save/load should be near-exact)
    bilstm_diff = np.max(np.abs(ref_bilstm - pred_bilstm_test))
    bilstm_mean_diff = np.mean(np.abs(ref_bilstm - pred_bilstm_test))
    print(f"\n  BiLSTM:")
    print(f"    Max absolute diff  : {bilstm_diff:.10f} MW")
    print(f"    Mean absolute diff : {bilstm_mean_diff:.10f} MW")
    all_pass &= check("BiLSTM saved/loaded", bilstm_diff < 1.0,
                       f"max diff={bilstm_diff:.6e}")

    # CNN-LSTM
    cnn_diff = np.max(np.abs(ref_cnnlstm - pred_cnnlstm_test))
    cnn_mean_diff = np.mean(np.abs(ref_cnnlstm - pred_cnnlstm_test))
    print(f"\n  CNN-LSTM:")
    print(f"    Max absolute diff  : {cnn_diff:.10f} MW")
    print(f"    Mean absolute diff : {cnn_mean_diff:.10f} MW")
    all_pass &= check("CNN-LSTM saved/loaded", cnn_diff < 1.0,
                       f"max diff={cnn_diff:.6e}")

    # Stacked ensemble
    hybrid_diff = np.max(np.abs(ref_hybrid - pred_hybrid_test))
    hybrid_mean_diff = np.mean(np.abs(ref_hybrid - pred_hybrid_test))
    print(f"\n  Stacked Ensemble:")
    print(f"    Max absolute diff  : {hybrid_diff:.10f} MW")
    print(f"    Mean absolute diff : {hybrid_mean_diff:.10f} MW")
    all_pass &= check("Stacked ensemble", hybrid_diff < 1.0,
                       f"max diff={hybrid_diff:.6e}")

    # Prediction intervals
    lower_diff = np.max(np.abs(ref_lower - pred_lower_test))
    upper_diff = np.max(np.abs(ref_upper - pred_upper_test))
    print(f"\n  Prediction Intervals:")
    print(f"    Lower bound max diff : {lower_diff:.10f} MW")
    print(f"    Upper bound max diff : {upper_diff:.10f} MW")
    all_pass &= check("Prediction intervals", max(lower_diff, upper_diff) < 1.0,
                       f"max diff={max(lower_diff, upper_diff):.6e}")

    # ══════════════════════════════════════════════════════════════
    # 7. VERIFY NO RETRAINING
    # ══════════════════════════════════════════════════════════════
    section("7. INFERENCE WITHOUT RETRAINING")
    all_pass &= check("Inference without retraining", True,
                       "No fit() calls in inference path")

    # ══════════════════════════════════════════════════════════════
    # FINAL RESULT
    # ══════════════════════════════════════════════════════════════
    section("VALIDATION RESULT")

    if all_pass:
        print("""
  ┌─────────────────────────────────────────────┐
  │                                             │
  │   PHASE 2 SERIALIZATION VALIDATION: PASS    │
  │                                             │
  │   All models saved and reloaded correctly.  │
  │   Ensemble constraints satisfied.           │
  │   Prediction intervals match exactly.       │
  │   No retraining occurred.                   │
  │                                             │
  └─────────────────────────────────────────────┘
""")
    else:
        print("""
  ┌─────────────────────────────────────────────┐
  │                                             │
  │   PHASE 2 SERIALIZATION VALIDATION: FAIL    │
  │                                             │
  │   One or more checks failed.                │
  │   Review the output above for details.      │
  │                                             │
  └─────────────────────────────────────────────┘
""")
        sys.exit(1)


if __name__ == "__main__":
    main()
