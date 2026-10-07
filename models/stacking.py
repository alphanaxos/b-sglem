import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Bidirectional, LSTM, Conv1D, MaxPooling1D

def optimise_convex_stack_weights(stack_train, y_train_stack, step=0.01):
    """
    Find non-negative ensemble weights that sum to one.

    This is intentionally conservative for an academic forecasting project:
    with only three base learners, a grid search over the simplex avoids
    unstable Ridge coefficients, negative weights, and intercept drift.
    """
    best_rmse = np.inf
    best_weights = np.array([1.0, 0.0, 0.0])

    grid = np.arange(0.0, 1.0 + step / 2, step)
    for w_xgb in grid:
        for w_bilstm in grid:
            w_cnn = 1.0 - w_xgb - w_bilstm
            if w_cnn < -1e-12:
                continue
            weights = np.array([w_xgb, w_bilstm, max(w_cnn, 0.0)])
            pred = stack_train @ weights
            rmse = np.sqrt(mean_squared_error(y_train_stack, pred))
            if rmse < best_rmse:
                best_rmse = rmse
                best_weights = weights

    return best_weights, best_rmse

def perform_stacking(X_train, y_train, X_scaled, y_scaled, scaler_y, time_steps=24):
    """
    Performs out-of-fold (OOF) cross-validation predictions for XGBoost, BiLSTM, and CNN-LSTM,
    aligns predictions, performs bias correction, and optimizes convex stack weights.

    Args:
        X_train (pd.DataFrame): Training feature DataFrame.
        y_train (pd.Series): Training target Series.
        X_scaled (np.ndarray): Full scaled feature matrix.
        y_scaled (np.ndarray): Full scaled target vector.
        scaler_y (MinMaxScaler): Scaler object for target variable.
        time_steps (int): Sequence window length. Defaults to 24.

    Returns:
        tuple: (ensemble_weights, model_bias, residuals_stacked_train, best_oof_model, base_oof_rmse, stack_oof_rmse)
    """
    tscv = TimeSeriesSplit(n_splits=5)

    # ---------------------------
    # 1. OOF for XGBoost
    # ---------------------------
    print("  Calculating XGBoost OOF predictions...")
    oof_xgb = np.zeros(len(X_train))

    for train_idx, val_idx in tscv.split(X_train):
        model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            verbosity=0
        )
        model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        oof_xgb[val_idx] = model.predict(X_train.iloc[val_idx])

    # Helper function for seq generation in CV
    def seq(data, target):
        Xs, ys = [], []
        for i in range(len(data) - time_steps):
            Xs.append(data[i:i+time_steps])
            ys.append(target[i+time_steps])
        return np.array(Xs), np.array(ys)

    # ---------------------------
    # 2. OOF for BiLSTM
    # ---------------------------
    print("  Calculating BiLSTM OOF predictions...")
    oof_bilstm = np.zeros(len(X_train))

    for train_idx, val_idx in tscv.split(X_train):
        X_tr = X_scaled[train_idx]
        y_tr = y_scaled[train_idx]
        X_val = X_scaled[val_idx]
        
        X_tr_seq, y_tr_seq = seq(X_tr, y_tr)
        X_val_seq, _ = seq(X_val, y_scaled[val_idx])
        
        if len(X_tr_seq) == 0 or len(X_val_seq) == 0:
            continue

        model = Sequential([
            Bidirectional(LSTM(64, return_sequences=True), input_shape=(time_steps, X_train.shape[1])),
            Dropout(0.2),
            Bidirectional(LSTM(32)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_tr_seq, y_tr_seq, epochs=10, batch_size=64, verbose=0)
        
        preds = model.predict(X_val_seq, verbose=0)
        preds = scaler_y.inverse_transform(preds).flatten()
        
        oof_bilstm[val_idx[time_steps:]] = preds[:len(val_idx[time_steps:])]

    # ---------------------------
    # 3. OOF for CNN-LSTM
    # ---------------------------
    print("  Calculating CNN-LSTM OOF predictions...")
    oof_cnn = np.zeros(len(X_train))

    for train_idx, val_idx in tscv.split(X_train):
        X_tr = X_scaled[train_idx]
        y_tr = y_scaled[train_idx]
        X_val = X_scaled[val_idx]
        
        X_tr_seq, y_tr_seq = seq(X_tr, y_tr)
        X_val_seq, _ = seq(X_val, y_scaled[val_idx])
        
        if len(X_tr_seq) == 0 or len(X_val_seq) == 0:
            continue

        inp = Input(shape=(time_steps, X_train.shape[1]))
        x = Conv1D(64, 3, activation='relu', padding='same')(inp)
        x = MaxPooling1D(2)(x)
        x = LSTM(48)(x)
        out = Dense(1)(x)
        
        model = Model(inp, out)
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_tr_seq, y_tr_seq, epochs=10, batch_size=64, verbose=0)
        
        preds = model.predict(X_val_seq, verbose=0)
        preds = scaler_y.inverse_transform(preds).flatten()
        
        oof_cnn[val_idx[time_steps:]] = preds[:len(val_idx[time_steps:])]

    # ---------------------------
    # ALIGN OOF
    # ---------------------------
    valid_idx = np.where((oof_xgb != 0) & (oof_bilstm != 0) & (oof_cnn != 0))[0]

    stack_train = np.column_stack([
        oof_xgb[valid_idx],
        oof_bilstm[valid_idx],
        oof_cnn[valid_idx]
    ])

    y_train_stack = y_train.values[valid_idx]

    # ---------------------------
    # BASE-MODEL BIAS CORRECTION
    # ---------------------------
    model_bias = np.mean(stack_train - y_train_stack.reshape(-1, 1), axis=0)
    stack_train_corrected = stack_train - model_bias

    # ---------------------------
    # ROBUST CONVEX STACKING
    # ---------------------------
    ensemble_weights, stack_oof_rmse = optimise_convex_stack_weights(
        stack_train_corrected,
        y_train_stack,
        step=0.01
    )
    stack_train_pred = stack_train_corrected @ ensemble_weights
    residuals_stacked_train = y_train_stack - stack_train_pred

    base_oof_rmse = {
        "XGBoost": np.sqrt(mean_squared_error(y_train_stack, stack_train_corrected[:, 0])),
        "BiLSTM": np.sqrt(mean_squared_error(y_train_stack, stack_train_corrected[:, 1])),
        "CNN-LSTM": np.sqrt(mean_squared_error(y_train_stack, stack_train_corrected[:, 2])),
    }
    best_oof_model = min(base_oof_rmse, key=base_oof_rmse.get)

    return ensemble_weights, model_bias, residuals_stacked_train, best_oof_model, base_oof_rmse, stack_oof_rmse
