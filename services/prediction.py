import numpy as np
from models.sequences import make_sequences
from services.artifact_manager import load_model_artifacts

class LoadForecaster:
    """
    Forecaster service that encapsulates trained models, scaling objects, bias corrections,
    and meta-learner ensemble weights to perform unified predictions.
    """
    def __init__(self, xgb_model, bilstm_model, cnn_lstm_model, ensemble_weights, model_bias, scaler_X, scaler_y):
        """
        Initializes the forecaster.

        Args:
            xgb_model (XGBRegressor): Trained XGBoost model.
            bilstm_model (Sequential): Trained BiLSTM model.
            cnn_lstm_model (Model): Trained CNN-LSTM model.
            ensemble_weights (np.ndarray): Stacking ensemble weights.
            model_bias (np.ndarray): Stacking OOF model biases.
            scaler_X (MinMaxScaler): Scaler used for input features.
            scaler_y (MinMaxScaler): Scaler used for target variable.
        """
        self.xgb_model = xgb_model
        self.bilstm_model = bilstm_model
        self.cnn_lstm_model = cnn_lstm_model
        self.ensemble_weights = ensemble_weights
        self.model_bias = model_bias
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y

        # Inference-specific attributes (populated by load())
        self.q05 = None
        self.q95 = None
        self.residuals = None
        self.feature_columns = None
        self.time_steps = None
        self.n_features = None

    @classmethod
    def load(cls, model_dir):
        """
        Loads all saved model artifacts from disk and constructs a LoadForecaster.

        Models, scalers, and ensemble parameters are loaded ONCE and kept in memory.
        Subsequent calls to predict() do NOT reload from disk.

        Args:
            model_dir (str): Path to the directory containing saved model artifacts.

        Returns:
            LoadForecaster: Fully initialized forecaster ready for prediction.

        Raises:
            FileNotFoundError: If any required artifact is missing.
        """
        artifacts = load_model_artifacts(model_dir)

        instance = cls(
            xgb_model=artifacts["xgb_model"],
            bilstm_model=artifacts["bilstm_model"],
            cnn_lstm_model=artifacts["cnn_lstm_model"],
            ensemble_weights=artifacts["ensemble_weights"],
            model_bias=artifacts["model_bias"],
            scaler_X=artifacts["scaler_X"],
            scaler_y=artifacts["scaler_y"],
        )

        # Store inference-specific parameters
        instance.q05 = artifacts["q05"]
        instance.q95 = artifacts["q95"]
        instance.residuals = artifacts["residuals"]
        instance.feature_columns = artifacts["feature_columns"]
        instance.time_steps = artifacts["time_steps"]
        instance.n_features = artifacts["n_features"]

        return instance

    def predict_xgb(self, X):
        """
        Runs prediction for the XGBoost model.
        """
        return self.xgb_model.predict(X)

    def predict_bilstm(self, X_seq):
        """
        Runs prediction for the BiLSTM model.
        """
        preds = self.bilstm_model.predict(X_seq, verbose=0)
        return self.scaler_y.inverse_transform(preds.reshape(-1, 1)).flatten()

    def predict_cnnlstm(self, X_seq):
        """
        Runs prediction for the CNN-LSTM model.
        """
        preds = self.cnn_lstm_model.predict(X_seq, verbose=0)
        return self.scaler_y.inverse_transform(preds.reshape(-1, 1)).flatten()

    def predict_stacked(self, X_test, X_te_seq):
        """
        Performs predictions across all base models and applies bias correction and stacking.

        Args:
            X_test (pd.DataFrame): Test features for XGBoost.
            X_te_seq (np.ndarray): Test sequences for BiLSTM and CNN-LSTM.

        Returns:
            tuple: (y_pred_hybrid, y_pred_xgb_aligned, y_pred_bi_aligned, y_pred_cnn_aligned, min_len)
        """
        y_pred_xgb = self.predict_xgb(X_test)
        y_pred_bi = self.predict_bilstm(X_te_seq)
        y_pred_cnn = self.predict_cnnlstm(X_te_seq)

        min_len = min(len(y_pred_xgb), len(y_pred_bi), len(y_pred_cnn))

        y_pred_xgb_a = y_pred_xgb[-min_len:]
        y_pred_bi_a = y_pred_bi[-min_len:]
        y_pred_cnn_a = y_pred_cnn[-min_len:]

        stack_test = np.column_stack([y_pred_xgb_a, y_pred_bi_a, y_pred_cnn_a])
        stack_test_corrected = stack_test - self.model_bias

        y_pred_hybrid = stack_test_corrected @ self.ensemble_weights

        return y_pred_hybrid, y_pred_xgb_a, y_pred_bi_a, y_pred_cnn_a, min_len

    def predict(self, df):
        """
        Generates predictions from a preprocessed + feature-engineered DataFrame.

        This method uses the feature-only preparation path: it does NOT require the
        target/load column as a prediction target. The DataFrame must contain historical
        load observations (for lag/rolling features computed by feature engineering),
        but the load column is NOT used as a target here.

        No model.fit(), scaler.fit(), or scaler.fit_transform() calls are made.
        All models, scalers, and parameters were loaded once during load().

        Args:
            df (pd.DataFrame): Preprocessed + feature-engineered DataFrame.
                Must contain all columns listed in self.feature_columns.

        Returns:
            dict: Prediction results with keys:
                y_pred_xgb, y_pred_bilstm, y_pred_cnnlstm, y_pred_hybrid,
                lower_bound, upper_bound, q05, q95, min_len

        Raises:
            ValueError: If required feature columns are missing or insufficient data.
        """
        if self.feature_columns is None:
            raise ValueError(
                "Feature configuration not available. "
                "Use LoadForecaster.load() to create an inference-ready forecaster."
            )

        # ── Validate required feature columns ───────────────────
        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required feature columns for inference: {missing}\n"
                f"The incoming DataFrame must contain all features used during training."
            )

        # ── Feature-only preparation (no target required) ───────
        X = df[self.feature_columns].dropna().reset_index(drop=True)

        if len(X) == 0:
            raise ValueError(
                "No valid rows remaining after dropping NaN values. "
                "Ensure sufficient historical data for lag/rolling feature computation."
            )

        # ── Validate minimum rows for sequence generation ───────
        if len(X) <= self.time_steps:
            raise ValueError(
                f"At least {self.time_steps + 1} sequential observations are required "
                f"for BiLSTM/CNN-LSTM inference. Got {len(X)} rows."
            )

        # ── XGBoost prediction (unscaled features) ──────────────
        y_pred_xgb = self.xgb_model.predict(X)

        # ── Scale features for neural networks ──────────────────
        # Uses saved scaler_X.transform() — NEVER .fit() or .fit_transform()
        X_scaled = self.scaler_X.transform(X)

        # ── Create sequences ────────────────────────────────────
        # make_sequences needs a y array for indexing, but we only use X sequences
        # Use a dummy y array since we don't have target values during inference
        dummy_y = np.zeros(len(X_scaled))
        X_seq, _ = make_sequences(X_scaled, dummy_y, time_steps=self.time_steps)

        # ── Neural network predictions ──────────────────────────
        y_pred_bilstm = self.predict_bilstm(X_seq)
        y_pred_cnnlstm = self.predict_cnnlstm(X_seq)

        # ── Align to shortest prediction length ─────────────────
        min_len = min(len(y_pred_xgb), len(y_pred_bilstm), len(y_pred_cnnlstm))

        y_pred_xgb_a = y_pred_xgb[-min_len:]
        y_pred_bi_a = y_pred_bilstm[-min_len:]
        y_pred_cnn_a = y_pred_cnnlstm[-min_len:]

        # ── Bias-corrected convex ensemble ──────────────────────
        stack_test = np.column_stack([y_pred_xgb_a, y_pred_bi_a, y_pred_cnn_a])
        stack_test_corrected = stack_test - self.model_bias
        y_pred_hybrid = stack_test_corrected @ self.ensemble_weights

        # ── Prediction intervals from saved q05/q95 ────────────
        lower_bound = y_pred_hybrid + self.q05
        upper_bound = y_pred_hybrid + self.q95

        return {
            "y_pred_xgb": y_pred_xgb_a,
            "y_pred_bilstm": y_pred_bi_a,
            "y_pred_cnnlstm": y_pred_cnn_a,
            "y_pred_hybrid": y_pred_hybrid,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "q05": self.q05,
            "q95": self.q95,
            "min_len": min_len,
        }
