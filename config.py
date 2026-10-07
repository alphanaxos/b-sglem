import os

# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════
TIME_STEPS  = 24
TRAIN_RATIO = 0.80
LSTM_EPOCHS = 20
LSTM_UNITS  = 64
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.getenv("OUTPUT_DIR", os.path.join(BASE_DIR, "results_v2"))
MODEL_DIR   = os.getenv("MODEL_DIR", os.path.join(BASE_DIR, "models_saved"))
DATA_DIR    = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
ACTIVE_DATASET = os.getenv("ACTIVE_DATASET", "2024")  # "2024", "2023", or "combined"
DATA_SOURCE_MODE = os.getenv("DATA_SOURCE_MODE", "historical")  # "historical", "simulated", "live"
ENABLE_SIMULATION_MODE = os.getenv("ENABLE_SIMULATION_MODE", "false").lower() in ("true", "1")
API_HOST    = os.getenv("HOST", "127.0.0.1")
API_PORT    = int(os.getenv("PORT", "8000"))
PEAK_HOURS  = list(range(18, 22))   # 18:00–21:00 evening peak

N_SIMULATIONS = 1000
ENABLE_WEATHER_UNCERTAINTY = True
MONTE_CARLO_DIR = os.path.join(OUTPUT_DIR, "monte_carlo")
RANDOM_SEED = 42
LOAD_SHEDDING_COST = 100.0   # Reliability penalty per MWh of unmet demand
ENABLE_RESIDUAL_TAIL_CLIPPING = True
RESIDUAL_CLIP_LOW_PERCENTILE = 5
RESIDUAL_CLIP_HIGH_PERCENTILE = 95

FEATURE_COLS = [
    'is_weekend', 'is_holiday', 'is_festival',
    'is_pre_holiday', 'is_post_holiday',
    'is_pre_festival', 'is_post_festival',
    'workday_after_holiday',
    'temperature', 'humidity', 'wind_speed', 'solar_irradiance',
    'lag_1', 'lag_24', 'lag_168',
    'rolling_mean_24', 'rolling_std_24',
    # ── NEW: Fourier encodings ─────────────────────────────────
    'hour_sin', 'hour_cos',
    'day_sin',  'day_cos',
    'month_sin','month_cos',
    # ── NEW: Extra lags ───────────────────────────────────────
    'lag_48', 'lag_72',
    # ── NEW: Rolling extremes ─────────────────────────────────
    'rolling_max_24', 'rolling_min_24',
    # ── NEW: Interaction feature ──────────────────────────────
    'temp_x_hour_sin',
]
TARGET_COL = 'load'
