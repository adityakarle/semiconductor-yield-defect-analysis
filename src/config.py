"""Central configuration: data sources, paths, and modeling constants.

Data source: the UCI SECOM dataset (Semiconductor Manufacturing Process
Control), real sensor readings from an actual semiconductor fab line with
pass/fail yield labels. No API key or auth required.
"""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

UCI_BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom"
SOURCES = {
    "features": {
        "url": f"{UCI_BASE}/secom.data",
        "file": RAW_DIR / "secom.data",
    },
    "labels": {
        "url": f"{UCI_BASE}/secom_labels.data",
        "file": RAW_DIR / "secom_labels.data",
    },
}

DATASET_PATH = PROCESSED_DIR / "secom_clean.parquet"
FEATURE_IMPORTANCE_PATH = PROCESSED_DIR / "feature_importance.csv"
METRICS_PATH = PROCESSED_DIR / "metrics.json"
MODEL_PATH = PROCESSED_DIR / "model.joblib"
SELECTED_FEATURES_PATH = PROCESSED_DIR / "selected_features.json"

# Cleaning thresholds
MAX_MISSING_FRAC = 0.45      # drop a sensor column if more than this fraction is missing
NEAR_CONSTANT_VARIANCE = 1e-6  # drop near-zero-variance sensor columns
CORR_DROP_THRESHOLD = 0.95   # drop one of a pair of sensors this collinear

# Feature selection
TOP_K_FEATURES = 40

# Modeling
RANDOM_STATE = 42
TEST_SIZE = 0.2
