"""End-to-end pipeline: extract -> clean -> select features -> train -> evaluate.

Usage:
    python -m src.pipeline                # full run
    python -m src.pipeline --refresh       # force re-download source files
"""
import argparse
import json
import logging

from src.config import (
    DATASET_PATH,
    FEATURE_IMPORTANCE_PATH,
    METRICS_PATH,
    MODEL_PATH,
    PROCESSED_DIR,
    SELECTED_FEATURES_PATH,
)
from src.extract import extract_all
from src.features import select_top_features
from src.model import save_artifacts, train_and_select_best
from src.transform import transform_all

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline(refresh: bool = False) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Step 1/4: extract")
    extract_all(refresh=refresh)

    logger.info("Step 2/4: clean sensor data")
    clean = transform_all()
    clean.to_parquet(DATASET_PATH, index=False)
    logger.info("Cleaned dataset: %d units x %d sensors, %.1f%% fail rate", clean.shape[0], clean.shape[1] - 2, clean["fail"].mean() * 100)

    X = clean.drop(columns=["fail", "timestamp"])
    y = clean["fail"]

    logger.info("Step 3/4: select top features")
    selected, importance = select_top_features(X, y)
    importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    with open(SELECTED_FEATURES_PATH, "w") as f:
        json.dump(selected, f, indent=2)

    logger.info("Step 4/4: train + evaluate classifiers")
    best_name, best_model, results_df, pr_curve = train_and_select_best(X[selected], y)

    metrics = {
        "best_model": best_name,
        "n_units": int(clean.shape[0]),
        "n_sensors_raw": 590,
        "n_sensors_after_cleaning": int(X.shape[1]),
        "n_sensors_selected": len(selected),
        "fail_rate": float(y.mean()),
        "model_comparison": results_df.to_dict(orient="records"),
        "best_pr_curve": pr_curve,
    }
    save_artifacts(best_model, MODEL_PATH, metrics, METRICS_PATH)

    logger.info("Pipeline complete. Artifacts in %s", PROCESSED_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SECOM yield/defect ETL + ML pipeline.")
    parser.add_argument("--refresh", action="store_true", help="Force re-download of source files.")
    args = parser.parse_args()
    run_pipeline(refresh=args.refresh)


if __name__ == "__main__":
    main()
