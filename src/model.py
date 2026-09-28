"""Train and evaluate a defect classifier over the imbalanced SECOM labels.

Only ~7% of units fail, so accuracy is meaningless here. Models are compared
on PR-AUC (average precision) — the same metric used for the fraud-detection
project's imbalanced-classification evaluation — since it focuses on how well
the model ranks the rare failing units rather than overall accuracy.
"""
import json
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import RANDOM_STATE, TEST_SIZE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_candidates(scale_pos_weight: float) -> dict:
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced",
            max_depth=None,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def evaluate_model(name: str, model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    pr_auc = average_precision_score(y_test, proba)
    roc_auc = roc_auc_score(y_test, proba)
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, preds).tolist()

    logger.info("%s: PR-AUC=%.4f  ROC-AUC=%.4f", name, pr_auc, roc_auc)
    return {
        "model": name,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "precision_fail": report["1"]["precision"],
        "recall_fail": report["1"]["recall"],
        "f1_fail": report["1"]["f1-score"],
        "confusion_matrix": cm,
    }


def train_and_select_best(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    n_pass = (y_train == 0).sum()
    n_fail = (y_train == 1).sum()
    scale_pos_weight = n_pass / max(n_fail, 1)

    candidates = build_candidates(scale_pos_weight)
    results = []
    fitted = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        fitted[name] = model
        results.append(evaluate_model(name, model, X_test, y_test))

    results_df = pd.DataFrame(results).sort_values("pr_auc", ascending=False, ignore_index=True)
    best_name = results_df.iloc[0]["model"]
    best_model = fitted[best_name]
    logger.info("Best model: %s (PR-AUC=%.4f)", best_name, results_df.iloc[0]["pr_auc"])

    proba = best_model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, proba)
    pr_curve = {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": thresholds.tolist(),
    }

    return best_name, best_model, results_df, pr_curve


def save_artifacts(model, model_path, metrics: dict, metrics_path) -> None:
    joblib.dump(model, model_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)
    logger.info("Saved model to %s and metrics to %s", model_path, metrics_path)
