"""Feature selection: rank cleaned sensors by predictive power and keep the top K.

Uses a Random Forest's impurity-based importances (fit with balanced class
weights, since failures are rare) rather than a plain correlation ranking,
since sensor/yield relationships in process data are frequently non-linear.
"""
import logging

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.config import RANDOM_STATE, TOP_K_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def rank_features(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    ranker = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    ranker.fit(X, y)
    importance = pd.DataFrame(
        {"sensor": X.columns, "importance": ranker.feature_importances_}
    ).sort_values("importance", ascending=False, ignore_index=True)
    return importance


def select_top_features(X: pd.DataFrame, y: pd.Series, top_k: int = TOP_K_FEATURES) -> tuple[list[str], pd.DataFrame]:
    importance = rank_features(X, y)
    selected = importance.head(top_k)["sensor"].tolist()
    logger.info("Selected top %d of %d sensors by importance", len(selected), X.shape[1])
    return selected, importance
