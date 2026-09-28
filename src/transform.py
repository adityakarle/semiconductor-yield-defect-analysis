"""Transform stage: clean the raw SECOM sensor matrix into a modeling-ready table.

SECOM is notoriously messy: 590 anonymous sensor columns, heavy missingness,
many near-constant (dead) sensors, and strong collinearity between sensors on
the same process step. This stage handles all of that explicitly rather than
papering over it with a blind `fillna(0)`.
"""
import logging

import numpy as np
import pandas as pd

from src.config import (
    CORR_DROP_THRESHOLD,
    MAX_MISSING_FRAC,
    NEAR_CONSTANT_VARIANCE,
    SOURCES,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the raw whitespace-delimited SECOM feature matrix and labels."""
    features = pd.read_csv(SOURCES["features"]["file"], sep=r"\s+", header=None, na_values=["NaN"])
    features.columns = [f"sensor_{i}" for i in range(features.shape[1])]

    labels = pd.read_csv(
        SOURCES["labels"]["file"],
        sep=r"\s+",
        header=None,
        names=["label", "timestamp"],
        na_values=["NaN"],
    )
    labels["timestamp"] = pd.to_datetime(labels["timestamp"], format="%d/%m/%Y %H:%M:%S")
    # UCI encodes -1 = pass, 1 = fail. Recode to the more intuitive 1 = fail (defect).
    labels["fail"] = (labels["label"] == 1).astype(int)

    return features, labels


def drop_high_missing(df: pd.DataFrame, max_missing_frac: float = MAX_MISSING_FRAC) -> pd.DataFrame:
    missing_frac = df.isna().mean()
    keep = missing_frac[missing_frac <= max_missing_frac].index
    dropped = df.shape[1] - len(keep)
    logger.info("Dropped %d sensors with >%.0f%% missing values", dropped, max_missing_frac * 100)
    return df[keep]


def drop_near_constant(df: pd.DataFrame, min_variance: float = NEAR_CONSTANT_VARIANCE) -> pd.DataFrame:
    variances = df.var(skipna=True)
    keep = variances[variances > min_variance].index
    dropped = df.shape[1] - len(keep)
    logger.info("Dropped %d near-constant (dead) sensors", dropped)
    return df[keep]


def impute_median(df: pd.DataFrame) -> pd.DataFrame:
    return df.fillna(df.median(numeric_only=True))


def drop_collinear(df: pd.DataFrame, threshold: float = CORR_DROP_THRESHOLD) -> pd.DataFrame:
    """Greedily drop one sensor from each pair correlated above `threshold`."""
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    logger.info("Dropped %d sensors redundant with another sensor (|corr| > %.2f)", len(to_drop), threshold)
    return df.drop(columns=to_drop)


def clean_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = drop_high_missing(raw)
    df = drop_near_constant(df)
    df = impute_median(df)
    df = drop_collinear(df)
    logger.info("Feature matrix reduced from %d to %d sensors", raw.shape[1], df.shape[1])
    return df


def transform_all() -> pd.DataFrame:
    raw_features, labels = load_raw()
    clean = clean_features(raw_features).copy()
    clean["fail"] = labels["fail"].values
    clean["timestamp"] = labels["timestamp"].values
    return clean


if __name__ == "__main__":
    result = transform_all()
    print(result.shape)
    print(result["fail"].value_counts())
