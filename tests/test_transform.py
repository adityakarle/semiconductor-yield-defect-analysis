import numpy as np
import pandas as pd

from src.transform import drop_collinear, drop_high_missing, drop_near_constant, impute_median


def test_drop_high_missing_removes_sparse_columns():
    df = pd.DataFrame(
        {
            "keep": [1, 2, 3, 4, 5],
            "mostly_missing": [np.nan, np.nan, np.nan, np.nan, 1.0],
        }
    )
    out = drop_high_missing(df, max_missing_frac=0.45)
    assert list(out.columns) == ["keep"]


def test_drop_near_constant_removes_dead_sensors():
    df = pd.DataFrame(
        {
            "varying": [1.0, 2.0, 3.0, 4.0, 5.0],
            "dead": [7.0, 7.0, 7.0, 7.0, 7.0],
        }
    )
    out = drop_near_constant(df, min_variance=1e-6)
    assert list(out.columns) == ["varying"]


def test_impute_median_fills_missing_with_column_median():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    out = impute_median(df)
    assert out["a"].iloc[1] == 2.0
    assert not out["a"].isna().any()


def test_drop_collinear_keeps_one_of_a_correlated_pair():
    base = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    df = pd.DataFrame({"a": base, "b": base * 2 + 1, "c": [5.0, 1.0, 4.0, 2.0, 6.0, 3.0]})
    out = drop_collinear(df, threshold=0.95)
    assert "c" in out.columns
    assert ("a" in out.columns) != ("b" in out.columns)
