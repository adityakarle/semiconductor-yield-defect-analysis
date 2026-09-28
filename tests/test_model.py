import numpy as np
import pandas as pd

from src.model import train_and_select_best


def test_train_and_select_best_returns_a_model_that_beats_random_guessing():
    rng = np.random.RandomState(0)
    n = 600
    y = pd.Series(rng.binomial(1, 0.15, size=n))
    signal = y * 3 + rng.normal(0, 1, size=n)
    noise = rng.normal(0, 1, size=n)
    X = pd.DataFrame({"signal": signal, "noise": noise})

    best_name, best_model, results_df, pr_curve = train_and_select_best(X, y)

    assert best_name in results_df["model"].values
    # Base rate is ~0.15; a model using real signal should clear it comfortably.
    assert results_df["pr_auc"].max() > 0.3
    assert "precision" in pr_curve and "recall" in pr_curve
