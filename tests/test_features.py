import numpy as np
import pandas as pd

from src.features import select_top_features


def test_select_top_features_ranks_the_informative_sensor_first():
    rng = np.random.RandomState(0)
    n = 300
    y = pd.Series(rng.binomial(1, 0.3, size=n))
    informative = y * 5 + rng.normal(0, 0.5, size=n)
    noise = rng.normal(0, 1, size=n)
    X = pd.DataFrame({"informative_sensor": informative, "noise_sensor": noise})

    selected, importance = select_top_features(X, y, top_k=1)

    assert selected == ["informative_sensor"]
    assert importance.iloc[0]["sensor"] == "informative_sensor"
