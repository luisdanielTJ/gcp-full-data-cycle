import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from ml.explain import compute_feature_importance


def test_compute_feature_importance_returns_one_value_per_feature():
    rng = np.random.default_rng(0)
    n = 60
    X = pd.DataFrame({
        "f1": rng.uniform(0, 1, n),
        "f2": rng.uniform(0, 1, n),
        "f3": rng.uniform(0, 1, n),
    })
    y = (X["f1"] + X["f2"] > 1.0).astype(int)
    model = XGBClassifier(n_estimators=20, max_depth=3, learning_rate=0.1, eval_metric="logloss")
    model.fit(X, y)

    result = compute_feature_importance(model, X)

    assert set(result.keys()) == {"f1", "f2", "f3"}
    assert all(isinstance(v, float) for v in result.values())
    assert all(v >= 0 for v in result.values())
    # f1/f2 drive the label; f3 is noise, so it should rank below at least one of them.
    assert result["f3"] < max(result["f1"], result["f2"])
