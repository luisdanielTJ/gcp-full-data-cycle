import numpy as np
import pandas as pd

from ml.train import time_based_split, train_model


def _synthetic_dataset(n=100):
    times = pd.date_range("2026-01-01", periods=n, freq="4h", tz="UTC")
    rng = np.random.default_rng(42)
    rsi = rng.uniform(20, 80, n)
    label = (rsi > 50).astype(int)
    return pd.DataFrame({
        "asset": "XBTUSD",
        "open_time": times,
        "rsi_14": rsi,
        "computed_at": pd.Timestamp.now(tz="UTC"),
        "return_pct": rng.uniform(-0.02, 0.02, n),
        "label": label,
    })


def test_time_based_split_preserves_order_and_ratio():
    dataset = _synthetic_dataset(100)

    train_df, test_df = time_based_split(dataset)

    assert len(train_df) == 80
    assert len(test_df) == 20
    assert train_df["open_time"].max() < test_df["open_time"].min()


def test_time_based_split_no_shuffling():
    dataset = _synthetic_dataset(10).sample(frac=1, random_state=1).reset_index(drop=True)

    train_df, test_df = time_based_split(dataset)

    combined = pd.concat([train_df, test_df])
    expected_order = dataset.sort_values("open_time")["open_time"].tolist()
    assert combined["open_time"].tolist() == expected_order


def test_train_model_returns_fitted_model_and_test_data():
    dataset = _synthetic_dataset(100)
    train_df, test_df = time_based_split(dataset)

    model, X_test, y_test = train_model(train_df, test_df)

    assert len(X_test) == len(test_df)
    assert list(y_test) == list(test_df["label"])
    preds = model.predict(X_test)
    assert len(preds) == len(X_test)


def test_train_model_excludes_non_feature_columns():
    dataset = _synthetic_dataset(100)
    train_df, test_df = time_based_split(dataset)

    _, X_test, _ = train_model(train_df, test_df)

    assert "label" not in X_test.columns
    assert "return_pct" not in X_test.columns
    assert "asset" not in X_test.columns
    assert "open_time" not in X_test.columns
    assert "computed_at" not in X_test.columns
    assert "rsi_14" in X_test.columns
