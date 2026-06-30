import pandas as pd

from ml.dataset import build_training_dataset


def test_build_training_dataset_inner_joins_on_asset_and_open_time():
    open_time = pd.Timestamp("2026-01-01T00:00:00Z")
    features_df = pd.DataFrame({
        "asset": ["XBTUSD"], "open_time": [open_time], "rsi_14": [55.0],
    })
    labels_df = pd.DataFrame({
        "asset": ["XBTUSD"], "open_time": [open_time], "return_pct": [0.02], "label": [1],
    })

    result = build_training_dataset(features_df, labels_df)

    assert len(result) == 1
    assert result["rsi_14"].iloc[0] == 55.0
    assert result["label"].iloc[0] == 1


def test_build_training_dataset_drops_unmatched_rows():
    features_df = pd.DataFrame({
        "asset": ["XBTUSD", "ETHUSD"],
        "open_time": pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"]),
        "rsi_14": [55.0, 60.0],
    })
    labels_df = pd.DataFrame({
        "asset": ["XBTUSD"],
        "open_time": [pd.Timestamp("2026-01-01T00:00:00Z")],
        "return_pct": [0.02],
        "label": [1],
    })

    result = build_training_dataset(features_df, labels_df)

    assert len(result) == 1
    assert result["asset"].iloc[0] == "XBTUSD"


def test_build_training_dataset_handles_empty_features():
    labels_df = pd.DataFrame({
        "asset": ["XBTUSD"], "open_time": [pd.Timestamp("2026-01-01T00:00:00Z")],
        "return_pct": [0.02], "label": [1],
    })

    result = build_training_dataset(pd.DataFrame(), labels_df)

    assert result.empty


def test_build_training_dataset_handles_empty_labels():
    features_df = pd.DataFrame({
        "asset": ["XBTUSD"], "open_time": [pd.Timestamp("2026-01-01T00:00:00Z")], "rsi_14": [55.0],
    })

    result = build_training_dataset(features_df, pd.DataFrame())

    assert result.empty
