from unittest.mock import MagicMock, patch

import pandas as pd

from gold import run


def test_compute_and_write_features_writes_when_new_rows():
    mock_warehouse = MagicMock()
    ohlcv_df = pd.DataFrame({"asset": ["XBTUSD"], "open_time": [pd.Timestamp.now(tz="UTC")]})
    mock_warehouse.read_table.side_effect = [
        ohlcv_df, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    ]
    open_time = pd.Timestamp("2026-06-01T00:00:00Z")
    indicators_df = pd.DataFrame({
        "asset": ["XBTUSD"], "open_time": [open_time], "rsi_14": [55.0],
    })
    sentiment_df = pd.DataFrame({
        "open_time": [open_time], "sentiment_4h": [0.0], "sentiment_24h": [0.0],
        "sentiment_72h": [0.0], "news_sentiment_24h": [0.0], "post_volume_spike": [False],
    })

    with (
        patch("gold.run.compute_indicators", return_value=indicators_df) as mock_ind,
        patch("gold.run.compute_sentiment_features", return_value=sentiment_df) as mock_sent,
    ):
        run.compute_and_write_features(mock_warehouse)

    mock_ind.assert_called_once()
    mock_sent.assert_called_once()
    mock_warehouse.write_table.assert_called_once()
    args, kwargs = mock_warehouse.write_table.call_args
    written_df = args[0]
    assert written_df["asset"].iloc[0] == "XBTUSD"
    assert "computed_at" in written_df.columns
    assert args[1:] == ("gold", "ml_features")
    assert kwargs == {"mode": "append"}


def test_compute_and_write_features_skips_write_when_indicators_empty():
    mock_warehouse = MagicMock()
    mock_warehouse.read_table.side_effect = [
        pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    ]

    with patch("gold.run.compute_indicators", return_value=pd.DataFrame()):
        run.compute_and_write_features(mock_warehouse)

    mock_warehouse.write_table.assert_not_called()


def test_compute_and_write_features_skips_write_when_all_keys_already_exist():
    mock_warehouse = MagicMock()
    open_time = pd.Timestamp("2026-06-01T00:00:00Z")
    ohlcv_df = pd.DataFrame({"asset": ["XBTUSD"], "open_time": [open_time]})
    gold_df = pd.DataFrame({"asset": ["XBTUSD"], "open_time": [open_time]})
    mock_warehouse.read_table.side_effect = [ohlcv_df, pd.DataFrame(), pd.DataFrame(), gold_df]

    indicators_df = pd.DataFrame({"asset": ["XBTUSD"], "open_time": [open_time], "rsi_14": [55.0]})
    sentiment_df = pd.DataFrame({
        "open_time": [open_time], "sentiment_4h": [0.0], "sentiment_24h": [0.0],
        "sentiment_72h": [0.0], "news_sentiment_24h": [0.0], "post_volume_spike": [False],
    })

    with (
        patch("gold.run.compute_indicators", return_value=indicators_df),
        patch("gold.run.compute_sentiment_features", return_value=sentiment_df),
    ):
        run.compute_and_write_features(mock_warehouse)

    mock_warehouse.write_table.assert_not_called()


def test_run_gold_cycle_calls_compute_and_write_features():
    mock_warehouse = MagicMock()
    with patch("gold.run.compute_and_write_features") as mock_compute:
        run.run_gold_cycle(mock_warehouse)

    mock_compute.assert_called_once_with(mock_warehouse)
