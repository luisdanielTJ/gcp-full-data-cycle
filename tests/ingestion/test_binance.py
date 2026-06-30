from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from ingestion.binance import BinanceClient


def _mock_kline_response():
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = [
        [
            1718870400000, "50000.00", "50500.00", "49800.00", "50250.00", "1234.5",
            1718884799999, "61912345.0", 1000, "600.0", "30000000.0", "0",
        ]
    ]
    return response


def test_fetch_ohlcv_returns_expected_schema():
    client = BinanceClient()
    with patch("requests.get", return_value=_mock_kline_response()):
        df = client.fetch_ohlcv("BTCUSDT")
    expected_cols = {"asset", "open_time", "open", "high", "low", "close",
                     "volume", "close_time", "ingested_at"}
    assert expected_cols.issubset(set(df.columns))
    assert len(df) == 1


def test_fetch_ohlcv_parses_values_correctly():
    client = BinanceClient()
    with patch("requests.get", return_value=_mock_kline_response()):
        df = client.fetch_ohlcv("BTCUSDT")
    assert df["asset"].iloc[0] == "BTCUSDT"
    assert df["close"].iloc[0] == pytest.approx(50250.0)
    assert df["open"].iloc[0] == pytest.approx(50000.0)
    assert df["volume"].iloc[0] == pytest.approx(1234.5)


def test_fetch_ohlcv_uses_4h_interval_and_symbol():
    client = BinanceClient()
    with patch("requests.get", return_value=_mock_kline_response()) as mock_get:
        client.fetch_ohlcv("ETHUSDT")
    _, kwargs = mock_get.call_args
    assert kwargs["params"]["interval"] == "4h"
    assert kwargs["params"]["symbol"] == "ETHUSDT"
    assert kwargs["params"]["limit"] == 1


def test_fetch_ohlcv_timestamps_are_utc():
    client = BinanceClient()
    with patch("requests.get", return_value=_mock_kline_response()):
        df = client.fetch_ohlcv("BTCUSDT")
    assert df["open_time"].iloc[0].tzinfo is not None
    assert df["ingested_at"].iloc[0].tzinfo is not None
