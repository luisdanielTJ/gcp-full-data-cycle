from unittest.mock import MagicMock, patch

import pytest

from ingestion.kraken import KrakenClient


def _mock_response(pair: str = "XXBTZUSD") -> MagicMock:
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {
        "error": [],
        "result": {
            pair: [
                [1718870400, "50000.00", "50500.00", "49800.00",
                 "50250.00", "50100.00", "1234.5", 500],
            ],
            "last": 1718870400,
        },
    }
    return response


def test_fetch_ohlcv_returns_expected_schema():
    client = KrakenClient()
    with patch("requests.get", return_value=_mock_response()):
        df = client.fetch_ohlcv("XBTUSD")
    expected_cols = {"asset", "open_time", "open", "high", "low", "close", "volume", "ingested_at"}
    assert expected_cols.issubset(set(df.columns))
    assert len(df) == 1


def test_fetch_ohlcv_parses_values_correctly():
    client = KrakenClient()
    with patch("requests.get", return_value=_mock_response()):
        df = client.fetch_ohlcv("XBTUSD")
    assert df["asset"].iloc[0] == "XBTUSD"
    assert df["close"].iloc[0] == pytest.approx(50250.0)
    assert df["open"].iloc[0] == pytest.approx(50000.0)
    assert df["volume"].iloc[0] == pytest.approx(1234.5)


def test_fetch_ohlcv_uses_4h_interval_and_pair():
    client = KrakenClient()
    with patch("requests.get", return_value=_mock_response()) as mock_get:
        client.fetch_ohlcv("ETHUSD")
    _, kwargs = mock_get.call_args
    assert kwargs["params"]["interval"] == 240
    assert kwargs["params"]["pair"] == "ETHUSD"


def test_fetch_ohlcv_timestamps_are_utc():
    client = KrakenClient()
    with patch("requests.get", return_value=_mock_response()):
        df = client.fetch_ohlcv("XBTUSD")
    assert df["open_time"].iloc[0].tzinfo is not None
    assert df["ingested_at"].iloc[0].tzinfo is not None


def test_fetch_ohlcv_returns_all_candles():
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {
        "error": [],
        "result": {
            "XXBTZUSD": [
                [1718870400, "50000.00", "50500.00", "49800.00", "50250.00", "50100.00", "100.0", 50],
                [1718884800, "50250.00", "51000.00", "50100.00", "50900.00", "50500.00", "150.0", 75],
            ],
            "last": 1718884800,
        },
    }
    client = KrakenClient()
    with patch("requests.get", return_value=response):
        df = client.fetch_ohlcv("XBTUSD")
    assert len(df) == 2
    assert df["open_time"].iloc[0] < df["open_time"].iloc[1]
