from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from ingestion.cryptopanic import CryptoPanicClient


def _mock_response(results: list[dict]) -> MagicMock:
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {"results": results}
    return response


def _make_result(title: str, published_at: str, url: str = "https://cryptopanic.com/news/1") -> dict:
    return {"title": title, "published_at": published_at, "url": url}


def test_fetch_news_returns_expected_schema():
    client = CryptoPanicClient(api_key="test-key")
    results = [_make_result("BTC hits ATH", "2024-06-20T10:00:00Z")]

    with patch("requests.get", return_value=_mock_response(results)):
        df = client.fetch_news(currencies=["BTC"])

    expected_cols = {"title", "published_at", "url", "currency", "ingested_at"}
    assert expected_cols.issubset(set(df.columns))
    assert len(df) == 1


def test_fetch_news_passes_auth_and_currency():
    client = CryptoPanicClient(api_key="my-api-key")
    results = [_make_result("ETH upgrade", "2024-06-20T12:00:00Z")]

    with patch("requests.get", return_value=_mock_response(results)) as mock_get:
        client.fetch_news(currencies=["ETH"])

    _, kwargs = mock_get.call_args
    assert kwargs["params"]["auth_token"] == "my-api-key"
    assert kwargs["params"]["currencies"] == "ETH"


def test_fetch_news_queries_all_currencies():
    client = CryptoPanicClient(api_key="test-key")

    def get_side_effect(url, params, timeout):
        currency = params["currencies"]
        return _mock_response([_make_result(f"{currency} news", "2024-06-20T10:00:00Z")])

    with patch("requests.get", side_effect=get_side_effect):
        df = client.fetch_news(currencies=["BTC", "ETH"])

    assert len(df) == 2
    currencies_found = set(df["currency"].tolist())
    assert currencies_found == {"BTC", "ETH"}


def test_fetch_news_parses_timestamps_as_utc():
    client = CryptoPanicClient(api_key="test-key")
    results = [_make_result("BTC news", "2024-06-20T10:00:00Z")]

    with patch("requests.get", return_value=_mock_response(results)):
        df = client.fetch_news(currencies=["BTC"])

    assert df["published_at"].iloc[0].tzinfo is not None
    assert df["ingested_at"].iloc[0].tzinfo is not None
