from unittest.mock import MagicMock, patch

import pandas as pd

from ingestion import run


def test_ingest_ohlcv_writes_to_bronze_for_each_pair():
    fake_df = pd.DataFrame({"asset": ["XBTUSD"], "close": [50000.0]})
    mock_warehouse = MagicMock()
    with patch("ingestion.run.KrakenClient") as mock_cls:
        mock_client = MagicMock()
        mock_client.fetch_ohlcv.return_value = fake_df
        mock_cls.return_value = mock_client

        run.ingest_ohlcv(mock_warehouse)

    assert mock_client.fetch_ohlcv.call_count == len(run.ASSETS)
    assert mock_warehouse.write_table.call_count == len(run.ASSETS)
    args = mock_warehouse.write_table.call_args[0]
    assert args[1] == "bronze"
    assert args[2] == "ohlcv"
    assert mock_warehouse.write_table.call_args[1]["mode"] == "append"


def test_ingest_reddit_writes_to_bronze():
    fake_df = pd.DataFrame({"title": ["BTC post"]})
    mock_warehouse = MagicMock()
    with patch("ingestion.run.RedditClient") as mock_cls:
        mock_client = MagicMock()
        mock_client.fetch_posts.return_value = fake_df
        mock_cls.return_value = mock_client

        run.ingest_reddit(mock_warehouse)

    mock_client.fetch_posts.assert_called_once_with(run.SUBREDDITS)
    mock_warehouse.write_table.assert_called_once_with(
        fake_df, "bronze", "reddit_posts", mode="append"
    )


def test_ingest_news_writes_to_bronze():
    fake_df = pd.DataFrame({"title": ["BTC ATH"]})
    mock_warehouse = MagicMock()
    with patch("ingestion.run.CryptoNewsClient") as mock_cls:
        mock_client = MagicMock()
        mock_client.fetch_news.return_value = fake_df
        mock_cls.return_value = mock_client

        run.ingest_news(mock_warehouse)

    mock_client.fetch_news.assert_called_once()
    mock_warehouse.write_table.assert_called_once_with(
        fake_df, "bronze", "crypto_news", mode="append"
    )


def test_run_ingestion_cycle_calls_all_three_ingesters():
    mock_warehouse = MagicMock()
    with (
        patch("ingestion.run.ingest_ohlcv") as mock_ohlcv,
        patch("ingestion.run.ingest_reddit") as mock_red,
        patch("ingestion.run.ingest_news") as mock_news,
    ):
        run.run_ingestion_cycle(mock_warehouse)

    mock_ohlcv.assert_called_once_with(mock_warehouse)
    mock_red.assert_called_once_with(mock_warehouse)
    mock_news.assert_called_once_with(mock_warehouse)
