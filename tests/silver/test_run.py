from unittest.mock import MagicMock, patch

import pandas as pd

from silver import run


def test_clean_and_write_ohlcv_writes_when_new_rows():
    mock_warehouse = MagicMock()
    bronze_df = pd.DataFrame({"asset": ["XBTUSD"], "open_time": [pd.Timestamp.now(tz="UTC")]})
    mock_warehouse.read_table.side_effect = [bronze_df, pd.DataFrame()]
    cleaned_df = pd.DataFrame({"asset": ["XBTUSD"], "close": [50000.0]})

    with patch("silver.run.clean_ohlcv", return_value=cleaned_df) as mock_clean:
        run.clean_and_write_ohlcv(mock_warehouse)

    mock_clean.assert_called_once()
    mock_warehouse.write_table.assert_called_once_with(
        cleaned_df, "silver", "ohlcv", mode="append"
    )


def test_clean_and_write_ohlcv_skips_write_when_empty():
    mock_warehouse = MagicMock()
    mock_warehouse.read_table.side_effect = [pd.DataFrame(), pd.DataFrame()]

    with patch("silver.run.clean_ohlcv", return_value=pd.DataFrame()):
        run.clean_and_write_ohlcv(mock_warehouse)

    mock_warehouse.write_table.assert_not_called()


def test_score_and_write_reddit_writes_when_new_rows():
    mock_warehouse = MagicMock()
    mock_llm = MagicMock()
    bronze_df = pd.DataFrame({"url": ["https://example.com/1"], "title": ["BTC moon"]})
    mock_warehouse.read_table.side_effect = [bronze_df, pd.DataFrame()]
    scored_df = pd.DataFrame({"url": ["https://example.com/1"], "sentiment": [1]})

    with patch("silver.run.score_new_posts", return_value=scored_df) as mock_score:
        run.score_and_write_reddit(mock_warehouse, mock_llm)

    mock_score.assert_called_once()
    mock_warehouse.write_table.assert_called_once_with(
        scored_df, "silver", "reddit_posts", mode="append"
    )


def test_score_and_write_news_writes_when_new_rows():
    mock_warehouse = MagicMock()
    mock_llm = MagicMock()
    bronze_df = pd.DataFrame({"url": ["https://example.com/n1"], "title": ["ETH ATH"]})
    mock_warehouse.read_table.side_effect = [bronze_df, pd.DataFrame()]
    scored_df = pd.DataFrame({"url": ["https://example.com/n1"], "sentiment": [1]})

    with patch("silver.run.score_new_posts", return_value=scored_df) as mock_score:
        run.score_and_write_news(mock_warehouse, mock_llm)

    mock_score.assert_called_once()
    mock_warehouse.write_table.assert_called_once_with(
        scored_df, "silver", "crypto_news", mode="append"
    )


def test_run_silver_cycle_calls_all_three():
    mock_warehouse = MagicMock()
    mock_llm = MagicMock()
    with (
        patch("silver.run.clean_and_write_ohlcv") as mock_ohlcv,
        patch("silver.run.score_and_write_reddit") as mock_reddit,
        patch("silver.run.score_and_write_news") as mock_news,
    ):
        run.run_silver_cycle(mock_warehouse, mock_llm)

    mock_ohlcv.assert_called_once_with(mock_warehouse)
    mock_reddit.assert_called_once_with(mock_warehouse, mock_llm)
    mock_news.assert_called_once_with(mock_warehouse, mock_llm)
