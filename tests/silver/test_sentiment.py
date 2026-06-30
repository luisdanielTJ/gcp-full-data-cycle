from unittest.mock import MagicMock

import pandas as pd
import pytest

from silver.sentiment import score_new_posts


def _bronze_row(url="https://example.com/1", title="BTC moon"):
    return {
        "subreddit": "Bitcoin",
        "title": title,
        "url": url,
        "published_at": pd.Timestamp.now(tz="UTC"),
        "ingested_at": pd.Timestamp.now(tz="UTC"),
    }


def test_score_new_posts_filters_existing_urls():
    bronze_df = pd.DataFrame([
        _bronze_row(url="https://example.com/old"),
        _bronze_row(url="https://example.com/new"),
    ])
    llm = MagicMock()
    llm.score_sentiment.return_value = {"sentiment": 1, "confidence": 0.9, "reason": "bullish"}

    result = score_new_posts(bronze_df, existing_urls={"https://example.com/old"}, llm=llm)

    assert len(result) == 1
    assert result["url"].iloc[0] == "https://example.com/new"
    llm.score_sentiment.assert_called_once_with("BTC moon")


def test_score_new_posts_adds_sentiment_columns():
    bronze_df = pd.DataFrame([_bronze_row()])
    llm = MagicMock()
    llm.score_sentiment.return_value = {
        "sentiment": -1, "confidence": 0.7, "reason": "bearish news",
    }

    result = score_new_posts(bronze_df, existing_urls=set(), llm=llm)

    assert result["sentiment"].iloc[0] == -1
    assert result["confidence"].iloc[0] == pytest.approx(0.7)
    assert result["reason"].iloc[0] == "bearish news"
    assert result["scored_at"].iloc[0].tzinfo is not None


def test_score_new_posts_skips_failed_rows_and_continues():
    bronze_df = pd.DataFrame([
        _bronze_row(url="https://example.com/fails", title="bad post"),
        _bronze_row(url="https://example.com/ok", title="good post"),
    ])
    llm = MagicMock()
    llm.score_sentiment.side_effect = [
        Exception("rate limited"),
        {"sentiment": 0, "confidence": 0.5, "reason": "neutral"},
    ]

    result = score_new_posts(bronze_df, existing_urls=set(), llm=llm)

    assert len(result) == 1
    assert result["url"].iloc[0] == "https://example.com/ok"


def test_score_new_posts_returns_empty_when_nothing_new():
    bronze_df = pd.DataFrame([_bronze_row(url="https://example.com/old")])
    llm = MagicMock()

    result = score_new_posts(bronze_df, existing_urls={"https://example.com/old"}, llm=llm)

    assert result.empty
    llm.score_sentiment.assert_not_called()


def test_score_new_posts_handles_empty_bronze_df():
    llm = MagicMock()

    result = score_new_posts(pd.DataFrame(), existing_urls=set(), llm=llm)

    assert result.empty
    llm.score_sentiment.assert_not_called()
