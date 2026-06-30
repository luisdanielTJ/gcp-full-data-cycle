from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from ingestion.reddit import RedditClient


def _make_submission(title: str, score: int, created_utc: float, url: str = "https://reddit.com/r/test") -> MagicMock:
    sub = MagicMock()
    sub.title = title
    sub.score = score
    sub.created_utc = created_utc
    sub.url = url
    sub.subreddit.display_name = "CryptoCurrency"
    return sub


def _now_utc() -> float:
    return pd.Timestamp.now(tz="UTC").timestamp()


def test_fetch_posts_returns_expected_schema():
    client = RedditClient(client_id="id", client_secret="secret", user_agent="test/0.1")
    recent_ts = _now_utc() - 3600  # 1 hour ago
    submissions = [_make_submission("BTC moon", score=100, created_utc=recent_ts)]

    with patch("praw.Reddit") as mock_reddit_cls:
        mock_reddit = mock_reddit_cls.return_value
        mock_reddit.subreddit.return_value.new.return_value = iter(submissions)
        df = client.fetch_posts(subreddits=["CryptoCurrency"], hours=4, min_upvotes=10)

    expected_cols = {"subreddit", "title", "score", "url", "created_utc", "ingested_at"}
    assert expected_cols.issubset(set(df.columns))
    assert len(df) == 1


def test_fetch_posts_filters_low_score():
    client = RedditClient(client_id="id", client_secret="secret", user_agent="test/0.1")
    recent_ts = _now_utc() - 3600
    submissions = [
        _make_submission("High score post", score=50, created_utc=recent_ts),
        _make_submission("Low score post", score=5, created_utc=recent_ts),
    ]

    with patch("praw.Reddit") as mock_reddit_cls:
        mock_reddit = mock_reddit_cls.return_value
        mock_reddit.subreddit.return_value.new.return_value = iter(submissions)
        df = client.fetch_posts(subreddits=["CryptoCurrency"], hours=4, min_upvotes=10)

    assert len(df) == 1
    assert df["title"].iloc[0] == "High score post"


def test_fetch_posts_filters_old_posts():
    client = RedditClient(client_id="id", client_secret="secret", user_agent="test/0.1")
    old_ts = _now_utc() - 6 * 3600  # 6 hours ago — outside 4h window
    recent_ts = _now_utc() - 3600   # 1 hour ago
    submissions = [
        _make_submission("Old post", score=100, created_utc=old_ts),
        _make_submission("Recent post", score=100, created_utc=recent_ts),
    ]

    with patch("praw.Reddit") as mock_reddit_cls:
        mock_reddit = mock_reddit_cls.return_value
        mock_reddit.subreddit.return_value.new.return_value = iter(submissions)
        df = client.fetch_posts(subreddits=["CryptoCurrency"], hours=4, min_upvotes=10)

    assert len(df) == 1
    assert df["title"].iloc[0] == "Recent post"


def test_fetch_posts_queries_all_subreddits():
    client = RedditClient(client_id="id", client_secret="secret", user_agent="test/0.1")
    recent_ts = _now_utc() - 1800

    def subreddit_side_effect(name: str):
        sub_mock = MagicMock()
        s = MagicMock()
        s.title = f"Post from {name}"
        s.score = 50
        s.created_utc = recent_ts
        s.url = f"https://reddit.com/r/{name}/post"
        s.subreddit.display_name = name
        sub_mock.new.return_value = iter([s])
        return sub_mock

    with patch("praw.Reddit") as mock_reddit_cls:
        mock_reddit = mock_reddit_cls.return_value
        mock_reddit.subreddit.side_effect = subreddit_side_effect
        df = client.fetch_posts(subreddits=["CryptoCurrency", "Bitcoin"], hours=4, min_upvotes=10)

    assert len(df) == 2
    subreddits_found = set(df["subreddit"].tolist())
    assert subreddits_found == {"CryptoCurrency", "Bitcoin"}
