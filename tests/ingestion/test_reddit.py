from unittest.mock import MagicMock, patch

import pandas as pd

from ingestion.reddit import RedditClient


def _mock_json_response(posts: list[dict]) -> MagicMock:
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {"data": {"children": [{"data": p} for p in posts]}}
    return response


def _make_post(title: str, score: int, created_utc: float, subreddit: str = "CryptoCurrency") -> dict:
    return {
        "title": title,
        "score": score,
        "created_utc": created_utc,
        "url": f"https://reddit.com/r/{subreddit}/comments/abc",
        "subreddit": subreddit,
    }


def _now_utc() -> float:
    return pd.Timestamp.now(tz="UTC").timestamp()


def test_fetch_posts_returns_expected_schema():
    client = RedditClient()
    posts = [_make_post("BTC moon", score=100, created_utc=_now_utc() - 3600)]

    with patch("requests.get", return_value=_mock_json_response(posts)):
        df = client.fetch_posts(subreddits=["CryptoCurrency"], hours=4, min_upvotes=10)

    expected_cols = {"subreddit", "title", "score", "url", "created_utc", "ingested_at"}
    assert expected_cols.issubset(set(df.columns))
    assert len(df) == 1


def test_fetch_posts_filters_low_score():
    client = RedditClient()
    recent_ts = _now_utc() - 3600
    posts = [
        _make_post("High score post", score=50, created_utc=recent_ts),
        _make_post("Low score post", score=5, created_utc=recent_ts),
    ]

    with patch("requests.get", return_value=_mock_json_response(posts)):
        df = client.fetch_posts(subreddits=["CryptoCurrency"], hours=4, min_upvotes=10)

    assert len(df) == 1
    assert df["title"].iloc[0] == "High score post"


def test_fetch_posts_filters_old_posts():
    client = RedditClient()
    posts = [
        _make_post("Old post", score=100, created_utc=_now_utc() - 6 * 3600),
        _make_post("Recent post", score=100, created_utc=_now_utc() - 3600),
    ]

    with patch("requests.get", return_value=_mock_json_response(posts)):
        df = client.fetch_posts(subreddits=["CryptoCurrency"], hours=4, min_upvotes=10)

    assert len(df) == 1
    assert df["title"].iloc[0] == "Recent post"


def test_fetch_posts_queries_all_subreddits():
    client = RedditClient()
    recent_ts = _now_utc() - 1800

    def get_side_effect(url, params, headers, timeout):
        sub = url.split("/r/")[1].split("/")[0]
        return _mock_json_response(
            [_make_post(f"Post from {sub}", score=50, created_utc=recent_ts, subreddit=sub)]
        )

    with patch("requests.get", side_effect=get_side_effect):
        df = client.fetch_posts(subreddits=["CryptoCurrency", "Bitcoin"], hours=4, min_upvotes=10)

    assert len(df) == 2
    assert set(df["subreddit"].tolist()) == {"CryptoCurrency", "Bitcoin"}
