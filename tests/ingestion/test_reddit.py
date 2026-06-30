from unittest.mock import MagicMock, patch

import pandas as pd

from ingestion.reddit import RedditClient

_ATOM_NS = "http://www.w3.org/2005/Atom"
_BTC_URL = "https://www.reddit.com/r/Bitcoin/comments/abc/"
_CC_URL = "https://www.reddit.com/r/CryptoCurrency/comments/abc/"


def _atom_xml(entries: list[dict]) -> bytes:
    entries_xml = "".join(
        f'<entry xmlns="{_ATOM_NS}">'
        f"<title>{e['title']}</title>"
        f'<link href="{e["url"]}" />'
        f"<updated>{e['updated']}</updated>"
        f"</entry>"
        for e in entries
    )
    return (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f'<feed xmlns="{_ATOM_NS}">{entries_xml}</feed>'
    ).encode()


def _mock_response(entries: list[dict]) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.content = _atom_xml(entries)
    return resp


def _hours_ago(h: float) -> str:
    return (pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=h)).isoformat()


def test_fetch_posts_returns_expected_schema():
    client = RedditClient()
    entries = [{"title": "BTC moon", "url": _BTC_URL, "updated": _hours_ago(1)}]
    with patch("requests.get", return_value=_mock_response(entries)):
        df = client.fetch_posts(subreddits=["Bitcoin"], hours=4)

    expected_cols = {"subreddit", "title", "url", "published_at", "ingested_at"}
    assert expected_cols.issubset(set(df.columns))
    assert len(df) == 1
    assert df["subreddit"].iloc[0] == "Bitcoin"


def test_fetch_posts_filters_old_posts():
    client = RedditClient()
    entries = [
        {"title": "Old post", "url": _BTC_URL, "updated": _hours_ago(6)},
        {"title": "Recent post", "url": _BTC_URL, "updated": _hours_ago(1)},
    ]
    with patch("requests.get", return_value=_mock_response(entries)):
        df = client.fetch_posts(subreddits=["Bitcoin"], hours=4)

    assert len(df) == 1
    assert df["title"].iloc[0] == "Recent post"


def test_fetch_posts_makes_single_combined_request():
    client = RedditClient()
    entries = [
        {"title": "Post 1", "url": _BTC_URL, "updated": _hours_ago(1)},
        {"title": "Post 2", "url": _CC_URL, "updated": _hours_ago(1)},
    ]
    with patch("requests.get", return_value=_mock_response(entries)) as mock_get:
        df = client.fetch_posts(subreddits=["Bitcoin", "CryptoCurrency"], hours=4)

    assert mock_get.call_count == 1
    called_url = mock_get.call_args[0][0]
    assert "Bitcoin+CryptoCurrency" in called_url
    assert len(df) == 2


def test_fetch_posts_parses_timestamps_as_utc():
    client = RedditClient()
    entries = [{"title": "Post", "url": _BTC_URL, "updated": _hours_ago(1)}]
    with patch("requests.get", return_value=_mock_response(entries)):
        df = client.fetch_posts(subreddits=["Bitcoin"], hours=4)

    assert df["published_at"].iloc[0].tzinfo is not None
    assert df["ingested_at"].iloc[0].tzinfo is not None
