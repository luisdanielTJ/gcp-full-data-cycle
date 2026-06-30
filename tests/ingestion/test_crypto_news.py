import time
from email.utils import formatdate
from unittest.mock import MagicMock, patch

from ingestion.crypto_news import CryptoNewsClient


def _rss_xml(items: list[dict]) -> bytes:
    items_xml = "".join(
        f"<item>"
        f"<title>{i['title']}</title>"
        f"<link>{i['url']}</link>"
        f"<pubDate>{i['pub_date']}</pubDate>"
        f"</item>"
        for i in items
    )
    return (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f"<rss version=\"2.0\"><channel>{items_xml}</channel></rss>"
    ).encode()


def _mock_response(items: list[dict]) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.content = _rss_xml(items)
    return resp


def _pub_date(seconds_ago: float) -> str:
    return formatdate(time.time() - seconds_ago)


def test_fetch_news_returns_expected_schema():
    client = CryptoNewsClient(feeds={"coindesk": "http://fake/rss"})
    items = [{"title": "BTC hits ATH", "url": "https://coindesk.com/1",
               "pub_date": _pub_date(3600)}]

    with patch("requests.get", return_value=_mock_response(items)):
        df = client.fetch_news(hours=4)

    expected_cols = {"source", "title", "url", "published_at", "ingested_at"}
    assert expected_cols.issubset(set(df.columns))
    assert len(df) == 1
    assert df["source"].iloc[0] == "coindesk"


def test_fetch_news_filters_old_items():
    client = CryptoNewsClient(feeds={"coindesk": "http://fake/rss"})
    items = [
        {"title": "Recent news", "url": "https://coindesk.com/1", "pub_date": _pub_date(3600)},
        {"title": "Old news", "url": "https://coindesk.com/2", "pub_date": _pub_date(6 * 3600)},
    ]

    with patch("requests.get", return_value=_mock_response(items)):
        df = client.fetch_news(hours=4)

    assert len(df) == 1
    assert df["title"].iloc[0] == "Recent news"


def test_fetch_news_queries_all_feeds():
    feeds = {
        "coindesk": "http://fake/coindesk",
        "cointelegraph": "http://fake/cointelegraph",
    }
    client = CryptoNewsClient(feeds=feeds)

    def get_side_effect(url, headers, timeout):
        source = "coindesk" if "coindesk" in url else "cointelegraph"
        items = [{"title": f"{source} headline", "url": url + "/1", "pub_date": _pub_date(1800)}]
        return _mock_response(items)

    with patch("requests.get", side_effect=get_side_effect):
        df = client.fetch_news(hours=4)

    assert len(df) == 2
    assert set(df["source"].tolist()) == {"coindesk", "cointelegraph"}


def test_fetch_news_parses_timestamps_as_utc():
    client = CryptoNewsClient(feeds={"coindesk": "http://fake/rss"})
    items = [{"title": "News", "url": "https://coindesk.com/1", "pub_date": _pub_date(3600)}]

    with patch("requests.get", return_value=_mock_response(items)):
        df = client.fetch_news(hours=4)

    assert df["published_at"].iloc[0].tzinfo is not None
    assert df["ingested_at"].iloc[0].tzinfo is not None
