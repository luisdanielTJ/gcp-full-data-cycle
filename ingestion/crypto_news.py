import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

import pandas as pd

RSS_FEEDS = {
    "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "cointelegraph": "https://cointelegraph.com/rss",
}


class CryptoNewsClient:
    def __init__(self, feeds: dict[str, str] = RSS_FEEDS) -> None:
        self._feeds = feeds

    def fetch_news(self, hours: int = 4) -> pd.DataFrame:
        import requests

        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours)
        rows = []
        for source, url in self._feeds.items():
            response = requests.get(
                url,
                headers={"User-Agent": "crypto-edge-ingestion/0.1"},
                timeout=10,
            )
            response.raise_for_status()
            root = ET.fromstring(response.content)
            for item in root.findall(".//item"):
                title = item.findtext("title", "").strip()
                link = item.findtext("link", "").strip()
                pub_date = item.findtext("pubDate", "").strip()
                if not pub_date:
                    continue
                try:
                    published = pd.Timestamp(parsedate_to_datetime(pub_date)).tz_convert("UTC")
                except Exception:
                    try:
                        ts = pd.Timestamp(pub_date)
                        published = (
                            ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
                        )
                    except Exception:
                        continue
                if published < cutoff:
                    continue
                rows.append(
                    {
                        "source": source,
                        "title": title,
                        "url": link,
                        "published_at": published,
                        "ingested_at": pd.Timestamp.now(tz="UTC"),
                    }
                )
        return pd.DataFrame(rows)
