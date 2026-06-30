import xml.etree.ElementTree as ET

import pandas as pd

_ATOM_NS = "http://www.w3.org/2005/Atom"


def _subreddit_from_url(url: str) -> str:
    """Extract subreddit name from a Reddit post URL."""
    try:
        parts = url.split("/r/")
        return parts[1].split("/")[0] if len(parts) > 1 else ""
    except Exception:
        return ""


class RedditClient:
    _FEED_URL = "https://www.reddit.com/r/{subreddits}/new.rss"

    def __init__(self, user_agent: str = "crypto-edge-ingestion/0.1") -> None:
        self._user_agent = user_agent

    def fetch_posts(
        self,
        subreddits: list[str],
        hours: int = 4,
    ) -> pd.DataFrame:
        import requests

        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours)
        combined = "+".join(subreddits)
        response = requests.get(
            self._FEED_URL.format(subreddits=combined),
            headers={"User-Agent": self._user_agent},
            timeout=10,
        )
        response.raise_for_status()
        root = ET.fromstring(response.content)
        rows = []
        for entry in root.findall(f"{{{_ATOM_NS}}}entry"):
            title = (entry.findtext(f"{{{_ATOM_NS}}}title") or "").strip()
            link_el = entry.find(f"{{{_ATOM_NS}}}link")
            url = link_el.get("href", "") if link_el is not None else ""
            updated = entry.findtext(f"{{{_ATOM_NS}}}updated") or ""
            if not updated:
                continue
            try:
                published = pd.Timestamp(updated).tz_convert("UTC")
            except Exception:
                continue
            if published < cutoff:
                continue
            rows.append(
                {
                    "subreddit": _subreddit_from_url(url),
                    "title": title,
                    "url": url,
                    "published_at": published,
                    "ingested_at": pd.Timestamp.now(tz="UTC"),
                }
            )
        return pd.DataFrame(rows)
