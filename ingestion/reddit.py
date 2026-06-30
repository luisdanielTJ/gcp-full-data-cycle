import pandas as pd


class RedditClient:
    _BASE_URL = "https://www.reddit.com/r/{subreddit}/new.json"

    def __init__(self, user_agent: str = "crypto-edge-ingestion/0.1") -> None:
        self._user_agent = user_agent

    def fetch_posts(
        self,
        subreddits: list[str],
        hours: int = 4,
        min_upvotes: int = 10,
    ) -> pd.DataFrame:
        import requests

        cutoff_ts = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours)).timestamp()
        rows = []
        for name in subreddits:
            response = requests.get(
                self._BASE_URL.format(subreddit=name),
                params={"limit": 100},
                headers={"User-Agent": self._user_agent},
                timeout=10,
            )
            response.raise_for_status()
            for post in response.json()["data"]["children"]:
                d = post["data"]
                if d["created_utc"] < cutoff_ts or d["score"] < min_upvotes:
                    continue
                rows.append(
                    {
                        "subreddit": d["subreddit"],
                        "title": d["title"],
                        "score": d["score"],
                        "url": d["url"],
                        "created_utc": pd.Timestamp(d["created_utc"], unit="s", tz="UTC"),
                        "ingested_at": pd.Timestamp.now(tz="UTC"),
                    }
                )
        return pd.DataFrame(rows)
