import pandas as pd


class RedditClient:
    _FETCH_LIMIT = 100

    def __init__(self, client_id: str, client_secret: str, user_agent: str) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._user_agent = user_agent
        self._reddit = None

    def _get_reddit(self):
        if self._reddit is None:
            import praw

            self._reddit = praw.Reddit(
                client_id=self._client_id,
                client_secret=self._client_secret,
                user_agent=self._user_agent,
            )
        return self._reddit

    def fetch_posts(
        self,
        subreddits: list[str],
        hours: int = 4,
        min_upvotes: int = 10,
    ) -> pd.DataFrame:
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours)
        cutoff_ts = cutoff.timestamp()
        rows = []
        for name in subreddits:
            for post in self._get_reddit().subreddit(name).new(limit=self._FETCH_LIMIT):
                if post.created_utc < cutoff_ts:
                    continue
                if post.score < min_upvotes:
                    continue
                rows.append(
                    {
                        "subreddit": post.subreddit.display_name,
                        "title": post.title,
                        "score": post.score,
                        "url": post.url,
                        "created_utc": pd.Timestamp(post.created_utc, unit="s", tz="UTC"),
                        "ingested_at": pd.Timestamp.now(tz="UTC"),
                    }
                )
        return pd.DataFrame(rows)
