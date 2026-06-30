import pandas as pd

_WINDOWS = {
    "sentiment_4h": pd.Timedelta(hours=4),
    "sentiment_24h": pd.Timedelta(hours=24),
    "sentiment_72h": pd.Timedelta(hours=72),
}
_NEWS_WINDOW = pd.Timedelta(hours=24)
_SPIKE_WINDOW = pd.Timedelta(hours=4)
_SPIKE_LOOKBACK = pd.Timedelta(days=7)
_SPIKE_CANDLES = 42  # 7 days of 4h candles

_POST_COLUMNS = ["published_at", "sentiment", "confidence"]


def _normalize(posts: pd.DataFrame) -> pd.DataFrame:
    if posts.empty:
        return pd.DataFrame(columns=_POST_COLUMNS)
    posts = posts.copy()
    posts["published_at"] = pd.to_datetime(posts["published_at"], utc=True)
    return posts


def _weighted_sentiment(posts: pd.DataFrame, end: pd.Timestamp, window: pd.Timedelta) -> float:
    in_window = posts[
        (posts["published_at"] > end - window) & (posts["published_at"] <= end)
    ]
    if in_window.empty:
        return 0.0
    weights = in_window["confidence"]
    if weights.sum() == 0:
        return 0.0
    return float((in_window["sentiment"] * weights).sum() / weights.sum())


def _post_volume_spike(posts: pd.DataFrame, open_time: pd.Timestamp) -> bool:
    if posts.empty:
        return False
    earliest = posts["published_at"].min()
    if earliest > open_time - _SPIKE_LOOKBACK:
        return False

    current_count = len(posts[
        (posts["published_at"] > open_time - _SPIKE_WINDOW)
        & (posts["published_at"] <= open_time)
    ])
    lookback_count = len(posts[
        (posts["published_at"] > open_time - _SPIKE_LOOKBACK)
        & (posts["published_at"] <= open_time)
    ])
    avg_per_candle = lookback_count / _SPIKE_CANDLES
    if avg_per_candle == 0:
        return False
    return bool(current_count > 2 * avg_per_candle)


def compute_sentiment_features(
    reddit_df: pd.DataFrame, news_df: pd.DataFrame, candle_times: pd.Series
) -> pd.DataFrame:
    reddit_df = _normalize(reddit_df)
    news_df = _normalize(news_df)
    combined_df = pd.concat([reddit_df, news_df], ignore_index=True)

    rows = []
    for open_time in sorted(set(candle_times)):
        row = {"open_time": open_time}
        for col, window in _WINDOWS.items():
            row[col] = _weighted_sentiment(combined_df, open_time, window)
        row["news_sentiment_24h"] = _weighted_sentiment(news_df, open_time, _NEWS_WINDOW)
        row["post_volume_spike"] = _post_volume_spike(combined_df, open_time)
        rows.append(row)

    result = pd.DataFrame(rows)
    result["post_volume_spike"] = result["post_volume_spike"].astype(object)
    return result
