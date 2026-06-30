import pandas as pd
import pytest

from gold.sentiment_features import compute_sentiment_features


def _post(published_at, sentiment, confidence=0.8):
    return {
        "published_at": pd.Timestamp(published_at),
        "sentiment": sentiment,
        "confidence": confidence,
    }


def test_compute_sentiment_features_weighted_average_within_window():
    reddit_df = pd.DataFrame([
        _post("2026-06-01T08:00:00Z", sentiment=1, confidence=0.9),
        _post("2026-06-01T09:00:00Z", sentiment=-1, confidence=0.1),
    ])
    news_df = pd.DataFrame(columns=["published_at", "sentiment", "confidence"])
    candle_times = pd.Series([pd.Timestamp("2026-06-01T10:00:00Z")])

    result = compute_sentiment_features(reddit_df, news_df, candle_times)

    assert result["sentiment_4h"].iloc[0] == pytest.approx(0.8)


def test_compute_sentiment_features_excludes_posts_outside_window():
    reddit_df = pd.DataFrame([
        _post("2026-06-01T00:00:00Z", sentiment=1, confidence=0.9),
    ])
    news_df = pd.DataFrame(columns=["published_at", "sentiment", "confidence"])
    candle_times = pd.Series([pd.Timestamp("2026-06-01T10:00:00Z")])

    result = compute_sentiment_features(reddit_df, news_df, candle_times)

    assert result["sentiment_4h"].iloc[0] == pytest.approx(0.0)
    assert result["sentiment_24h"].iloc[0] == pytest.approx(1.0)


def test_compute_sentiment_features_defaults_to_neutral_when_no_posts():
    reddit_df = pd.DataFrame(columns=["published_at", "sentiment", "confidence"])
    news_df = pd.DataFrame(columns=["published_at", "sentiment", "confidence"])
    candle_times = pd.Series([pd.Timestamp("2026-06-01T10:00:00Z")])

    result = compute_sentiment_features(reddit_df, news_df, candle_times)

    assert result["sentiment_4h"].iloc[0] == 0.0
    assert result["sentiment_24h"].iloc[0] == 0.0
    assert result["sentiment_72h"].iloc[0] == 0.0
    assert result["news_sentiment_24h"].iloc[0] == 0.0
    assert result["post_volume_spike"].iloc[0] is False


def test_compute_sentiment_features_news_only_score_excludes_reddit():
    reddit_df = pd.DataFrame([_post("2026-06-01T09:00:00Z", sentiment=1, confidence=0.9)])
    news_df = pd.DataFrame([_post("2026-06-01T09:00:00Z", sentiment=-1, confidence=0.9)])
    candle_times = pd.Series([pd.Timestamp("2026-06-01T10:00:00Z")])

    result = compute_sentiment_features(reddit_df, news_df, candle_times)

    assert result["news_sentiment_24h"].iloc[0] == pytest.approx(-1.0)
    assert result["sentiment_24h"].iloc[0] == pytest.approx(0.0)


def test_compute_sentiment_features_post_volume_spike_true():
    open_time = pd.Timestamp("2026-06-08T00:00:00Z")
    baseline_posts = [
        _post(open_time - pd.Timedelta(hours=4 * i), sentiment=0, confidence=0.5)
        for i in range(2, 43)
    ]
    spike_posts = [
        _post(open_time - pd.Timedelta(hours=h), sentiment=0, confidence=0.5)
        for h in (1, 2, 3)
    ]
    reddit_df = pd.DataFrame(baseline_posts + spike_posts)
    news_df = pd.DataFrame(columns=["published_at", "sentiment", "confidence"])
    candle_times = pd.Series([open_time])

    result = compute_sentiment_features(reddit_df, news_df, candle_times)

    assert result["post_volume_spike"].iloc[0] == True  # noqa: E712


def test_compute_sentiment_features_no_spike_without_seven_days_history():
    open_time = pd.Timestamp("2026-06-01T00:00:00Z")
    reddit_df = pd.DataFrame([
        _post(open_time - pd.Timedelta(hours=1), sentiment=1, confidence=0.9),
        _post(open_time - pd.Timedelta(hours=2), sentiment=1, confidence=0.9),
        _post(open_time - pd.Timedelta(hours=3), sentiment=1, confidence=0.9),
    ])
    news_df = pd.DataFrame(columns=["published_at", "sentiment", "confidence"])
    candle_times = pd.Series([open_time])

    result = compute_sentiment_features(reddit_df, news_df, candle_times)

    assert result["post_volume_spike"].iloc[0] == False  # noqa: E712
