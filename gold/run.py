import pandas as pd

from adapters import get_warehouse
from gold.indicators import compute_indicators
from gold.sentiment_features import compute_sentiment_features


def compute_and_write_features(warehouse) -> None:
    ohlcv_df = warehouse.read_table("silver", "ohlcv")
    reddit_df = warehouse.read_table("silver", "reddit_posts")
    news_df = warehouse.read_table("silver", "crypto_news")
    gold_df = warehouse.read_table("gold", "ml_features")

    indicators_df = compute_indicators(ohlcv_df)
    if indicators_df.empty:
        print("[gold] no new rows")
        return

    sentiment_df = compute_sentiment_features(reddit_df, news_df, indicators_df["open_time"])
    merged = indicators_df.merge(sentiment_df, on="open_time", how="left")

    existing_keys = set(zip(gold_df.get("asset", []), gold_df.get("open_time", [])))
    keys = list(zip(merged["asset"], merged["open_time"]))
    merged = merged[[k not in existing_keys for k in keys]].reset_index(drop=True)

    if merged.empty:
        print("[gold] no new rows")
        return

    merged["computed_at"] = pd.Timestamp.now(tz="UTC")
    warehouse.write_table(merged, "gold", "ml_features", mode="append")
    print(f"[gold] wrote {len(merged)} row(s)")


def run_gold_cycle(warehouse) -> None:
    compute_and_write_features(warehouse)


if __name__ == "__main__":
    wh = get_warehouse()
    run_gold_cycle(wh)
    print("Gold cycle complete")
