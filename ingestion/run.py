from adapters import get_warehouse
from adapters.config import REDDIT_USER_AGENT
from ingestion.binance import BinanceClient
from ingestion.crypto_news import CryptoNewsClient
from ingestion.reddit import RedditClient

ASSETS = ["BTCUSDT", "ETHUSDT"]
SUBREDDITS = ["Bitcoin", "ethereum", "CryptoCurrency"]


def ingest_binance(warehouse) -> None:
    client = BinanceClient()
    for asset in ASSETS:
        df = client.fetch_ohlcv(asset)
        warehouse.write_table(df, "bronze", "binance_ohlcv", mode="append")
        print(f"[binance] wrote {len(df)} row(s) for {asset}")


def ingest_reddit(warehouse) -> None:
    client = RedditClient(user_agent=REDDIT_USER_AGENT)
    df = client.fetch_posts(SUBREDDITS)
    warehouse.write_table(df, "bronze", "reddit_posts", mode="append")
    print(f"[reddit] wrote {len(df)} row(s)")


def ingest_news(warehouse) -> None:
    client = CryptoNewsClient()
    df = client.fetch_news()
    warehouse.write_table(df, "bronze", "crypto_news", mode="append")
    print(f"[news] wrote {len(df)} row(s)")


def run_ingestion_cycle(warehouse) -> None:
    ingest_binance(warehouse)
    ingest_reddit(warehouse)
    ingest_news(warehouse)


if __name__ == "__main__":
    wh = get_warehouse()
    run_ingestion_cycle(wh)
    print("Bronze ingestion cycle complete")
