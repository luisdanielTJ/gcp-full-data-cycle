from adapters import get_warehouse
from adapters.config import REDDIT_USER_AGENT
from ingestion.crypto_news import CryptoNewsClient
from ingestion.kraken import KrakenClient
from ingestion.reddit import RedditClient

ASSETS = ["XBTUSD", "ETHUSD"]
SUBREDDITS = ["Bitcoin", "ethereum", "CryptoCurrency"]


def ingest_ohlcv(warehouse) -> None:
    client = KrakenClient()
    for pair in ASSETS:
        df = client.fetch_ohlcv(pair)
        warehouse.write_table(df, "bronze", "ohlcv", mode="append")
        print(f"[kraken] wrote {len(df)} row(s) for {pair}")


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
    ingest_ohlcv(warehouse)
    ingest_reddit(warehouse)
    ingest_news(warehouse)


if __name__ == "__main__":
    wh = get_warehouse()
    run_ingestion_cycle(wh)
    print("Bronze ingestion cycle complete")
