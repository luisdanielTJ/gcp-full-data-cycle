from adapters import get_warehouse
from adapters.config import (
    CRYPTOPANIC_API_KEY,
    REDDIT_CLIENT_ID,
    REDDIT_CLIENT_SECRET,
    REDDIT_USER_AGENT,
)
from ingestion.binance import BinanceClient
from ingestion.cryptopanic import CryptoPanicClient
from ingestion.reddit import RedditClient

ASSETS = ["BTCUSDT", "ETHUSDT"]
SUBREDDITS = ["Bitcoin", "ethereum", "CryptoCurrency"]
CURRENCIES = ["BTC", "ETH"]


def ingest_binance(warehouse) -> None:
    client = BinanceClient()
    for asset in ASSETS:
        df = client.fetch_ohlcv(asset)
        warehouse.write_table(df, "bronze", "binance_ohlcv", mode="append")
        print(f"[binance] wrote {len(df)} row(s) for {asset}")


def ingest_reddit(warehouse) -> None:
    client = RedditClient(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
    )
    df = client.fetch_posts(SUBREDDITS)
    warehouse.write_table(df, "bronze", "reddit_posts", mode="append")
    print(f"[reddit] wrote {len(df)} row(s)")


def ingest_cryptopanic(warehouse) -> None:
    client = CryptoPanicClient(api_key=CRYPTOPANIC_API_KEY)
    df = client.fetch_news(currencies=CURRENCIES)
    warehouse.write_table(df, "bronze", "cryptopanic_news", mode="append")
    print(f"[cryptopanic] wrote {len(df)} row(s)")


def run_ingestion_cycle(warehouse) -> None:
    ingest_binance(warehouse)
    ingest_reddit(warehouse)
    ingest_cryptopanic(warehouse)


if __name__ == "__main__":
    wh = get_warehouse()
    run_ingestion_cycle(wh)
    print("Bronze ingestion cycle complete")
