from adapters import get_llm, get_warehouse
from silver.ohlcv import clean_ohlcv
from silver.sentiment import score_new_posts


def clean_and_write_ohlcv(warehouse) -> None:
    bronze_df = warehouse.read_table("bronze", "ohlcv")
    silver_df = warehouse.read_table("silver", "ohlcv")
    existing_keys = set(zip(silver_df.get("asset", []), silver_df.get("open_time", [])))

    cleaned = clean_ohlcv(bronze_df, existing_keys)
    if cleaned.empty:
        print("[silver-ohlcv] no new rows")
        return
    warehouse.write_table(cleaned, "silver", "ohlcv", mode="append")
    print(f"[silver-ohlcv] wrote {len(cleaned)} row(s)")


def score_and_write_reddit(warehouse, llm) -> None:
    bronze_df = warehouse.read_table("bronze", "reddit_posts")
    silver_df = warehouse.read_table("silver", "reddit_posts")
    existing_urls = set(silver_df.get("url", []))

    scored = score_new_posts(bronze_df, existing_urls, llm)
    if scored.empty:
        print("[silver-reddit] no new rows")
        return
    warehouse.write_table(scored, "silver", "reddit_posts", mode="append")
    print(f"[silver-reddit] wrote {len(scored)} row(s)")


def score_and_write_news(warehouse, llm) -> None:
    bronze_df = warehouse.read_table("bronze", "crypto_news")
    silver_df = warehouse.read_table("silver", "crypto_news")
    existing_urls = set(silver_df.get("url", []))

    scored = score_new_posts(bronze_df, existing_urls, llm)
    if scored.empty:
        print("[silver-news] no new rows")
        return
    warehouse.write_table(scored, "silver", "crypto_news", mode="append")
    print(f"[silver-news] wrote {len(scored)} row(s)")


def run_silver_cycle(warehouse, llm) -> None:
    clean_and_write_ohlcv(warehouse)
    score_and_write_reddit(warehouse, llm)
    score_and_write_news(warehouse, llm)


if __name__ == "__main__":
    wh = get_warehouse()
    model = get_llm()
    run_silver_cycle(wh, model)
    print("Silver cycle complete")
