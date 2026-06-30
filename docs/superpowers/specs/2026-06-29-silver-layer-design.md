# Silver Layer — Design Spec

**Date:** 2026-06-29
**Status:** Approved

---

## 1. Overview

Plan 3 in the crypto-edge build. Builds the Silver layer of the medallion architecture: cleaning bronze OHLCV data and scoring sentiment on bronze Reddit posts / crypto news headlines, using the LLM adapter already built in Plan 1.

This deviates from the original design spec (`2026-06-29-crypto-edge-design.md`, section 5.2) in two ways, both already established by earlier deviations in this project:

- **No dbt.** The original spec called for dbt-bigquery models. Since ingestion (Plan 2) used a plain Python + adapter pattern instead of Cloud Composer, Silver continues that pattern for consistency. No new toolchain.
- **No `ML.GENERATE_TEXT`.** The original spec scored sentiment via BigQuery's `ML.GENERATE_TEXT`, which is Vertex AI/Gemini-specific and doesn't support OpenAI. Sentiment scoring instead calls `LLMAdapter.score_sentiment()` directly from Python (already implemented in Plan 1, supports both Gemini and OpenAI).

## 2. Scope

**In scope:**
- Clean bronze OHLCV: type casting, dedup, gap logging (no fabrication)
- Score sentiment on new Reddit posts and crypto news via `LLMAdapter`
- Incremental processing — never re-score a post already in Silver
- Wire into the existing Cloud Run Job, same 4h schedule as ingestion

**Out of scope (Plan 4 — Gold layer):**
- Technical indicators (RSI, MACD, Bollinger Bands, ATR, momentum)
- Rolling/aggregated sentiment scores across time windows
- ML feature table and label generation

## 3. Adapter Change

`WarehouseAdapter` currently exposes `write_table` and `run_query` only. Reading a full table back requires backend-specific SQL: BigQuery uses `{dataset}.{table}`, DuckDB uses `{dataset}__{table}` (see `adapters/warehouse.py`). Callers should not need to know this.

Add a symmetric method:

```python
def read_table(self, dataset: str, table: str) -> pd.DataFrame:
    """Returns the full table contents, or an empty DataFrame if the table doesn't exist yet."""
```

Implemented in both `DuckDBWarehouse` and `BigQueryWarehouse`. Returns an empty DataFrame (not an exception) when the table doesn't exist — this is the normal case on the very first run, before any Silver table has been created.

## 4. Components

### `silver/ohlcv.py`

```python
def clean_ohlcv(bronze_df: pd.DataFrame, existing_keys: set[tuple]) -> pd.DataFrame:
    """
    - Filters out rows whose (asset, open_time) already exist in Silver.
    - Casts open/high/low/close/volume to float, open_time to UTC timestamp.
    - Dedupes remaining rows by (asset, open_time).
    - For each asset, sorts by open_time and logs a warning for any gap
      where the difference between consecutive open_times isn't exactly 4 hours.
    - Does NOT fabricate forward-filled rows for gaps.
    - Adds a `cleaned_at` UTC timestamp column.
    """
```

### `silver/sentiment.py`

```python
def score_new_posts(bronze_df: pd.DataFrame, existing_urls: set[str], llm) -> pd.DataFrame:
    """
    - Filters bronze_df to rows whose `url` is not in existing_urls.
    - For each remaining row, calls llm.score_sentiment(title).
    - On a per-row failure (exception from the LLM call or invalid response),
      logs a warning with the URL and skips that row — it stays unscored in
      bronze and will be retried next cycle since it won't be in Silver yet.
    - Successful rows get `sentiment`, `confidence`, `reason` columns appended
      from the LLM response, plus a `scored_at` UTC timestamp.
    - Returns only the successfully scored rows (empty DataFrame if none).
    """
```

Used for both Reddit posts and crypto news — same function, different bronze/silver table names passed by the caller in `silver/run.py`.

### `silver/run.py`

```python
def clean_and_write_ohlcv(warehouse) -> None:
    """Reads bronze.ohlcv and silver.ohlcv, cleans new rows, appends to silver.ohlcv."""

def score_and_write_reddit(warehouse, llm) -> None:
    """Reads bronze.reddit_posts and silver.reddit_posts, scores new rows, appends to silver.reddit_posts."""

def score_and_write_news(warehouse, llm) -> None:
    """Reads bronze.crypto_news and silver.crypto_news, scores new rows, appends to silver.crypto_news."""

def run_silver_cycle(warehouse, llm) -> None:
    """Calls all three in sequence."""
```

Mirrors the structure of `ingestion/run.py`.

### `pipeline.py` (new, repo root)

```python
from adapters import get_llm, get_warehouse
from ingestion.run import run_ingestion_cycle
from silver.run import run_silver_cycle

if __name__ == "__main__":
    wh = get_warehouse()
    llm = get_llm()
    run_ingestion_cycle(wh)
    run_silver_cycle(wh, llm)
    print("Pipeline cycle complete")
```

Replaces `ingestion.run` as the Docker `CMD` entrypoint. `ingestion/run.py` keeps its own `if __name__ == "__main__"` block for standalone manual testing of ingestion alone.

## 5. Tables

| Table | Columns |
|---|---|
| `silver.ohlcv` | `asset, open_time, open, high, low, close, volume, cleaned_at` |
| `silver.reddit_posts` | `subreddit, title, url, published_at, sentiment, confidence, reason, scored_at` |
| `silver.crypto_news` | `source, title, url, published_at, sentiment, confidence, reason, scored_at` |

All three datasets (`bronze`, `silver`, `gold`, `predictions`, `trades`) already exist in BigQuery from Plan 1 (foundation infrastructure).

## 6. Data Flow Per Cycle

1. Ingestion (existing, unchanged) writes new rows to `bronze.ohlcv`, `bronze.reddit_posts`, `bronze.crypto_news`.
2. Silver reads the full bronze table and the full silver table for each of the three pairs (`read_table`).
3. Computes the new rows via a pandas set difference on the natural key (`(asset, open_time)` for OHLCV, `url` for posts/news).
4. Cleans (OHLCV) or scores (posts/news) only the new rows.
5. Appends results to the corresponding silver table.

Full-table reads are intentional and fine at this scale — bronze tables grow by a handful of rows every 4 hours (personal-scale project, not high-volume).

## 7. Error Handling

- **Per-post LLM failure:** caught, logged with the post URL, skipped. Retried automatically next cycle (anti-join naturally picks it up again since it's absent from Silver).
- **OHLCV gap:** logged as a warning, not raised. Missing candles stay missing — no fabricated data enters Silver or downstream model training.
- **Silver table doesn't exist yet (first run):** `read_table` returns an empty DataFrame, so the "existing keys" set is empty and everything in bronze is treated as new. No special-casing needed in `silver/run.py`.

## 8. Testing

TDD throughout, following the existing project pattern:

- `tests/adapters/test_warehouse.py` — add tests for `read_table` (DuckDB: real read after write; both backends: empty DataFrame when table doesn't exist)
- `tests/silver/test_ohlcv.py` — `clean_ohlcv`: dedup, type casting, gap-warning logging, exclusion of already-Silver rows
- `tests/silver/test_sentiment.py` — `score_new_posts`: anti-join filtering, mocked LLM success path, mocked LLM per-row failure (skip + continue), empty result when nothing is new
- `tests/silver/test_run.py` — orchestration: each `*_and_write_*` function calls `read_table` correctly and writes via `write_table`; `run_silver_cycle` calls all three

## 9. Out of Scope / Deferred to Plan 4 (Gold Layer)

- Technical indicators (RSI, MACD, Bollinger Bands, ATR, momentum)
- Rolling sentiment aggregation (4h/24h/72h windows), post volume spike detection
- `gold.ml_features` table and label generation
