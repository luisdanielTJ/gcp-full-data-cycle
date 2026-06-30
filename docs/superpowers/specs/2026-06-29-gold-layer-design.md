# Gold Layer — Design Spec

**Date:** 2026-06-29
**Status:** Approved

---

## 1. Overview

Plan 4 in the crypto-edge build. Builds the Gold layer of the medallion architecture: technical indicators per asset and aggregated sentiment features, joined into a single ML feature table consumed by future model training (Plan 5).

This deviates from the original design spec (`2026-06-29-crypto-edge-design.md`, sections 5.3/5.4) the same way Bronze and Silver already did:

- **No dbt.** Continues the plain Python + adapter pattern established in Plan 2/3.
- **No label column at Gold time.** The original spec computes the binary label ("did price close >1% higher next candle?") at the Gold layer via a look-ahead join. Bronze/Silver/Gold in this project are append-only — there's no update-in-place path, and the next candle needed for the label doesn't exist yet at the time a row is written. The label is deferred to Plan 5 (ML training), which will join Gold features against `silver.ohlcv` at training time, when the future candle is actually available.

## 2. Scope

**In scope:**
- Technical indicators per asset per 4h candle: RSI(14), MACD(12/26/9), Bollinger Bands(20, 2σ), ATR(14), volume ratio, price momentum (1/3/6 candles)
- Sentiment aggregates, shared identically across BTC and ETH rows for the same `open_time`: rolling sentiment (4h/24h/72h, Reddit+news combined), a separate news-only 24h sentiment score, and a post-volume-spike flag
- Incremental processing — never recompute a `(asset, open_time)` row already in Gold
- Wire into the existing Cloud Run Job, same 4h schedule as ingestion and Silver

**Out of scope (Plan 5 — ML Training):**
- Label computation (next-candle >1% threshold)
- XGBoost training, hyperparameter tuning, model registry promotion
- Prediction serving

## 3. New Dependency

`pandas-ta-classic` (PyPI, v0.6.52+), imported as `pandas_ta_classic`. The original `pandas-ta` package is no longer viable for this project: the actively maintained release on PyPI now requires Python ≥3.12 (this project pins 3.11), and the older 3.11-compatible releases hard-fail on numpy ≥2.0 (`ImportError` from a removed `numpy.NaN` alias — confirmed against this project's installed numpy 2.4.6). `pandas-ta-classic` is a community-maintained fork with the same API, confirmed working in this project's environment (Python 3.11, numpy 2.4.6, pandas 3.0.3) — RSI, MACD, Bollinger Bands, and ATR all verified to produce correct output.

## 4. Components

### `gold/indicators.py`

```python
def compute_indicators(ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups ohlcv_df by `asset`. For each asset's full historical series
    (sorted by open_time), computes via pandas_ta_classic:
      - RSI(14)                          -> rsi_14
      - MACD(12, 26, 9)                  -> macd_line, macd_signal, macd_hist
      - Bollinger Bands(20, 2σ)          -> bb_upper, bb_lower, bb_width
      - ATR(14)                          -> atr_14
    Plus manually computed:
      - volume_ratio = volume / rolling_mean(volume, 20)
      - momentum_1/3/6 = pct_change(close, periods=1/3/6)

    Rows without enough trailing history for a given indicator come out
    NaN from pandas_ta_classic / pct_change and are dropped before
    returning — never fabricated or zero-filled. Returns one row per
    (asset, open_time) that has a complete set of indicators.
    """
```

### `gold/sentiment_features.py`

```python
def compute_sentiment_features(
    reddit_df: pd.DataFrame,
    news_df: pd.DataFrame,
    candle_times: pd.Series,
) -> pd.DataFrame:
    """
    For each timestamp in candle_times, looks back from that open_time and computes:
      - sentiment_4h / sentiment_24h / sentiment_72h:
            confidence-weighted average of `sentiment` across reddit_df and
            news_df combined, restricted to published_at in
            (open_time - window, open_time]. Defaults to 0.0 (neutral)
            if no posts fall in the window.
      - news_sentiment_24h:
            same weighted-average calculation, restricted to news_df only,
            24h window.
      - post_volume_spike:
            True if the combined post count in (open_time - 4h, open_time]
            exceeds 2x the average combined post count per 4h window over
            the trailing 7 days (42 candles). False (not null) when there's
            not yet 7 days of history.

    Returns one row per candle_time with columns:
      open_time, sentiment_4h, sentiment_24h, sentiment_72h,
      news_sentiment_24h, post_volume_spike
    """
```

These values are computed once per `open_time` and reused for both BTC and ETH rows in `gold/run.py` — sentiment is market-wide, not asset-specific.

### `gold/run.py`

```python
def compute_and_write_features(warehouse) -> None:
    """
    - Reads silver.ohlcv, silver.reddit_posts, silver.crypto_news, and
      the existing gold.ml_features (for keys already present).
    - Calls compute_indicators(ohlcv_df) to get indicator rows per
      (asset, open_time).
    - Calls compute_sentiment_features(...) once per distinct open_time
      present in the indicator rows.
    - Merges indicators with sentiment features on open_time (sentiment
      broadcasts to both assets).
    - Anti-joins against existing (asset, open_time) keys in gold.ml_features.
    - Adds a `computed_at` UTC timestamp column to the merged rows.
    - Appends only the new rows. If there are none, logs and returns
      without writing.
    """

def run_gold_cycle(warehouse) -> None:
    """Calls compute_and_write_features. Mirrors run_silver_cycle's role in pipeline.py."""
```

Mirrors the structure of `silver/run.py`. No LLM adapter needed — Gold only reads sentiment that Silver already scored.

### `pipeline.py` (modify)

```python
from adapters import get_llm, get_warehouse
from gold.run import run_gold_cycle
from ingestion.run import run_ingestion_cycle
from silver.run import run_silver_cycle

if __name__ == "__main__":
    wh = get_warehouse()
    llm = get_llm()
    run_ingestion_cycle(wh)
    run_silver_cycle(wh, llm)
    run_gold_cycle(wh)
    print("Pipeline cycle complete")
```

### `Dockerfile` (modify)

Add `COPY gold/ gold/` alongside the existing `COPY silver/ silver/` line.

### `pyproject.toml` (modify)

Add `"pandas-ta-classic>=0.6.52"` to `dependencies`, and `"gold"` to `[tool.hatch.build.targets.wheel].packages`.

## 5. Tables

| Table | Columns |
|---|---|
| `gold.ml_features` | `asset, open_time, rsi_14, macd_line, macd_signal, macd_hist, bb_upper, bb_lower, bb_width, atr_14, volume_ratio, momentum_1, momentum_3, momentum_6, sentiment_4h, sentiment_24h, sentiment_72h, news_sentiment_24h, post_volume_spike, computed_at` |

No label column. `gold.ml_features` is append-only, like every other table in this project — Plan 5 computes labels at training time by joining against `silver.ohlcv`, not by writing back into Gold.

## 6. Data Flow Per Cycle

1. Silver (existing, unchanged) writes new cleaned OHLCV and scored sentiment rows.
2. Gold reads the full `silver.ohlcv`, `silver.reddit_posts`, `silver.crypto_news`, and the full `gold.ml_features` (for existing keys).
3. Computes indicators across each asset's entire historical OHLCV series (pandas-ta indicators need trailing history, not just the newest row), then drops rows with incomplete history.
4. Computes sentiment aggregates once per distinct `open_time` appearing in the indicator output.
5. Merges indicators + sentiment, anti-joins on `(asset, open_time)` against what's already in `gold.ml_features`, appends only the new rows.

Full-table reads are intentional and consistent with Bronze/Silver — this is a personal-scale project, not high-volume.

## 7. Error Handling

- **Insufficient OHLCV history for an indicator** (e.g., fewer than 26 candles for a newly-added asset): that `(asset, open_time)` row is dropped from the indicator output, not fabricated. Resolves itself automatically as more candles accumulate over subsequent cycles.
- **No Reddit/news posts in a sentiment window:** sentiment defaults to `0.0` (neutral), not null or fabricated as bullish/bearish.
- **Fewer than 7 days of post history for volume-spike baseline:** `post_volume_spike` defaults to `False`, not null.
- **No new candles since the last Gold cycle:** skip the write, log and continue — same pattern as Silver's "no new rows" handling.

## 8. Testing

TDD throughout, following the existing project pattern:

- `tests/gold/test_indicators.py` — `compute_indicators`: correct indicator values on a known synthetic OHLCV series, NaN/incomplete-history rows dropped, grouping by asset is correct (no cross-asset leakage)
- `tests/gold/test_sentiment_features.py` — `compute_sentiment_features`: weighted-average calculation across windows, neutral default when no posts in window, post-volume-spike true/false cases, news-only score excludes Reddit rows
- `tests/gold/test_run.py` — orchestration: `compute_and_write_features` reads via `read_table`, anti-joins correctly against existing Gold keys, writes via `write_table`, skips write when nothing new; `run_gold_cycle` calls it

## 9. Out of Scope / Deferred to Plan 5 (ML Training)

- Label generation (binary, >1% next-candle threshold) via look-ahead join against `silver.ohlcv` at training time
- XGBoost binary classifier training, threshold tuning (>0.65 BUY, <0.35 SELL)
- Model registry promotion gate (precision/recall/F1/AUC-ROC, signal accuracy >55%, Sharpe vs buy-and-hold)
- SHAP feature importance
- Prediction serving
