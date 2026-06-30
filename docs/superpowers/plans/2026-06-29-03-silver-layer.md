# Silver Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clean bronze OHLCV data and score sentiment on bronze Reddit posts / crypto news via the existing LLM adapter, writing results to BigQuery `silver` tables, running in the same Cloud Run Job as ingestion.

**Architecture:** New `silver/` Python package (mirrors `ingestion/`) reads bronze tables, computes only-new rows via a pandas anti-join against existing silver tables, cleans/scores them, and appends to silver. A new root-level `pipeline.py` entrypoint runs ingestion then silver in one Cloud Run Job execution. Requires adding a `read_table()` method to `WarehouseAdapter`.

**Tech Stack:** Python 3.11, pandas, existing `adapters.WarehouseAdapter` / `adapters.LLMAdapter`, pytest, ruff, Docker, Cloud Run Jobs, Cloud Scheduler (unchanged), BigQuery, Secret Manager.

**Reference spec:** `docs/superpowers/specs/2026-06-29-silver-layer-design.md`

---

### Task 1: Add `read_table()` to WarehouseAdapter

**Files:**
- Modify: `adapters/warehouse.py`
- Test: `tests/adapters/test_warehouse.py`
- Create: `tests/adapters/test_bigquery_warehouse.py`

- [ ] **Step 1: Write the failing DuckDB tests**

Add to the end of `tests/adapters/test_warehouse.py`:

```python
def test_read_table_returns_written_data(warehouse):
    df = pd.DataFrame({
        "asset": ["BTC"], "close": [50000.0],
        "ts": [pd.Timestamp("2024-01-01", tz="UTC")],
    })
    warehouse.write_table(df, "bronze", "raw_prices", mode="replace")
    result = warehouse.read_table("bronze", "raw_prices")
    assert len(result) == 1
    assert result["asset"].iloc[0] == "BTC"


def test_read_table_returns_empty_when_table_missing(warehouse):
    result = warehouse.read_table("silver", "does_not_exist")
    assert result.empty
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/adapters/test_warehouse.py -v`
Expected: FAIL with `AttributeError: 'DuckDBWarehouse' object has no attribute 'read_table'`

- [ ] **Step 3: Write the failing BigQuery tests**

Create `tests/adapters/test_bigquery_warehouse.py`:

```python
from unittest.mock import MagicMock, patch

import pandas as pd

from adapters.warehouse import BigQueryWarehouse


def test_read_table_returns_dataframe():
    with patch("google.cloud.bigquery.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_query_job = MagicMock()
        mock_query_job.to_dataframe.return_value = pd.DataFrame({"asset": ["BTC"]})
        mock_client.query.return_value = mock_query_job

        warehouse = BigQueryWarehouse(project_id="test-project")
        result = warehouse.read_table("silver", "ohlcv")

    assert result["asset"].iloc[0] == "BTC"
    called_sql = mock_client.query.call_args[0][0]
    assert "test-project.silver.ohlcv" in called_sql


def test_read_table_returns_empty_on_not_found():
    from google.api_core.exceptions import NotFound

    with patch("google.cloud.bigquery.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.query.side_effect = NotFound("not found")

        warehouse = BigQueryWarehouse(project_id="test-project")
        result = warehouse.read_table("silver", "does_not_exist")

    assert result.empty
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `uv run pytest tests/adapters/test_bigquery_warehouse.py -v`
Expected: FAIL with `AttributeError: 'BigQueryWarehouse' object has no attribute 'read_table'`

- [ ] **Step 5: Implement `read_table()`**

In `adapters/warehouse.py`, add the abstract method to `WarehouseAdapter`:

```python
class WarehouseAdapter(ABC):
    @abstractmethod
    def run_query(self, sql: str) -> pd.DataFrame:
        ...

    @abstractmethod
    def write_table(
        self, df: pd.DataFrame, dataset: str, table: str, mode: str = "append"
    ) -> None:
        ...

    @abstractmethod
    def read_table(self, dataset: str, table: str) -> pd.DataFrame:
        """Returns the full table contents, or an empty DataFrame if it doesn't exist yet."""
        ...
```

Add to `DuckDBWarehouse`:

```python
    def read_table(self, dataset: str, table: str) -> pd.DataFrame:
        table_name = f"{dataset}__{table}"
        existing = self.conn.execute("SHOW TABLES").fetchdf()
        if table_name not in existing["name"].values:
            return pd.DataFrame()
        return self.conn.execute(f"SELECT * FROM {table_name}").df()
```

Add to `BigQueryWarehouse`:

```python
    def read_table(self, dataset: str, table: str) -> pd.DataFrame:
        from google.api_core.exceptions import NotFound

        table_ref = f"{self.project_id}.{dataset}.{table}"
        try:
            return self.client.query(f"SELECT * FROM `{table_ref}`").to_dataframe()
        except NotFound:
            return pd.DataFrame()
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/adapters/test_warehouse.py tests/adapters/test_bigquery_warehouse.py -v`
Expected: All PASS

- [ ] **Step 7: Lint and commit**

Run: `uv run ruff check .`
Expected: All checks passed!

```bash
git add adapters/warehouse.py tests/adapters/test_warehouse.py tests/adapters/test_bigquery_warehouse.py
git commit -m "feat: add read_table() to WarehouseAdapter for incremental silver reads"
```

---

### Task 2: Create the `silver` package skeleton

**Files:**
- Create: `silver/__init__.py`
- Create: `tests/silver/__init__.py`

- [ ] **Step 1: Create empty package files**

Create `silver/__init__.py` with empty content.
Create `tests/silver/__init__.py` with empty content.

- [ ] **Step 2: Verify the package imports**

Run: `uv run python -c "import silver"`
Expected: no output, exit code 0

- [ ] **Step 3: Commit**

```bash
git add silver/__init__.py tests/silver/__init__.py
git commit -m "chore: create silver package skeleton"
```

---

### Task 3: `silver/ohlcv.py` — OHLCV cleaning

**Files:**
- Create: `silver/ohlcv.py`
- Test: `tests/silver/test_ohlcv.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/silver/test_ohlcv.py`:

```python
import logging

import pandas as pd
import pytest

from silver.ohlcv import clean_ohlcv


def _bronze_row(asset="XBTUSD", open_time="2026-06-01T00:00:00Z", close=50000.0):
    return {
        "asset": asset,
        "open_time": pd.Timestamp(open_time),
        "open": 49000.0,
        "high": 50500.0,
        "low": 48900.0,
        "close": close,
        "volume": 100.0,
        "ingested_at": pd.Timestamp.now(tz="UTC"),
    }


def test_clean_ohlcv_excludes_existing_keys():
    bronze_df = pd.DataFrame([
        _bronze_row(open_time="2026-06-01T00:00:00Z"),
        _bronze_row(open_time="2026-06-01T04:00:00Z"),
    ])
    existing_keys = {("XBTUSD", pd.Timestamp("2026-06-01T00:00:00Z", tz="UTC"))}

    result = clean_ohlcv(bronze_df, existing_keys)

    assert len(result) == 1
    assert result["open_time"].iloc[0] == pd.Timestamp("2026-06-01T04:00:00Z", tz="UTC")


def test_clean_ohlcv_casts_types_and_adds_cleaned_at():
    bronze_df = pd.DataFrame([_bronze_row()])

    result = clean_ohlcv(bronze_df, existing_keys=set())

    assert result["close"].iloc[0] == pytest.approx(50000.0)
    assert "cleaned_at" in result.columns
    assert result["cleaned_at"].iloc[0].tzinfo is not None
    assert "ingested_at" not in result.columns


def test_clean_ohlcv_dedupes_duplicate_rows():
    bronze_df = pd.DataFrame([_bronze_row(), _bronze_row()])

    result = clean_ohlcv(bronze_df, existing_keys=set())

    assert len(result) == 1


def test_clean_ohlcv_logs_warning_on_gap(caplog):
    bronze_df = pd.DataFrame([
        _bronze_row(open_time="2026-06-01T00:00:00Z"),
        _bronze_row(open_time="2026-06-01T12:00:00Z"),
    ])

    with caplog.at_level(logging.WARNING):
        clean_ohlcv(bronze_df, existing_keys=set())

    assert "gap" in caplog.text.lower()


def test_clean_ohlcv_no_warning_when_consecutive_4h(caplog):
    bronze_df = pd.DataFrame([
        _bronze_row(open_time="2026-06-01T00:00:00Z"),
        _bronze_row(open_time="2026-06-01T04:00:00Z"),
    ])

    with caplog.at_level(logging.WARNING):
        clean_ohlcv(bronze_df, existing_keys=set())

    assert "gap" not in caplog.text.lower()


def test_clean_ohlcv_handles_empty_bronze_df():
    result = clean_ohlcv(pd.DataFrame(), existing_keys=set())

    assert result.empty
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/silver/test_ohlcv.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'silver.ohlcv'`

- [ ] **Step 3: Implement `clean_ohlcv()`**

Create `silver/ohlcv.py`:

```python
import logging

import pandas as pd

logger = logging.getLogger(__name__)

_EXPECTED_INTERVAL = pd.Timedelta(hours=4)


def clean_ohlcv(bronze_df: pd.DataFrame, existing_keys: set[tuple]) -> pd.DataFrame:
    if bronze_df.empty:
        return bronze_df

    df = bronze_df.copy()
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    keys = list(zip(df["asset"], df["open_time"]))
    df = df[[k not in existing_keys for k in keys]]
    df = df.drop_duplicates(subset=["asset", "open_time"])

    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)

    for asset, group in df.groupby("asset"):
        sorted_times = group.sort_values("open_time")["open_time"]
        gaps = sorted_times.diff().dropna()
        for gap in gaps[gaps != _EXPECTED_INTERVAL]:
            logger.warning("OHLCV gap detected for %s: %s between candles", asset, gap)

    df["cleaned_at"] = pd.Timestamp.now(tz="UTC")
    return df.drop(columns=["ingested_at"], errors="ignore").reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/silver/test_ohlcv.py -v`
Expected: All PASS

- [ ] **Step 5: Lint and commit**

Run: `uv run ruff check .`
Expected: All checks passed!

```bash
git add silver/ohlcv.py tests/silver/test_ohlcv.py
git commit -m "feat: add clean_ohlcv() for silver OHLCV cleaning"
```

---

### Task 4: `silver/sentiment.py` — sentiment scoring

**Files:**
- Create: `silver/sentiment.py`
- Test: `tests/silver/test_sentiment.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/silver/test_sentiment.py`:

```python
from unittest.mock import MagicMock

import pandas as pd
import pytest

from silver.sentiment import score_new_posts


def _bronze_row(url="https://example.com/1", title="BTC moon"):
    return {
        "subreddit": "Bitcoin",
        "title": title,
        "url": url,
        "published_at": pd.Timestamp.now(tz="UTC"),
        "ingested_at": pd.Timestamp.now(tz="UTC"),
    }


def test_score_new_posts_filters_existing_urls():
    bronze_df = pd.DataFrame([
        _bronze_row(url="https://example.com/old"),
        _bronze_row(url="https://example.com/new"),
    ])
    llm = MagicMock()
    llm.score_sentiment.return_value = {"sentiment": 1, "confidence": 0.9, "reason": "bullish"}

    result = score_new_posts(bronze_df, existing_urls={"https://example.com/old"}, llm=llm)

    assert len(result) == 1
    assert result["url"].iloc[0] == "https://example.com/new"
    llm.score_sentiment.assert_called_once_with("BTC moon")


def test_score_new_posts_adds_sentiment_columns():
    bronze_df = pd.DataFrame([_bronze_row()])
    llm = MagicMock()
    llm.score_sentiment.return_value = {
        "sentiment": -1, "confidence": 0.7, "reason": "bearish news",
    }

    result = score_new_posts(bronze_df, existing_urls=set(), llm=llm)

    assert result["sentiment"].iloc[0] == -1
    assert result["confidence"].iloc[0] == pytest.approx(0.7)
    assert result["reason"].iloc[0] == "bearish news"
    assert result["scored_at"].iloc[0].tzinfo is not None


def test_score_new_posts_skips_failed_rows_and_continues():
    bronze_df = pd.DataFrame([
        _bronze_row(url="https://example.com/fails", title="bad post"),
        _bronze_row(url="https://example.com/ok", title="good post"),
    ])
    llm = MagicMock()
    llm.score_sentiment.side_effect = [
        Exception("rate limited"),
        {"sentiment": 0, "confidence": 0.5, "reason": "neutral"},
    ]

    result = score_new_posts(bronze_df, existing_urls=set(), llm=llm)

    assert len(result) == 1
    assert result["url"].iloc[0] == "https://example.com/ok"


def test_score_new_posts_returns_empty_when_nothing_new():
    bronze_df = pd.DataFrame([_bronze_row(url="https://example.com/old")])
    llm = MagicMock()

    result = score_new_posts(bronze_df, existing_urls={"https://example.com/old"}, llm=llm)

    assert result.empty
    llm.score_sentiment.assert_not_called()


def test_score_new_posts_handles_empty_bronze_df():
    llm = MagicMock()

    result = score_new_posts(pd.DataFrame(), existing_urls=set(), llm=llm)

    assert result.empty
    llm.score_sentiment.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/silver/test_sentiment.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'silver.sentiment'`

- [ ] **Step 3: Implement `score_new_posts()`**

Create `silver/sentiment.py`:

```python
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def score_new_posts(bronze_df: pd.DataFrame, existing_urls: set[str], llm) -> pd.DataFrame:
    if bronze_df.empty:
        return bronze_df

    new_rows = bronze_df[~bronze_df["url"].isin(existing_urls)]

    scored = []
    for _, row in new_rows.iterrows():
        try:
            result = llm.score_sentiment(row["title"])
        except Exception:
            logger.warning("Sentiment scoring failed for %s", row["url"], exc_info=True)
            continue
        scored_row = row.to_dict()
        scored_row["sentiment"] = result["sentiment"]
        scored_row["confidence"] = result["confidence"]
        scored_row["reason"] = result["reason"]
        scored_row["scored_at"] = pd.Timestamp.now(tz="UTC")
        scored.append(scored_row)

    return pd.DataFrame(scored)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/silver/test_sentiment.py -v`
Expected: All PASS

- [ ] **Step 5: Lint and commit**

Run: `uv run ruff check .`
Expected: All checks passed!

```bash
git add silver/sentiment.py tests/silver/test_sentiment.py
git commit -m "feat: add score_new_posts() for silver sentiment scoring"
```

---

### Task 5: `silver/run.py` — orchestrator

**Files:**
- Create: `silver/run.py`
- Test: `tests/silver/test_run.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/silver/test_run.py`:

```python
from unittest.mock import MagicMock, patch

import pandas as pd

from silver import run


def test_clean_and_write_ohlcv_writes_when_new_rows():
    mock_warehouse = MagicMock()
    bronze_df = pd.DataFrame({"asset": ["XBTUSD"], "open_time": [pd.Timestamp.now(tz="UTC")]})
    mock_warehouse.read_table.side_effect = [bronze_df, pd.DataFrame()]
    cleaned_df = pd.DataFrame({"asset": ["XBTUSD"], "close": [50000.0]})

    with patch("silver.run.clean_ohlcv", return_value=cleaned_df) as mock_clean:
        run.clean_and_write_ohlcv(mock_warehouse)

    mock_clean.assert_called_once()
    mock_warehouse.write_table.assert_called_once_with(
        cleaned_df, "silver", "ohlcv", mode="append"
    )


def test_clean_and_write_ohlcv_skips_write_when_empty():
    mock_warehouse = MagicMock()
    mock_warehouse.read_table.side_effect = [pd.DataFrame(), pd.DataFrame()]

    with patch("silver.run.clean_ohlcv", return_value=pd.DataFrame()):
        run.clean_and_write_ohlcv(mock_warehouse)

    mock_warehouse.write_table.assert_not_called()


def test_score_and_write_reddit_writes_when_new_rows():
    mock_warehouse = MagicMock()
    mock_llm = MagicMock()
    bronze_df = pd.DataFrame({"url": ["https://example.com/1"], "title": ["BTC moon"]})
    mock_warehouse.read_table.side_effect = [bronze_df, pd.DataFrame()]
    scored_df = pd.DataFrame({"url": ["https://example.com/1"], "sentiment": [1]})

    with patch("silver.run.score_new_posts", return_value=scored_df) as mock_score:
        run.score_and_write_reddit(mock_warehouse, mock_llm)

    mock_score.assert_called_once()
    mock_warehouse.write_table.assert_called_once_with(
        scored_df, "silver", "reddit_posts", mode="append"
    )


def test_score_and_write_news_writes_when_new_rows():
    mock_warehouse = MagicMock()
    mock_llm = MagicMock()
    bronze_df = pd.DataFrame({"url": ["https://example.com/n1"], "title": ["ETH ATH"]})
    mock_warehouse.read_table.side_effect = [bronze_df, pd.DataFrame()]
    scored_df = pd.DataFrame({"url": ["https://example.com/n1"], "sentiment": [1]})

    with patch("silver.run.score_new_posts", return_value=scored_df) as mock_score:
        run.score_and_write_news(mock_warehouse, mock_llm)

    mock_score.assert_called_once()
    mock_warehouse.write_table.assert_called_once_with(
        scored_df, "silver", "crypto_news", mode="append"
    )


def test_run_silver_cycle_calls_all_three():
    mock_warehouse = MagicMock()
    mock_llm = MagicMock()
    with (
        patch("silver.run.clean_and_write_ohlcv") as mock_ohlcv,
        patch("silver.run.score_and_write_reddit") as mock_reddit,
        patch("silver.run.score_and_write_news") as mock_news,
    ):
        run.run_silver_cycle(mock_warehouse, mock_llm)

    mock_ohlcv.assert_called_once_with(mock_warehouse)
    mock_reddit.assert_called_once_with(mock_warehouse, mock_llm)
    mock_news.assert_called_once_with(mock_warehouse, mock_llm)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/silver/test_run.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'silver.run'`

- [ ] **Step 3: Implement the orchestrator**

Create `silver/run.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/silver/test_run.py -v`
Expected: All PASS

- [ ] **Step 5: Lint and commit**

Run: `uv run ruff check .`
Expected: All checks passed!

```bash
git add silver/run.py tests/silver/test_run.py
git commit -m "feat: add silver layer orchestrator"
```

---

### Task 6: `pipeline.py` entrypoint + Dockerfile + packaging

**Files:**
- Create: `pipeline.py`
- Modify: `Dockerfile`
- Modify: `pyproject.toml`

- [ ] **Step 1: Create the pipeline entrypoint**

Create `pipeline.py` at the repo root:

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

- [ ] **Step 2: Update `pyproject.toml` packaging list**

In `pyproject.toml`, change:

```toml
[tool.hatch.build.targets.wheel]
packages = ["adapters", "ingestion"]
```

to:

```toml
[tool.hatch.build.targets.wheel]
packages = ["adapters", "ingestion", "silver"]
```

- [ ] **Step 3: Update the Dockerfile**

Replace the contents of `Dockerfile` with:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev

COPY adapters/ adapters/
COPY ingestion/ ingestion/
COPY silver/ silver/
COPY pipeline.py ./

ENV PYTHONPATH=/app

CMD ["uv", "run", "python", "pipeline.py"]
```

- [ ] **Step 4: Verify the pipeline imports locally**

Run: `uv run python -c "import pipeline"`
Expected: no output, exit code 0 (module-level code only runs under `__main__`, so this just checks imports resolve)

- [ ] **Step 5: Commit**

```bash
git add pipeline.py Dockerfile pyproject.toml
git commit -m "feat: wire ingestion + silver into a single pipeline.py entrypoint"
```

---

### Task 7: Local end-to-end verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest -v`
Expected: All tests pass (33 existing + new silver/warehouse tests)

- [ ] **Step 2: Run lint**

Run: `uv run ruff check .`
Expected: All checks passed!

- [ ] **Step 3: Run the full pipeline locally against DuckDB**

Run: `uv run python pipeline.py`

Expected output (exact counts will vary with live RSS/Kraken data, but the shape should match):

```
[kraken] wrote 1 row(s) for XBTUSD
[kraken] wrote 1 row(s) for ETHUSD
[reddit] wrote N row(s)
[news] wrote M row(s)
[silver-ohlcv] wrote 2 row(s)
[silver-reddit] wrote N row(s)
[silver-news] wrote M row(s)
Pipeline cycle complete
```

- [ ] **Step 4: Verify silver tables locally via DuckDB**

Run:

```bash
uv run python -c "
from adapters import get_warehouse
wh = get_warehouse()
print(wh.read_table('silver', 'ohlcv'))
print(wh.read_table('silver', 'reddit_posts')[['title', 'sentiment', 'confidence']])
print(wh.read_table('silver', 'crypto_news')[['title', 'sentiment', 'confidence']])
"
```

Expected: three non-empty DataFrames printed, with `sentiment` values in `{-1, 0, 1}` and `confidence` between 0.0 and 1.0.

- [ ] **Step 5: Run the pipeline a second time to verify incremental behavior**

Run: `uv run python pipeline.py`

Expected: `[silver-ohlcv]`, `[silver-reddit]`, `[silver-news]` lines show either `wrote 0 row(s)` worth of *new* posts/candles relative to the first run, or only the handful of genuinely new RSS/Kraken items that appeared since — confirming already-scored/cleaned rows are not reprocessed (no duplicate OpenAI calls for the same URLs).

---

### Task 8: Deploy — Secret Manager, Docker rebuild, Cloud Run Job update

**Files:** none (infrastructure only)

- [ ] **Step 1: Create the OpenAI API key secret**

The deployed Cloud Run Job currently has no `OPENAI_API_KEY` or `LLM_MODE` — it never called `get_llm()` before. Create the secret from your local `.env` value:

```powershell
$openaiKey = (Get-Content .env | Where-Object { $_ -match '^OPENAI_API_KEY=' }) -replace '^OPENAI_API_KEY=', ''
$openaiKey | gcloud secrets create openai-api-key --data-file=- --project crypto-edge-500922
```

Expected: `Created secret [openai-api-key].`

- [ ] **Step 2: Grant the service account access to the new secret**

```powershell
gcloud secrets add-iam-policy-binding openai-api-key `
  --member="serviceAccount:crypto-edge-dev@crypto-edge-500922.iam.gserviceaccount.com" `
  --role="roles/secretmanager.secretAccessor" `
  --project crypto-edge-500922
```

Expected: updated IAM policy printed, includes the new binding.

- [ ] **Step 3: Rebuild the Docker image**

```powershell
docker build -t us-central1-docker.pkg.dev/crypto-edge-500922/crypto-edge/ingestion:latest .
```

Expected: `DONE` with no errors; image includes `silver/` and `pipeline.py`.

- [ ] **Step 4: Push the image**

```powershell
docker push us-central1-docker.pkg.dev/crypto-edge-500922/crypto-edge/ingestion:latest
```

Expected: `latest: digest: sha256:... size: ...`

- [ ] **Step 5: Update the Cloud Run Job with the new env var and secret**

```powershell
gcloud run jobs update crypto-edge-ingestion `
  --region us-central1 `
  --project crypto-edge-500922 `
  --update-env-vars LLM_MODE=openai `
  --update-secrets OPENAI_API_KEY=openai-api-key:latest
```

Expected: `Job [crypto-edge-ingestion] has been successfully updated.`

- [ ] **Step 6: Execute the job and confirm success**

```powershell
gcloud run jobs execute crypto-edge-ingestion --region us-central1 --project crypto-edge-500922 --wait
```

Expected: `Execution [crypto-edge-ingestion-xxxxx] has successfully completed.`

- [ ] **Step 7: Verify silver tables in BigQuery**

```powershell
bq query --use_legacy_sql=false --project_id=crypto-edge-500922 "SELECT 'ohlcv' as t, COUNT(*) as cnt FROM silver.ohlcv UNION ALL SELECT 'reddit_posts', COUNT(*) FROM silver.reddit_posts UNION ALL SELECT 'crypto_news', COUNT(*) FROM silver.crypto_news"
```

Expected: non-zero counts for all three tables.

```powershell
bq query --use_legacy_sql=false --project_id=crypto-edge-500922 "SELECT title, sentiment, confidence, reason FROM silver.reddit_posts LIMIT 5"
```

Expected: rows with `sentiment` in `{-1, 0, 1}`, `confidence` between 0 and 1, and a non-empty `reason` string.

---

### Task 9: Full test suite, lint, and tag

**Files:** none (verification + git tag only)

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest -v`
Expected: all tests pass

- [ ] **Step 2: Run lint**

Run: `uv run ruff check .`
Expected: All checks passed!

- [ ] **Step 3: Tag the release**

```bash
git tag v0.3.0-silver
```

- [ ] **Step 4: Confirm the tag**

Run: `git log --oneline -1 && git describe --tags`
Expected: shows the latest commit with `v0.3.0-silver` tag attached
