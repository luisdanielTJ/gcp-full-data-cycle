# Open-Source Migration Implementation Plan (Plan 7)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate crypto-edge from GCP + Render to Supabase PostgreSQL + GitHub Actions + Streamlit Community Cloud.

**Architecture:** Add `SupabaseWarehouseAdapter` (SQLAlchemy + psycopg2) to the existing `WarehouseAdapter` interface, wire it into `get_warehouse()`, create three GitHub Actions cron workflows, rewrite `app/dashboard.py` to call the warehouse directly (no FastAPI layer), then delete all GCP and Render artifacts.

**Tech Stack:** SQLAlchemy 2.x, psycopg2-binary, GitHub Actions, Streamlit Community Cloud, uv

---

## File Map

| Action | File | Purpose |
|---|---|---|
| Modify | `pyproject.toml` | Add sqlalchemy/psycopg2-binary now; remove GCP/fastapi/httpx in Task 6 |
| Modify | `adapters/warehouse.py` | Add `SupabaseWarehouseAdapter`; remove `BigQueryWarehouse` in Task 6 |
| Modify | `adapters/config.py` | Add `DATABASE_URL`; remove GCP/Gemini vars in Task 6 |
| Modify | `adapters/__init__.py` | Add supabase branch; remove BigQuery/Gemini in Task 6 |
| Modify | `adapters/llm.py` | Remove `GeminiAdapter` in Task 6 |
| Rewrite | `app/dashboard.py` | Use `get_warehouse()` directly (Task 5) |
| Create | `tests/adapters/test_supabase_warehouse.py` | Unit tests for new adapter (Task 2) |
| Create | `.github/workflows/pipeline.yml` | Ingestion cron (Task 4) |
| Create | `.github/workflows/predict.yml` | Prediction cron (Task 4) |
| Create | `.github/workflows/train.yml` | Training cron (Task 4) |
| Delete | `app/api.py` | FastAPI layer (Task 6) |
| Delete | `Dockerfile.app`, `start.sh`, `render.yaml` | Render artifacts (Task 6) |
| Delete | `tests/app/test_api.py` | API tests (Task 6) |
| Delete | `tests/adapters/test_bigquery_warehouse.py` | BigQuery tests (Task 6) |
| Modify | `tests/adapters/test_llm.py` | Remove Gemini fixture + tests (Task 6) |
| Modify | `tests/adapters/test_warehouse.py` | Add factory test for supabase (Task 3) |

---

### Task 1: Add new dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add sqlalchemy and psycopg2-binary to pyproject.toml**

In `pyproject.toml`, add two lines to the `dependencies` list (after the `plotly` line):

```toml
dependencies = [
    "google-cloud-bigquery[pandas]>=3.25.0",
    "google-cloud-storage>=2.18.0",
    "google-genai>=1.0.0",
    "pandas>=2.2.0",
    "python-dotenv>=1.0.0",
    "fastapi>=0.115.0",
    "uvicorn>=0.32.0",
    "streamlit>=1.40.0",
    "openai>=2.44.0",
    "requests>=2.34.2",
    "duckdb>=1.1.0",
    "pandas-ta-classic>=0.6.52",
    "httpx>=0.27.0",
    "plotly>=5.24.0",
    "sqlalchemy>=2.0.0",
    "psycopg2-binary>=2.9.0",
]
```

- [ ] **Step 2: Install new dependencies**

Run: `uv sync`
Expected: no errors, `uv.lock` updated to include sqlalchemy and psycopg2-binary

- [ ] **Step 3: Run baseline tests to confirm nothing is broken**

Run: `uv run pytest --tb=short -q`
Expected: all tests that were passing before still pass (may see warnings about unused deps, that's fine)

- [ ] **Step 4: Commit dependency addition**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add sqlalchemy and psycopg2-binary dependencies"
```

---

### Task 2: SupabaseWarehouseAdapter (TDD)

**Files:**
- Create: `tests/adapters/test_supabase_warehouse.py`
- Modify: `adapters/warehouse.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/adapters/test_supabase_warehouse.py`:

```python
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from adapters.warehouse import SupabaseWarehouseAdapter


@pytest.fixture
def warehouse():
    mock_engine = MagicMock()
    with patch("sqlalchemy.create_engine", return_value=mock_engine):
        wh = SupabaseWarehouseAdapter("postgresql://fake/db")
    return wh, mock_engine


def test_read_table_calls_correct_sql(warehouse):
    wh, mock_engine = warehouse
    expected_df = pd.DataFrame({"asset": ["BTC"]})
    with patch("pandas.read_sql", return_value=expected_df) as mock_read_sql:
        result = wh.read_table("silver", "ohlcv")
    mock_read_sql.assert_called_once_with("SELECT * FROM silver.ohlcv", mock_engine)
    assert result["asset"].iloc[0] == "BTC"


def test_run_query_calls_read_sql(warehouse):
    wh, mock_engine = warehouse
    expected_df = pd.DataFrame({"count": [42]})
    with patch("pandas.read_sql", return_value=expected_df) as mock_read_sql:
        result = wh.run_query("SELECT COUNT(*) FROM silver.ohlcv")
    mock_read_sql.assert_called_once_with("SELECT COUNT(*) FROM silver.ohlcv", mock_engine)
    assert result["count"].iloc[0] == 42


def test_write_table_replace(warehouse):
    wh, mock_engine = warehouse
    df = pd.DataFrame({"asset": ["BTC"], "close": [50000.0]})
    with patch.object(df, "to_sql") as mock_to_sql:
        wh.write_table(df, "silver", "ohlcv", mode="replace")
    mock_to_sql.assert_called_once_with(
        "ohlcv", mock_engine, schema="silver", if_exists="replace", index=False
    )


def test_write_table_append(warehouse):
    wh, mock_engine = warehouse
    df = pd.DataFrame({"asset": ["BTC"], "close": [50000.0]})
    with patch.object(df, "to_sql") as mock_to_sql:
        wh.write_table(df, "silver", "ohlcv", mode="append")
    mock_to_sql.assert_called_once_with(
        "ohlcv", mock_engine, schema="silver", if_exists="append", index=False
    )


def test_write_table_creates_schema(warehouse):
    wh, mock_engine = warehouse
    df = pd.DataFrame({"asset": ["BTC"]})
    with patch.object(df, "to_sql"):
        wh.write_table(df, "silver", "ohlcv", mode="replace")
    conn = mock_engine.begin.return_value.__enter__.return_value
    executed_clause = conn.execute.call_args[0][0]
    assert "CREATE SCHEMA IF NOT EXISTS silver" in str(executed_clause)


def test_write_table_invalid_mode_raises(warehouse):
    wh, _ = warehouse
    df = pd.DataFrame({"asset": ["BTC"]})
    with pytest.raises(ValueError, match="mode must be"):
        wh.write_table(df, "silver", "ohlcv", mode="invalid")


def test_read_table_returns_empty_when_table_missing(warehouse):
    from sqlalchemy.exc import ProgrammingError
    wh, _ = warehouse
    with patch("pandas.read_sql", side_effect=ProgrammingError("table not found", None, None)):
        result = wh.read_table("silver", "does_not_exist")
    assert result.empty
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/adapters/test_supabase_warehouse.py -v`
Expected: FAIL with `ImportError: cannot import name 'SupabaseWarehouseAdapter'`

- [ ] **Step 3: Add SupabaseWarehouseAdapter to adapters/warehouse.py**

Append the following class to the end of `adapters/warehouse.py` (after `BigQueryWarehouse`):

```python


class SupabaseWarehouseAdapter(WarehouseAdapter):
    def __init__(self, database_url: str):
        from sqlalchemy import create_engine, text
        self._engine = create_engine(database_url)
        self._text = text

    def run_query(self, sql: str) -> pd.DataFrame:
        return pd.read_sql(sql, self._engine)

    def write_table(
        self, df: pd.DataFrame, dataset: str, table: str, mode: str = "append"
    ) -> None:
        if mode not in ("append", "replace"):
            raise ValueError(f"mode must be 'append' or 'replace', got {mode!r}")
        with self._engine.begin() as conn:
            conn.execute(self._text(f"CREATE SCHEMA IF NOT EXISTS {dataset}"))
        if_exists = "replace" if mode == "replace" else "append"
        df.to_sql(table, self._engine, schema=dataset, if_exists=if_exists, index=False)

    def read_table(self, dataset: str, table: str) -> pd.DataFrame:
        from sqlalchemy.exc import ProgrammingError
        try:
            return pd.read_sql(f"SELECT * FROM {dataset}.{table}", self._engine)
        except ProgrammingError:
            return pd.DataFrame()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/adapters/test_supabase_warehouse.py -v`
Expected: all 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add tests/adapters/test_supabase_warehouse.py adapters/warehouse.py
git commit -m "feat: add SupabaseWarehouseAdapter"
```

---

### Task 3: Config + factory update

**Files:**
- Modify: `adapters/config.py`
- Modify: `adapters/__init__.py`
- Modify: `tests/adapters/test_warehouse.py`

- [ ] **Step 1: Add DATABASE_URL to adapters/config.py**

The current file ends with `API_BASE_URL`. Add `DATABASE_URL` after `DUCKDB_PATH`:

Replace the full content of `adapters/config.py` with:

```python
import os

from dotenv import load_dotenv

load_dotenv()

GCP_PROJECT_ID: str = os.getenv("GCP_PROJECT_ID", "")
GCP_REGION: str = os.getenv("GCP_REGION", "us-central1")
WAREHOUSE_MODE: str = os.getenv("WAREHOUSE_MODE", "duckdb")
LLM_MODE: str = os.getenv("LLM_MODE", "gemini")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
REDDIT_USER_AGENT: str = os.getenv("REDDIT_USER_AGENT", "crypto-edge-ingestion/0.1")
DUCKDB_PATH: str = os.getenv("DUCKDB_PATH", ":memory:")
DATABASE_URL: str = os.getenv("DATABASE_URL", "")
API_BASE_URL: str = os.getenv("API_BASE_URL", "http://localhost:8000")
```

- [ ] **Step 2: Update adapters/__init__.py to add supabase branch**

Replace the full content of `adapters/__init__.py` with:

```python
from adapters.config import (
    DATABASE_URL,
    DUCKDB_PATH,
    GCP_PROJECT_ID,
    GEMINI_API_KEY,
    LLM_MODE,
    OPENAI_API_KEY,
    WAREHOUSE_MODE,
)
from adapters.llm import GeminiAdapter, LLMAdapter, OpenAIAdapter
from adapters.model_registry import ModelRegistryAdapter, WarehouseModelRegistry
from adapters.warehouse import (
    BigQueryWarehouse,
    DuckDBWarehouse,
    SupabaseWarehouseAdapter,
    WarehouseAdapter,
)


def get_warehouse() -> WarehouseAdapter:
    if WAREHOUSE_MODE == "supabase":
        return SupabaseWarehouseAdapter(database_url=DATABASE_URL)
    if WAREHOUSE_MODE == "bigquery":
        return BigQueryWarehouse(project_id=GCP_PROJECT_ID)
    return DuckDBWarehouse(db_path=DUCKDB_PATH)


def get_llm() -> LLMAdapter:
    if LLM_MODE == "gemini":
        return GeminiAdapter(api_key=GEMINI_API_KEY)
    if LLM_MODE == "openai":
        return OpenAIAdapter(api_key=OPENAI_API_KEY)
    raise ValueError(f"Unknown LLM_MODE: {LLM_MODE!r}. Set LLM_MODE=gemini or openai in .env")


def get_model_registry() -> ModelRegistryAdapter:
    return WarehouseModelRegistry(get_warehouse())
```

- [ ] **Step 3: Add factory test to tests/adapters/test_warehouse.py**

Add the following test to the end of `tests/adapters/test_warehouse.py`:

```python
def test_get_warehouse_returns_supabase_adapter():
    from unittest.mock import MagicMock, patch

    from adapters import get_warehouse
    from adapters.warehouse import SupabaseWarehouseAdapter

    with patch("adapters.WAREHOUSE_MODE", "supabase"):
        with patch("adapters.DATABASE_URL", "postgresql://fake/db"):
            with patch("sqlalchemy.create_engine", return_value=MagicMock()):
                result = get_warehouse()

    assert isinstance(result, SupabaseWarehouseAdapter)
```

- [ ] **Step 4: Run full warehouse test suite**

Run: `uv run pytest tests/adapters/test_warehouse.py tests/adapters/test_supabase_warehouse.py -v`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add adapters/config.py adapters/__init__.py tests/adapters/test_warehouse.py
git commit -m "feat: wire SupabaseWarehouseAdapter into get_warehouse() factory"
```

---

### Task 4: GitHub Actions workflows

**Files:**
- Create: `.github/workflows/pipeline.yml`
- Create: `.github/workflows/predict.yml`
- Create: `.github/workflows/train.yml`

- [ ] **Step 1: Create the .github/workflows directory if it doesn't exist**

Run: `mkdir -p .github/workflows`

- [ ] **Step 2: Create pipeline.yml**

Create `.github/workflows/pipeline.yml`:

```yaml
name: Pipeline (Ingestion + Silver + Gold)

on:
  schedule:
    - cron: "0 */4 * * *"
  workflow_dispatch:

jobs:
  run:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync --frozen --no-dev
      - run: uv run python pipeline.py
        env:
          WAREHOUSE_MODE: ${{ secrets.WAREHOUSE_MODE }}
          DATABASE_URL: ${{ secrets.DATABASE_URL }}
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          LLM_MODE: openai
```

- [ ] **Step 3: Create predict.yml**

Create `.github/workflows/predict.yml`:

```yaml
name: Predict (Gold + Signals)

on:
  schedule:
    - cron: "15 */4 * * *"
  workflow_dispatch:

jobs:
  run:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync --frozen --no-dev --extra ml
      - run: uv run python predict.py
        env:
          WAREHOUSE_MODE: ${{ secrets.WAREHOUSE_MODE }}
          DATABASE_URL: ${{ secrets.DATABASE_URL }}
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          LLM_MODE: openai
```

- [ ] **Step 4: Create train.yml**

Create `.github/workflows/train.yml`:

```yaml
name: Train (ML Model)

on:
  schedule:
    - cron: "0 2 * * *"
  workflow_dispatch:

jobs:
  run:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync --frozen --no-dev --extra ml
      - run: uv run python train.py
        env:
          WAREHOUSE_MODE: ${{ secrets.WAREHOUSE_MODE }}
          DATABASE_URL: ${{ secrets.DATABASE_URL }}
```

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/pipeline.yml .github/workflows/predict.yml .github/workflows/train.yml
git commit -m "feat: add GitHub Actions cron workflows for pipeline, predict, and train"
```

---

### Task 5: Rewrite dashboard

**Files:**
- Rewrite: `app/dashboard.py`

- [ ] **Step 1: Rewrite app/dashboard.py to use warehouse directly**

Replace the entire content of `app/dashboard.py` with:

```python
import json
import uuid
from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from adapters import get_warehouse
from app.pnl import match_signal_for_trade, summarize_performance
from app.positions import enrich_open_positions

st.set_page_config(page_title="crypto-edge", layout="centered")

ASSETS = ["XBTUSD", "ETHUSD"]
ASSET_LABELS = {"XBTUSD": "BTC", "ETHUSD": "ETH"}

warehouse = get_warehouse()


def _ago(ts) -> str:
    delta = datetime.now(timezone.utc) - pd.Timestamp(ts).to_pydatetime()
    hours, remainder = divmod(int(delta.total_seconds()), 3600)
    minutes = remainder // 60
    return f"{hours}h {minutes}m ago"


def _latest_for_asset(df: pd.DataFrame, asset: str, sort_col: str) -> pd.Series | None:
    if df.empty or "asset" not in df.columns:
        return None
    rows = df[df["asset"] == asset]
    if rows.empty:
        return None
    return rows.sort_values(sort_col).iloc[-1]


# ── Signals ──────────────────────────────────────────────────────────────────

st.title("crypto-edge")
st.header("Signals")

signals_df = warehouse.read_table("predictions", "signals")
narrations_df = warehouse.read_table("predictions", "narrations")

cols = st.columns(2)
for col, asset in zip(cols, ASSETS):
    with col:
        st.subheader(ASSET_LABELS[asset])
        signal_row = _latest_for_asset(signals_df, asset, "predicted_at")
        if signal_row is None:
            st.info("No signal yet")
            continue
        badge = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}[signal_row["signal"]]
        st.metric(f"{badge} {signal_row['signal']}", f"{signal_row['confidence']:.0%} confidence")
        st.caption(_ago(signal_row["predicted_at"]))

        narration_row = _latest_for_asset(narrations_df, asset, "predicted_at")
        if narration_row is not None:
            st.write(narration_row["narration"])

        st.link_button(
            "Trade on Binance",
            f"https://www.binance.com/en/trade/{ASSET_LABELS[asset]}_USDT",
        )

# ── Charts ────────────────────────────────────────────────────────────────────

st.header("Charts")
ohlcv_df = warehouse.read_table("silver", "ohlcv")

for asset in ASSETS:
    st.subheader(ASSET_LABELS[asset])
    if ohlcv_df.empty or "asset" not in ohlcv_df.columns:
        st.info("No chart data yet")
        continue
    df = ohlcv_df[ohlcv_df["asset"] == asset].sort_values("open_time").copy()
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=7)
    df = df[df["open_time"] >= cutoff]
    if df.empty:
        st.info("No chart data yet")
        continue
    df["open_time"] = pd.to_datetime(df["open_time"])
    fig = go.Figure(data=[go.Candlestick(
        x=df["open_time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
    )])

    if not signals_df.empty and "asset" in signals_df.columns:
        hist_df = signals_df[signals_df["asset"] == asset].copy()
        hist_df["predicted_at"] = pd.to_datetime(hist_df["predicted_at"])
        hist_df = hist_df[hist_df["predicted_at"] >= df["open_time"].min()]
        if not hist_df.empty:
            merged = pd.merge_asof(
                hist_df.sort_values("predicted_at"), df.sort_values("open_time"),
                left_on="predicted_at", right_on="open_time", direction="nearest",
            )
            for signal_type, color in (("BUY", "green"), ("SELL", "red")):
                points = merged[merged["signal"] == signal_type]
                if not points.empty:
                    symbol = "triangle-up" if signal_type == "BUY" else "triangle-down"
                    fig.add_trace(go.Scatter(
                        x=points["open_time"], y=points["close"], mode="markers",
                        marker=dict(color=color, size=10, symbol=symbol),
                        name=signal_type,
                    ))

    st.plotly_chart(fig, use_container_width=True)

# ── Feature Breakdown ─────────────────────────────────────────────────────────

st.header("Feature Breakdown")
for asset in ASSETS:
    signal_row = _latest_for_asset(signals_df, asset, "predicted_at")
    if signal_row is None:
        continue
    top5 = json.loads(signal_row["shap_top5"])
    if not top5:
        continue
    st.subheader(ASSET_LABELS[asset])
    bar_df = pd.DataFrame(top5)
    colors = ["green" if v > 0 else "red" for v in bar_df["value"]]
    fig = go.Figure(go.Bar(
        x=bar_df["value"], y=bar_df["feature"], orientation="h", marker_color=colors,
    ))
    st.plotly_chart(fig, use_container_width=True)

# ── Sentiment Feed ────────────────────────────────────────────────────────────

st.header("Sentiment Feed")
reddit_df = warehouse.read_table("silver", "reddit_posts")
news_df = warehouse.read_table("silver", "crypto_news")
feed_rows = []
if not reddit_df.empty:
    r = reddit_df.rename(columns={"subreddit": "source"})
    feed_rows.append(r[["source", "title", "url", "published_at", "sentiment", "confidence"]])
if not news_df.empty:
    feed_rows.append(
        news_df[["source", "title", "url", "published_at", "sentiment", "confidence"]]
    )
if feed_rows:
    combined = pd.concat(feed_rows, ignore_index=True).sort_values(
        "published_at", ascending=False
    )
    st.dataframe(combined.head(5)[["source", "sentiment", "title"]])
else:
    st.info("No sentiment data yet")

# ── Trade Journal ─────────────────────────────────────────────────────────────

st.header("Trade Journal")

with st.form("log_trade"):
    form_asset = st.selectbox("Asset", ASSETS, format_func=lambda a: ASSET_LABELS[a])
    direction = st.selectbox("Direction", ["LONG", "SHORT"])
    entry_price = st.number_input("Entry price", min_value=0.0)
    amount_usd = st.number_input("Amount (USD)", min_value=0.0)
    default_opened_at = datetime.now(timezone.utc).isoformat()
    opened_at = st.text_input("Opened at (ISO timestamp)", value=default_opened_at)
    submitted = st.form_submit_button("Log trade")
    if submitted:
        row = pd.DataFrame([{
            "id": str(uuid.uuid4()),
            "asset": form_asset,
            "direction": direction,
            "entry_price": entry_price,
            "amount_usd": amount_usd,
            "opened_at": pd.Timestamp(opened_at),
            "closed_at": pd.NaT,
            "exit_price": float("nan"),
        }])
        warehouse.write_table(row, "trades", "journal", mode="append")
        st.rerun()

st.subheader("Open positions")
journal_df = warehouse.read_table("trades", "journal")
if not journal_df.empty and "closed_at" in journal_df.columns:
    open_trades = journal_df[journal_df["closed_at"].isna()]
    if not open_trades.empty:
        current_ohlcv = warehouse.read_table("silver", "ohlcv")
        enriched = enrich_open_positions(open_trades.reset_index(drop=True), current_ohlcv)
        st.dataframe(
            enriched[["asset", "direction", "entry_price", "current_price", "unrealized_pnl"]]
        )
        for _, pos in enriched.iterrows():
            with st.expander(f"Close {ASSET_LABELS.get(pos['asset'], pos['asset'])} position"):
                exit_price = st.number_input(
                    "Exit price", min_value=0.0, key=f"exit_{pos['id']}"
                )
                if st.button("Close", key=f"close_{pos['id']}"):
                    idx = journal_df[journal_df["id"] == pos["id"]].index[0]
                    journal_df.loc[idx, "exit_price"] = exit_price
                    journal_df.loc[idx, "closed_at"] = pd.Timestamp.now().floor("s")
                    warehouse.write_table(journal_df, "trades", "journal", mode="replace")
                    st.rerun()
    else:
        st.info("No open positions")
else:
    st.info("No open positions")

st.subheader("Closed trades")
if not journal_df.empty and "closed_at" in journal_df.columns:
    closed = journal_df[journal_df["closed_at"].notna()]
    if not closed.empty:
        st.dataframe(closed[["asset", "direction", "entry_price", "exit_price"]])
    else:
        st.info("No closed trades yet")
else:
    st.info("No closed trades yet")

st.subheader("Performance summary")
if not journal_df.empty and "closed_at" in journal_df.columns:
    closed = journal_df[journal_df["closed_at"].notna()].copy()
    if not closed.empty:
        if not signals_df.empty:
            closed["matched_signal"] = closed.apply(
                lambda r: match_signal_for_trade(signals_df, r["asset"], r["opened_at"]),
                axis=1,
            )
        performance = summarize_performance(closed)
        perf_cols = st.columns(5)
        perf_cols[0].metric("Total P&L", f"${performance['total_pnl']:.2f}")
        win_rate = performance["win_rate"]
        perf_cols[1].metric("Win rate", f"{win_rate:.0%}" if win_rate is not None else "N/A")
        best = performance["best_trade_pnl"]
        worst = performance["worst_trade_pnl"]
        perf_cols[2].metric("Best trade", f"${best:.2f}" if best is not None else "N/A")
        perf_cols[3].metric("Worst trade", f"${worst:.2f}" if worst is not None else "N/A")
        sig_acc = performance["signal_accuracy"]
        perf_cols[4].metric(
            "Signal accuracy", f"{sig_acc:.0%}" if sig_acc is not None else "N/A"
        )
```

- [ ] **Step 2: Run tests to make sure existing app tests still pass**

Run: `uv run pytest tests/app/ -v`
Expected: `test_pnl.py` and `test_positions.py` pass; `test_api.py` still passes (api.py not deleted yet)

- [ ] **Step 3: Commit**

```bash
git add app/dashboard.py
git commit -m "feat: rewrite dashboard to use warehouse directly (remove FastAPI HTTP layer)"
```

---

### Task 6: Delete GCP + FastAPI artifacts

**Files:**
- Delete: `tests/app/test_api.py`
- Delete: `tests/adapters/test_bigquery_warehouse.py`
- Modify: `tests/adapters/test_llm.py`
- Delete: `app/api.py`
- Delete: `Dockerfile.app`, `start.sh`, `render.yaml`
- Modify: `adapters/warehouse.py`
- Modify: `adapters/llm.py`
- Modify: `adapters/__init__.py`
- Modify: `adapters/config.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Delete test files that test removed code**

```bash
git rm tests/app/test_api.py
git rm tests/adapters/test_bigquery_warehouse.py
```

- [ ] **Step 2: Remove Gemini tests from tests/adapters/test_llm.py**

Replace the full content of `tests/adapters/test_llm.py` with (keep only OpenAI tests):

```python
from unittest.mock import MagicMock, patch

import pytest

from adapters.llm import OpenAIAdapter


@pytest.fixture
def mock_openai():
    with patch("openai.OpenAI") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        adapter = OpenAIAdapter(api_key="fake-key")
        yield adapter


def _openai_response(text: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=text))]
    return response


def test_openai_score_sentiment_bullish(mock_openai):
    mock_openai.client.chat.completions.create.return_value = _openai_response(
        '{"sentiment": 1, "confidence": 0.85, "reason": "Positive ETF approval news"}'
    )
    result = mock_openai.score_sentiment("Bitcoin ETF approved by the SEC today")
    assert result["sentiment"] == 1
    assert result["confidence"] == pytest.approx(0.85)
    assert isinstance(result["reason"], str)


def test_openai_score_sentiment_bearish(mock_openai):
    mock_openai.client.chat.completions.create.return_value = _openai_response(
        '{"sentiment": -1, "confidence": 0.72, "reason": "Regulatory crackdown fears"}'
    )
    result = mock_openai.score_sentiment("SEC sues major crypto exchange for fraud")
    assert result["sentiment"] == -1
    assert 0.0 <= result["confidence"] <= 1.0


def test_openai_score_sentiment_validates_range(mock_openai):
    mock_openai.client.chat.completions.create.return_value = _openai_response(
        '{"sentiment": 0, "confidence": 0.5, "reason": "No strong signal"}'
    )
    result = mock_openai.score_sentiment("Crypto markets quiet today")
    assert result["sentiment"] in (-1, 0, 1)
    assert 0.0 <= result["confidence"] <= 1.0


def test_openai_narrate_signal_returns_non_empty_string(mock_openai):
    mock_openai.client.chat.completions.create.return_value = _openai_response(
        "BTC shows a BUY signal at 71% confidence driven by MACD crossover and positive sentiment."
    )
    context = {
        "signal": "BUY",
        "asset": "BTC",
        "confidence": 0.71,
        "top_features": ["RSI(14)=42: not overbought", "MACD: bullish crossover 8h ago"],
        "sentiment_summary": "Positive 24h score: 0.6, ETF inflow mentions up",
        "recent_prices": [48000.0, 49500.0, 50200.0],
    }
    result = mock_openai.narrate_signal(context)
    assert isinstance(result, str)
    assert len(result) > 20
```

- [ ] **Step 3: Delete source files for removed services**

```bash
git rm app/api.py
git rm Dockerfile.app
git rm start.sh
git rm render.yaml
```

- [ ] **Step 4: Remove BigQueryWarehouse from adapters/warehouse.py**

Replace the full content of `adapters/warehouse.py` with (keep only `WarehouseAdapter`, `DuckDBWarehouse`, and `SupabaseWarehouseAdapter`):

```python
from abc import ABC, abstractmethod

import pandas as pd


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


class DuckDBWarehouse(WarehouseAdapter):
    def __init__(self, db_path: str = ":memory:"):
        import duckdb
        self.conn = duckdb.connect(db_path)

    def run_query(self, sql: str) -> pd.DataFrame:
        return self.conn.execute(sql).df()

    def write_table(
        self, df: pd.DataFrame, dataset: str, table: str, mode: str = "append"
    ) -> None:
        if mode not in ("append", "replace"):
            raise ValueError(f"mode must be 'append' or 'replace', got {mode!r}")
        table_name = f"{dataset}__{table}"
        if mode == "replace":
            self.conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM df")
        else:
            existing = self.conn.execute("SHOW TABLES").fetchdf()
            if table_name in existing["name"].values:
                self.conn.execute(f"INSERT INTO {table_name} SELECT * FROM df")
            else:
                self.conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")

    def read_table(self, dataset: str, table: str) -> pd.DataFrame:
        table_name = f"{dataset}__{table}"
        existing = self.conn.execute("SHOW TABLES").fetchdf()
        if table_name not in existing["name"].values:
            return pd.DataFrame()
        return self.conn.execute(f"SELECT * FROM {table_name}").df()


class SupabaseWarehouseAdapter(WarehouseAdapter):
    def __init__(self, database_url: str):
        from sqlalchemy import create_engine, text
        self._engine = create_engine(database_url)
        self._text = text

    def run_query(self, sql: str) -> pd.DataFrame:
        return pd.read_sql(sql, self._engine)

    def write_table(
        self, df: pd.DataFrame, dataset: str, table: str, mode: str = "append"
    ) -> None:
        if mode not in ("append", "replace"):
            raise ValueError(f"mode must be 'append' or 'replace', got {mode!r}")
        with self._engine.begin() as conn:
            conn.execute(self._text(f"CREATE SCHEMA IF NOT EXISTS {dataset}"))
        if_exists = "replace" if mode == "replace" else "append"
        df.to_sql(table, self._engine, schema=dataset, if_exists=if_exists, index=False)

    def read_table(self, dataset: str, table: str) -> pd.DataFrame:
        from sqlalchemy.exc import ProgrammingError
        try:
            return pd.read_sql(f"SELECT * FROM {dataset}.{table}", self._engine)
        except ProgrammingError:
            return pd.DataFrame()
```

- [ ] **Step 5: Remove GeminiAdapter from adapters/llm.py**

Replace the full content of `adapters/llm.py` with (keep only `LLMAdapter` and `OpenAIAdapter`):

```python
import json
from abc import ABC, abstractmethod


class LLMAdapter(ABC):
    @abstractmethod
    def score_sentiment(self, text: str) -> dict:
        """Returns {"sentiment": -1|0|1, "confidence": float, "reason": str}."""
        ...

    @abstractmethod
    def narrate_signal(self, context: dict) -> str:
        """Returns 3-4 sentence plain English explanation of a trading signal."""
        ...


class OpenAIAdapter(LLMAdapter):
    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def score_sentiment(self, text: str) -> dict:
        prompt = (
            "Analyze the sentiment of this crypto-related text toward Bitcoin/Ethereum price "
            "direction.\n\n"
            f"Text: {text}\n\n"
            "Respond with ONLY valid JSON in this exact format:\n"
            '{"sentiment": -1, "confidence": 0.8, "reason": "one sentence explanation"}\n\n'
            "sentiment must be exactly -1 (bearish), 0 (neutral), or 1 (bullish).\n"
            "confidence must be a float between 0.0 and 1.0."
        )
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return json.loads(response.choices[0].message.content.strip())

    def narrate_signal(self, context: dict) -> str:
        features = "\n".join(f"- {f}" for f in context["top_features"])
        prices = ", ".join(str(p) for p in context["recent_prices"])
        prompt = (
            "You are a crypto trading assistant. Explain this trading signal in 3-4 plain "
            "English sentences.\n\n"
            f"Signal: {context['signal']} for {context['asset']}\n"
            f"Confidence: {context['confidence']:.0%}\n"
            f"Top factors:\n{features}\n"
            f"Recent sentiment: {context['sentiment_summary']}\n"
            f"Last 3 prices (4h closes, USD): {prices}\n\n"
            "Be specific and factual. Do not give financial advice."
        )
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()
```

- [ ] **Step 6: Update adapters/__init__.py to remove all GCP/Gemini references**

Replace the full content of `adapters/__init__.py` with:

```python
from adapters.config import (
    DATABASE_URL,
    DUCKDB_PATH,
    LLM_MODE,
    OPENAI_API_KEY,
    WAREHOUSE_MODE,
)
from adapters.llm import LLMAdapter, OpenAIAdapter
from adapters.model_registry import ModelRegistryAdapter, WarehouseModelRegistry
from adapters.warehouse import DuckDBWarehouse, SupabaseWarehouseAdapter, WarehouseAdapter


def get_warehouse() -> WarehouseAdapter:
    if WAREHOUSE_MODE == "supabase":
        return SupabaseWarehouseAdapter(database_url=DATABASE_URL)
    return DuckDBWarehouse(db_path=DUCKDB_PATH)


def get_llm() -> LLMAdapter:
    if LLM_MODE == "openai":
        return OpenAIAdapter(api_key=OPENAI_API_KEY)
    raise ValueError(f"Unknown LLM_MODE: {LLM_MODE!r}. Set LLM_MODE=openai in .env")


def get_model_registry() -> ModelRegistryAdapter:
    return WarehouseModelRegistry(get_warehouse())
```

- [ ] **Step 7: Update adapters/config.py to remove GCP/Gemini/API vars**

Replace the full content of `adapters/config.py` with:

```python
import os

from dotenv import load_dotenv

load_dotenv()

WAREHOUSE_MODE: str = os.getenv("WAREHOUSE_MODE", "duckdb")
LLM_MODE: str = os.getenv("LLM_MODE", "openai")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
REDDIT_USER_AGENT: str = os.getenv("REDDIT_USER_AGENT", "crypto-edge-ingestion/0.1")
DUCKDB_PATH: str = os.getenv("DUCKDB_PATH", ":memory:")
DATABASE_URL: str = os.getenv("DATABASE_URL", "")
```

- [ ] **Step 8: Remove GCP/FastAPI/httpx deps from pyproject.toml**

Replace the full content of `pyproject.toml` with:

```toml
[project]
name = "crypto-edge"
version = "0.1.0"
description = "Personal crypto trading signal platform for BTC/ETH"
requires-python = ">=3.11"
dependencies = [
    "pandas>=2.2.0",
    "python-dotenv>=1.0.0",
    "streamlit>=1.40.0",
    "openai>=2.44.0",
    "requests>=2.34.2",
    "duckdb>=1.1.0",
    "pandas-ta-classic>=0.6.52",
    "plotly>=5.24.0",
    "sqlalchemy>=2.0.0",
    "psycopg2-binary>=2.9.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3.0",
    "pytest-cov>=6.0.0",
    "ruff>=0.8.0",
    "mypy>=1.13.0",
]
ml = [
    "xgboost>=2.1.0,<3.0",
    "shap>=0.46.0,<0.50.0",
    "scikit-learn>=1.5.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["adapters", "ingestion", "silver", "gold", "ml", "app"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I"]
```

- [ ] **Step 9: Update the lock file**

Run: `uv sync --extra dev --extra ml`
Expected: uv removes google-cloud-bigquery, google-cloud-storage, google-genai, fastapi, uvicorn, httpx from the environment and updates `uv.lock`

- [ ] **Step 10: Run the full test suite**

Run: `uv run pytest --tb=short -q`
Expected: all tests pass, no import errors

- [ ] **Step 11: Commit everything**

```bash
git add adapters/warehouse.py adapters/llm.py adapters/__init__.py adapters/config.py
git add tests/adapters/test_llm.py pyproject.toml uv.lock
git commit -m "chore: remove GCP/FastAPI/Render artifacts, trim deps to open-source stack"
```

---

## After all tasks: Deploy to Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) → New app
2. Repo: `luisdanielTJ/gcp-full-data-cycle`, branch: `master`, main file: `app/dashboard.py`
3. Set these secrets in the Streamlit UI (Advanced settings → Secrets):
   ```toml
   WAREHOUSE_MODE = "supabase"
   DATABASE_URL = "postgresql://postgres.[ref]:[password]@pooler.supabase.com:6543/postgres"
   OPENAI_API_KEY = "sk-..."
   LLM_MODE = "openai"
   ```
4. Deploy — the dashboard will be accessible at a public URL (phone-friendly)

## After all tasks: Configure GitHub Actions secrets

In your GitHub repo → Settings → Secrets and variables → Actions → New repository secret:

| Secret name | Value |
|---|---|
| `WAREHOUSE_MODE` | `supabase` |
| `DATABASE_URL` | Supabase transaction pooler connection string |
| `OPENAI_API_KEY` | Your OpenAI key |
