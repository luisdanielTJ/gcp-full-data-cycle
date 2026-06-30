# Open-Source Migration Design (Plan 7)

## Goal

Migrate crypto-edge from GCP (BigQuery + Cloud Run + Cloud Scheduler) to a fully open-source, zero-cost stack: Supabase PostgreSQL as the warehouse, GitHub Actions as the scheduler/runner, and Streamlit Community Cloud as the dashboard host.

---

## 1. Architecture Overview

| Layer | Before | After |
|---|---|---|
| Warehouse | BigQuery | Supabase PostgreSQL |
| Scheduling | Cloud Scheduler + Cloud Run Jobs | GitHub Actions cron workflows |
| Dashboard host | Render (Docker, FastAPI + Streamlit) | Streamlit Community Cloud |
| Dashboard data source | FastAPI HTTP layer | `get_warehouse()` directly |
| Local dev | `WAREHOUSE_MODE=duckdb` | unchanged |

No GCP services remain. All pipelines run on GitHub's free tier (~1,050 min/month estimated, well within 2,000 min/month for private repos).

Schedule:
- Ingestion + Silver: every 4 hours (`0 */4 * * *`)
- Gold + Predictions: 15 min after ingestion (`15 */4 * * *`)
- ML Training: daily at 2am UTC (`0 2 * * *`)

---

## 2. SupabaseWarehouseAdapter

New adapter in `adapters/warehouse.py` selected when `WAREHOUSE_MODE=supabase`.

**Connection:** SQLAlchemy engine created from `DATABASE_URL` (Supabase transaction pooler connection string). `psycopg2-binary` is the driver.

**PostgreSQL schema mapping:** BigQuery datasets map to PostgreSQL schemas. `read_table("silver", "ohlcv")` executes `SELECT * FROM silver.ohlcv`. Schema creation (`CREATE SCHEMA IF NOT EXISTS {dataset}`) happens automatically before first write.

**Interface implementation:**

| Method | Implementation |
|---|---|
| `read_table(dataset, table)` | `pd.read_sql("SELECT * FROM {dataset}.{table}", engine)` |
| `write_table(df, dataset, table, mode)` | `df.to_sql(table, engine, schema=dataset, if_exists="replace"\|"append", index=False)` |
| `run_query(sql)` | `pd.read_sql(sql, engine)` |

`mode="replace"` maps to `if_exists="replace"` (TRUNCATE + recreate); `mode="append"` maps to `if_exists="append"` (INSERT).

**New config values in `adapters/config.py`:**
```python
DATABASE_URL = os.getenv("DATABASE_URL", "")
```

Remove: `GCP_PROJECT_ID`, `GCP_DATASET_ID`, `BQ_DATASET`.

**New dependencies:** `sqlalchemy>=2.0.0`, `psycopg2-binary>=2.9.0`

---

## 3. GitHub Actions Workflows

Three workflow files in `.github/workflows/`.

### `pipeline.yml` — Ingestion (Bronze + Silver)
```
schedule: "0 */4 * * *"
command:  uv run python pipeline.py
```

### `predict.yml` — Gold + Predictions
```
schedule: "15 */4 * * *"
command:  uv run python predict.py
```

### `train.yml` — ML Training
```
schedule: "0 2 * * *"
command:  uv run python train.py
```

All three workflows share the same structure:
1. `actions/checkout@v4`
2. `astral-sh/setup-uv@v5` (installs uv)
3. `uv sync --frozen --no-dev`
4. Run the script with env vars from GitHub Secrets

All three include `workflow_dispatch:` for manual triggering from the GitHub UI.

**GitHub Secrets required (set once in repo settings):**
```
DATABASE_URL        # Supabase pooler connection string
OPENAI_API_KEY      # LLM narration
WAREHOUSE_MODE      # = supabase
```

---

## 4. Dashboard Simplification

`app/dashboard.py` is rewritten to call `get_warehouse()` directly.

**Removed:**
- `import httpx`
- `API_BASE_URL` config dependency
- All `client.get(...)` / `client.post(...)` / `client.patch(...)` HTTP calls

**Replaced with:**
- `warehouse = get_warehouse()` at module level
- `warehouse.read_table(dataset, table)` for all reads
- `warehouse.write_table(df, "trades", "journal", mode="append")` for trade opens/closes
- `warehouse.run_query(sql)` for filtered queries

**Unchanged:**
- All five dashboard sections: Signals, Charts (candlestick + SHAP overlays), Feature Breakdown, Sentiment Feed, Trade Journal
- `ASSET_LABELS = {"XBTUSD": "BTC", "ETHUSD": "ETH"}`
- `app/pnl.py` and `app/positions.py` — called the same way, unchanged

**Deployment:**
1. Go to share.streamlit.io → New app
2. Repo: `luisdanielTJ/gcp-full-data-cycle`, branch: `master`, file: `app/dashboard.py`
3. Set secrets in the Streamlit UI: `DATABASE_URL`, `OPENAI_API_KEY`, `WAREHOUSE_MODE=supabase`
4. Dashboard accessible via public URL (phone-friendly)

---

## 5. Cleanup

**Files to delete:**
- `app/api.py` — FastAPI layer
- `Dockerfile.app` — Render Docker image
- `start.sh` — uvicorn + streamlit launcher
- `render.yaml` — Render Blueprint config

**Files to keep:**
- `pipeline.py`, `predict.py`, `train.py` — GitHub Actions entrypoints
- `app/pnl.py`, `app/positions.py` — reused by dashboard

**`adapters/warehouse.py`:**
- Delete `BigQueryWarehouse` class
- `get_warehouse()` retains only `"duckdb"` and `"supabase"` branches

**`pyproject.toml` — remove:**
```
google-cloud-bigquery[pandas]
google-cloud-storage
google-genai
google-cloud-aiplatform  (ml extra)
fastapi
uvicorn[standard]
httpx
```

**`pyproject.toml` — add:**
```
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.0
```

---

## 6. Testing

The `SupabaseWarehouseAdapter` gets a unit test in `tests/adapters/test_supabase_warehouse.py` using a mock SQLAlchemy engine — same pattern as the existing BigQuery adapter test. The mock verifies:
- `read_table` calls `pd.read_sql` with the correct schema-qualified SQL
- `write_table(mode="replace")` calls `df.to_sql` with `if_exists="replace"`
- `write_table(mode="append")` calls `df.to_sql` with `if_exists="append"`
- Schema creation is triggered before first write

The `get_warehouse()` factory in `adapters/warehouse.py` gains a `"supabase"` branch that instantiates `SupabaseWarehouseAdapter(DATABASE_URL)`. A test in `tests/adapters/test_warehouse.py` verifies the factory returns the correct type for `WAREHOUSE_MODE=supabase`.

Existing tests for `pipeline.py`, `train.py`, `predict.py`, `app/pnl.py`, and `app/positions.py` require no changes — they use `InMemoryWarehouse` or mocks that are adapter-agnostic.
