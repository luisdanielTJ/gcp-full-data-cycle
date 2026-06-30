# Plan 6: Prediction Serving + Dashboard Design

**Status:** Approved  
**Date:** 2026-06-30  
**Builds on:** Plan 5 (ML Training — `v0.5.0-ml-training`)

---

## 1. Goal

Add a prediction-serving batch job and a Streamlit dashboard (FastAPI + Streamlit on Render) that displays BTC/ETH trading signals, SHAP feature breakdowns, AI narration, sentiment feed, and a personal trade journal.

---

## 2. Architecture & Data Flow

```
[Cloud Scheduler — every 4h]
        │
        ▼
[Cloud Run Job: predict.py]
  - load xgboost_signal (production) via model_registry.load_model()
  - read gold.ml_features
  - compute predictions (BUY/HOLD/SELL + confidence)
  - compute per-prediction SHAP (top 5 features per asset)
  - call OpenAI → narration text (3–4 sentences)
  - write → predictions.signals
  - write → predictions.narrations
        │
        ▼
[BigQuery / DuckDB: predictions dataset]
        │
        ▼
[Render Web Service: app/]
  FastAPI (port 8000) ← internal, read/write warehouse
  Streamlit (port 8501) ← public HTTPS, reads FastAPI
```

The existing pipeline (`pipeline.py` → ingestion → silver → gold) and training job (`train.py` / `Dockerfile.training`) are **unchanged**. The prediction job is a new, independent Cloud Run Job.

---

## 3. Batch Prediction Job

### 3.1 Entrypoint: `predict.py`

Root-level entrypoint, mirrors `train.py`:

```python
from adapters import get_llm, get_model_registry, get_warehouse
from ml.predict import run_prediction_cycle

wh = get_warehouse()
registry = get_model_registry()
llm = get_llm()
run_prediction_cycle(wh, registry, llm)
```

### 3.2 Orchestrator: `ml/predict.py`

`run_prediction_cycle(warehouse, model_registry, llm)`:

1. Load production model: `model_registry.load_model("xgboost_signal", version="production")`. If no production model exists, log a warning and return (no crash).
2. Read `gold.ml_features` from warehouse.
3. For each asset (BTC, ETH):
   - Run `model.predict_proba(features)` → signal (argmax class label) + confidence (max probability).
   - Compute SHAP values using `shap.TreeExplainer(model)`. Extract top 5 features by absolute SHAP value as a JSON-serialisable list: `[{"feature": "rsi_14", "value": 0.12}, …]`.
   - Call LLM adapter with context: asset, signal, confidence %, top 5 SHAP features, last-24h sentiment summary from `silver.sentiment`. Returns 3–4 sentence narration string.
4. Write one row per asset to `predictions.signals`.
5. Write one row per asset to `predictions.narrations`.

### 3.3 Dockerfile: `Dockerfile.predict`

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --extra ml --no-dev
COPY adapters/ adapters/
COPY ml/ ml/
COPY predict.py ./
ENV PYTHONPATH=/app
CMD ["uv", "run", "python", "predict.py"]
```

### 3.4 Cloud Scheduler

New job: cron `0 */4 * * *`, triggers the predict Cloud Run Job. Runs after the pipeline job completes (offset by ~15 minutes, same pattern as the training job).

---

## 4. Data Model

All tables use the existing `warehouse.write_table()` / `warehouse.read_table()` adapter. Three new datasets: `predictions` and `trades`.

### 4.1 `predictions.signals`

| column | type | notes |
|---|---|---|
| `asset` | STRING | `BTC` or `ETH` |
| `signal` | STRING | `BUY`, `HOLD`, or `SELL` |
| `confidence` | FLOAT | 0–1, max class probability |
| `predicted_at` | TIMESTAMP | cycle timestamp |
| `model_version` | STRING | version from registry |
| `shap_top5` | STRING | JSON-encoded list of `{feature, value}` dicts |

### 4.2 `predictions.narrations`

| column | type | notes |
|---|---|---|
| `asset` | STRING | `BTC` or `ETH` |
| `narration` | STRING | 3–4 sentence OpenAI output |
| `predicted_at` | TIMESTAMP | matches `signals.predicted_at` |

### 4.3 `trades.journal`

| column | type | notes |
|---|---|---|
| `id` | STRING | UUID generated at insert |
| `asset` | STRING | `BTC` or `ETH` |
| `direction` | STRING | `LONG` or `SHORT` |
| `entry_price` | FLOAT | user-entered |
| `amount_usd` | FLOAT | user-entered |
| `opened_at` | TIMESTAMP | user-entered |
| `closed_at` | TIMESTAMP | null if open |
| `exit_price` | FLOAT | null if open |

Unrealized P&L for open trades: `(current_price - entry_price) / entry_price * amount_usd` (LONG; inverted for SHORT), where `current_price` = latest `silver.ohlcv` close for the asset. Realized P&L uses `exit_price` set when the user closes a trade via the dashboard.

---

## 5. FastAPI Backend (`app/api.py`)

Warehouse and model registry adapters instantiated **once at app startup**, not per-request.

No authentication — personal-use tool.

### Endpoints

| method | path | description |
|---|---|---|
| `GET` | `/signals/{asset}` | Latest row from `predictions.signals` |
| `GET` | `/narration/{asset}` | Latest row from `predictions.narrations` |
| `GET` | `/ohlcv/{asset}` | Last 7 days of candles from `silver.ohlcv` |
| `GET` | `/sentiment/{asset}` | Last 5 rows from `silver.sentiment` |
| `GET` | `/trades` | All rows from `trades.journal` |
| `POST` | `/trades` | Insert one trade row |
| `PATCH` | `/trades/{id}/close` | Set `closed_at` + `exit_price` on an open trade |

Streamlit is the **only** client of this API. Streamlit never accesses the warehouse directly.

---

## 6. Streamlit Dashboard (`app/dashboard.py`)

`st.set_page_config(layout="centered")` — mobile-friendly, accessible via the public Render HTTPS URL on any device.

Reads from FastAPI via `httpx`. All panels render from pre-computed data — no AI calls at dashboard load time.

### 6.1 Signal Panel

Displayed side-by-side for BTC and ETH:
- Large badge: signal (`BUY` / `HOLD` / `SELL`) + confidence %
- Countdown: time elapsed since `predicted_at` vs 4h cycle (e.g., "2h 13m ago")
- Narration text below the badge
- "Trade on Binance" link button (static URL; Phase 2 hook for order placement via `TRADING_MODE` flag — out of scope Phase 1)

### 6.2 Chart Panel

Per asset:
- 4h candlestick chart (last 7 days) using `plotly`
- Past `BUY` / `SELL` signals overlaid as green / red markers at the candle's close price (from `predictions.signals` history)
- No backtest performance overlay (Phase 1 out of scope per §15)

### 6.3 Feature Breakdown Panel

Per asset:
- Horizontal bar chart of top 5 SHAP features from `signals.shap_top5`
- Green bars = positive contribution to signal, red = negative
- Feature names on y-axis

### 6.4 Sentiment Feed Panel

Per asset:
- Last 5 posts / headlines from `silver.sentiment`
- Each row: source tag, sentiment score, headline text

### 6.5 Trade Journal Panel

- **Log a trade** form: asset (BTC/ETH), direction (LONG/SHORT), entry price, USD amount, opened_at datetime → `POST /trades`
- **Open positions** table: asset, direction, entry price, current price (latest OHLCV close), unrealized P&L — "Close" button per row → `PATCH /trades/{id}/close` (prompts for exit price)
- **Closed trades** table: entry/exit price, realized P&L per trade
- **Performance summary**: total P&L, win rate, best/worst trade, signal accuracy (% of followed BUY/SELL signals that resulted in a winning trade)

---

## 7. Deployment

### 7.1 App Container (`Dockerfile.app`)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY adapters/ adapters/
COPY app/ app/
COPY start.sh ./
RUN chmod +x start.sh
ENV PYTHONPATH=/app
CMD ["./start.sh"]
```

`start.sh` starts both processes:

```bash
#!/bin/sh
uvicorn app.api:app --host 0.0.0.0 --port 8000 &
streamlit run app/dashboard.py --server.port 8501 --server.address 0.0.0.0
```

### 7.2 Render Configuration (`render.yaml`)

```yaml
services:
  - type: web
    name: crypto-edge
    runtime: docker
    dockerfilePath: Dockerfile.app
    plan: free
    port: 8501
    envVars:
      - key: WAREHOUSE_BACKEND
        sync: false
      - key: BIGQUERY_PROJECT
        sync: false
      - key: OPENAI_API_KEY
        sync: false
      - key: GOOGLE_APPLICATION_CREDENTIALS_JSON
        sync: false
```

Render exposes the service on a public HTTPS URL (e.g., `https://crypto-edge.onrender.com`). Streamlit on port 8501 is the user-facing UI; FastAPI on port 8000 is internal to the container.

**Note:** Render's free tier suspends after 15 minutes of inactivity. Acceptable for personal use.

### 7.3 Prediction Job

- `Dockerfile.predict` → pushed to GCR → Cloud Run Job (`predict-job`)
- Cloud Scheduler: `0 */4 * * *`, triggers `predict-job`
- Same deployment pattern as `Dockerfile.training` / `train-job`

---

## 8. Pyproject Changes

Add `app` to the `packages` list in `[tool.hatch.build.targets.wheel]`:

```toml
packages = ["adapters", "ingestion", "silver", "gold", "ml", "app"]
```

Add `httpx` to core dependencies (needed by Streamlit to call FastAPI). All other required packages — `fastapi`, `uvicorn`, `streamlit`, `openai` — are already listed.

---

## 9. Out of Scope (Phase 1)

Carried forward from the original design spec §15:

- Automated or manual Binance order placement (Phase 2 — `TRADING_MODE` flag hook already noted)
- Altcoins beyond BTC and ETH
- Sub-4h timeframes
- Multi-user authentication
- Portfolio optimization across multiple positions
- Backtesting engine beyond simple signal overlay on chart
- Narration regeneration on demand (narration is pre-computed in batch job)
