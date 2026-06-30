# ML Training Layer — Design Spec

**Date:** 2026-06-30
**Status:** Approved

---

## 1. Overview

Plan 5 in the crypto-edge build. Generates training labels from `silver.ohlcv`, joins them against `gold.ml_features` (Plan 4), and trains an XGBoost binary classifier to predict whether BTC/ETH closes >1% higher on the next 4h candle. Computes standard ML metrics plus a financial promotion gate (simulated long-only backtest vs. buy-and-hold), and persists trained models in a new warehouse-backed model registry. Runs as its own weekly Cloud Run Job, fully independent of the existing 4h ingestion/silver/gold pipeline.

**Out of scope for this plan:**
- Prediction serving (writing to a `predictions` table, batch prediction job)
- Dashboard / FastAPI app
- LLM signal narration or second-opinion suggestion (the LLM does not make or influence the trading decision — XGBoost's probability threshold is the sole signal source)
- Per-prediction SHAP at serving time (this plan computes training-time global feature importance only)
- Automated hyperparameter tuning (fixed hyperparameters for now — Vertex AI Vizier isn't available under the no-billing-upgrade constraint; tuning can be revisited once there's real data to tune against)
- Automated trade execution (Phase 2, unchanged from the original project spec)

**Data availability note:** as of this plan being written, `gold.ml_features` has 0 real rows (needs ~26-35 real candles per asset to start producing rows, i.e. several more days at the 4h cadence). This plan is built and fully tested now regardless — the first real training run simply won't have enough data to pass the minimum-row guard (Section 4) until more candles accumulate. This is expected, not a bug.

---

## 2. Components

New `ml/` package, mirroring the structure of `ingestion/`, `silver/`, and `gold/`:

| File | Responsibility |
|---|---|
| `ml/labels.py` | `compute_labels(ohlcv_df) -> pd.DataFrame` — per-asset binary label + return_pct |
| `ml/dataset.py` | `build_training_dataset(features_df, labels_df) -> pd.DataFrame` — inner join |
| `ml/train.py` | `time_based_split(...)`, `train_model(...)` — XGBoost training |
| `ml/evaluate.py` | `evaluate_model(...) -> dict` — ML metrics + financial promotion-gate metrics |
| `ml/explain.py` | `compute_feature_importance(...) -> dict` — SHAP global importance |
| `ml/run.py` | `run_training_cycle(warehouse, model_registry)` — orchestrator |

Plus a new top-level **`train.py`** entrypoint (mirrors `pipeline.py`), and a new `WarehouseModelRegistry` class added to `adapters/model_registry.py`.

---

## 3. Label Generation (`ml/labels.py`)

`compute_labels(ohlcv_df: pd.DataFrame) -> pd.DataFrame`

For each asset, sorted by `open_time`:
- `next_close` = `close` shifted back one row (the following candle's close)
- `return_pct` = `next_close / close - 1`
- `label` = `1` if `return_pct > 0.01` else `0`

The most recent candle per asset has no next candle yet, so it's dropped — it has no label. Returns columns: `asset, open_time, return_pct, label`.

---

## 4. Dataset Assembly (`ml/dataset.py`)

`build_training_dataset(features_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame`

Inner join on `(asset, open_time)`. A row survives only if it has both engineered Gold features *and* a known label. This means the most recent Gold row per asset (no label yet) and any Gold row whose corresponding OHLCV candle never produced a label are naturally excluded — no special-casing needed beyond the join itself.

**Minimum data guard:** if the joined dataset has fewer than 50 total rows (across both assets), `run_training_cycle` prints `[train] insufficient data (N rows), skipping` and returns without training — mirroring Gold's `[gold] no new rows` guard from Plan 4. 50 is a tunable constant (`MIN_TRAINING_ROWS` in `ml/run.py`), chosen to avoid a meaningless train/test split on a handful of rows while still being low enough to be exercised by Plan 5's own tests with realistic synthetic data sizes.

---

## 5. Training (`ml/train.py`)

**Split:** Sort the full dataset by `open_time` (combining both assets), then split 80/20 by position — no shuffling. This preserves temporal order and prevents look-ahead leakage between train and test.

**Model:** XGBoost binary classifier (`xgboost.XGBClassifier`), fixed hyperparameters:
```python
XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    eval_metric="logloss",
)
```

**Features:** all Gold feature columns except `asset`, `open_time`, `computed_at` (the identifier/metadata columns). **Target:** `label`.

**Function signature:** `train_model(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[XGBClassifier, pd.DataFrame, pd.Series]` — fits on `train_df`'s features/label, returns `(model, X_test, y_test)` where `X_test`/`y_test` are the feature columns and `label` column extracted from `test_df`.

---

## 6. Evaluation (`ml/evaluate.py`)

`evaluate_model(model, X_test, y_test, test_df) -> dict`

**Standard ML metrics** (sklearn, on the test split): `precision`, `recall`, `f1`, `auc_roc`.

**Financial metrics** — a long-only backtest simulation over the test split:
- Derive a signal per row from `model.predict_proba()`: `proba > 0.65 → BUY`, `proba < 0.35 → SELL`, else `HOLD`.
- `signal_accuracy` = fraction of BUY-signal rows where `label == 1`. `None` if there are zero BUY signals in the test set (and the promotion gate fails in that case — see below).
- `strategy_returns` = `return_pct` on BUY rows, `0` elsewhere (no shorting — matches manual spot trading; no position taken on SELL or HOLD).
- `buy_and_hold_returns` = `return_pct` on every row (always long — the baseline).
- `simulated_pnl` = cumulative product of `(1 + strategy_returns)`, minus 1.
- `sharpe` = `mean(strategy_returns) / std(strategy_returns) * sqrt(2190)` (2190 = trading periods per year for 4h candles: 6/day × 365); `0` if `std == 0`. The same formula computes `buy_and_hold_sharpe` from `buy_and_hold_returns`.

**Promotion gate** (all must hold; short-circuits on the first false condition, so `signal_accuracy` is never compared when `None`):
```python
has_buy_signal = (signals == "BUY").any()
gate_passed = bool(
    has_buy_signal
    and signal_accuracy is not None
    and signal_accuracy > 0.55
    and sharpe > buy_and_hold_sharpe
)
```

The model is **always** logged to the registry regardless of gate outcome (so every training run's metrics are inspectable). It is only **promoted** to production if the gate passes.

`evaluate_model()` returns a dict with exactly these keys: `precision, recall, f1, auc_roc, signal_accuracy, simulated_pnl, sharpe, buy_and_hold_sharpe, gate_passed`.

---

## 7. Explainability (`ml/explain.py`)

`compute_feature_importance(model, X_test) -> dict[str, float]`

`shap.TreeExplainer(model).shap_values(X_test)`, then mean absolute SHAP value per feature column. Returns a `{feature_name: importance}` dict, stored inside the logged model's `metrics_json` under a `feature_importance` key. This is training-time global importance only — per-prediction SHAP at serving time is out of scope (Section 1).

---

## 8. Model Registry (`adapters/model_registry.py`)

The existing `ModelRegistryAdapter` interface (from Plan 1) is unchanged:
```python
class ModelRegistryAdapter(ABC):
    def log_model(self, model, metrics: dict, params: dict, name: str) -> str: ...
    def load_model(self, name: str, version: str = "production") -> Any: ...
    def promote_model(self, name: str, version: str) -> None: ...
```

`InMemoryModelRegistry` (Plan 1 stub) is kept as-is for fast in-process unit tests. A new **`WarehouseModelRegistry`** is added, taking any `WarehouseAdapter` in its constructor — it works against BigQuery and DuckDB identically, with no new GCP service, the same way `gold/run.py` and `silver/run.py` take a `warehouse` rather than a backend-specific class.

**Storage:** two new append-only tables (same convention as Bronze/Silver/Gold — no update-in-place):

| Table | Columns |
|---|---|
| `models.registry` | `name, version, model_bytes, metrics_json, params_json, created_at` |
| `models.promotions` | `name, version, promoted_at` |

- `log_model(model, metrics, params, name)`: pickles `model` to bytes, serializes `metrics`/`params` to JSON strings, appends one row to `models.registry`. `version` = `str(existing_count_for_name + 1)`, matching `InMemoryModelRegistry`'s existing scheme. Returns the version string.
- `load_model(name, version="production")`: if `version == "production"`, finds the row with the latest `promoted_at` in `models.promotions` for that `name` (raises `ValueError(f"No production model for '{name}'")` if none exists — matches the existing test in `tests/adapters/test_model_registry.py`); otherwise uses the given version directly. Looks up the matching row in `models.registry`, unpickles `model_bytes`, returns it.
- `promote_model(name, version)`: appends one row to `models.promotions` with the current timestamp.

Pickling makes the registry generic — it round-trips arbitrary Python objects (satisfying the existing dict-based tests written in Plan 1) as well as real `XGBClassifier` instances.

---

## 9. Orchestration (`ml/run.py`, `train.py`)

```python
MIN_TRAINING_ROWS = 50

def run_training_cycle(warehouse, model_registry) -> None:
    ohlcv_df = warehouse.read_table("silver", "ohlcv")
    features_df = warehouse.read_table("gold", "ml_features")

    labels_df = compute_labels(ohlcv_df)
    dataset = build_training_dataset(features_df, labels_df)

    if len(dataset) < MIN_TRAINING_ROWS:
        print(f"[train] insufficient data ({len(dataset)} rows), skipping")
        return

    train_df, test_df = time_based_split(dataset)
    model, X_test, y_test = train_model(train_df, test_df)
    metrics = evaluate_model(model, X_test, y_test, test_df)
    metrics["feature_importance"] = compute_feature_importance(model, X_test)

    params = model.get_params()
    version = model_registry.log_model(model, metrics, params, name="xgboost_signal")
    print(f"[train] logged version {version}, metrics={metrics}")

    if metrics["gate_passed"]:
        model_registry.promote_model("xgboost_signal", version)
        print(f"[train] promoted version {version} to production")
    else:
        print(f"[train] version {version} did not pass promotion gate")
```

`train.py` (top-level, mirrors `pipeline.py`):
```python
from adapters import get_model_registry, get_warehouse
from ml.run import run_training_cycle

if __name__ == "__main__":
    wh = get_warehouse()
    registry = get_model_registry()
    run_training_cycle(wh, registry)
    print("Training cycle complete")
```

`get_model_registry()` in `adapters/__init__.py` is updated to return `WarehouseModelRegistry(get_warehouse())` instead of `InMemoryModelRegistry()`.

---

## 10. Infra & Deployment

- **`Dockerfile.training`** — separate image from the existing ingestion `Dockerfile`. `python:3.11-slim` base, `uv sync --frozen --extra ml --no-dev` (installs the `ml` optional-dependency group — xgboost, shap, scikit-learn — without dev tooling). Copies only `adapters/`, `ml/`, and `train.py` — no need for `ingestion/`, `silver/`, or `gold/` source at runtime. The exact `--extra ml --no-dev` flag combination needs to be verified during implementation (this project has already hit `uv sync` extras quirks once, in Plan 4).
- New image: `us-central1-docker.pkg.dev/crypto-edge-500922/crypto-edge/training:latest`.
- New Cloud Run Job **`crypto-edge-training`** — same region (`us-central1`) and service account (`crypto-edge-dev@crypto-edge-500922.iam.gserviceaccount.com`) as the existing `crypto-edge-ingestion` job.
- New **weekly Cloud Scheduler** job triggering `gcloud run jobs execute crypto-edge-training` (e.g. Mondays 00:00 UTC) — fully independent of the existing 4h ingestion/silver/gold scheduler, so a training failure can't break the signal pipeline.
- No new secrets needed — training only reads from BigQuery via the existing service account.

---

## 11. Testing

TDD throughout, mirroring `tests/gold/`:

- `tests/ml/test_labels.py` — correct `label`/`return_pct` on synthetic OHLCV; last candle per asset dropped (no label yet)
- `tests/ml/test_dataset.py` — inner join correctness; rows missing from either side are dropped
- `tests/ml/test_train.py` — `time_based_split`: order preserved, correct ratio, no shuffling
- `tests/ml/test_evaluate.py` — metric calculations against hand-computed expected values; gate boolean cases (pass / fail on signal accuracy / fail on Sharpe / fail on zero BUY signals)
- `tests/ml/test_explain.py` — feature importance output shape/keys match input feature columns
- `tests/ml/test_run.py` — orchestration: minimum-row guard, mocked `warehouse`/`model_registry`, correct call sequencing, promote-vs-skip-promote branching
- `tests/adapters/test_model_registry.py` — extended with a parallel `WarehouseModelRegistry` suite (same test cases as `InMemoryModelRegistry`: log returns version string, log-and-load round trip, promote-and-load-production, load-production-without-promotion raises, versions increment), backed by `DuckDBWarehouse(":memory:")`
