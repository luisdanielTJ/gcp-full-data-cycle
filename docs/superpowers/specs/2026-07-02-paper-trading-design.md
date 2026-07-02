# Paper Trading Simulation — Design Spec

## Goal

Automatically open and close simulated trades based on BUY/SELL signals from the XGBoost model, record P&L history in Supabase, and display results in the Streamlit dashboard to validate signal accuracy before using real money.

## Constraints

- Long-only (no short positions)
- Fixed $1,000 simulated position size per trade
- Only act on signals with confidence ≥ 70%
- HOLD signal while in a position → keep holding
- One open position per asset at a time (no pyramiding)
- Zero new OpenAI calls — paper trading is pure computation

---

## Signal → Action Rules

| Signal | Confidence | Open position? | Action |
|--------|-----------|----------------|--------|
| BUY | ≥ 70% | No | Open LONG at current close price |
| SELL | ≥ 70% | Yes | Close LONG, record P&L |
| HOLD | any | any | No action |
| BUY | < 70% | any | No action |
| SELL | < 70% | any | No action |
| BUY | ≥ 70% | Yes | No action (already in) |
| SELL | ≥ 70% | No | No action (nothing to close) |

---

## Data Layer

**Table:** `paper_trades.journal`

| Column | Type | Nullable | Notes |
|--------|------|----------|-------|
| `id` | text (uuid) | No | |
| `asset` | text | No | XBTUSD or ETHUSD |
| `position_size_usd` | float | No | Always 1000.0 |
| `entry_price` | float | No | Close price at BUY time |
| `exit_price` | float | Yes | Close price at SELL time |
| `entry_confidence` | float | No | Model confidence at open |
| `exit_confidence` | float | Yes | Model confidence at close |
| `opened_at` | timestamp (UTC) | No | |
| `closed_at` | timestamp (UTC) | Yes | Null while position is open |
| `pnl_usd` | float | Yes | Null while open; (exit−entry)/entry × 1000 |
| `pnl_pct` | float | Yes | Null while open; (exit−entry)/entry |

---

## Architecture

### New module: `paper_trading/`

**`paper_trading/trade.py`**

Single public function:

```
run_paper_trade_cycle(warehouse, signals_df, ohlcv_df) -> None
```

Steps:
1. Read `paper_trades.journal` (returns empty DataFrame if table doesn't exist yet)
2. Get latest close price per asset from `ohlcv_df`
3. For each asset independently:
   - Get the latest signal row from `signals_df`
   - Find any open position (row where `closed_at` is null) for this asset
   - Apply the rule table above
   - On open: append new row with entry fields, null exit fields
   - On close: update row in-memory, write entire table back with `mode="replace"`
4. Print action taken per asset

**`paper_trading/__init__.py`** — empty

### Integration point: `ml/predict.py`

After writing signals and narrations to Supabase, add:

```python
from paper_trading.trade import run_paper_trade_cycle
run_paper_trade_cycle(warehouse, pd.DataFrame(signal_rows), ohlcv_df)
```

`ohlcv_df` is already loaded earlier in `run_prediction_cycle` — no additional DB read.

No changes to `.github/workflows/predict.yml` — paper trading runs inside the existing `uv run python predict.py` step.

---

## Dashboard

New **"Paper Trading"** section added to `app/dashboard.py` (below Trade Journal):

**Summary metrics (4 columns):**
- Total P&L ($) — sum of all closed `pnl_usd`
- Win Rate (%) — closed trades where `pnl_usd > 0` / total closed
- Open Positions — count of rows where `closed_at` is null
- Total Trades — count of all rows

**Open positions table:**
- Columns: Asset, Entry Price, Current Price, Unrealized P&L ($), Unrealized P&L (%), Opened At
- Current price from `silver.ohlcv` (already loaded for charts — no extra query)
- "No open positions" info message when empty

**Closed trades table:**
- Columns: Asset, Entry Price, Exit Price, P&L ($), P&L (%), Opened At, Closed At
- Sorted newest-first
- "No closed trades yet" info message when empty

---

## Testing

- `tests/paper_trading/test_trade.py`
- Test: BUY signal + no position → new row appended
- Test: SELL signal + open position → position closed, P&L computed correctly
- Test: HOLD signal → no action
- Test: BUY signal + confidence < 0.70 → no action
- Test: BUY signal + position already open → no action
- Test: SELL signal + no open position → no action
- Test: P&L calculation (long: exit > entry → positive; exit < entry → negative)
