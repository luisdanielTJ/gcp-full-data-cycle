import uuid

import pandas as pd

POSITION_SIZE_USD = 1000.0
MIN_CONFIDENCE = 0.70


def _latest_close(ohlcv_df: pd.DataFrame, asset: str) -> float | None:
    rows = ohlcv_df[ohlcv_df["asset"] == asset]
    if rows.empty:
        return None
    return float(rows.sort_values("open_time").iloc[-1]["close"])


def _normalize_journal(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    for col in ("opened_at", "closed_at"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True)
    return df


def _open_position(journal_df: pd.DataFrame, asset: str) -> pd.Series | None:
    if journal_df.empty:
        return None
    open_rows = journal_df[
        (journal_df["asset"] == asset) & (journal_df["closed_at"].isna())
    ]
    if open_rows.empty:
        return None
    return open_rows.iloc[0]


def run_paper_trade_cycle(
    warehouse, signals_df: pd.DataFrame, ohlcv_df: pd.DataFrame
) -> None:
    journal_df = _normalize_journal(warehouse.read_table("paper_trades", "journal"))

    for asset in signals_df["asset"].unique():
        latest = signals_df[signals_df["asset"] == asset].sort_values("predicted_at").iloc[-1]
        signal = latest["signal"]
        confidence = float(latest["confidence"])

        current_price = _latest_close(ohlcv_df, asset)
        if current_price is None:
            print(f"[paper] {asset}: no price data, skipping")
            continue

        open_pos = _open_position(journal_df, asset)

        if signal == "BUY" and confidence >= MIN_CONFIDENCE and open_pos is None:
            new_row = pd.DataFrame([{
                "id": str(uuid.uuid4()),
                "asset": asset,
                "position_size_usd": POSITION_SIZE_USD,
                "entry_price": current_price,
                "exit_price": float("nan"),
                "entry_confidence": confidence,
                "exit_confidence": float("nan"),
                "opened_at": pd.Timestamp.now(tz="UTC"),
                "closed_at": pd.NaT,
                "pnl_usd": float("nan"),
                "pnl_pct": float("nan"),
            }])
            warehouse.write_table(new_row, "paper_trades", "journal", mode="append")
            journal_df = (
                pd.concat([journal_df, new_row], ignore_index=True)
                if not journal_df.empty
                else new_row
            )
            print(
                f"[paper] {asset}: opened LONG at ${current_price:,.2f}"
                f" (confidence={confidence:.0%})"
            )

        elif signal == "SELL" and confidence >= MIN_CONFIDENCE and open_pos is not None:
            pnl_pct = (current_price - float(open_pos["entry_price"])) / float(open_pos["entry_price"])
            pnl_usd = pnl_pct * POSITION_SIZE_USD
            idx = journal_df[journal_df["id"] == open_pos["id"]].index[0]
            journal_df.loc[idx, "exit_price"] = current_price
            journal_df.loc[idx, "exit_confidence"] = confidence
            journal_df.loc[idx, "closed_at"] = pd.Timestamp.now(tz="UTC").floor("s")
            journal_df.loc[idx, "pnl_usd"] = pnl_usd
            journal_df.loc[idx, "pnl_pct"] = pnl_pct
            warehouse.write_table(journal_df, "paper_trades", "journal", mode="replace")
            sign = "+" if pnl_usd >= 0 else ""
            print(
                f"[paper] {asset}: closed LONG at ${current_price:,.2f}"
                f" → P&L {sign}${pnl_usd:.2f} ({sign}{pnl_pct:.1%})"
            )

        else:
            print(f"[paper] {asset}: {signal} @ {confidence:.0%} — no action")
