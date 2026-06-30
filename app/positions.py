import pandas as pd

from app.pnl import unrealized_pnl


def latest_close_prices(ohlcv_df: pd.DataFrame) -> dict[str, float]:
    if ohlcv_df.empty:
        return {}
    latest = (
        ohlcv_df.sort_values("open_time")
        .groupby("asset", as_index=False)
        .tail(1)
    )
    return dict(zip(latest["asset"], latest["close"]))


def enrich_open_positions(open_trades: pd.DataFrame, ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    if open_trades.empty:
        return pd.DataFrame()

    prices = latest_close_prices(ohlcv_df)
    rows = []
    for _, row in open_trades.iterrows():
        current_price = prices.get(row["asset"])
        if current_price is None:
            continue
        upnl = unrealized_pnl(
            row["direction"], row["entry_price"], current_price, row["amount_usd"]
        )
        rows.append({**row.to_dict(), "current_price": current_price, "unrealized_pnl": upnl})

    return pd.DataFrame(rows)
