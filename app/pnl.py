import pandas as pd

_DIRECTION_TO_SIGNAL = {"LONG": "BUY", "SHORT": "SELL"}


def unrealized_pnl(
    direction: str, entry_price: float, current_price: float, amount_usd: float
) -> float:
    pct_change = (current_price - entry_price) / entry_price
    if direction == "SHORT":
        pct_change = -pct_change
    return pct_change * amount_usd


def realized_pnl(direction: str, entry_price: float, exit_price: float, amount_usd: float) -> float:
    return unrealized_pnl(direction, entry_price, exit_price, amount_usd)


def match_signal_for_trade(
    signals_df: pd.DataFrame, asset: str, opened_at: pd.Timestamp
) -> str | None:
    candidates = signals_df[
        (signals_df["asset"] == asset) & (signals_df["predicted_at"] <= opened_at)
    ]
    if candidates.empty:
        return None
    return candidates.sort_values("predicted_at").iloc[-1]["signal"]


def summarize_performance(closed_trades: pd.DataFrame) -> dict:
    if closed_trades.empty:
        return {
            "total_pnl": 0.0,
            "win_rate": None,
            "best_trade_pnl": None,
            "worst_trade_pnl": None,
            "signal_accuracy": None,
        }

    pnls = closed_trades.apply(
        lambda r: realized_pnl(r["direction"], r["entry_price"], r["exit_price"], r["amount_usd"]),
        axis=1,
    )

    matched = closed_trades[
        closed_trades.apply(
            lambda r: r["matched_signal"] == _DIRECTION_TO_SIGNAL.get(r["direction"]), axis=1
        )
    ]
    matched_pnls = pnls[matched.index]
    signal_accuracy = float((matched_pnls > 0).mean()) if not matched.empty else None

    return {
        "total_pnl": float(pnls.sum()),
        "win_rate": float((pnls > 0).mean()),
        "best_trade_pnl": float(pnls.max()),
        "worst_trade_pnl": float(pnls.min()),
        "signal_accuracy": signal_accuracy,
    }
