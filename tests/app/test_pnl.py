import pandas as pd
import pytest

from app.pnl import match_signal_for_trade, summarize_performance, unrealized_pnl


def test_unrealized_pnl_long_position_gain():
    pnl = unrealized_pnl(direction="LONG", entry_price=100.0, current_price=110.0, amount_usd=1000.0)
    assert pnl == 100.0


def test_unrealized_pnl_long_position_loss():
    pnl = unrealized_pnl(direction="LONG", entry_price=100.0, current_price=90.0, amount_usd=1000.0)
    assert pnl == -100.0


def test_unrealized_pnl_short_position_gain():
    pnl = unrealized_pnl(direction="SHORT", entry_price=100.0, current_price=90.0, amount_usd=1000.0)
    assert pnl == 100.0


def test_unrealized_pnl_short_position_loss():
    pnl = unrealized_pnl(direction="SHORT", entry_price=100.0, current_price=110.0, amount_usd=1000.0)
    assert pnl == -100.0


def test_match_signal_for_trade_finds_most_recent_signal_at_or_before_open():
    signals_df = pd.DataFrame([
        {"asset": "XBTUSD", "signal": "BUY", "predicted_at": pd.Timestamp("2026-06-01T00:00Z")},
        {"asset": "XBTUSD", "signal": "SELL", "predicted_at": pd.Timestamp("2026-06-01T04:00Z")},
        {"asset": "XBTUSD", "signal": "BUY", "predicted_at": pd.Timestamp("2026-06-01T08:00Z")},
    ])
    opened_at = pd.Timestamp("2026-06-01T05:00Z")

    matched = match_signal_for_trade(signals_df, asset="XBTUSD", opened_at=opened_at)

    assert matched == "SELL"


def test_match_signal_for_trade_returns_none_when_no_prior_signal():
    signals_df = pd.DataFrame([
        {"asset": "XBTUSD", "signal": "BUY", "predicted_at": pd.Timestamp("2026-06-01T08:00Z")},
    ])
    opened_at = pd.Timestamp("2026-06-01T05:00Z")

    matched = match_signal_for_trade(signals_df, asset="XBTUSD", opened_at=opened_at)

    assert matched is None


def test_summarize_performance_computes_totals_win_rate_and_signal_accuracy():
    closed_trades = pd.DataFrame([
        {"asset": "XBTUSD", "direction": "LONG", "entry_price": 100.0, "exit_price": 110.0,
         "amount_usd": 1000.0, "opened_at": pd.Timestamp("2026-06-01T05:00Z"),
         "matched_signal": "BUY"},
        {"asset": "XBTUSD", "direction": "LONG", "entry_price": 100.0, "exit_price": 90.0,
         "amount_usd": 1000.0, "opened_at": pd.Timestamp("2026-06-02T05:00Z"),
         "matched_signal": None},
        {"asset": "ETHUSD", "direction": "SHORT", "entry_price": 100.0, "exit_price": 90.0,
         "amount_usd": 1000.0, "opened_at": pd.Timestamp("2026-06-03T05:00Z"),
         "matched_signal": None},
    ])

    summary = summarize_performance(closed_trades)

    assert summary["total_pnl"] == 100.0  # +100 -100 +100
    assert summary["win_rate"] == pytest.approx(2 / 3)
    assert summary["best_trade_pnl"] == 100.0
    assert summary["worst_trade_pnl"] == -100.0
    assert summary["signal_accuracy"] == 1.0  # 1/1 matched BUY trades won
