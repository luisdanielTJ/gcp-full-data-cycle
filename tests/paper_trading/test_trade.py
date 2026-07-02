import pandas as pd
import pytest
from unittest.mock import MagicMock

from paper_trading.trade import run_paper_trade_cycle, POSITION_SIZE_USD, MIN_CONFIDENCE


def _signal_df(asset, signal, confidence):
    return pd.DataFrame([{
        "asset": asset,
        "signal": signal,
        "confidence": confidence,
        "predicted_at": pd.Timestamp("2026-07-02 00:00:00+00:00"),
    }])


def _ohlcv_df(asset, close):
    return pd.DataFrame([{
        "asset": asset,
        "open_time": pd.Timestamp("2026-07-02 00:00:00+00:00"),
        "close": close,
    }])


def _journal_with_open(asset, entry_price):
    return pd.DataFrame([{
        "id": "test-uuid-1",
        "asset": asset,
        "position_size_usd": POSITION_SIZE_USD,
        "entry_price": entry_price,
        "exit_price": float("nan"),
        "entry_confidence": 0.85,
        "exit_confidence": float("nan"),
        "opened_at": pd.Timestamp("2026-07-01 00:00:00+00:00"),
        "closed_at": pd.NaT,
        "pnl_usd": float("nan"),
        "pnl_pct": float("nan"),
    }])


def test_buy_signal_opens_position():
    wh = MagicMock()
    wh.read_table.return_value = pd.DataFrame()

    run_paper_trade_cycle(wh, _signal_df("XBTUSD", "BUY", 0.82), _ohlcv_df("XBTUSD", 60000.0))

    wh.write_table.assert_called_once()
    args, kwargs = wh.write_table.call_args
    written_df, dataset, table = args[0], args[1], args[2]
    assert dataset == "paper_trades"
    assert table == "journal"
    assert kwargs["mode"] == "append"
    assert written_df.iloc[0]["asset"] == "XBTUSD"
    assert written_df.iloc[0]["entry_price"] == pytest.approx(60000.0)
    assert pd.isna(written_df.iloc[0]["exit_price"])
    assert pd.isna(written_df.iloc[0]["closed_at"])


def test_sell_signal_closes_position():
    wh = MagicMock()
    wh.read_table.return_value = _journal_with_open("XBTUSD", entry_price=60000.0)

    run_paper_trade_cycle(wh, _signal_df("XBTUSD", "SELL", 0.88), _ohlcv_df("XBTUSD", 63000.0))

    wh.write_table.assert_called_once()
    args, kwargs = wh.write_table.call_args
    written_df = args[0]
    assert kwargs["mode"] == "replace"
    closed_row = written_df[written_df["id"] == "test-uuid-1"].iloc[0]
    assert closed_row["exit_price"] == pytest.approx(63000.0)
    assert closed_row["pnl_pct"] == pytest.approx(0.05)
    assert closed_row["pnl_usd"] == pytest.approx(50.0)
    assert pd.notna(closed_row["closed_at"])


def test_hold_signal_no_action():
    wh = MagicMock()
    wh.read_table.return_value = pd.DataFrame()

    run_paper_trade_cycle(wh, _signal_df("XBTUSD", "HOLD", 0.50), _ohlcv_df("XBTUSD", 60000.0))

    wh.write_table.assert_not_called()


def test_buy_below_confidence_threshold_no_action():
    wh = MagicMock()
    wh.read_table.return_value = pd.DataFrame()

    run_paper_trade_cycle(
        wh,
        _signal_df("XBTUSD", "BUY", MIN_CONFIDENCE - 0.01),
        _ohlcv_df("XBTUSD", 60000.0),
    )

    wh.write_table.assert_not_called()


def test_buy_with_open_position_no_action():
    wh = MagicMock()
    wh.read_table.return_value = _journal_with_open("XBTUSD", entry_price=58000.0)

    run_paper_trade_cycle(wh, _signal_df("XBTUSD", "BUY", 0.80), _ohlcv_df("XBTUSD", 60000.0))

    wh.write_table.assert_not_called()


def test_sell_with_no_open_position_no_action():
    wh = MagicMock()
    wh.read_table.return_value = pd.DataFrame()

    run_paper_trade_cycle(wh, _signal_df("XBTUSD", "SELL", 0.85), _ohlcv_df("XBTUSD", 60000.0))

    wh.write_table.assert_not_called()


def test_pnl_loss_when_exit_below_entry():
    wh = MagicMock()
    wh.read_table.return_value = _journal_with_open("XBTUSD", entry_price=60000.0)

    run_paper_trade_cycle(wh, _signal_df("XBTUSD", "SELL", 0.88), _ohlcv_df("XBTUSD", 57000.0))

    args, _ = wh.write_table.call_args
    closed_row = args[0][args[0]["id"] == "test-uuid-1"].iloc[0]
    assert closed_row["pnl_pct"] == pytest.approx(-0.05)
    assert closed_row["pnl_usd"] == pytest.approx(-50.0)
