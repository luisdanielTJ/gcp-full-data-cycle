import pandas as pd

from app.positions import enrich_open_positions, latest_close_prices


def test_latest_close_prices_returns_most_recent_close_per_asset():
    ohlcv_df = pd.DataFrame([
        {"asset": "XBTUSD", "open_time": pd.Timestamp("2026-06-01T00:00Z"), "close": 49000.0},
        {"asset": "XBTUSD", "open_time": pd.Timestamp("2026-06-01T04:00Z"), "close": 50000.0},
        {"asset": "ETHUSD", "open_time": pd.Timestamp("2026-06-01T00:00Z"), "close": 3000.0},
    ])

    prices = latest_close_prices(ohlcv_df)

    assert prices == {"XBTUSD": 50000.0, "ETHUSD": 3000.0}


def test_enrich_open_positions_adds_unrealized_pnl():
    open_trades = pd.DataFrame([{
        "id": "t1", "asset": "XBTUSD", "direction": "LONG",
        "entry_price": 50000.0, "amount_usd": 1000.0,
        "opened_at": pd.Timestamp("2026-06-01T00:00Z"),
    }])
    ohlcv_df = pd.DataFrame([{
        "asset": "XBTUSD", "open_time": pd.Timestamp("2026-06-01T04:00Z"), "close": 51000.0,
    }])

    result = enrich_open_positions(open_trades, ohlcv_df)

    assert len(result) == 1
    assert result.iloc[0]["current_price"] == 51000.0
    assert result.iloc[0]["unrealized_pnl"] == 20.0  # (51k-50k)/50k * 1000


def test_enrich_open_positions_returns_empty_when_no_open_trades():
    result = enrich_open_positions(pd.DataFrame(), pd.DataFrame())

    assert result.empty
