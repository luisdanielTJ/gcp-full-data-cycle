import numpy as np
import pandas as pd

from gold.indicators import compute_indicators


def _ohlcv_series(asset, n=60, start="2026-01-01T00:00:00Z", start_price=50000.0, step=5.0):
    times = pd.date_range(start=start, periods=n, freq="4h", tz="UTC")
    closes = start_price + np.arange(n) * step
    return pd.DataFrame({
        "asset": asset,
        "open_time": times,
        "open": closes - 5,
        "high": closes + 10,
        "low": closes - 10,
        "close": closes,
        "volume": 100.0 + np.arange(n),
    })


def test_compute_indicators_returns_expected_columns():
    df = _ohlcv_series("XBTUSD", n=60)

    result = compute_indicators(df)

    expected_cols = {
        "rsi_14", "macd_line", "macd_signal", "macd_hist",
        "bb_upper", "bb_lower", "bb_width", "atr_14", "volume_ratio",
        "momentum_1", "momentum_3", "momentum_6",
    }
    assert expected_cols.issubset(result.columns)
    assert not result.empty
    assert result[list(expected_cols)].isna().sum().sum() == 0


def test_compute_indicators_drops_rows_with_insufficient_history():
    df = _ohlcv_series("XBTUSD", n=5)

    result = compute_indicators(df)

    assert result.empty


def test_compute_indicators_keeps_one_asset_when_other_lacks_history():
    sufficient = _ohlcv_series("XBTUSD", n=60)
    insufficient = _ohlcv_series("ETHUSD", n=5, start_price=3000.0)
    df = pd.concat([sufficient, insufficient], ignore_index=True)

    result = compute_indicators(df)

    assert set(result["asset"].unique()) == {"XBTUSD"}
    assert not result.empty


def test_compute_indicators_handles_empty_input():
    result = compute_indicators(pd.DataFrame())

    assert result.empty
