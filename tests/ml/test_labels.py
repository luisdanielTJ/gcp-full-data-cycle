import pandas as pd
import pytest

from ml.labels import compute_labels


def test_compute_labels_marks_buy_when_return_exceeds_threshold():
    df = pd.DataFrame({
        "asset": ["XBTUSD", "XBTUSD"],
        "open_time": pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T04:00:00Z"]),
        "close": [100.0, 102.0],
    })

    result = compute_labels(df)

    assert len(result) == 1
    assert result["return_pct"].iloc[0] == pytest.approx(0.02)
    assert result["label"].iloc[0] == 1


def test_compute_labels_marks_no_buy_when_return_below_threshold():
    df = pd.DataFrame({
        "asset": ["XBTUSD", "XBTUSD"],
        "open_time": pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T04:00:00Z"]),
        "close": [100.0, 100.5],
    })

    result = compute_labels(df)

    assert result["label"].iloc[0] == 0


def test_compute_labels_drops_last_candle_per_asset():
    df = pd.DataFrame({
        "asset": ["XBTUSD", "XBTUSD", "ETHUSD"],
        "open_time": pd.to_datetime([
            "2026-01-01T00:00:00Z", "2026-01-01T04:00:00Z", "2026-01-01T00:00:00Z",
        ]),
        "close": [100.0, 102.0, 3000.0],
    })

    result = compute_labels(df)

    assert len(result) == 1
    assert result["asset"].iloc[0] == "XBTUSD"


def test_compute_labels_handles_empty_input():
    result = compute_labels(pd.DataFrame())

    assert result.empty
    assert list(result.columns) == ["asset", "open_time", "return_pct", "label"]
