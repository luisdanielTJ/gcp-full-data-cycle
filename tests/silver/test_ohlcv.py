import logging

import pandas as pd
import pytest

from silver.ohlcv import clean_ohlcv


def _bronze_row(asset="XBTUSD", open_time="2026-06-01T00:00:00Z", close=50000.0):
    return {
        "asset": asset,
        "open_time": pd.Timestamp(open_time),
        "open": 49000.0,
        "high": 50500.0,
        "low": 48900.0,
        "close": close,
        "volume": 100.0,
        "ingested_at": pd.Timestamp.now(tz="UTC"),
    }


def test_clean_ohlcv_excludes_existing_keys():
    bronze_df = pd.DataFrame([
        _bronze_row(open_time="2026-06-01T00:00:00Z"),
        _bronze_row(open_time="2026-06-01T04:00:00Z"),
    ])
    existing_keys = {("XBTUSD", pd.Timestamp("2026-06-01T00:00:00Z", tz="UTC"))}

    result = clean_ohlcv(bronze_df, existing_keys)

    assert len(result) == 1
    assert result["open_time"].iloc[0] == pd.Timestamp("2026-06-01T04:00:00Z", tz="UTC")


def test_clean_ohlcv_casts_types_and_adds_cleaned_at():
    bronze_df = pd.DataFrame([_bronze_row()])

    result = clean_ohlcv(bronze_df, existing_keys=set())

    assert result["close"].iloc[0] == pytest.approx(50000.0)
    assert "cleaned_at" in result.columns
    assert result["cleaned_at"].iloc[0].tzinfo is not None
    assert "ingested_at" not in result.columns


def test_clean_ohlcv_dedupes_duplicate_rows():
    bronze_df = pd.DataFrame([_bronze_row(), _bronze_row()])

    result = clean_ohlcv(bronze_df, existing_keys=set())

    assert len(result) == 1


def test_clean_ohlcv_logs_warning_on_gap(caplog):
    bronze_df = pd.DataFrame([
        _bronze_row(open_time="2026-06-01T00:00:00Z"),
        _bronze_row(open_time="2026-06-01T12:00:00Z"),
    ])

    with caplog.at_level(logging.WARNING):
        clean_ohlcv(bronze_df, existing_keys=set())

    assert "gap" in caplog.text.lower()


def test_clean_ohlcv_no_warning_when_consecutive_4h(caplog):
    bronze_df = pd.DataFrame([
        _bronze_row(open_time="2026-06-01T00:00:00Z"),
        _bronze_row(open_time="2026-06-01T04:00:00Z"),
    ])

    with caplog.at_level(logging.WARNING):
        clean_ohlcv(bronze_df, existing_keys=set())

    assert "gap" not in caplog.text.lower()


def test_clean_ohlcv_handles_empty_bronze_df():
    result = clean_ohlcv(pd.DataFrame(), existing_keys=set())

    assert result.empty
