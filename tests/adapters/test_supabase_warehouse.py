from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from adapters.warehouse import SupabaseWarehouseAdapter


@pytest.fixture
def warehouse():
    mock_engine = MagicMock()
    with patch("sqlalchemy.create_engine", return_value=mock_engine):
        wh = SupabaseWarehouseAdapter("postgresql://fake/db")
    return wh, mock_engine


def test_read_table_calls_correct_sql(warehouse):
    wh, mock_engine = warehouse
    expected_df = pd.DataFrame({"asset": ["BTC"]})
    with patch("pandas.read_sql", return_value=expected_df) as mock_read_sql:
        result = wh.read_table("silver", "ohlcv")
    mock_read_sql.assert_called_once_with("SELECT * FROM silver.ohlcv", mock_engine)
    assert result["asset"].iloc[0] == "BTC"


def test_run_query_calls_read_sql(warehouse):
    wh, mock_engine = warehouse
    expected_df = pd.DataFrame({"count": [42]})
    with patch("pandas.read_sql", return_value=expected_df) as mock_read_sql:
        result = wh.run_query("SELECT COUNT(*) FROM silver.ohlcv")
    mock_read_sql.assert_called_once_with("SELECT COUNT(*) FROM silver.ohlcv", mock_engine)
    assert result["count"].iloc[0] == 42


def test_write_table_replace(warehouse):
    wh, mock_engine = warehouse
    df = pd.DataFrame({"asset": ["BTC"], "close": [50000.0]})
    with patch.object(df, "to_sql") as mock_to_sql:
        wh.write_table(df, "silver", "ohlcv", mode="replace")
    mock_to_sql.assert_called_once_with(
        "ohlcv", mock_engine, schema="silver", if_exists="replace", index=False
    )


def test_write_table_append(warehouse):
    wh, mock_engine = warehouse
    df = pd.DataFrame({"asset": ["BTC"], "close": [50000.0]})
    with patch.object(df, "to_sql") as mock_to_sql:
        wh.write_table(df, "silver", "ohlcv", mode="append")
    mock_to_sql.assert_called_once_with(
        "ohlcv", mock_engine, schema="silver", if_exists="append", index=False
    )


def test_write_table_creates_schema(warehouse):
    wh, mock_engine = warehouse
    df = pd.DataFrame({"asset": ["BTC"]})
    with patch.object(df, "to_sql"):
        wh.write_table(df, "silver", "ohlcv", mode="replace")
    conn = mock_engine.begin.return_value.__enter__.return_value
    executed_clause = conn.execute.call_args[0][0]
    assert "CREATE SCHEMA IF NOT EXISTS silver" in str(executed_clause)


def test_write_table_invalid_mode_raises(warehouse):
    wh, _ = warehouse
    df = pd.DataFrame({"asset": ["BTC"]})
    with pytest.raises(ValueError, match="mode must be"):
        wh.write_table(df, "silver", "ohlcv", mode="invalid")


def test_read_table_returns_empty_when_table_missing(warehouse):
    from sqlalchemy.exc import ProgrammingError
    wh, _ = warehouse
    with patch("pandas.read_sql", side_effect=ProgrammingError("table not found", None, None)):
        result = wh.read_table("silver", "does_not_exist")
    assert result.empty
