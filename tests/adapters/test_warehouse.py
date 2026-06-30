import pandas as pd
import pytest

from adapters.warehouse import DuckDBWarehouse


@pytest.fixture
def warehouse():
    return DuckDBWarehouse(db_path=":memory:")


def test_write_replace_and_read(warehouse):
    df = pd.DataFrame({
        "asset": ["BTC"],
        "close": [50000.0],
        "ts": [pd.Timestamp("2024-01-01", tz="UTC")],
    })
    warehouse.write_table(df, "bronze", "raw_prices", mode="replace")
    result = warehouse.run_query("SELECT * FROM bronze__raw_prices")
    assert len(result) == 1
    assert result["asset"].iloc[0] == "BTC"
    assert result["close"].iloc[0] == 50000.0


def test_write_append(warehouse):
    df1 = pd.DataFrame({
        "asset": ["BTC"], "close": [50000.0],
        "ts": [pd.Timestamp("2024-01-01", tz="UTC")],
    })
    df2 = pd.DataFrame({
        "asset": ["ETH"], "close": [3000.0],
        "ts": [pd.Timestamp("2024-01-01", tz="UTC")],
    })
    warehouse.write_table(df1, "bronze", "raw_prices", mode="replace")
    warehouse.write_table(df2, "bronze", "raw_prices", mode="append")
    result = warehouse.run_query("SELECT COUNT(*) AS cnt FROM bronze__raw_prices")
    assert result["cnt"].iloc[0] == 2


def test_run_query_with_filter(warehouse):
    df = pd.DataFrame({
        "asset": ["BTC", "ETH"],
        "close": [50000.0, 3000.0],
        "ts": [pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-01", tz="UTC")],
    })
    warehouse.write_table(df, "bronze", "raw_prices", mode="replace")
    result = warehouse.run_query("SELECT * FROM bronze__raw_prices WHERE asset = 'BTC'")
    assert len(result) == 1
    assert result["asset"].iloc[0] == "BTC"


def test_invalid_mode_raises(warehouse):
    df = pd.DataFrame({"asset": ["BTC"], "close": [50000.0],
                        "ts": [pd.Timestamp("2024-01-01", tz="UTC")]})
    with pytest.raises(ValueError, match="mode must be"):
        warehouse.write_table(df, "bronze", "raw_prices", mode="invalid")


def test_read_table_returns_written_data(warehouse):
    df = pd.DataFrame({
        "asset": ["BTC"], "close": [50000.0],
        "ts": [pd.Timestamp("2024-01-01", tz="UTC")],
    })
    warehouse.write_table(df, "bronze", "raw_prices", mode="replace")
    result = warehouse.read_table("bronze", "raw_prices")
    assert len(result) == 1
    assert result["asset"].iloc[0] == "BTC"


def test_read_table_returns_empty_when_table_missing(warehouse):
    result = warehouse.read_table("silver", "does_not_exist")
    assert result.empty
