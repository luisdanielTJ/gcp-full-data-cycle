from abc import ABC, abstractmethod

import pandas as pd


class WarehouseAdapter(ABC):
    @abstractmethod
    def run_query(self, sql: str) -> pd.DataFrame:
        ...

    @abstractmethod
    def write_table(
        self, df: pd.DataFrame, dataset: str, table: str, mode: str = "append"
    ) -> None:
        ...

    @abstractmethod
    def read_table(self, dataset: str, table: str) -> pd.DataFrame:
        """Returns the full table contents, or an empty DataFrame if it doesn't exist yet."""
        ...


class DuckDBWarehouse(WarehouseAdapter):
    def __init__(self, db_path: str = ":memory:"):
        import duckdb
        self.conn = duckdb.connect(db_path)

    def run_query(self, sql: str) -> pd.DataFrame:
        return self.conn.execute(sql).df()

    def write_table(
        self, df: pd.DataFrame, dataset: str, table: str, mode: str = "append"
    ) -> None:
        if mode not in ("append", "replace"):
            raise ValueError(f"mode must be 'append' or 'replace', got {mode!r}")
        table_name = f"{dataset}__{table}"
        if mode == "replace":
            self.conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM df")
        else:
            existing = self.conn.execute("SHOW TABLES").fetchdf()
            if table_name in existing["name"].values:
                self.conn.execute(f"INSERT INTO {table_name} SELECT * FROM df")
            else:
                self.conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")

    def read_table(self, dataset: str, table: str) -> pd.DataFrame:
        table_name = f"{dataset}__{table}"
        existing = self.conn.execute("SHOW TABLES").fetchdf()
        if table_name not in existing["name"].values:
            return pd.DataFrame()
        return self.conn.execute(f"SELECT * FROM {table_name}").df()


class SupabaseWarehouseAdapter(WarehouseAdapter):
    def __init__(self, database_url: str):
        from sqlalchemy import create_engine, text
        self._engine = create_engine(
            database_url,
            connect_args={"connect_timeout": 10},
            pool_pre_ping=True,
        )
        self._text = text

    def run_query(self, sql: str) -> pd.DataFrame:
        return pd.read_sql(sql, self._engine)

    def write_table(
        self, df: pd.DataFrame, dataset: str, table: str, mode: str = "append"
    ) -> None:
        if mode not in ("append", "replace"):
            raise ValueError(f"mode must be 'append' or 'replace', got {mode!r}")
        with self._engine.begin() as conn:
            conn.execute(self._text(f"CREATE SCHEMA IF NOT EXISTS {dataset}"))
        if_exists = "replace" if mode == "replace" else "append"
        df.to_sql(table, self._engine, schema=dataset, if_exists=if_exists, index=False)

    def read_table(self, dataset: str, table: str) -> pd.DataFrame:
        try:
            return pd.read_sql(f"SELECT * FROM {dataset}.{table}", self._engine)
        except Exception:
            return pd.DataFrame()
