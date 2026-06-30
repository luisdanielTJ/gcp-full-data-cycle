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


class BigQueryWarehouse(WarehouseAdapter):
    def __init__(self, project_id: str):
        from google.cloud import bigquery
        self.client = bigquery.Client(project=project_id)
        self.project_id = project_id

    def run_query(self, sql: str) -> pd.DataFrame:
        return self.client.query(sql).to_dataframe()

    def write_table(
        self, df: pd.DataFrame, dataset: str, table: str, mode: str = "append"
    ) -> None:
        if mode not in ("append", "replace"):
            raise ValueError(f"mode must be 'append' or 'replace', got {mode!r}")
        from google.cloud import bigquery
        disposition = (
            bigquery.WriteDisposition.WRITE_APPEND
            if mode == "append"
            else bigquery.WriteDisposition.WRITE_TRUNCATE
        )
        job_config = bigquery.LoadJobConfig(write_disposition=disposition)
        table_ref = f"{self.project_id}.{dataset}.{table}"
        self.client.load_table_from_dataframe(df, table_ref, job_config=job_config).result()

    def read_table(self, dataset: str, table: str) -> pd.DataFrame:
        from google.api_core.exceptions import NotFound

        table_ref = f"{self.project_id}.{dataset}.{table}"
        try:
            return self.client.query(f"SELECT * FROM `{table_ref}`").to_dataframe()
        except NotFound:
            return pd.DataFrame()
