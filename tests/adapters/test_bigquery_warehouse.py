from unittest.mock import MagicMock, patch

import pandas as pd

from adapters.warehouse import BigQueryWarehouse


def test_read_table_returns_dataframe():
    with patch("google.cloud.bigquery.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_query_job = MagicMock()
        mock_query_job.to_dataframe.return_value = pd.DataFrame({"asset": ["BTC"]})
        mock_client.query.return_value = mock_query_job

        warehouse = BigQueryWarehouse(project_id="test-project")
        result = warehouse.read_table("silver", "ohlcv")

    assert result["asset"].iloc[0] == "BTC"
    called_sql = mock_client.query.call_args[0][0]
    assert "test-project.silver.ohlcv" in called_sql


def test_read_table_returns_empty_on_not_found():
    from google.api_core.exceptions import NotFound

    with patch("google.cloud.bigquery.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.query.side_effect = NotFound("not found")

        warehouse = BigQueryWarehouse(project_id="test-project")
        result = warehouse.read_table("silver", "does_not_exist")

    assert result.empty
