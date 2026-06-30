from unittest.mock import MagicMock, patch

import pandas as pd

from ml import run


def test_run_training_cycle_skips_when_insufficient_data():
    mock_warehouse = MagicMock()
    mock_registry = MagicMock()
    mock_warehouse.read_table.side_effect = [pd.DataFrame(), pd.DataFrame()]

    with (
        patch("ml.run.compute_labels", return_value=pd.DataFrame()),
        patch("ml.run.build_training_dataset", return_value=pd.DataFrame({"a": range(10)})),
    ):
        run.run_training_cycle(mock_warehouse, mock_registry)

    mock_registry.log_model.assert_not_called()


def test_run_training_cycle_logs_and_promotes_when_gate_passes():
    mock_warehouse = MagicMock()
    mock_registry = MagicMock()
    mock_registry.log_model.return_value = "1"
    mock_warehouse.read_table.side_effect = [pd.DataFrame(), pd.DataFrame()]

    dataset = pd.DataFrame({"a": range(60)})
    train_df, test_df = dataset.iloc[:48], dataset.iloc[48:]
    fake_model = MagicMock()
    fake_model.get_params.return_value = {"n_estimators": 100}

    with (
        patch("ml.run.compute_labels", return_value=pd.DataFrame()),
        patch("ml.run.build_training_dataset", return_value=dataset),
        patch("ml.run.time_based_split", return_value=(train_df, test_df)),
        patch("ml.run.train_model", return_value=(fake_model, "X_test", "y_test")),
        patch("ml.run.evaluate_model", return_value={"gate_passed": True}),
        patch("ml.run.compute_feature_importance", return_value={"f1": 0.5}),
    ):
        run.run_training_cycle(mock_warehouse, mock_registry)

    mock_registry.log_model.assert_called_once()
    args, kwargs = mock_registry.log_model.call_args
    assert args[0] == fake_model
    assert args[1]["gate_passed"] is True
    assert args[1]["feature_importance"] == {"f1": 0.5}
    assert kwargs["name"] == "xgboost_signal"
    mock_registry.promote_model.assert_called_once_with("xgboost_signal", "1")


def test_run_training_cycle_logs_without_promoting_when_gate_fails():
    mock_warehouse = MagicMock()
    mock_registry = MagicMock()
    mock_registry.log_model.return_value = "1"
    mock_warehouse.read_table.side_effect = [pd.DataFrame(), pd.DataFrame()]

    dataset = pd.DataFrame({"a": range(60)})
    train_df, test_df = dataset.iloc[:48], dataset.iloc[48:]
    fake_model = MagicMock()
    fake_model.get_params.return_value = {}

    with (
        patch("ml.run.compute_labels", return_value=pd.DataFrame()),
        patch("ml.run.build_training_dataset", return_value=dataset),
        patch("ml.run.time_based_split", return_value=(train_df, test_df)),
        patch("ml.run.train_model", return_value=(fake_model, "X_test", "y_test")),
        patch("ml.run.evaluate_model", return_value={"gate_passed": False}),
        patch("ml.run.compute_feature_importance", return_value={}),
    ):
        run.run_training_cycle(mock_warehouse, mock_registry)

    mock_registry.log_model.assert_called_once()
    mock_registry.promote_model.assert_not_called()
