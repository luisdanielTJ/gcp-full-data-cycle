from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from ml import predict


def _features_row(asset, open_time):
    return {
        "asset": asset,
        "open_time": open_time,
        "rsi_14": 55.0,
        "computed_at": pd.Timestamp.now(tz="UTC"),
    }


def test_run_prediction_cycle_skips_when_no_production_model():
    mock_warehouse = MagicMock()
    mock_registry = MagicMock()
    mock_llm = MagicMock()
    mock_registry.load_model.side_effect = ValueError("No production model for 'xgboost_signal'")

    predict.run_prediction_cycle(mock_warehouse, mock_registry, mock_llm)

    mock_warehouse.write_table.assert_not_called()


def test_run_prediction_cycle_skips_when_no_features():
    mock_warehouse = MagicMock()
    mock_registry = MagicMock()
    mock_llm = MagicMock()
    mock_registry.load_model.return_value = MagicMock()
    mock_warehouse.read_table.return_value = pd.DataFrame()

    predict.run_prediction_cycle(mock_warehouse, mock_registry, mock_llm)

    mock_warehouse.write_table.assert_not_called()


def test_run_prediction_cycle_writes_one_signal_and_narration_row_per_asset():
    mock_warehouse = MagicMock()
    mock_registry = MagicMock()
    mock_llm = MagicMock()

    now = pd.Timestamp.now(tz="UTC")
    features_df = pd.DataFrame([
        _features_row("XBTUSD", now - pd.Timedelta(hours=4)),
        _features_row("XBTUSD", now),
        _features_row("ETHUSD", now),
    ])
    ohlcv_df = pd.DataFrame([
        {"asset": "XBTUSD", "open_time": now, "close": 50000.0},
        {"asset": "ETHUSD", "open_time": now, "close": 3000.0},
    ])
    mock_warehouse.read_table.side_effect = lambda dataset, table: (
        features_df if (dataset, table) == ("gold", "ml_features")
        else ohlcv_df if (dataset, table) == ("silver", "ohlcv")
        else pd.DataFrame()
    )

    fake_model = MagicMock()
    fake_model.predict_proba.return_value = np.array([[0.2, 0.8], [0.2, 0.8]])
    mock_registry.load_model.return_value = fake_model
    mock_registry.get_production_version.return_value = "3"
    mock_llm.narrate_signal.return_value = "This is a narration."

    with patch("ml.predict.shap.TreeExplainer") as mock_explainer_cls:
        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = np.array([[0.5], [0.5]])
        mock_explainer_cls.return_value = mock_explainer

        predict.run_prediction_cycle(mock_warehouse, mock_registry, mock_llm)

    write_calls = mock_warehouse.write_table.call_args_list
    signal_writes = [c for c in write_calls if c.args[1:3] == ("predictions", "signals")]
    narration_writes = [c for c in write_calls if c.args[1:3] == ("predictions", "narrations")]

    assert len(signal_writes) == 1
    assert len(narration_writes) == 1

    signals_df = signal_writes[0].args[0]
    assert len(signals_df) == 2
    assert set(signals_df["asset"]) == {"XBTUSD", "ETHUSD"}
    assert set(signals_df["signal"]) == {"BUY"}
    assert (signals_df["model_version"] == "3").all()

    narrations_df = narration_writes[0].args[0]
    assert len(narrations_df) == 2
    assert (narrations_df["narration"] == "This is a narration.").all()


def test_run_prediction_cycle_uses_only_latest_row_per_asset():
    mock_warehouse = MagicMock()
    mock_registry = MagicMock()
    mock_llm = MagicMock()

    now = pd.Timestamp.now(tz="UTC")
    older = now - pd.Timedelta(hours=4)
    older_row = _features_row("XBTUSD", older)
    older_row["rsi_14"] = 40.0
    newer_row = _features_row("XBTUSD", now)
    newer_row["rsi_14"] = 70.0
    features_df = pd.DataFrame([older_row, newer_row])
    ohlcv_df = pd.DataFrame([{"asset": "XBTUSD", "open_time": now, "close": 50000.0}])
    mock_warehouse.read_table.side_effect = lambda dataset, table: (
        features_df if (dataset, table) == ("gold", "ml_features")
        else ohlcv_df if (dataset, table) == ("silver", "ohlcv")
        else pd.DataFrame()
    )

    fake_model = MagicMock()
    fake_model.predict_proba.return_value = np.array([[0.5, 0.5]])
    mock_registry.load_model.return_value = fake_model
    mock_registry.get_production_version.return_value = "1"
    mock_llm.narrate_signal.return_value = "narration"

    with patch("ml.predict.shap.TreeExplainer") as mock_explainer_cls:
        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = np.array([[0.1]])
        mock_explainer_cls.return_value = mock_explainer

        predict.run_prediction_cycle(mock_warehouse, mock_registry, mock_llm)

    assert fake_model.predict_proba.call_count == 1
    called_with = fake_model.predict_proba.call_args.args[0]
    assert len(called_with) == 1
    assert called_with["rsi_14"].iloc[0] == 70.0
