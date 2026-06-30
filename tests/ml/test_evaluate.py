import numpy as np
import pandas as pd

from ml.evaluate import evaluate_model


class _FakeModel:
    """Duck-typed stand-in for XGBClassifier — only predict/predict_proba are used."""

    def __init__(self, proba):
        self._proba = np.array(proba)

    def predict(self, X):
        return (self._proba > 0.5).astype(int)

    def predict_proba(self, X):
        return np.column_stack([1 - self._proba, self._proba])


def _frame(n):
    return pd.DataFrame({"f": range(n)})


def test_evaluate_model_computes_ml_metrics_and_passes_gate():
    model = _FakeModel([0.9, 0.8, 0.2, 0.1])
    y_test = pd.Series([1, 1, 0, 0])
    test_df = pd.DataFrame({"return_pct": [0.02, 0.015, -0.01, -0.02]})

    result = evaluate_model(model, _frame(4), y_test, test_df)

    assert set(result.keys()) == {
        "precision", "recall", "f1", "auc_roc", "signal_accuracy",
        "simulated_pnl", "sharpe", "buy_and_hold_sharpe", "gate_passed",
    }
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0
    assert result["auc_roc"] == 1.0
    assert result["signal_accuracy"] == 1.0
    assert result["sharpe"] > result["buy_and_hold_sharpe"]
    assert result["gate_passed"] is True


def test_evaluate_model_gate_fails_on_low_signal_accuracy():
    model = _FakeModel([0.9, 0.8, 0.7, 0.1])
    y_test = pd.Series([0, 0, 0, 1])
    test_df = pd.DataFrame({"return_pct": [0.01, 0.01, 0.01, -0.05]})

    result = evaluate_model(model, _frame(4), y_test, test_df)

    assert result["signal_accuracy"] == 0.0
    assert result["gate_passed"] is False


def test_evaluate_model_gate_fails_when_strategy_underperforms_buy_and_hold():
    model = _FakeModel([0.9, 0.8, 0.2, 0.1])
    y_test = pd.Series([1, 1, 0, 0])
    # BUY rows (0, 1) lose money; non-BUY rows (2, 3) would have gained a lot.
    test_df = pd.DataFrame({"return_pct": [-0.01, -0.01, 0.05, 0.05]})

    result = evaluate_model(model, _frame(4), y_test, test_df)

    assert result["signal_accuracy"] == 1.0
    assert result["sharpe"] < result["buy_and_hold_sharpe"]
    assert result["gate_passed"] is False


def test_evaluate_model_gate_fails_when_no_buy_signals():
    model = _FakeModel([0.5, 0.5, 0.5, 0.5])
    y_test = pd.Series([1, 0, 1, 0])
    test_df = pd.DataFrame({"return_pct": [0.01, -0.01, 0.02, -0.02]})

    result = evaluate_model(model, _frame(4), y_test, test_df)

    assert result["signal_accuracy"] is None
    assert result["gate_passed"] is False
