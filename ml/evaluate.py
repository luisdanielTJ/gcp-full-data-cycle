import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

_BUY_THRESHOLD = 0.65
_SELL_THRESHOLD = 0.35
_PERIODS_PER_YEAR = 2190  # 6 four-hour candles/day * 365 days
_MIN_SIGNAL_ACCURACY = 0.55


def _derive_signals(proba: np.ndarray) -> pd.Series:
    signals = pd.Series("HOLD", index=range(len(proba)))
    signals[proba > _BUY_THRESHOLD] = "BUY"
    signals[proba < _SELL_THRESHOLD] = "SELL"
    return signals


def _sharpe(returns: pd.Series) -> float:
    std = returns.std()
    if std == 0:
        return 0.0
    return float(returns.mean() / std * np.sqrt(_PERIODS_PER_YEAR))


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, test_df: pd.DataFrame) -> dict:
    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    auc_roc = float(roc_auc_score(y_test, proba)) if y_test.nunique() > 1 else float("nan")

    signals = _derive_signals(proba).reset_index(drop=True)
    return_pct = test_df["return_pct"].reset_index(drop=True)
    label = pd.Series(y_test).reset_index(drop=True)

    is_buy = signals == "BUY"
    has_buy_signal = bool(is_buy.any())
    signal_accuracy = float(label[is_buy].eq(1).mean()) if has_buy_signal else None

    strategy_returns = return_pct.where(is_buy, 0.0)
    buy_and_hold_returns = return_pct

    simulated_pnl = float((1 + strategy_returns).cumprod().iloc[-1] - 1)
    sharpe = _sharpe(strategy_returns)
    buy_and_hold_sharpe = _sharpe(buy_and_hold_returns)

    gate_passed = bool(
        has_buy_signal
        and signal_accuracy is not None
        and signal_accuracy > _MIN_SIGNAL_ACCURACY
        and sharpe > buy_and_hold_sharpe
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc_roc": auc_roc,
        "signal_accuracy": signal_accuracy,
        "simulated_pnl": simulated_pnl,
        "sharpe": sharpe,
        "buy_and_hold_sharpe": buy_and_hold_sharpe,
        "gate_passed": gate_passed,
    }
