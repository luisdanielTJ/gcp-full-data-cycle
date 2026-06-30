import pandas as pd
from xgboost import XGBClassifier

_TEST_SPLIT_RATIO = 0.2
_NON_FEATURE_COLS = ["asset", "open_time", "computed_at", "return_pct", "label"]
_XGB_PARAMS = {
    "n_estimators": 100,
    "max_depth": 4,
    "learning_rate": 0.1,
    "eval_metric": "logloss",
}


def time_based_split(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sorted_df = dataset.sort_values("open_time").reset_index(drop=True)
    split_index = int(len(sorted_df) * (1 - _TEST_SPLIT_RATIO))
    return sorted_df.iloc[:split_index], sorted_df.iloc[split_index:]


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _NON_FEATURE_COLS]


def train_model(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[XGBClassifier, pd.DataFrame, pd.Series]:
    feature_cols = _feature_columns(train_df)
    model = XGBClassifier(**_XGB_PARAMS)
    model.fit(train_df[feature_cols], train_df["label"])

    X_test = test_df[feature_cols]
    y_test = test_df["label"]
    return model, X_test, y_test
