import pandas as pd

_RETURN_THRESHOLD = 0.01

_COLUMNS = ["asset", "open_time", "return_pct", "label"]


def compute_labels(ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    if ohlcv_df.empty:
        return pd.DataFrame(columns=_COLUMNS)

    results = []
    for _, group in ohlcv_df.groupby("asset"):
        df = group.sort_values("open_time").reset_index(drop=True)
        df["next_close"] = df["close"].shift(-1)
        df = df.dropna(subset=["next_close"])
        df["return_pct"] = df["next_close"] / df["close"] - 1
        df["label"] = (df["return_pct"] > _RETURN_THRESHOLD).astype(int)
        results.append(df[_COLUMNS])

    return pd.concat(results, ignore_index=True)
