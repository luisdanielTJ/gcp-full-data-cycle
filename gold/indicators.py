import pandas as pd
import pandas_ta_classic as ta

_INDICATOR_COLS = [
    "rsi_14",
    "macd_line",
    "macd_signal",
    "macd_hist",
    "bb_upper",
    "bb_lower",
    "bb_width",
    "atr_14",
    "volume_ratio",
    "momentum_1",
    "momentum_3",
    "momentum_6",
]


def _compute_for_asset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("open_time").reset_index(drop=True)

    rsi = ta.rsi(df["close"], length=14)
    df["rsi_14"] = rsi if rsi is not None else float("nan")

    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    if macd is not None:
        df["macd_line"] = macd["MACD_12_26_9"]
        df["macd_signal"] = macd["MACDs_12_26_9"]
        df["macd_hist"] = macd["MACDh_12_26_9"]
    else:
        df["macd_line"] = float("nan")
        df["macd_signal"] = float("nan")
        df["macd_hist"] = float("nan")

    bbands = ta.bbands(df["close"], length=20)
    if bbands is not None:
        df["bb_upper"] = bbands["BBU_20_2.0"]
        df["bb_lower"] = bbands["BBL_20_2.0"]
        df["bb_width"] = bbands["BBB_20_2.0"]
    else:
        df["bb_upper"] = float("nan")
        df["bb_lower"] = float("nan")
        df["bb_width"] = float("nan")

    atr = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["atr_14"] = atr if atr is not None else float("nan")

    df["volume_ratio"] = df["volume"] / df["volume"].rolling(window=20).mean()

    df["momentum_1"] = df["close"].pct_change(periods=1)
    df["momentum_3"] = df["close"].pct_change(periods=3)
    df["momentum_6"] = df["close"].pct_change(periods=6)

    return df


def compute_indicators(ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    if ohlcv_df.empty:
        return pd.DataFrame(columns=["asset", "open_time"] + _INDICATOR_COLS)

    results = []
    for _, group in ohlcv_df.groupby("asset"):
        results.append(_compute_for_asset(group.copy()))

    combined = pd.concat(results, ignore_index=True)
    combined = combined.dropna(subset=_INDICATOR_COLS)

    return combined[["asset", "open_time"] + _INDICATOR_COLS].reset_index(drop=True)
