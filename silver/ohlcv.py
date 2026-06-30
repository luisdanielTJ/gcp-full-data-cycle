import logging

import pandas as pd

logger = logging.getLogger(__name__)

_EXPECTED_INTERVAL = pd.Timedelta(hours=4)


def clean_ohlcv(bronze_df: pd.DataFrame, existing_keys: set[tuple]) -> pd.DataFrame:
    if bronze_df.empty:
        return bronze_df

    df = bronze_df.copy()
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    keys = list(zip(df["asset"], df["open_time"]))
    df = df[[k not in existing_keys for k in keys]]
    df = df.drop_duplicates(subset=["asset", "open_time"])

    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)

    for asset, group in df.groupby("asset"):
        sorted_times = group.sort_values("open_time")["open_time"]
        gaps = sorted_times.diff().dropna()
        for gap in gaps[gaps != _EXPECTED_INTERVAL]:
            logger.warning("OHLCV gap detected for %s: %s between candles", asset, gap)

    df["cleaned_at"] = pd.Timestamp.now(tz="UTC")
    return df.drop(columns=["ingested_at"], errors="ignore").reset_index(drop=True)
