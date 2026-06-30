import pandas as pd


def build_training_dataset(features_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    if features_df.empty or labels_df.empty:
        return pd.DataFrame()
    return features_df.merge(labels_df, on=["asset", "open_time"], how="inner")
