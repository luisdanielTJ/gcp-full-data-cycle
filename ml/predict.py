import json

import numpy as np
import pandas as pd
import shap

from ml.evaluate import derive_signals
from ml.train import feature_columns

MODEL_NAME = "xgboost_signal"
_TOP_K_FEATURES = 5
_SENTIMENT_COLS = ["sentiment_4h", "sentiment_24h", "sentiment_72h", "news_sentiment_24h"]


def _latest_features_per_asset(features_df: pd.DataFrame) -> pd.DataFrame:
    return (
        features_df.sort_values("open_time")
        .groupby("asset", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )


def _top_features(shap_row: np.ndarray, columns: list[str]) -> list[dict]:
    order = np.argsort(-np.abs(shap_row))[:_TOP_K_FEATURES]
    return [{"feature": columns[i], "value": float(shap_row[i])} for i in order]


def _sentiment_summary(row: pd.Series) -> str:
    parts = [f"{col}={row[col]:.2f}" for col in _SENTIMENT_COLS if col in row.index]
    return ", ".join(parts) if parts else "no sentiment data"


def _recent_prices(ohlcv_df: pd.DataFrame, asset: str) -> list[float]:
    asset_rows = ohlcv_df[ohlcv_df["asset"] == asset].sort_values("open_time")
    return asset_rows["close"].tail(3).round(2).tolist()


def run_prediction_cycle(warehouse, model_registry, llm) -> None:
    try:
        model = model_registry.load_model(MODEL_NAME, version="production")
    except ValueError as exc:
        print(f"[predict] {exc}, skipping")
        return

    features_df = warehouse.read_table("gold", "ml_features")
    if features_df.empty:
        print("[predict] no features available, skipping")
        return

    ohlcv_df = warehouse.read_table("silver", "ohlcv")
    latest_df = _latest_features_per_asset(features_df)
    cols = feature_columns(latest_df)

    proba = model.predict_proba(latest_df[cols])[:, 1]
    signals = derive_signals(proba).reset_index(drop=True)
    confidence = np.maximum(proba, 1 - proba)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(latest_df[cols])

    version = model_registry.get_production_version(MODEL_NAME)
    predicted_at = pd.Timestamp.now(tz="UTC")

    signal_rows = []
    narration_rows = []
    for i, row in latest_df.reset_index(drop=True).iterrows():
        asset = row["asset"]
        top5 = _top_features(shap_values[i], cols)

        signal_rows.append({
            "asset": asset,
            "signal": signals.iloc[i],
            "confidence": float(confidence[i]),
            "predicted_at": predicted_at,
            "model_version": version,
            "shap_top5": json.dumps(top5),
        })

        narration = llm.narrate_signal({
            "asset": asset,
            "signal": signals.iloc[i],
            "confidence": float(confidence[i]),
            "top_features": [f["feature"] for f in top5],
            "sentiment_summary": _sentiment_summary(row),
            "recent_prices": _recent_prices(ohlcv_df, asset),
        })
        narration_rows.append({
            "asset": asset,
            "narration": narration,
            "predicted_at": predicted_at,
        })

    warehouse.write_table(pd.DataFrame(signal_rows), "predictions", "signals", mode="append")
    warehouse.write_table(pd.DataFrame(narration_rows), "predictions", "narrations", mode="append")
    print(f"[predict] wrote {len(signal_rows)} signal(s) and narration(s)")
