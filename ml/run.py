from adapters import get_model_registry, get_warehouse
from ml.dataset import build_training_dataset
from ml.evaluate import evaluate_model
from ml.explain import compute_feature_importance
from ml.labels import compute_labels
from ml.train import time_based_split, train_model

MIN_TRAINING_ROWS = 50


def run_training_cycle(warehouse, model_registry) -> None:
    ohlcv_df = warehouse.read_table("silver", "ohlcv")
    features_df = warehouse.read_table("gold", "ml_features")

    labels_df = compute_labels(ohlcv_df)
    dataset = build_training_dataset(features_df, labels_df)

    if len(dataset) < MIN_TRAINING_ROWS:
        print(f"[train] insufficient data ({len(dataset)} rows), skipping")
        return

    train_df, test_df = time_based_split(dataset)
    neg = (train_df["label"] == 0).sum()
    pos = (train_df["label"] == 1).sum()
    scale_pos_weight = float(neg / pos) if pos > 0 else 1.0
    print(f"[train] class balance: {pos} positive, {neg} negative, scale_pos_weight={scale_pos_weight:.2f}")
    model, X_test, y_test = train_model(train_df, test_df, scale_pos_weight=scale_pos_weight)
    metrics = evaluate_model(model, X_test, y_test, test_df)
    metrics["feature_importance"] = compute_feature_importance(model, X_test)

    params = model.get_params()
    version = model_registry.log_model(model, metrics, params, name="xgboost_signal")
    print(f"[train] logged version {version}, metrics={metrics}")

    if metrics["gate_passed"]:
        model_registry.promote_model("xgboost_signal", version)
        print(f"[train] promoted version {version} to production")
    else:
        print(f"[train] version {version} did not pass promotion gate")


if __name__ == "__main__":
    wh = get_warehouse()
    registry = get_model_registry()
    run_training_cycle(wh, registry)
    print("Training cycle complete")
