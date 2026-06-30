import numpy as np
import pandas as pd
import shap


def compute_feature_importance(model, X_test: pd.DataFrame) -> dict[str, float]:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    importance = np.abs(shap_values).mean(axis=0)
    return dict(zip(X_test.columns, importance.tolist()))
