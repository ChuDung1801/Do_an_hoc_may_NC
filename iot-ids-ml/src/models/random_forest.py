"""
random_forest.py
-----------------
Ensemble Learning - Bagging nhiều Decision Tree, kết quả tổng hợp
bằng voting/averaging. Hỗ trợ feature_importances_ để phân tích
Feature Importance (mục 9 SKILL.md).
"""

from sklearn.ensemble import RandomForestClassifier

RANDOM_SEED = 42


def build_model(**kwargs) -> RandomForestClassifier:
    params = dict(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        max_features="sqrt",
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    params.update(kwargs)
    return RandomForestClassifier(**params)


def get_feature_importance(model: RandomForestClassifier, feature_names: list):
    """Trả về DataFrame Feature Importance sắp xếp giảm dần."""
    import pandas as pd

    importances = model.feature_importances_
    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


DEFAULT_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5, 10],
    "max_features": ["sqrt", "log2"],
}
