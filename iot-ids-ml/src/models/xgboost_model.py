"""
xgboost_model.py
-----------------
Gradient Boosting: xây dựng các cây TUẦN TỰ, mỗi cây sau học để sửa lỗi
(residual) của các cây trước, khác với Random Forest (các cây độc lập,
bagging + voting). Không mô tả XGBoost đơn giản là "nhiều cây cùng vote".
"""

from xgboost import XGBClassifier

RANDOM_SEED = 42


def build_model(num_class: int | None = None, **kwargs) -> XGBClassifier:
    """
    num_class: nếu > 2 (multi-class), tự động chuyển objective phù hợp.
    """
    if num_class is not None and num_class > 2:
        objective = "multi:softprob"
        extra = {"num_class": num_class}
    else:
        objective = "binary:logistic"
        extra = {}

    params = dict(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective=objective,
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    params.update(extra)
    params.update(kwargs)
    return XGBClassifier(**params)


def get_feature_importance(model: XGBClassifier, feature_names: list):
    import pandas as pd

    importances = model.feature_importances_
    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


DEFAULT_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 6, 9],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
}
