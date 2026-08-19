"""
Model registry - cung cấp cách gọi thống nhất tới 5 model bắt buộc
(SKILL.md mục 5): Logistic Regression, Decision Tree, SVM, Random Forest, XGBoost.
"""

from . import logistic_regression
from . import decision_tree
from . import svm
from . import random_forest
from . import xgboost_model

MODEL_REGISTRY = {
    "logistic_regression": logistic_regression,
    "decision_tree": decision_tree,
    "svm": svm,
    "random_forest": random_forest,
    "xgboost": xgboost_model,
}


def build_all_models(num_class: int = 2, overrides: dict | None = None) -> dict:
    """
    Khởi tạo cả 5 model với tham số mặc định (có thể override từng model).

    overrides: dict dạng {"random_forest": {"n_estimators": 300}, ...}
    """
    overrides = overrides or {}
    models = {}
    for name, module in MODEL_REGISTRY.items():
        kwargs = overrides.get(name, {})
        if name == "xgboost":
            models[name] = module.build_model(num_class=num_class, **kwargs)
        else:
            models[name] = module.build_model(**kwargs)
    return models
