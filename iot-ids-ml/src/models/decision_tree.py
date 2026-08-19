"""
decision_tree.py
-----------------
Model dễ giải thích, phù hợp trình bày trong báo cáo/bảo vệ đồ án.

Chú ý: max_depth cần được kiểm soát để tránh overfitting.
"""

from sklearn.tree import DecisionTreeClassifier

RANDOM_SEED = 42


def build_model(**kwargs) -> DecisionTreeClassifier:
    params = dict(
        criterion="gini",   # có thể đổi 'entropy' để so sánh Information Gain
        max_depth=None,     # nên tune, None dễ overfitting trên dataset lớn
        min_samples_split=2,
        random_state=RANDOM_SEED,
    )
    params.update(kwargs)
    return DecisionTreeClassifier(**params)


DEFAULT_PARAM_GRID = {
    "criterion": ["gini", "entropy"],
    "max_depth": [5, 10, 15, 20, None],
    "min_samples_split": [2, 5, 10],
}
