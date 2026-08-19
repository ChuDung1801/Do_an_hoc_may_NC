"""
logistic_regression.py
-----------------------
Baseline model cho bài toán Normal vs Attack.

Cần scaling dữ liệu trước khi huấn luyện (đã xử lý ở preprocessing.py).
"""

from sklearn.linear_model import LogisticRegression

RANDOM_SEED = 42


def build_model(**kwargs) -> LogisticRegression:
    """
    Trả về Logistic Regression model với seed cố định để tái lập kết quả.

    Tham số mặc định phù hợp cho binary classification (Normal/Attack).
    Có thể override qua kwargs, ví dụ: build_model(C=0.5, max_iter=500)
    """
    params = dict(
        max_iter=1000,
        random_state=RANDOM_SEED,
        class_weight=None,  # đổi thành 'balanced' nếu dữ liệu mất cân bằng
    )
    params.update(kwargs)
    return LogisticRegression(**params)


DEFAULT_PARAM_GRID = {
    "C": [0.01, 0.1, 1, 10],
    "penalty": ["l2"],
    "solver": ["lbfgs"],
}
