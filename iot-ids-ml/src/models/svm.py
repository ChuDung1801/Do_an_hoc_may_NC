"""
svm.py
------
Support Vector Machine. Bắt buộc dữ liệu đã được scale trước khi train
(SVM rất nhạy với thang đo feature).

Chú ý: với dataset IoT lớn, SVM (đặc biệt kernel RBF) có thể chậm.
Cân nhắc lấy mẫu (sampling) khi thử nghiệm nhanh, nhưng báo cáo cuối
cùng nên chạy trên toàn bộ / tập đại diện đủ lớn.
"""

from sklearn.svm import SVC

RANDOM_SEED = 42


def build_model(**kwargs) -> SVC:
    params = dict(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=True,  # cần True để tính ROC-AUC / predict_proba
        random_state=RANDOM_SEED,
    )
    params.update(kwargs)
    return SVC(**params)


DEFAULT_PARAM_GRID = {
    "C": [0.1, 1, 10],
    "gamma": ["scale", "auto"],
    "kernel": ["rbf", "linear"],
}
