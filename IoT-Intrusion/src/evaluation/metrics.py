"""
metrics.py
----------
Tính toán các chỉ số đánh giá bắt buộc (SKILL.md mục 6):
Accuracy, Precision, Recall, F1-score, ROC-AUC, Training Time, Prediction Time.

QUY TẮC: Không hard-code hoặc bịa kết quả. Mọi số liệu phải xuất phát
từ model đã được fit và predict trên dữ liệu thực tế.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


@dataclass
class EvalResult:
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    training_time_sec: float
    prediction_time_sec: float
    confusion_matrix: np.ndarray
    n_false_negative: int
    n_false_positive: int
    report: str = field(repr=False, default="")

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "training_time_sec": self.training_time_sec,
            "prediction_time_sec": self.prediction_time_sec,
            "n_false_negative": self.n_false_negative,
            "n_false_positive": self.n_false_positive,
        }


def train_and_evaluate(
    model,
    model_name: str,
    X_train,
    y_train,
    X_test,
    y_test,
    average: str = "binary",
) -> EvalResult:
    """
    Huấn luyện model, đo thời gian, và tính đầy đủ các chỉ số đánh giá
    trên tập test thực tế (không dùng lại train).

    average: 'binary' cho Normal/Attack, 'macro' hoặc 'weighted' cho multi-class.
    """
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    prediction_time = time.perf_counter() - t0

    roc_auc = None
    try:
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
            if average == "binary" and y_proba.shape[1] == 2:
                roc_auc = roc_auc_score(y_test, y_proba[:, 1])
            else:
                roc_auc = roc_auc_score(
                    y_test, y_proba, multi_class="ovr", average="macro"
                )
    except Exception as e:  # pragma: no cover
        print(f"[metrics] Không tính được ROC-AUC cho {model_name}: {e}")

    cm = confusion_matrix(y_test, y_pred)

    # False Negative / False Positive chỉ có ý nghĩa rõ ràng ở binary.
    # Với binary [ [TN, FP], [FN, TP] ]
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        fn, fp = -1, -1  # multi-class: xem chi tiết trong confusion matrix đầy đủ

    result = EvalResult(
        model_name=model_name,
        accuracy=accuracy_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred, average=average, zero_division=0),
        recall=recall_score(y_test, y_pred, average=average, zero_division=0),
        f1=f1_score(y_test, y_pred, average=average, zero_division=0),
        roc_auc=roc_auc,
        training_time_sec=training_time,
        prediction_time_sec=prediction_time,
        confusion_matrix=cm,
        n_false_negative=int(fn),
        n_false_positive=int(fp),
        report=classification_report(y_test, y_pred, zero_division=0),
    )
    return result


def summarize_results(results: list[EvalResult]):
    """Trả về DataFrame tổng hợp để so sánh nhiều model (mục 10, 16)."""
    import pandas as pd

    rows = [r.to_dict() for r in results]
    df = pd.DataFrame(rows)
    return df.sort_values(
        by=["recall", "f1", "precision", "roc_auc"], ascending=False
    ).reset_index(drop=True)


def select_best_model(
    results: list[EvalResult],
    priority: tuple[str, ...] = ("recall", "f1", "precision", "roc_auc"),
) -> EvalResult:
    """
    Chọn model tốt nhất theo thứ tự ưu tiên mặc định trong SKILL.md mục 10:
    Recall > F1 > Precision > ROC-AUC > Prediction Time > Training Time.

    KHÔNG chọn chỉ dựa vào Accuracy.
    """
    def sort_key(r: EvalResult):
        values = []
        for p in priority:
            v = getattr(r, p)
            values.append(v if v is not None else -1)
        # thời gian: thấp hơn tốt hơn -> đảo dấu
        values.append(-r.prediction_time_sec)
        values.append(-r.training_time_sec)
        return tuple(values)

    return max(results, key=sort_key)
