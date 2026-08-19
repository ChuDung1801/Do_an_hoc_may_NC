"""
visualization.py
-----------------
Bộ biểu đồ tối thiểu (SKILL.md mục 11):
Confusion Matrix, ROC Curve, Feature Importance, Model Comparison.

Ngoài ra hỗ trợ thêm: Precision-Recall Curve, Class Distribution,
Attack Distribution, Training/Prediction Time Comparison.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, ConfusionMatrixDisplay


def plot_confusion_matrix(cm: np.ndarray, labels: list, title: str = "Confusion Matrix", ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    return ax


def plot_roc_curve(y_true, y_proba_positive, model_name: str = "", ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    fpr, tpr, _ = roc_curve(y_true, y_proba_positive)
    ax.plot(fpr, tpr, label=model_name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    return ax


def plot_precision_recall_curve(y_true, y_proba_positive, model_name: str = "", ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    precision, recall, _ = precision_recall_curve(y_true, y_proba_positive)
    ax.plot(recall, precision, label=model_name)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    return ax


def plot_class_distribution(y, labels: list | None = None, ax=None, title="Class Distribution"):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    values, counts = np.unique(y, return_counts=True)
    x_labels = labels if labels is not None else values
    ax.bar([str(v) for v in x_labels], counts)
    ax.set_title(title)
    ax.set_ylabel("Số lượng mẫu")
    return ax


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 15, ax=None,
                             title: str = "Feature Importance"):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    top = importance_df.head(top_n).iloc[::-1]
    ax.barh(top["feature"], top["importance"])
    ax.set_title(title)
    ax.set_xlabel("Importance")
    return ax


def plot_model_comparison(summary_df: pd.DataFrame, metric: str = "f1", ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(summary_df["model_name"], summary_df[metric])
    ax.set_title(f"Model Comparison - {metric}")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=30)
    return ax


def plot_time_comparison(summary_df: pd.DataFrame, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(summary_df))
    width = 0.35
    ax.bar(x - width / 2, summary_df["training_time_sec"], width, label="Training Time")
    ax.bar(x + width / 2, summary_df["prediction_time_sec"], width, label="Prediction Time")
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["model_name"], rotation=30)
    ax.set_ylabel("Seconds")
    ax.set_title("Training vs Prediction Time")
    ax.legend()
    return ax
