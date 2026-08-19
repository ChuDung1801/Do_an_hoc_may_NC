"""
preprocessing.py
-----------------
Pipeline tiền xử lý dữ liệu Edge-IIoTset, tuân thủ nghiêm ngặt quy tắc
chống Data Leakage trong SKILL.md (mục 4 & 7):

    Dataset -> Train/Test Split -> Fit Scaler/Encoder trên Train ->
    Transform Train & Test -> SMOTE chỉ trên Train

KHÔNG được: fit scaler/encoder hoặc chạy SMOTE trên toàn bộ dataset
trước khi chia train/test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


RANDOM_SEED = 42


@dataclass
class PreprocessResult:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    scaler: Optional[StandardScaler]
    label_encoder: LabelEncoder
    feature_columns: list


def check_required_columns(df: pd.DataFrame, label_col: str) -> None:
    """Bắt buộc kiểm tra cột nhãn tồn tại trước khi xử lý gì khác."""
    if label_col not in df.columns:
        raise ValueError(
            f"Cột nhãn '{label_col}' không tồn tại trong dataset. "
            f"Các cột hiện có: {list(df.columns)}"
        )


def drop_duplicates_and_report(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"[Data Cleaning] Removed {before - after} duplicate rows "
          f"({before} -> {after})")
    return df


def handle_missing_values(df: pd.DataFrame, strategy: str = "drop") -> pd.DataFrame:
    """
    strategy:
        'drop'   -> loại bỏ dòng có giá trị thiếu
        'median' -> điền median cho numeric, mode cho categorical
    """
    n_missing = df.isna().sum().sum()
    print(f"[Missing Value] Tổng số ô thiếu dữ liệu: {n_missing}")

    if n_missing == 0:
        return df

    if strategy == "drop":
        return df.dropna()

    if strategy == "median":
        df = df.copy()
        for col in df.columns:
            if df[col].isna().sum() == 0:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                mode = df[col].mode(dropna=True)
                df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "unknown")
        return df

    raise ValueError(f"Unknown missing-value strategy: {strategy}")


def encode_categorical_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    One-hot encode categorical columns. Encoder logic được 'học' cấu trúc
    cột chỉ từ train, sau đó align lại test để tránh leakage / lệch cột.
    """
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    if not cat_cols:
        return X_train, X_test

    print(f"[Encoding] Categorical columns: {cat_cols}")

    X_train_enc = pd.get_dummies(X_train, columns=cat_cols, dummy_na=False)
    X_test_enc = pd.get_dummies(X_test, columns=cat_cols, dummy_na=False)

    # Align test columns to train columns (tránh lộ thông tin, chỉ đồng bộ schema)
    X_train_enc, X_test_enc = X_train_enc.align(
        X_test_enc, join="left", axis=1, fill_value=0
    )
    return X_train_enc, X_test_enc


def scale_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit StandardScaler CHỈ trên train, transform cả train và test."""
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    return X_train_scaled, X_test_scaled, scaler


def apply_smote_on_train(
    X_train: pd.DataFrame, y_train: np.ndarray, random_state: int = RANDOM_SEED
):
    """
    Áp dụng SMOTE CHỈ trên tập train. KHÔNG BAO GIỜ áp dụng cho test.
    Chỉ gọi hàm này sau khi đã kiểm tra mất cân bằng lớp (xem check_class_balance).
    """
    from imblearn.over_sampling import SMOTE

    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"[SMOTE] Train size before: {len(y_train)}, after: {len(y_res)}")
    return X_res, y_res


def check_class_balance(y, threshold: float = 0.3) -> bool:
    """
    Trả về True nếu dữ liệu bị mất cân bằng đáng kể (lớp thiểu số < threshold
    tỉ lệ so với lớp đa số). Agent phải gọi hàm này TRƯỚC KHI đề xuất SMOTE,
    không tự động áp dụng SMOTE nếu chưa kiểm tra.
    """
    values, counts = np.unique(y, return_counts=True)
    ratio = counts.min() / counts.max()
    print(f"[Class Distribution] {dict(zip(values, counts))}, "
          f"minority/majority ratio = {ratio:.3f}")
    return ratio < threshold


def run_preprocessing_pipeline(
    df: pd.DataFrame,
    label_col: str,
    test_size: float = 0.2,
    do_scale: bool = True,
    apply_smote: bool = False,
    missing_strategy: str = "drop",
    random_state: int = RANDOM_SEED,
) -> PreprocessResult:
    """
    Pipeline đầy đủ, tuân thủ thứ tự bắt buộc trong SKILL.md mục 4:

    Raw -> Inspect -> Clean -> Missing -> Duplicate -> (Encode) ->
    Train/Test Split -> Scaling (fit train) -> SMOTE (chỉ train)
    """
    check_required_columns(df, label_col)

    df = drop_duplicates_and_report(df)
    df = handle_missing_values(df, strategy=missing_strategy)

    y_raw = df[label_col]
    X_raw = df.drop(columns=[label_col])

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    # Train/Test split TRƯỚC khi encode/scale/SMOTE để tránh leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train, X_test = encode_categorical_features(X_train, X_test)

    scaler = None
    if do_scale:
        X_train, X_test, scaler = scale_features(X_train, X_test)

    if apply_smote:
        is_imbalanced = check_class_balance(y_train)
        if is_imbalanced:
            X_train, y_train = apply_smote_on_train(X_train, y_train, random_state)
        else:
            print("[SMOTE] Bỏ qua vì phân bố lớp chưa mất cân bằng nghiêm trọng.")

    return PreprocessResult(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        scaler=scaler,
        label_encoder=label_encoder,
        feature_columns=list(X_train.columns),
    )
