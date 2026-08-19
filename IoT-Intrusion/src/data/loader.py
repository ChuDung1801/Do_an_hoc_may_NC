"""
loader.py
---------
Đọc dữ liệu Edge-IIoTset (hoặc dataset IoT tương tự dạng CSV).

Quy tắc áp dụng (theo SKILL.md mục 15):
- Không tự giả định tên cột / nhãn nếu chưa kiểm tra file thực tế.
- Phải kiểm tra tên cột trước khi xử lý ở bất kỳ bước nào phía sau.
"""

from __future__ import annotations

import os
import pandas as pd


class DatasetLoadError(Exception):
    pass


def load_raw_dataset(path: str) -> pd.DataFrame:
    """
    Đọc raw dataset từ CSV.

    Parameters
    ----------
    path : str
        Đường dẫn tới file CSV (ví dụ: data/raw/EdgeIIoTset.csv)

    Returns
    -------
    pd.DataFrame
    """
    if not os.path.exists(path):
        raise DatasetLoadError(
            f"Không tìm thấy file dataset tại: {path}. "
            f"Hãy đặt file Edge-IIoTset (.csv) vào thư mục data/raw/ trước khi chạy."
        )

    df = pd.read_csv(path, low_memory=False)

    if df.empty:
        raise DatasetLoadError("Dataset rỗng sau khi đọc, kiểm tra lại file gốc.")

    return df


def inspect_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trả về bảng tổng quan cột: tên cột, dtype, số giá trị null, số giá trị unique.
    Bước bắt buộc trước khi giả định bất kỳ tên cột nhãn / feature nào.
    """
    info = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "n_null": [df[c].isna().sum() for c in df.columns],
        "n_unique": [df[c].nunique(dropna=True) for c in df.columns],
    })
    return info.reset_index(drop=True)


def guess_label_columns(df: pd.DataFrame) -> list[str]:
    """
    Gợi ý (KHÔNG khẳng định) các cột có khả năng là nhãn, dựa trên tên cột
    thường gặp trong Edge-IIoTset (vd: 'Attack_label', 'Attack_type').

    Đây chỉ là gợi ý để người dùng xác nhận thủ công, agent không được
    tự động dùng kết quả này để huấn luyện mà chưa xác nhận.
    """
    candidates = []
    keywords = ["label", "attack", "class", "target"]
    for col in df.columns:
        col_lower = col.lower()
        if any(k in col_lower for k in keywords):
            candidates.append(col)
    return candidates


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Kiểm tra nhanh dataset Edge-IIoTset")
    parser.add_argument("--path", type=str, required=True, help="Đường dẫn file CSV")
    args = parser.parse_args()

    df = load_raw_dataset(args.path)
    print(f"Shape: {df.shape}")
    print("\n=== Column overview ===")
    print(inspect_columns(df).to_string(index=False))
    print("\n=== Gợi ý cột nhãn (cần xác nhận thủ công) ===")
    print(guess_label_columns(df))
