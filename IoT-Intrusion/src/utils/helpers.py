"""
helpers.py
----------
Tiện ích chung: set random seed toàn cục, lưu/tải model + preprocessing
artifacts (SKILL.md mục 12 - Model Persistence).
"""

from __future__ import annotations

import json
import os
import random
from datetime import datetime, timezone

import joblib
import numpy as np

RANDOM_SEED = 42


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    """Đặt seed cho random, numpy để đảm bảo tái lập kết quả."""
    random.seed(seed)
    np.random.seed(seed)


def save_artifacts(
    output_dir: str,
    model,
    model_name: str,
    scaler=None,
    encoder=None,
    label_encoder=None,
    feature_columns: list | None = None,
    extra_metadata: dict | None = None,
) -> None:
    """
    Lưu đầy đủ: model + scaler + encoder + label encoder + metadata.
    KHÔNG chỉ lưu model mà bỏ preprocessing (theo mục 12 SKILL.md).
    """
    os.makedirs(output_dir, exist_ok=True)

    joblib.dump(model, os.path.join(output_dir, "model.pkl"))

    if scaler is not None:
        joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))
    if encoder is not None:
        joblib.dump(encoder, os.path.join(output_dir, "encoder.pkl"))
    if label_encoder is not None:
        joblib.dump(label_encoder, os.path.join(output_dir, "label_encoder.pkl"))

    metadata = {
        "model_name": model_name,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_columns": feature_columns or [],
        "random_seed": RANDOM_SEED,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"[Persistence] Đã lưu artifacts vào: {output_dir}")


def load_artifacts(input_dir: str) -> dict:
    """Tải lại model + preprocessing artifacts để inference."""
    artifacts = {}

    model_path = os.path.join(input_dir, "model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy model.pkl trong {input_dir}")
    artifacts["model"] = joblib.load(model_path)

    for key, filename in [
        ("scaler", "scaler.pkl"),
        ("encoder", "encoder.pkl"),
        ("label_encoder", "label_encoder.pkl"),
    ]:
        path = os.path.join(input_dir, filename)
        artifacts[key] = joblib.load(path) if os.path.exists(path) else None

    metadata_path = os.path.join(input_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            artifacts["metadata"] = json.load(f)
    else:
        artifacts["metadata"] = {}

    return artifacts
