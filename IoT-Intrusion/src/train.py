"""
train.py
--------
Entry point chạy toàn bộ pipeline (SKILL.md mục 19):

Edge-IIoTset -> EDA (thủ công/notebook) -> Preprocessing ->
Train/Test -> 5 Models -> Evaluation -> Comparison -> Best Model ->
Save Model -> (sẵn sàng cho Web Monitoring)

Cách chạy:
    python -m src.train --data data/raw/EdgeIIoTset.csv --label Attack_label

Lưu ý: tên cột nhãn PHẢI được xác nhận thực tế trên file dataset
(chạy `python -m src.data.loader --path <file>` trước để kiểm tra cột).
"""

from __future__ import annotations

import argparse
import os

from src.data.loader import load_raw_dataset, inspect_columns
from src.data.preprocessing import run_preprocessing_pipeline
from src.models import build_all_models
from src.evaluation.metrics import train_and_evaluate, summarize_results, select_best_model
from src.utils.helpers import set_global_seed, save_artifacts, RANDOM_SEED


def main():
    parser = argparse.ArgumentParser(description="IoT IDS - Train & Compare Models")
    parser.add_argument("--data", type=str, required=True, help="Đường dẫn CSV Edge-IIoTset")
    parser.add_argument("--label", type=str, required=True, help="Tên cột nhãn (đã xác nhận)")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--smote", action="store_true", help="Cho phép SMOTE nếu mất cân bằng")
    parser.add_argument("--output-dir", type=str, default="artifacts")
    args = parser.parse_args()

    set_global_seed(RANDOM_SEED)

    print(f"[1/6] Đọc dataset: {args.data}")
    df = load_raw_dataset(args.data)
    print(f"Shape: {df.shape}")
    print(inspect_columns(df).to_string(index=False))

    print(f"\n[2/6] Tiền xử lý dữ liệu (label = '{args.label}')")
    result = run_preprocessing_pipeline(
        df,
        label_col=args.label,
        test_size=args.test_size,
        do_scale=True,
        apply_smote=args.smote,
    )

    n_classes = len(result.label_encoder.classes_)
    average = "binary" if n_classes == 2 else "macro"
    print(f"Số lớp: {n_classes} ({list(result.label_encoder.classes_)}), average metric = '{average}'")

    print("\n[3/6] Khởi tạo 5 model bắt buộc")
    models = build_all_models(num_class=n_classes)

    print("\n[4/6] Huấn luyện + đánh giá từng model")
    all_results = []
    for name, model in models.items():
        print(f"  -> Training {name} ...")
        eval_result = train_and_evaluate(
            model, name,
            result.X_train, result.y_train,
            result.X_test, result.y_test,
            average=average,
        )
        all_results.append(eval_result)
        print(f"     Accuracy={eval_result.accuracy:.4f} "
              f"Recall={eval_result.recall:.4f} F1={eval_result.f1:.4f} "
              f"ROC-AUC={eval_result.roc_auc}")

    print("\n[5/6] So sánh kết quả (ưu tiên Recall > F1 > Precision > ROC-AUC)")
    summary_df = summarize_results(all_results)
    print(summary_df.to_string(index=False))

    best = select_best_model(all_results)
    print(f"\n>>> Model tốt nhất theo tiêu chí ưu tiên: {best.model_name}")
    print(best.report)

    print("\n[6/6] Lưu model tốt nhất + preprocessing artifacts")
    best_model_obj = models[best.model_name]
    save_artifacts(
        output_dir=os.path.join(args.output_dir, "model"),
        model=best_model_obj,
        model_name=best.model_name,
        scaler=result.scaler,
        label_encoder=result.label_encoder,
        feature_columns=result.feature_columns,
        extra_metadata={"metrics": best.to_dict()},
    )

    summary_path = os.path.join(args.output_dir, "model_comparison.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Đã lưu bảng so sánh model vào: {summary_path}")


if __name__ == "__main__":
    main()
