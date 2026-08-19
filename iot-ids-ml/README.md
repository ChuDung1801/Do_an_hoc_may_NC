# IoT IDS Machine Learning

Nghiên cứu, so sánh và triển khai mô hình học máy phát hiện xâm nhập mạng IoT
sử dụng bộ dữ liệu **Edge-IIoTset**.

Cấu trúc project và toàn bộ quy tắc xử lý (chống data leakage, quy trình
đánh giá, thứ tự ưu tiên chọn model...) được xây dựng bám sát `SKILL.md`.

## 1. Cấu trúc thư mục

```text
iot-ids-ml/
│
├── data/
│   ├── raw/            # Đặt file Edge-IIoTset.csv (hoặc tương tự) vào đây
│   └── processed/       # Dữ liệu đã qua tiền xử lý (nếu muốn cache)
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_training.ipynb
│   └── 04_evaluation.ipynb
│
├── src/
│   ├── data/
│   │   ├── loader.py           # Đọc dataset + kiểm tra cột
│   │   └── preprocessing.py    # Pipeline chống leakage
│   │
│   ├── models/
│   │   ├── logistic_regression.py
│   │   ├── decision_tree.py
│   │   ├── svm.py
│   │   ├── random_forest.py
│   │   └── xgboost_model.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py          # Accuracy/Precision/Recall/F1/ROC-AUC/Time
│   │   └── visualization.py    # Confusion Matrix, ROC, Feature Importance...
│   │
│   ├── utils/
│   │   └── helpers.py          # Seed, save/load artifacts
│   │
│   └── train.py                # Entry point chạy toàn bộ pipeline
│
├── artifacts/
│   ├── model/                  # model.pkl, scaler.pkl, label_encoder.pkl, metadata.json
│   └── preprocessing/
│
├── app/
│   └── app.py                  # Web giám sát (Streamlit)
│
├── requirements.txt
└── README.md
```

## 2. Cài đặt

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Chuẩn bị dữ liệu

1. Tải Edge-IIoTset (CSV) và đặt vào `data/raw/`.
2. Kiểm tra tên cột trước khi train (bắt buộc — SKILL.md mục 15.10):

```bash
python -m src.data.loader --path data/raw/EdgeIIoTset.csv
```

Lệnh này in ra danh sách cột, kiểu dữ liệu, số giá trị null/unique, và
gợi ý (không khẳng định) cột nào có thể là nhãn. Xác nhận thủ công tên
cột nhãn thực tế (ví dụ `Attack_label` cho binary, `Attack_type` cho
multi-class) trước khi chạy bước huấn luyện.

## 4. Huấn luyện & so sánh 5 model

```bash
python -m src.train --data data/raw/EdgeIIoTset.csv --label Attack_label
```

Tuỳ chọn:

- `--smote` : cho phép áp dụng SMOTE trên train nếu phát hiện mất cân bằng lớp.
- `--test-size 0.2` : tỷ lệ tập test.
- `--output-dir artifacts` : nơi lưu model tốt nhất + bảng so sánh.

Pipeline sẽ:

1. Đọc & kiểm tra dataset.
2. Tiền xử lý (clean, missing, encode, split, scale, SMOTE-on-train-only).
3. Huấn luyện đủ 5 model: Logistic Regression, Decision Tree, SVM, Random
   Forest, XGBoost.
4. Đánh giá đầy đủ chỉ số: Accuracy, Precision, Recall, F1, ROC-AUC,
   Training Time, Prediction Time.
5. Chọn model tốt nhất theo thứ tự ưu tiên: **Recall > F1 > Precision >
   ROC-AUC > Prediction Time > Training Time** (không chỉ dựa Accuracy).
6. Lưu model + scaler + label encoder + metadata vào `artifacts/model/`.

## 5. Chạy Web giám sát

```bash
streamlit run app/app.py
```

Chức năng: upload CSV, tiền xử lý lại đúng theo pipeline đã lưu, dự đoán
Normal/Attack, thống kê tổng số mẫu / Normal / Attack / tỷ lệ Attack,
biểu đồ phân bố, bảng kết quả, tải kết quả, lịch sử dự đoán trong phiên.

## 6. Nguyên tắc quan trọng đã áp dụng trong code

- **Không leakage**: encode/scale fit trên train, transform trên test;
  SMOTE chỉ áp dụng trên train.
- **Không bịa kết quả**: mọi số liệu trong `metrics.py` đều tính từ model
  đã fit và predict thực tế, không có giá trị hard-code.
- **Chọn model không chỉ dựa Accuracy**: `select_best_model()` dùng thứ
  tự ưu tiên Recall/F1/Precision/ROC-AUC.
- **Random seed cố định** (`RANDOM_SEED = 42`) ở mọi model và split để
  tái lập kết quả.
- **Model Persistence đầy đủ**: lưu cả model, scaler, label encoder,
  metadata — không chỉ lưu model.

## 7. Việc cần làm tiếp (chưa tự động hoá, cần dữ liệu thật)

- Chạy `01_eda.ipynb` để khảo sát thực tế phân bố lớp, giá trị thiếu,
  kiểu dữ liệu — trước khi quyết định chiến lược xử lý missing/imbalance.
- Xác nhận tên cột nhãn chính xác trong file Edge-IIoTset đang dùng
  (binary `Attack_label` hay multi-class `Attack_type`).
- Sau khi có kết quả thực nghiệm, dùng `src/evaluation/visualization.py`
  để tạo biểu đồ và viết phần "So sánh kết quả" trong báo cáo — không
  trình bày số liệu giả định như kết quả thực nghiệm.
