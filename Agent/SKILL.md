---
name: iot-ids-ml
description: Hỗ trợ xây dựng, huấn luyện, đánh giá và triển khai mô hình học máy phát hiện xâm nhập mạng IoT từ bộ dữ liệu Edge-IIoTset. Sử dụng skill này khi người dùng yêu cầu phân tích dataset Edge-IIoTset, xây dựng pipeline tiền xử lý, huấn luyện/so sánh các mô hình (Logistic Regression, Decision Tree, SVM, Random Forest, XGBoost) cho bài toán IDS, đánh giá mô hình bằng các chỉ số Precision/Recall/F1/ROC-AUC, xử lý mất cân bằng dữ liệu (SMOTE), tránh data leakage, hoặc triển khai giao diện Streamlit/Flask để dự đoán Normal/Attack.
---

# IoT IDS ML

Skill hỗ trợ agent thực hiện toàn bộ vòng đời của đề tài "Nghiên cứu, so sánh và triển khai mô hình giám sát phát hiện xâm nhập mạng IoT" trên bộ dữ liệu Edge-IIoTset — từ phân tích dữ liệu, tiền xử lý, huấn luyện, đánh giá, đến triển khai.

## Instructions

Khi thực hiện nhiệm vụ liên quan đến đề tài này, luôn tuân theo các nguyên tắc sau:

- Phân tích dữ liệu Edge-IIoTset và xác định các thuộc tính đầu vào, nhãn và loại dữ liệu.
- Thực hiện tiền xử lý gồm xử lý missing values, duplicate, encoding và scaling.
- Kiểm tra mất cân bằng dữ liệu và đề xuất SMOTE khi phù hợp.
- Chia dữ liệu thành train/test **trước khi** thực hiện các bước có nguy cơ gây data leakage (scaling, SMOTE, ...).
- Huấn luyện và so sánh 5 mô hình: Logistic Regression, Decision Tree, SVM, Random Forest và XGBoost.
- Đánh giá bằng Accuracy, Precision, Recall, F1-score, ROC-AUC, confusion matrix, thời gian train và thời gian prediction.
- Xác định mô hình tốt nhất dựa trên F1-score, Recall và thời gian dự đoán — **không chỉ dựa vào Accuracy**.
- Phân tích feature importance và đề xuất SHAP khi cần giải thích mô hình.
- Lưu model, scaler và encoder để tái sử dụng khi inference.
- Hỗ trợ tạo API hoặc giao diện Streamlit/Flask để upload dữ liệu và trả về Normal/Attack.
- Khi giải thích kết quả, luôn phân biệt False Positive và False Negative, đặc biệt chú ý False Negative trong bài toán IDS (bỏ sót tấn công nguy hiểm hơn báo động giả).
- Không tự bịa kết quả thực nghiệm; nếu chưa chạy model thì phải nói rõ đó là dự đoán hoặc ví dụ minh họa, không phải số liệu thật.

## Workflow

1. **Kiểm tra dataset** — xác nhận file tồn tại, đọc cấu trúc, số dòng/cột, kiểu dữ liệu.
2. **Khám phá dữ liệu (EDA)** — phân bố nhãn, missing values, duplicate, các cột categorical.
3. **Tiền xử lý** — xử lý missing/duplicate, encoding (Label/One-Hot), fit scaler trên tập train.
4. **Chia train/test** — thực hiện trước mọi bước có nguy cơ leakage.
5. **Huấn luyện 5 model** — Logistic Regression, Decision Tree, SVM, Random Forest, XGBoost.
6. **Đánh giá và so sánh** — confusion matrix, Precision/Recall/F1/ROC-AUC, thời gian train/predict.
7. **Chọn model tốt nhất** — ưu tiên Recall/F1 cho bài toán IDS, giải thích ngắn gọn lý do lựa chọn.
8. **Lưu pipeline** — model, scaler, encoder (ví dụ `.pkl`) để tái sử dụng.
9. **Triển khai dashboard** — giao diện Streamlit/Flask cho phép upload CSV và dự đoán Normal/Attack.

## Quy tắc bắt buộc

1. Không để xảy ra data leakage (không fit scaler/SMOTE trên toàn bộ dataset trước khi chia train/test).
2. Không chỉ dùng Accuracy để kết luận mô hình tốt nhất.
3. Với IDS, ưu tiên phân tích Recall và False Negative hơn là chỉ nhìn tổng thể.
4. Không tự tạo số liệu kết quả khi chưa thực sự chạy mô hình.
5. Luôn giải thích ngắn gọn lý do lựa chọn mô hình cuối cùng.

## Ví dụ nhiệm vụ giao cho agent

```text
Bạn là AI Agent chuyên về Machine Learning cho hệ thống IDS mạng IoT.

Nhiệm vụ:
- Phân tích file Edge-IIoTset.csv.
- Xác định cột label và các feature.
- Kiểm tra missing, duplicate, categorical feature và class imbalance.
- Xây dựng pipeline preprocessing.
- Huấn luyện Logistic Regression, Decision Tree, SVM, Random Forest và XGBoost.
- So sánh Accuracy, Precision, Recall, F1, ROC-AUC, training time và prediction time.
- Vẽ confusion matrix và ROC curve.
- Chọn mô hình phù hợp nhất cho hệ thống IDS.
- Lưu mô hình và preprocessing pipeline.
- Tạo giao diện Streamlit cho phép upload CSV và dự đoán Normal/Attack.

Quy tắc:
1. Không để xảy ra data leakage.
2. Không chỉ dùng Accuracy để kết luận.
3. Với IDS, ưu tiên phân tích Recall và False Negative.
4. Không tự tạo số liệu kết quả khi chưa thực sự chạy mô hình.
5. Giải thích ngắn gọn lý do lựa chọn mô hình cuối cùng.
```
