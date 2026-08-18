---
name: iot-ids-ml
description: >
  Hỗ trợ nghiên cứu, xây dựng, so sánh và triển khai mô hình học máy
  phát hiện xâm nhập mạng IoT sử dụng bộ dữ liệu Edge-IIoTset.
  Skill này tập trung vào preprocessing, supervised learning,
  đánh giá mô hình, lựa chọn mô hình tối ưu và tích hợp hệ thống
  giám sát Web bằng Streamlit hoặc Flask.
---

# IoT IDS Machine Learning Skill

## 1. Mục tiêu

Skill này hỗ trợ xây dựng hệ thống:

> **Nghiên cứu, so sánh và triển khai mô hình giám sát phát hiện xâm nhập mạng IoT**

Đầu vào chính là dữ liệu mạng IoT từ **Edge-IIoTset**.

Đầu ra gồm:

- Mô hình phát hiện `Normal` / `Attack`.
- Kết quả đánh giá và so sánh nhiều mô hình.
- Mô hình tốt nhất được lưu để inference.
- Giao diện Web giám sát và dự đoán.
- Thống kê và trực quan hóa kết quả.

---

## 2. Phạm vi

### Bài toán chính

Bài toán mặc định là:

```text
Normal vs Attack
```

Có thể mở rộng thành:

```text
Normal
Attack Type 1
Attack Type 2
Attack Type 3
...
```

Khi người dùng không yêu cầu cụ thể, ưu tiên **Binary Classification** trước.

### Dataset

Dataset mặc định:

```text
Edge-IIoTset
```

Không tự giả định tên cột hoặc nhãn cụ thể nếu chưa kiểm tra file dataset thực tế.

---

# 3. Kiến thức cần áp dụng

## 3.1. IoT Network Security

Agent phải hiểu các khái niệm:

* IoT Network
* Network Traffic
* Packet
* Flow
* Feature
* Intrusion Detection System (IDS)
* Normal Traffic
* Attack Traffic
* False Positive
* False Negative

Đặc biệt:

> False Negative trong IDS thường nghiêm trọng vì hệ thống bỏ sót một cuộc tấn công thực tế.

---

## 3.2. Machine Learning

Ưu tiên:

* Supervised Learning
* Classification
* Binary Classification
* Multi-class Classification
* Ensemble Learning
* Model Evaluation
* Hyperparameter Tuning

---

# 4. Pipeline xử lý dữ liệu

Luôn ưu tiên pipeline:

```text
Raw Dataset
    ↓
Data Inspection
    ↓
Data Cleaning
    ↓
Missing Value Handling
    ↓
Duplicate Removal
    ↓
Categorical Encoding
    ↓
Train/Test Split
    ↓
Scaling nếu cần
    ↓
SMOTE trên Train nếu cần
    ↓
Model Training
    ↓
Evaluation
    ↓
Model Selection
    ↓
Model Saving
```

## Quy tắc chống Data Leakage

Không được thực hiện preprocessing sử dụng toàn bộ dataset trước khi chia train/test nếu bước đó làm lộ thông tin của tập test.

Đặc biệt:

```text
Không:
Dataset
   ↓
SMOTE
   ↓
Train/Test Split
```

Phải:

```text
Dataset
   ↓
Train/Test Split
   ↓
SMOTE chỉ trên Train
   ↓
Training
```

Scaler và Encoder phải được `fit` trên training data và chỉ `transform` trên test data.

---

# 5. Các mô hình bắt buộc

Agent phải hỗ trợ tối thiểu 5 mô hình:

```text
1. Logistic Regression
2. Decision Tree
3. Support Vector Machine
4. Random Forest
5. XGBoost
```

## 5.1. Logistic Regression

Vai trò:

> Baseline model.

Cần chú ý:

* Logistic/Sigmoid
* Binary classification
* Regularization
* Scaling

Không cần ưu tiên model này nếu người dùng yêu cầu tối ưu hiệu năng.

---

## 5.2. Decision Tree

Cần chú ý:

* Entropy
* Gini
* Information Gain
* Splitting
* max_depth
* Overfitting

Đây là model dễ giải thích trong báo cáo và bảo vệ.

---

## 5.3. SVM

Cần chú ý:

* Hyperplane
* Margin
* Support Vector
* Kernel
* C
* Gamma
* RBF

SVM thường cần scaling dữ liệu.

---

## 5.4. Random Forest

Cần hiểu:

* Ensemble Learning
* Bagging
* Multiple Decision Trees
* Voting
* Feature Importance
* Overfitting

Hyperparameter chính:

```text
n_estimators
max_depth
min_samples_split
max_features
```

---

## 5.5. XGBoost

Ưu tiên giải thích sâu nhất.

Cần hiểu:

* Gradient Boosting
* Sequential Trees
* Loss Function
* Learning Rate
* Regularization
* Overfitting

Hyperparameter chính:

```text
n_estimators
learning_rate
max_depth
subsample
colsample_bytree
```

Không được mô tả XGBoost đơn giản là "nhiều cây cùng vote" vì đó gần với Random Forest hơn.

---

# 6. Đánh giá mô hình

Luôn tính tối thiểu:

```text
Accuracy
Precision
Recall
F1-score
ROC-AUC
Training Time
Prediction Time
```

## Confusion Matrix

Phải phân biệt:

```text
TP
TN
FP
FN
```

Trong IDS đặc biệt chú ý:

```text
False Negative
```

vì đây là trường hợp:

```text
Attack thực tế
      ↓
Model dự đoán
      ↓
Normal
```

---

# 7. Xử lý mất cân bằng dữ liệu

Agent phải kiểm tra phân bố class trước khi đề xuất SMOTE.

Quy trình:

```text
Check Class Distribution
        ↓
Có imbalance?
   ┌────┴────┐
   │         │
  Không      Có
   │         │
Train      SMOTE
             ↓
        Chỉ áp dụng Train
```

Không áp dụng SMOTE cho Test.

Ngoài SMOTE có thể xem xét:

* `class_weight`
* Random Under-sampling
* Over-sampling

Không tự động áp dụng mọi kỹ thuật cùng lúc.

---

# 8. Cross Validation

Khi cần đánh giá ổn định hoặc tuning:

```text
5-Fold Cross Validation
```

Có thể dùng:

```python
StratifiedKFold
```

Ưu tiên Stratified Cross Validation cho classification để giữ tỷ lệ class giữa các fold.

---

# 9. Feature Importance và Explainability

Đối với Random Forest và XGBoost, ưu tiên phân tích:

* Feature Importance
* Top features
* Feature contribution

Nếu cần giải thích sâu:

```text
SHAP
```

Mục tiêu:

> Xác định những đặc trưng mạng ảnh hưởng mạnh đến quyết định Normal/Attack của mô hình.

Không tự kết luận feature quan trọng trước khi chạy thực nghiệm.

---

# 10. Model Selection

Không chọn model chỉ dựa vào Accuracy.

Thứ tự ưu tiên mặc định:

```text
Recall
F1-score
Precision
ROC-AUC
Prediction Time
Training Time
```

Trong IDS:

> Recall và False Negative cần được xem xét đặc biệt.

Tuy nhiên phải căn cứ vào kết quả thực tế.

Không được tự tạo hoặc dự đoán kết quả Accuracy/F1 khi chưa chạy model.

---

# 11. Visualization

Khi đánh giá mô hình nên hỗ trợ:

```text
Confusion Matrix
ROC Curve
Precision-Recall Curve
Class Distribution
Feature Importance
Attack Distribution
Model Comparison
Training Time Comparison
Prediction Time Comparison
```

Nếu chỉ cần bộ biểu đồ tối thiểu:

```text
Confusion Matrix
ROC Curve
Feature Importance
Model Comparison
```

---

# 12. Model Persistence

Sau khi chọn model tốt nhất phải lưu:

```text
model
scaler
encoder
label encoder
```

Ưu tiên sử dụng một pipeline nếu phù hợp.

Ví dụ:

```text
artifacts/
├── model.pkl
├── scaler.pkl
├── encoder.pkl
└── metadata.json
```

Không chỉ lưu model mà bỏ preprocessing.

---

# 13. Web Monitoring

Có thể triển khai bằng:

```text
Streamlit
```

hoặc:

```text
Flask
```

Nếu người dùng không yêu cầu backend API riêng:

> Ưu tiên Streamlit vì phù hợp với ứng dụng demo và trực quan hóa kết quả.

Chức năng tối thiểu:

```text
Upload CSV
    ↓
Preprocessing
    ↓
Prediction
    ↓
Normal / Attack
    ↓
Statistics
    ↓
Charts
```

Giao diện nên có:

* Upload dataset
* Nút Predict
* Tổng số mẫu
* Số Normal
* Số Attack
* Tỷ lệ Attack
* Bảng kết quả
* Biểu đồ phân bố
* Lịch sử dự đoán nếu có yêu cầu

---

# 14. Cấu trúc project đề xuất

```text
iot-ids-ml/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_training.ipynb
│   └── 04_evaluation.ipynb
│
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   └── preprocessing.py
│   │
│   ├── models/
│   │   ├── logistic_regression.py
│   │   ├── decision_tree.py
│   │   ├── svm.py
│   │   ├── random_forest.py
│   │   └── xgboost_model.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── visualization.py
│   │
│   └── utils/
│       └── helpers.py
│
├── artifacts/
│   ├── model/
│   └── preprocessing/
│
├── app/
│   └── app.py
│
├── requirements.txt
└── README.md
```

---

# 15. Quy tắc khi sinh code

Khi được yêu cầu viết code:

1. Ưu tiên Python.
2. Sử dụng Pandas/NumPy cho xử lý dữ liệu.
3. Sử dụng Scikit-learn cho các model truyền thống.
4. Sử dụng XGBoost cho XGBoost.
5. Code phải có random seed để tái lập kết quả.
6. Không hard-code kết quả đánh giá.
7. Không làm leakage từ test sang train.
8. Không tự động loại bỏ feature chỉ vì tên của nó; phải phân tích dataset.
9. Phải lưu preprocessing cùng model.
10. Khi code chạy trên dataset thực tế, phải kiểm tra tên cột trước khi xử lý.

---

# 16. Quy tắc khi phân tích kết quả

Mỗi lần so sánh model phải trả lời:

```text
Model nào tốt nhất?
Vì sao?
Recall bao nhiêu?
F1-score bao nhiêu?
Có nhiều False Negative không?
Prediction Time thế nào?
Có phù hợp triển khai thực tế không?
```

Không kết luận:

```text
Accuracy cao nhất = Model tốt nhất
```

nếu các chỉ số khác không ủng hộ kết luận đó.

---

# 17. Quy tắc khi viết báo cáo

Nội dung phải bám sát đề tài:

> **Nghiên cứu, so sánh và triển khai mô hình giám sát phát hiện xâm nhập mạng IoT**

Bố cục ưu tiên:

```text
1. Tổng quan IDS và IoT
2. Dataset Edge-IIoTset
3. Tiền xử lý dữ liệu
4. Cơ sở lý thuyết các mô hình
5. Thực nghiệm
6. So sánh kết quả
7. Lựa chọn mô hình
8. Xây dựng hệ thống giám sát
9. Đánh giá hệ thống
10. Kết luận
```

Khi viết báo cáo, phân biệt rõ:

```text
Lý thuyết
Thực nghiệm
Kết quả thực tế
Nhận xét
```

Không trình bày kết quả giả định như kết quả thực nghiệm.

---

# 18. Cách Agent phản hồi

Khi người dùng hỏi về đề tài:

* Trả lời theo đúng phạm vi IDS + IoT + Machine Learning.
* Ưu tiên giải thích dễ hiểu nhưng chính xác.
* Khi có code, giải thích những đoạn quan trọng.
* Khi có kết quả thực nghiệm, phân tích dựa trên số liệu thực tế.
* Khi chưa có dữ liệu hoặc chưa chạy model, phải nói rõ.
* Không tự bịa Accuracy, Precision, Recall, F1 hoặc ROC-AUC.
* Khi có nhiều phương án, ưu tiên phương án đơn giản, phù hợp với phạm vi đồ án.
* Không tự thêm Deep Learning nếu người dùng chưa yêu cầu.
* Không biến bài toán thành anomaly detection unsupervised nếu đề tài đang sử dụng supervised classification.

---

# 19. Mục tiêu cuối cùng

Agent phải hỗ trợ người dùng hoàn thành pipeline:

```text
Edge-IIoTset
      ↓
EDA
      ↓
Preprocessing
      ↓
Train/Test
      ↓
5 Machine Learning Models
      ↓
Evaluation
      ↓
Comparison
      ↓
Best Model
      ↓
Save Model
      ↓
Web Monitoring
      ↓
Normal / Attack Detection
```

Mục tiêu không chỉ là đạt Accuracy cao mà phải xây dựng được một **quy trình IDS hoàn chỉnh, có thể giải thích, đánh giá và triển khai**.
