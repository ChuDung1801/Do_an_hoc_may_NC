# Nghiên cứu, So sánh và Triển khai Mô hình Giám sát Phát hiện Xâm nhập Mạng IoT

Tài liệu ôn tập trọng tâm cho đề tài IDS trên bộ dữ liệu Edge-IIoTset. Không cần học lan man toàn bộ Machine Learning — tập trung vào 10 phần dưới đây.

## Mục lục

1. [Kiến thức nền về bài toán IDS](#1-kiến-thức-nền-về-bài-toán-ids)
2. [Kiến thức Machine Learning cần nắm](#2-kiến-thức-machine-learning-cần-nắm)
3. [Các mô hình phải đặc biệt chú ý](#3-các-mô-hình-phải-đặc-biệt-chú-ý)
4. [Phần đánh giá mô hình](#4-phần-đánh-giá-mô-hình--cực-kỳ-quan-trọng)
5. [Những kiến thức nâng cao nên có](#5-những-kiến-thức-nâng-cao-nên-có)
6. [Data Leakage](#6-data-leakage)
7. [Cross Validation](#7-cross-validation)
8. [Feature Selection / Feature Importance](#8-feature-selection--feature-importance)
9. [Kiến trúc code nên hiểu](#9-kiến-trúc-code-nên-hiểu)
10. [Mức độ ưu tiên học](#10-nếu-học-theo-mức-độ-ưu-tiên)

---

## 1. Kiến thức nền về bài toán IDS

Cần hiểu trước khi vào model:

- **IoT Network**: thiết bị, gateway, traffic, flow, packet.
- **Intrusion Detection System (IDS)** là gì.
- Phân biệt:
  - Normal traffic
  - Attack traffic
  - Anomaly Detection
  - Intrusion Detection
- Các dạng tấn công trong Edge-IIoTset: DoS/DDoS, Scanning, Injection, Malware, MITM, Password, XSS/SQL Injection... tùy nhãn được sử dụng.
- Bài toán của nhóm là:
  - **Binary Classification**: Normal / Attack
  - Có thể mở rộng thành **Multi-class Classification**: xác định loại attack.

> ⚠️ Điểm quan trọng khi bảo vệ: đừng nói đề tài là anomaly detection thuần túy, vì các model chính đang học từ dữ liệu có nhãn → đây chủ yếu là **Supervised Learning** cho IDS.

---

## 2. Kiến thức Machine Learning cần nắm

### A. Tiền xử lý dữ liệu

Đây gần như chắc chắn sẽ bị hỏi.

**Missing value**
- Phát hiện giá trị thiếu
- Xóa dòng/cột
- Thay thế bằng mean/median/mode

**Duplicate**
- Biết tại sao phải xử lý dữ liệu trùng.

**Encoding**

Edge-IIoTset có các thuộc tính dạng categorical. Cần hiểu:
- Label Encoding
- One-Hot Encoding

Ví dụ: `Protocol` (TCP, UDP, ICMP) → chuyển thành dạng số để model xử lý.

**Feature Scaling**

Đặc biệt cần hiểu **StandardScaler**:

```
z = (x - μ) / σ
```

Mục đích: đưa các feature về scale tương đồng.

Cực kỳ quan trọng với:
- Logistic Regression
- SVM

Trong khi Decision Tree, Random Forest, XGBoost thường **không bắt buộc** scaling.

> Đây là một câu hỏi bảo vệ rất dễ xuất hiện.

---

## 3. Các mô hình phải đặc biệt chú ý

Thứ tự: Logistic Regression → Decision Tree → SVM → Random Forest → XGBoost

### ⭐ Logistic Regression (baseline)

Cần biết: Sigmoid, xác suất dự đoán, decision boundary, binary classification, regularization.

```
P(y=1|x) = 1 / (1 + e^(-z))
```

Vai trò: mô hình đơn giản để làm mốc so sánh với các mô hình phức tạp hơn. Không cần đào quá sâu toán tối ưu nếu môn không yêu cầu.

### ⭐⭐ Decision Tree

Cần nắm: Node, Root, Branch, Leaf, splitting, Gini Impurity, Entropy, Information Gain, max_depth, overfitting.

```
          Packet Count > X?
             /       \
           Yes        No
           /           \
      Attack          Normal
```

Model rất dễ dùng để giải thích với giảng viên.

### ⭐⭐⭐ Random Forest

Có thể là một trong những model mạnh nhất trong bài.

**Bagging** — thay vì một Decision Tree:

```
Data
 ├── Tree 1
 ├── Tree 2
 ├── Tree 3
 ├── ...
 └── Tree N
```

Mỗi cây học trên một sample khác nhau → kết quả cuối là **voting** giữa nhiều cây.

Cần hiểu thêm: Random feature selection, Ensemble Learning, Bagging, giảm overfitting, feature importance.

```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42
)
```

Hyperparameter nên biết: `n_estimators`, `max_depth`, `min_samples_split`, `max_features`.

### ⭐⭐⭐ XGBoost

Model nên đầu tư nhiều nhất nếu muốn bài có chất Học máy nâng cao. Không cần học từng dòng source code, nhưng phải hiểu **Gradient Boosting**.

Khác Random Forest — các cây tương đối độc lập:

```
Random Forest
Tree 1 ──┐
Tree 2 ──┤
Tree 3 ──┼──> Voting
Tree 4 ──┤
Tree N ──┘
```

XGBoost — các cây được xây dựng tuần tự, cây sau sửa lỗi của cây trước:

```
Tree 1 → sai ở đâu? → Tree 2 học phần sai → sai tiếp ở đâu? → Tree 3 → ...
```

Cần hiểu: Boosting, Gradient Boosting, Loss function, Learning rate, Number of estimators, Tree depth, Regularization, Overfitting.

Hyperparameter quan trọng: `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`.

> **Câu hỏi bảo vệ thường gặp:** "Tại sao XGBoost thường mạnh hơn Decision Tree đơn?"
> **Trả lời:** Vì XGBoost sử dụng nhiều cây theo cơ chế boosting, các cây sau học cách giảm lỗi còn lại của các cây trước, đồng thời có cơ chế regularization giúp kiểm soát overfitting.

### ⭐⭐⭐ SVM

Cần hiểu về mặt ý tưởng, không nhất thiết đào toán quá sâu.

**Hyperplane** — SVM tìm một siêu phẳng phân chia các lớp, mục tiêu là tối đa hóa margin:

```
Attack       |       Normal
  ● ● ●      |      ○ ○ ○
  ● ● ●      |      ○ ○ ○
```

Cần biết: Hyperplane, Margin, Support Vector, Kernel, C, Gamma.

Các kernel: Linear, RBF, Polynomial.

> SVM rất nhạy với scale dữ liệu, vì vậy thường cần StandardScaler.

---

## 4. Phần đánh giá mô hình — cực kỳ quan trọng

Có thể còn quan trọng hơn việc biết code model.

### Confusion Matrix

|  | Pred Normal | Pred Attack |
|---|---|---|
| **Actual Normal** | TN | FP |
| **Actual Attack** | FN | TP |

Đặc biệt với IDS:
- **False Positive**: traffic bình thường nhưng hệ thống báo Attack → báo động giả.
- **False Negative**: traffic Attack nhưng hệ thống dự đoán Normal → rất nguy hiểm.

Do đó **Accuracy không đủ**.

### Precision

```
Precision = TP / (TP + FP)
```
Trả lời: trong những mẫu hệ thống báo Attack, có bao nhiêu mẫu thực sự là Attack?

### Recall

```
Recall = TP / (TP + FN)
```
Trả lời: trong toàn bộ Attack thực tế, hệ thống bắt được bao nhiêu?

> Với IDS: Recall rất quan trọng, vì bỏ sót một cuộc tấn công nguy hiểm hơn báo nhầm.

### F1-score

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
Dùng để cân bằng Precision và Recall.

### ROC-AUC

Phải hiểu: ROC curve, TPR, FPR, AUC. AUC càng gần 1 → khả năng phân biệt hai lớp càng tốt.

---

## 5. Những kiến thức nâng cao nên có

Giúp đề tài nhìn giống Học máy nâng cao chứ không chỉ là "train 5 model rồi lấy Accuracy".

### Data Imbalance

Dataset IDS thường mất cân bằng. Phải biết: class imbalance, minority class, majority class, **SMOTE** (tạo thêm mẫu cho lớp thiểu số).

> ⚠️ **Không được SMOTE trước khi chia Train/Test.**

Đúng:
```
Original Data → Train/Test → SMOTE trên Train → Train model → Test trên Test gốc
```

Nếu SMOTE trước khi chia → nguy cơ **data leakage**.

---

## 6. Data Leakage

Kiến thức rất đáng học vì thầy có thể hỏi.

Sai:
```
Scale toàn bộ dataset → chia train/test   (❌ không tốt)
```

Đúng:
```
Train → fit scaler → transform Train
Test  → transform bằng scaler của Train
```

Không được để thông tin từ Test "lọt" vào quá trình training.

---

## 7. Cross Validation

Nên biết **5-Fold Cross Validation**:

```
Dataset
 ├── Fold 1
 ├── Fold 2
 ├── Fold 3
 ├── Fold 4
 └── Fold 5
```

Mỗi lần dùng 4 fold → Train, 1 fold → Validation, rồi xoay vòng. Cuối cùng lấy trung bình.

Giúp đánh giá model ổn định hơn thay vì phụ thuộc vào một lần chia dữ liệu.

---

## 8. Feature Selection / Feature Importance

Nên có trong đề tài.

- **Random Forest**: có `feature_importances_`
- **XGBoost**: có nhiều cách đánh giá feature importance

Có thể trả lời: những đặc trưng nào ảnh hưởng nhiều nhất đến việc nhận diện Attack?

```
Feature A ██████████
Feature B ████████
Feature C █████
Feature D ███
```

> Nếu muốn nâng cấp đề tài nữa thì dùng **SHAP** để giải thích mô hình.

---

## 9. Kiến trúc code nên hiểu

```
dataset/
    Edge-IIoTset.csv

preprocessing/
    preprocess.py

models/
    logistic.py
    decision_tree.py
    svm.py
    random_forest.py
    xgboost.py

evaluation/
    evaluate.py
    plots.py

web/
    app.py

saved_models/
    best_model.pkl
    scaler.pkl
    encoder.pkl
```

### Pipeline

```
Edge-IIoTset
      ↓
Data Cleaning
      ↓
Encoding
      ↓
Feature Scaling
      ↓
Train/Test Split
      ↓
SMOTE (Train)
      ↓
Train 5 Models
      ↓
Evaluation
      ↓
Choose Best Model
      ↓
Save Model
      ↓
Flask / Streamlit
      ↓
Upload Data
      ↓
Prediction
      ↓
Normal / Attack
```

---

## 10. Nếu học theo mức độ ưu tiên

| Kiến thức | Mức độ |
|---|---|
| Confusion Matrix | 🔥🔥🔥🔥🔥 |
| Precision / Recall / F1 | 🔥🔥🔥🔥🔥 |
| Data preprocessing | 🔥🔥🔥🔥🔥 |
| Decision Tree | 🔥🔥🔥🔥 |
| Random Forest | 🔥🔥🔥🔥🔥 |
| XGBoost | 🔥🔥🔥🔥🔥 |
| SVM | 🔥🔥🔥🔥 |
| Logistic Regression | 🔥🔥🔥 |
| Imbalanced Data / SMOTE | 🔥🔥🔥🔥 |
| Data Leakage | 🔥🔥🔥🔥🔥 |
| Cross Validation | 🔥🔥🔥🔥 |
| Feature Importance | 🔥🔥🔥🔥 |
| SHAP | 🔥🔥🔥 |
| Streamlit/Flask | 🔥🔥 |
| IoT/IDS fundamentals | 🔥🔥🔥🔥 |

**Lộ trình học trọng tâm nếu ít thời gian:**

Edge-IIoTset → preprocessing → train/test → Logistic Regression → Decision Tree → Random Forest → SVM → XGBoost → Confusion Matrix → Precision/Recall/F1 → Cross Validation → SMOTE → Feature Importance → triển khai Streamlit.

> Trong 5 model, **XGBoost + Random Forest + Decision Tree** là ba cái nên hiểu sâu nhất; **Logistic Regression** làm baseline, còn **SVM** cần nắm chắc nguyên lý và lý do nó cần scaling.
