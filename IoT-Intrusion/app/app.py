"""
app.py
------
Web giám sát IDS bằng Streamlit (SKILL.md mục 13).

Chức năng tối thiểu:
Upload CSV -> Preprocessing -> Prediction -> Normal/Attack ->
Statistics -> Charts -> (tuỳ chọn) Lịch sử dự đoán.

Chạy:
    streamlit run app/app.py
"""

import os
import sys

import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.helpers import load_artifacts  # noqa: E402

ARTIFACTS_DIR = os.environ.get("IDS_ARTIFACTS_DIR", "artifacts/model")

st.set_page_config(page_title="IoT IDS Monitoring", layout="wide")
st.title("🛰️ Hệ thống giám sát phát hiện xâm nhập mạng IoT")
st.caption("Edge-IIoTset | Normal vs Attack")

if "history" not in st.session_state:
    st.session_state.history = []


@st.cache_resource
def get_artifacts(artifacts_dir: str):
    return load_artifacts(artifacts_dir)


def preprocess_for_inference(df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """
    Áp dụng lại đúng preprocessing đã fit khi training:
    align cột theo feature_columns, encode, scale bằng scaler đã lưu.

    Nếu cấu trúc cột đầu vào khác với lúc train, thông báo lỗi rõ ràng
    thay vì âm thầm suy đoán.
    """
    metadata = artifacts.get("metadata", {})
    feature_columns = metadata.get("feature_columns", [])

    df_proc = pd.get_dummies(df)
    df_proc = df_proc.reindex(columns=feature_columns, fill_value=0)

    scaler = artifacts.get("scaler")
    if scaler is not None:
        df_proc = pd.DataFrame(
            scaler.transform(df_proc), columns=df_proc.columns, index=df_proc.index
        )
    return df_proc


try:
    artifacts = get_artifacts(ARTIFACTS_DIR)
    model_loaded = True
except FileNotFoundError:
    artifacts = None
    model_loaded = False
    st.warning(
        f"Chưa tìm thấy model đã huấn luyện trong `{ARTIFACTS_DIR}`. "
        f"Hãy chạy `python -m src.train --data <file> --label <cột nhãn>` trước."
    )

uploaded_file = st.file_uploader("📤 Upload file CSV dữ liệu mạng để phân tích", type=["csv"])

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    st.subheader("Xem trước dữ liệu")
    st.dataframe(raw_df.head(20))

    predict_clicked = st.button("🔍 Predict", disabled=not model_loaded)

    if predict_clicked and model_loaded:
        X = preprocess_for_inference(raw_df, artifacts)
        model = artifacts["model"]
        label_encoder = artifacts.get("label_encoder")

        preds = model.predict(X)
        if label_encoder is not None:
            preds_label = label_encoder.inverse_transform(preds)
        else:
            preds_label = preds

        result_df = raw_df.copy()
        result_df["Prediction"] = preds_label

        total = len(result_df)
        is_attack = result_df["Prediction"].astype(str).str.lower() != "normal"
        n_normal = int((~is_attack).sum())
        n_attack = int(is_attack.sum())
        attack_ratio = n_attack / total if total else 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Tổng số mẫu", total)
        col2.metric("Normal", n_normal)
        col3.metric("Attack", n_attack)
        col4.metric("Tỷ lệ Attack", f"{attack_ratio:.1%}")

        st.subheader("📊 Phân bố kết quả dự đoán")
        st.bar_chart(result_df["Prediction"].value_counts())

        st.subheader("📋 Bảng kết quả")
        st.dataframe(result_df)

        st.download_button(
            "⬇️ Tải kết quả CSV",
            data=result_df.to_csv(index=False).encode("utf-8"),
            file_name="prediction_results.csv",
            mime="text/csv",
        )

        st.session_state.history.append(
            {
                "file": uploaded_file.name,
                "total": total,
                "n_normal": n_normal,
                "n_attack": n_attack,
                "attack_ratio": attack_ratio,
            }
        )

if st.session_state.history:
    st.subheader("🕘 Lịch sử dự đoán (phiên hiện tại)")
    st.dataframe(pd.DataFrame(st.session_state.history))
