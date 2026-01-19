import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from io import BytesIO
from datetime import datetime

# -----------------------------
# Load model and scaler safely
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(os.path.join(BASE_DIR, "model", "rf_smote_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "model", "scaler.pkl"))

# -----------------------------
# Feature names
# -----------------------------
feature_names = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "balance_diff",
    "type_encoded"
]

importances = model.feature_importances_

# -----------------------------
# Helper: Feature importance plot for PDF
# -----------------------------
def create_feature_importance_plot(feature_names, importances):
    fig, ax = plt.subplots(figsize=(6, 4))
    sorted_idx = np.argsort(importances)
    ax.barh(
        np.array(feature_names)[sorted_idx],
        importances[sorted_idx]
    )
    ax.set_title("Feature Importance (Fraud Model)")
    ax.set_xlabel("Importance")

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf

# -----------------------------
# Helper: PDF generator (Multi-page)
# -----------------------------
def generate_pdf_report(report_data, feature_names, importances):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # -------- PAGE 1 : SUMMARY --------
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(50, height - 50, "Fraud Detection Report")

    pdf.setFont("Helvetica", 10)
    pdf.drawString(
        50,
        height - 80,
        f"Generated On: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    y = height - 120
    pdf.setFont("Helvetica", 11)

    for key, value in report_data.items():
        pdf.drawString(50, y, f"{key}: {value}")
        y -= 18

    pdf.showPage()

    # -------- PAGE 2 : GRAPH --------
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(50, height - 50, "Model Explainability")

    pdf.setFont("Helvetica", 11)
    pdf.drawString(
        50,
        height - 80,
        "Feature Importance – Why the model made this decision"
    )

    graph_img = create_feature_importance_plot(feature_names, importances)
    pdf.drawImage(
        ImageReader(graph_img),
        50,
        height - 420,
        width=500,
        height=300
    )

    pdf.showPage()
    pdf.save()
    buffer.seek(0)
    return buffer

# -----------------------------
# App UI
# -----------------------------
st.title("💳 Fraud Detection System")
st.write("AI-powered fraud detection with explainability & risk control")

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Transaction Details")

threshold = st.sidebar.slider(
    "🎚 Fraud Probability Threshold",
    0.1, 0.9, 0.3, 0.05
)

step = st.sidebar.number_input("Step (Hour)", min_value=0)
amount = st.sidebar.number_input("Transaction Amount", min_value=0.0)
oldbalanceOrg = st.sidebar.number_input("Old Balance (Sender)", min_value=0.0)
newbalanceOrig = st.sidebar.number_input("New Balance (Sender)", min_value=0.0)
oldbalanceDest = st.sidebar.number_input("Old Balance (Receiver)", min_value=0.0)
newbalanceDest = st.sidebar.number_input("New Balance (Receiver)", min_value=0.0)

type_encoded = st.sidebar.selectbox(
    "Transaction Type",
    options=[0, 1, 2, 3, 4],
    format_func=lambda x: ["CASH-IN", "CASH-OUT", "DEBIT", "PAYMENT", "TRANSFER"][x]
)

# -----------------------------
# Feature Engineering
# -----------------------------
balance_diff = oldbalanceOrg - newbalanceOrig

input_data = np.array([[ 
    step, amount, oldbalanceOrg, newbalanceOrig,
    oldbalanceDest, newbalanceDest,
    balance_diff, type_encoded
]])

input_scaled = scaler.transform(input_data)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Fraud"):

    fraud_prob = model.predict_proba(input_scaled)[0][1]
    prediction = 1 if fraud_prob >= threshold else 0

    st.write(f"### 📊 Fraud Probability: **{fraud_prob:.2f}**")
    st.write(f"### 🎯 Threshold Used: **{threshold}**")

    # Risk Level
    st.subheader("🚨 Risk Level")
    if fraud_prob >= 0.7:
        st.error("HIGH RISK 🚨 — Immediate action required")
        risk_level = "HIGH RISK"
    elif fraud_prob >= 0.4:
        st.warning("MEDIUM RISK ⚠️ — Manual review recommended")
        risk_level = "MEDIUM RISK"
    else:
        st.success("LOW RISK ✅ — Transaction appears safe")
        risk_level = "LOW RISK"

    # Feature Importance Graph (UI)
    st.subheader("📊 Feature Importance")
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=True)

    fig, ax = plt.subplots()
    ax.barh(importance_df["Feature"], importance_df["Importance"])
    ax.set_title("Which features influence fraud prediction?")
    st.pyplot(fig)

    # Explainability
    st.subheader("🧠 AI Explanation")
    if amount > 100000:
        st.warning("• High transaction amount")
    if balance_diff > oldbalanceOrg * 0.8:
        st.warning("• Sudden balance depletion")
    if type_encoded in [1, 4]:
        st.warning("• High-risk transaction type")

    # AI Assistant
    st.subheader("🤖 AI Assistant")
    if prediction == 1:
        st.markdown(
            "**Fraud detected.** Recommended actions:\n"
            "- Block transaction\n"
            "- Trigger OTP\n"
            "- Alert fraud team"
        )
    else:
        st.markdown(
            "**Transaction is safe.** Recommended actions:\n"
            "- Approve transaction\n"
            "- Passive monitoring"
        )

    # -----------------------------
    # PDF REPORT DOWNLOAD
    # -----------------------------
    report_data = {
        "Transaction Amount": amount,
        "Transaction Type": ["CASH-IN","CASH-OUT","DEBIT","PAYMENT","TRANSFER"][type_encoded],
        "Fraud Probability": f"{fraud_prob:.2f}",
        "Threshold Used": threshold,
        "Risk Level": risk_level,
        "Model Decision": "FRAUD" if prediction == 1 else "LEGITIMATE"
    }

    pdf_file = generate_pdf_report(
        report_data,
        feature_names,
        importances
    )

    st.download_button(
        "📄 Download Fraud Report (PDF with Graph)",
        data=pdf_file,
        file_name="fraud_detection_report.pdf",
        mime="application/pdf"
    )
