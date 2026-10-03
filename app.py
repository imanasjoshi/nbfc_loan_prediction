import streamlit as st

st.set_page_config(
    page_title="NBFC Credit Risk Intelligence Platform",
    page_icon="🏦",
    layout="wide",
)

st.title("NBFC Credit Risk Intelligence & Loan Default Prediction Platform")

st.markdown("""
### Welcome to the Credit Risk Analytics Platform

This end-to-end platform provides insights into loan origination risk, portfolio monitoring, and explainable AI for credit decisions.

**Navigation:**
- 📊 **Executive Overview**: High-level portfolio KPIs and metrics.
- 📈 **Portfolio Analytics**: Deep dive into risk segments.
- 🧑‍💼 **Borrower Risk Assessment**: Predict default risk at origination for new applications.
- 🔍 **Risk Drivers & Explainability**: Understand model features driving the predictions.
- ⚙️ **Model Performance**: Evaluation metrics and threshold analysis.
- 📂 **Portfolio Explorer**: Interactive data view.
- ℹ️ **About**: Project background and methodology.

> **Note**: This application is a portfolio demonstration for predicting credit risk and understanding the drivers behind loan defaults.
""")
