import streamlit as st
import pandas as pd
import json

st.set_page_config(page_title="Model Performance", layout="wide")
st.title("Model Performance")

st.markdown("""
### Two Distinct Models
1. **Origination Risk Model:** Built using only data available at application time.
2. **Monitoring Risk Model:** Includes ongoing repayment metrics like `interest_receive` and `last_week_pay` for early warning.
""")

try:
    with open("artifacts/metrics/origination_metrics.json", "r") as f:
        orig_metrics = json.load(f)
    
    with open("artifacts/metrics/monitoring_metrics.json", "r") as f:
        mon_metrics = json.load(f)
        
    metrics_df = pd.DataFrame([
        {"Model": "Origination (Application Time)", **orig_metrics},
        {"Model": "Monitoring (Existing Loan)", **mon_metrics}
    ])
    
    st.dataframe(metrics_df.style.format(subset=["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "gini"], formatter="{:.4f}"))
    
except Exception as e:
    st.error("Metrics files not found. Please run the training pipeline.")
