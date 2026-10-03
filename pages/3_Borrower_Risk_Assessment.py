import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import joblib
import json
import os

st.set_page_config(page_title="Borrower Risk Assessment", layout="wide")
st.title("Borrower Risk Assessment (Origination)")

st.markdown("""
Use this tool to predict default risk at loan origination. 
**Note**: This model only uses data available at application time.
""")

@st.cache_resource
def load_model():
    model_path = "models/origination_model.cbm"
    if os.path.exists(model_path):
        cb = CatBoostClassifier()
        cb.load_model(model_path)
        return cb
    return None

@st.cache_data
def load_meta():
    meta_path = "models/metadata.json"
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            return json.load(f)
    return None

model = load_model()
meta = load_meta()

if model is None or meta is None:
    st.warning("Model or metadata not found. Please run the training pipeline first.")
else:
    with st.form("risk_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            loan_amnt = st.number_input("Loan Amount", value=10000)
            interest_rate = st.number_input("Interest Rate (%)", value=10.0)
            annual_income = st.number_input("Annual Income", value=60000)
            debt_to_income = st.number_input("Debt to Income", value=15.0)
            
        with col2:
            loan_grade = st.selectbox("Loan Grade", options=["A", "B", "C", "D", "E", "F", "G"])
            loan_term = st.selectbox("Loan Term", options=["3 years", "5 years"])
            home_ownership = st.selectbox("Home Ownership", options=["MORTGAGE", "RENT", "OWN"])
            job_experience = st.selectbox("Job Experience", options=["<5 Years", "6-10 years", "10+ years"])
            
        with col3:
            revolving_balance = st.number_input("Revolving Balance", value=5000)
            total_revolving_limit = st.number_input("Total Revolving Limit", value=20000)
            total_current_balance = st.number_input("Total Current Balance", value=50000)
            total_acc = st.number_input("Total Accounts", value=15)
            
        submit = st.form_submit_button("Predict Default Risk")
        
    if submit:
        # Create input df
        input_data = {
            'loan_amnt': [loan_amnt],
            'interest_rate': [interest_rate],
            'annual_income': [annual_income],
            'debt_to_income': [debt_to_income],
            'delinq_2yrs': [0],
            'public_records': [0],
            'revolving_balance': [revolving_balance],
            'total_acc': [total_acc],
            'total_current_balance': [total_current_balance],
            'total_revolving_limit': [total_revolving_limit],
            'loan_term': [loan_term],
            'loan_grade': [loan_grade],
            'loan_subgrade': [f"{loan_grade}1"],
            'job_experience': [job_experience],
            'home_ownership': [home_ownership],
            'income_verification_status': ["Source Verified"],
            'loan_purpose': ["debt_consolidation"],
            'state_code': ["CA"],
            'application_type': ["INDIVIDUAL"],
            
            # engineered
            'credit_utilization': [revolving_balance / total_revolving_limit if total_revolving_limit > 0 else 0],
            'loan_to_income': [loan_amnt / annual_income if annual_income > 0 else 0],
            'balance_to_income': [total_current_balance / annual_income if annual_income > 0 else 0],
            'revolving_to_income': [revolving_balance / annual_income if annual_income > 0 else 0],
            'loan_term_num': [3 if loan_term == "3 years" else 5]
        }
        
        df_input = pd.DataFrame(input_data)
        features = meta['origination_features']
        
        # Ensure all columns exist
        for col in features:
            if col not in df_input.columns:
                df_input[col] = "Missing" if col in meta['cat_cols'] else 0
                
        df_input = df_input[features]
        
        for col in meta['cat_cols']:
            if col in df_input.columns:
                df_input[col] = df_input[col].astype(str)
                
        prob = model.predict_proba(df_input)[0, 1]
        
        st.markdown("---")
        st.subheader("Prediction Results")
        
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.metric("Probability of Default", f"{prob*100:.1f}%")
            
        with col_res2:
            if prob < 0.15:
                risk = "Low"
                color = "green"
            elif prob < 0.30:
                risk = "Moderate"
                color = "orange"
            elif prob < 0.50:
                risk = "High"
                color = "red"
            else:
                risk = "Very High"
                color = "darkred"
            st.markdown(f"### Risk Band: <span style='color:{color}'>{risk}</span>", unsafe_allow_html=True)
            
        st.info("This model is an educational portfolio demonstration and should not be used to make real lending decisions.")
